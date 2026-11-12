"""
Copernicus DEM false-positive filter (optional, `--dem-filter`).

SAR calls a pixel water when it is dark. Some dark pixels are not water:
radar shadow behind steep slopes is dark on both dates, and still water
cannot stand on a slope anyway. Pixels steeper than `max_slope_deg` are
removed from both outputs (PWB and flood), so a false positive cannot move
from one to the other.

DEM: Copernicus GLO-30 (Earth Search `cop-dem-glo-30`, 1 arcsec ~ 30 m,
EPSG:4326), the tiles covering the AOI, resampled bilinear to the output
grid. Slope from central differences on that grid (metres, 10 m spacing).
Where there is no DEM the slope is NaN and nothing is removed.

Caveat: GLO-30 is a surface model (TanDEM-X X-band), so forest canopy edges
read as steep slopes even on flat ground.
"""

import logging
import urllib.request
from pathlib import Path

import boto3
import numpy as np
import rasterio
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig
from pystac_client import Client
from rasterio.warp import Resampling, reproject

from floodmask.config import Config
from floodmask.georef import Grid

log = logging.getLogger(__name__)

ASSET = "data"  # the DEM COG in each cop-dem-glo-30 item


def _download(href: str, dest: Path, region: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")  # no half-written file on interruption
    log.info("Downloading %s", href)
    if href.startswith("s3://"):
        bucket, _, key = href.removeprefix("s3://").partition("/")
        s3 = boto3.client("s3", region_name=region, config=BotoConfig(signature_version=UNSIGNED))
        s3.download_file(bucket, key, str(part))
    else:
        urllib.request.urlretrieve(href, part)
    part.rename(dest)


def fetch(cfg: Config) -> tuple[list[Path], dict]:
    """Find and download (once, into the cache) the DEM tiles covering the AOI."""
    items = list(Client.open(cfg.stac_url).search(collections=[cfg.dem_collection], bbox=cfg.aoi_bbox).items())
    if not items:
        raise RuntimeError(f"No {cfg.dem_collection} tile covers {cfg.aoi_bbox}")
    paths = []
    for it in items:
        href = it.assets[ASSET].href
        dest = cfg.cache_dir / "dem" / Path(href).name
        if not dest.exists():
            _download(href, dest, cfg.s3_region)
        paths.append(dest)
    return paths, {"collection": cfg.dem_collection, "item_ids": sorted(it.id for it in items)}


def slope_on_grid(dem_paths: list[Path], grid: Grid) -> np.ndarray:
    """DEM tiles -> slope in degrees on the output grid, float32; NaN where no DEM."""
    elev = np.full((grid.height, grid.width), np.nan, np.float32)
    for path in dem_paths:
        tile = np.full_like(elev, np.nan)
        with rasterio.open(path) as src:
            reproject(rasterio.band(src, 1), tile, dst_transform=grid.transform, dst_crs=grid.crs,
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
        elev = np.where(np.isnan(elev), tile, elev)  # tiles only touch; first valid wins
    dz_dy, dz_dx = np.gradient(elev, abs(grid.transform.e), grid.transform.a)
    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)
