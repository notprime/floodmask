"""
Georeference: GRD slices (radar geometry + GCPs) -> one calibrated sigma0 dB
stack on the common output grid (cfg.target_crs, 10 m native, AOI), optionally multilooked.

Per slice:
  1. find the radar-geometry window that covers the AOI, from the GCPs (+ margin)
  2. read only that window; mask border noise and calibrate DN -> noise-removed
     sigma0 there, where the LUTs are defined
  3. warp to the output grid: thin-plate spline through the GCPs, bilinear
     resampling of linear sigma0 (averaging power, not dB)
Slices of one pass are mosaicked (first valid pixel wins), optionally multilooked
to a coarser resolution (linear power), then converted to dB.

T0 and T1 land on the identical grid, so change detection is pixel-to-pixel.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.transform import Affine, GCPTransformer
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window

from floodmask.config import NATIVE_RES_M, Config
from floodmask.preprocess import CHANNELS, border_noise, calibrate, read_noise_lut, read_sigma_lut, to_db

log = logging.getLogger(__name__)

MARGIN_PX = 256  # extra radar pixels around the AOI window, covers GCP-fit error
MIN_VALID_FRACTION = 0.5  # multilook: output pixel is nodata below this share of valid input pixels


@dataclass
class Grid:
    crs: str
    transform: Affine
    width: int
    height: int


def target_grid(cfg: Config) -> Grid:
    """AOI bbox in the target CRS, snapped outward to whole native (10 m) pixels."""
    res = NATIVE_RES_M
    left, bottom, right, top = transform_bounds("EPSG:4326", cfg.target_crs, *cfg.aoi_bbox)
    left, bottom = math.floor(left / res) * res, math.floor(bottom / res) * res
    right, top = math.ceil(right / res) * res, math.ceil(top / res) * res
    return Grid(cfg.target_crs, Affine(res, 0, left, 0, -res, top),
                round((right - left) / res), round((top - bottom) / res))


def coarsen_grid(grid: Grid, factor: int) -> Grid:
    """Same CRS and upper-left corner, pixels `factor` times larger. The last row/column
    of blocks may extend past the original grid (see multilook)."""
    t = grid.transform
    return Grid(grid.crs, Affine(t.a * factor, 0, t.c, 0, t.e * factor, t.f),
                math.ceil(grid.width / factor), math.ceil(grid.height / factor))


def multilook(sigma0: np.ndarray, factor: int) -> np.ndarray:
    """(C, H, W) linear sigma0 -> (C, ceil(H/f), ceil(W/f)) mean over f x f blocks of valid pixels.

    Block mean rather than rasterio reproject(Resampling.average): for an integer factor
    on an already aligned grid it is exact, and it needs only numpy.
    Must be fed linear power: averaging dB is a geometric mean and biases low.
    Nodata (NaN) is left out of the mean; a block with less than MIN_VALID_FRACTION valid
    pixels is NaN. Edges: H, W are padded with NaN up to a multiple of `factor` (not cropped,
    so the AOI border is kept); a partial block is averaged over its real pixels only.
    """
    c, h, w = sigma0.shape
    ph, pw = -h % factor, -w % factor
    padded = np.pad(sigma0, ((0, 0), (0, ph), (0, pw)), constant_values=np.nan)
    blocks = padded.reshape(c, (h + ph) // factor, factor, (w + pw) // factor, factor)
    valid = np.isfinite(blocks)
    n = valid.sum(axis=(2, 4))
    total = np.where(valid, blocks, 0).sum(axis=(2, 4), dtype=np.float64)
    # Fraction of the full block, so padded cells count as missing too.
    enough = n >= MIN_VALID_FRACTION * factor * factor
    return np.where(enough, total / np.maximum(n, 1), np.nan).astype(np.float32)


def _check(src) -> dict:
    """Validate the metadata we rely on, before reading any pixels."""
    gcps, gcp_crs = src.gcps
    meta = {"width": src.width, "height": src.height, "count": src.count, "dtype": src.dtypes[0],
            "nodata": src.nodata, "gcp_count": len(gcps), "gcp_crs": str(gcp_crs)}
    if src.count != 1 or src.dtypes[0] != "uint16":
        raise ValueError(f"{src.name}: expected 1 band of uint16 DN, got {meta}")
    if not gcps or gcp_crs != CRS.from_epsg(4326):
        raise ValueError(f"{src.name}: expected GCPs in EPSG:4326 (radar geometry), got {meta}")
    return meta


def _aoi_window(gcps, bbox, width: int, height: int) -> Window | None:
    """Radar-geometry window covering the WGS84 bbox, or None if the slice misses it."""
    minx, miny, maxx, maxy = bbox
    with GCPTransformer(gcps) as tr:
        rows, cols = tr.rowcol([minx, maxx, maxx, minx], [miny, miny, maxy, maxy])
    r0, r1 = max(int(min(rows)) - MARGIN_PX, 0), min(int(max(rows)) + MARGIN_PX, height)
    c0, c1 = max(int(min(cols)) - MARGIN_PX, 0), min(int(max(cols)) + MARGIN_PX, width)
    if r0 >= r1 or c0 >= c1:
        return None
    return Window(c0, r0, c1 - c0, r1 - r0)


def _slice_sigma0(paths: dict[str, Path], bbox, grid: Grid) -> tuple[np.ndarray | None, dict]:
    """One GRD slice -> (2, H, W) linear sigma0 on the grid (NaN where not covered)."""
    out = np.full((len(CHANNELS), grid.height, grid.width), np.nan, np.float32)
    info = {}
    for i, pol in enumerate(c.lower() for c in CHANNELS):
        with rasterio.open(paths[pol]) as src:
            info[pol] = _check(src)
            gcps, gcp_crs = src.gcps
            win = _aoi_window(gcps, bbox, src.width, src.height)
            if win is None:
                return None, info
            dn = src.read(1, window=win)
            border = border_noise(dn, near=win.col_off == 0, far=win.col_off + win.width == src.width)
        sigma0 = calibrate(dn, read_sigma_lut(paths[f"schema-calibration-{pol}"], pol),
                           read_noise_lut(paths[f"schema-noise-{pol}"], pol), win.row_off, win.col_off)
        sigma0[border] = np.nan
        # GCPs refer to the full scene; shift them into the window.
        local = [GroundControlPoint(g.row - win.row_off, g.col - win.col_off, g.x, g.y, g.z) for g in gcps]
        # SRC_METHOD reaches GDAL's transformer (thin-plate spline is used); rasterio also
        # forwards it to the warp options, which log "not supported": harmless.
        reproject(sigma0, out[i], gcps=local, src_crs=gcp_crs, src_nodata=np.nan,
                  dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan,
                  resampling=Resampling.bilinear, SRC_METHOD="GCP_TPS")
        info[pol]["window"] = {"col_off": win.col_off, "row_off": win.row_off,
                               "width": win.width, "height": win.height}
        info[pol]["border_noise_px"] = int(border.sum())
    return out, info


def load_pass(slices: dict[str, dict[str, Path]], bbox, grid: Grid, factor: int = 1) -> tuple[np.ndarray, dict]:
    """{item_id: asset paths} of one acquisition -> ((2, H, W) sigma0 dB, manifest info).
    `grid` is the native grid; with factor > 1 the result is multilooked to factor x its pixel size."""
    sigma0 = np.full((len(CHANNELS), grid.height, grid.width), np.nan, np.float32)
    info = {"slices": {}}
    for item_id, paths in sorted(slices.items()):
        s, info["slices"][item_id] = _slice_sigma0(paths, bbox, grid)
        if s is None:
            log.info("%s does not cover the AOI, skipped", item_id)
            continue
        empty = np.isnan(sigma0)
        sigma0[empty] = s[empty]

    if factor > 1:
        sigma0 = multilook(sigma0, factor)  # linear power, before dB
    db = to_db(sigma0)
    valid = np.isfinite(db).all(axis=0)
    if not valid.any():
        raise RuntimeError("No valid SAR pixel inside the AOI")
    info["valid_fraction"] = float(valid.mean())
    info["db_p2_p50_p98"] = {c: [round(float(v), 2) for v in np.percentile(db[i][valid], [2, 50, 98])]
                             for i, c in enumerate(CHANNELS)}
    return db, info


def write_raster(path: Path, data: np.ndarray, grid: Grid, nodata, names: tuple[str, ...] | None = None) -> None:
    """(H, W) or (C, H, W) array -> tiled, compressed GeoTIFF on the grid, in the array's dtype."""
    data = data[None] if data.ndim == 2 else data
    profile = {"driver": "GTiff", "width": grid.width, "height": grid.height, "count": data.shape[0],
               "dtype": data.dtype.name, "crs": grid.crs, "transform": grid.transform, "nodata": nodata,
               "tiled": True, "blockxsize": 512, "blockysize": 512, "compress": "deflate"}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if names:
            dst.descriptions = names
