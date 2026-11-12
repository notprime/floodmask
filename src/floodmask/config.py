"""
All run parameters in one place. 
Default is the Bolivia event.

The CLI overrides individual fields; the final Config is dumped into
run_manifest.json so every run is reproducible from its manifest.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path


NATIVE_RES_M = 10.0   # pixel size of the GCP warp, the model's training resolution


@dataclass
class Config:
    # Event (from the Sen1Floods11 metadata)
    event: str = "Bolivia"
    t1_date: str = "2018-02-15"
    orbit_state: str = "descending"
    relative_orbit: int = 156
    # Beni river sub-AOI (Sen1Floods11 event area), WGS84 (minx, miny, maxx, maxy)
    aoi_bbox: tuple[float, float, float, float] = (-65.35, -14.65, -65.05, -14.35)
    t0_search_days: int = 60   # how far back to look for the pre-event scene

    # Data source
    stac_url: str = "https://earth-search.aws.element84.com/v1"
    collection: str = "sentinel-1-grd"
    s3_region: str = "eu-central-1"   # region of the GRD asset bucket

    # Output grid. The warp from radar geometry always happens at NATIVE_RES_M;
    # resolution_m is the final pixel size (a multiple of it, block-averaged).
    target_crs: str = "EPSG:32720"   # UTM 20S covers the sub-AOI
    resolution_m: float = 10.0

    # Model
    hf_repo: str = "Governor6191/sar-flood-extent-unet-resnet34"
    device: str = "cpu"

    # Patch inference contract
    patch_size: int = 512
    overlap: int = 64
    batch_size: int = 4
    water_threshold: float = 0.5   # on the blended probability; 0.5 = argmax of the 2 classes

    # Change detection: minimum mapping unit of the GeoJSON polygons.
    # 2500 m2 = 0.25 ha (25 px at 10 m), about 5 GRD resolution cells (~20 x 22 m each).
    # In metres so it means the same thing at any --resolution.
    min_area_m2: float = 2500.0

    # Optional: Copernicus DEM false-positive filter (off by default)
    dem_filter: bool = False
    dem_collection: str = "cop-dem-glo-30"
    max_slope_deg: float = 5.0   # PWB / flood pixels steeper than this are removed

    # Paths
    out_dir: Path = Path("outputs")
    cache_dir: Path = Path("cache")   # downloaded GRD assets, model weights

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: str(v) if isinstance(v, Path) else v for k, v in d.items()}
