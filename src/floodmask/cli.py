"""
Command-line entry point: `floodmask --out outputs`.
"""

import argparse
import logging
from pathlib import Path

from floodmask.config import NATIVE_RES_M, Config
from floodmask import pipeline


def parse_args(argv=None) -> Config:
    d = Config()
    p = argparse.ArgumentParser(prog="floodmask", description=__doc__)
    p.add_argument("--out", dest="out_dir", type=Path, default=None,
                   help=f"output directory (default {d.out_dir}, or {d.out_dir}_<res>m if --resolution is not 10)")
    p.add_argument("--resolution", dest="resolution_m", type=float, default=d.resolution_m,
                   help=f"output pixel size in metres, a multiple of {NATIVE_RES_M:g}; larger values "
                        f"block-average linear sigma0 (10 = no resampling)")
    p.add_argument("--cache", dest="cache_dir", type=Path, default=d.cache_dir, help="download/model cache")
    p.add_argument("--bbox", dest="aoi_bbox", nargs=4, type=float, default=d.aoi_bbox,
                   metavar=("MINX", "MINY", "MAXX", "MAXY"), help="AOI in WGS84")
    p.add_argument("--t1-date", default=d.t1_date)
    p.add_argument("--orbit", dest="relative_orbit", type=int, default=d.relative_orbit)
    p.add_argument("--patch-size", type=int, default=d.patch_size)
    p.add_argument("--overlap", type=int, default=d.overlap)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--threshold", dest="water_threshold", type=float, default=d.water_threshold,
                   help="water probability threshold")
    p.add_argument("--min-area-m2", type=float, default=d.min_area_m2,
                   help="minimum polygon (and hole) size in the GeoJSON outputs, m2")
    p.add_argument("--dem-filter", action="store_true",
                   help="remove PWB/flood pixels on slopes (Copernicus DEM GLO-30)")
    p.add_argument("--max-slope-deg", type=float, default=d.max_slope_deg,
                   help="slope above which the DEM filter removes water, degrees")
    p.add_argument("--device", default=d.device, help="cpu (default) or cuda (debug)")
    a = p.parse_args(argv)
    if a.resolution_m < NATIVE_RES_M or a.resolution_m % NATIVE_RES_M:
        p.error(f"--resolution must be a multiple of {NATIVE_RES_M:g} m, got {a.resolution_m:g}")
    if a.out_dir is None:
        a.out_dir = d.out_dir if a.resolution_m == NATIVE_RES_M else Path(f"{d.out_dir}_{a.resolution_m:g}m")

    cfg = Config()
    for k, v in vars(a).items():
        setattr(cfg, k, v)
    cfg.aoi_bbox = tuple(cfg.aoi_bbox)
    return cfg


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pipeline.run(parse_args(argv))


if __name__ == "__main__":
    main()
