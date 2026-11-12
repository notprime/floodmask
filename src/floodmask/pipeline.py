"""Orchestrates the stages in order. Each stage lives in its own module."""

import json
import logging
import platform
import time
from datetime import datetime, timezone

from floodmask import __version__, change, dem, download, georef, inference, stac
from floodmask.config import NATIVE_RES_M, Config
from floodmask.model import FloodDetector
from floodmask.preprocess import BORDER_DN, BORDER_SMOOTH_PX, CHANNELS, load_norm_constants

log = logging.getLogger(__name__)


def run(cfg: Config) -> None:
    run_start = time.perf_counter()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("Config: %s", cfg.to_dict())
    manifest = {"run": {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "floodmask_version": __version__, "python": platform.python_version()},
                "config": cfg.to_dict()}

    # Stage 1, discovery: pick the T0 / T1 acquisitions and prove we can read them
    items, manifest["stac"] = stac.discover(cfg)

    # Stage 2, model: frozen detector + the input contract it expects
    detector = FloodDetector(cfg.hf_repo, cfg.cache_dir / "model", cfg.device)
    manifest["model"] = detector.info

    # Stage 3, preprocessing: full GRD download -> calibrated sigma0 dB on one common grid
    native = georef.target_grid(cfg)   # the warp always runs at 10 m ...
    factor = round(cfg.resolution_m / NATIVE_RES_M)
    grid = georef.coarsen_grid(native, factor) if factor > 1 else native   # ... the outputs are on this grid
    manifest["grid"] = {"crs": grid.crs, "transform": list(grid.transform)[:6],
                        "width": grid.width, "height": grid.height}
    manifest["resampling"] = {
        "target_resolution_m": cfg.resolution_m,
        "native_resolution_m": NATIVE_RES_M,
        "factor": factor,
        "patch_footprint_km": round(cfg.patch_size * cfg.resolution_m / 1000, 2)}
    if factor > 1:
        manifest["resampling"].update({
            "method": f"area average (block mean, {factor}x{factor} = {factor ** 2} looks) of the 10 m "
                      "grid after the GCP warp and slice mosaic, before dB",
            "domain": "linear sigma0 (power), never dB",
            "nodata_rule": f"NaN pixels excluded from the mean; output is NaN if fewer than "
                           f"{georef.MIN_VALID_FRACTION:.0%} of the {factor ** 2} input pixels are valid",
            "edges": f"native grid {native.height}x{native.width} px padded with NaN to a multiple of {factor} "
                     "(not cropped); partial blocks averaged over their valid pixels, same valid-fraction rule",
            "native_grid": {"transform": list(native.transform)[:6], "width": native.width, "height": native.height},
            "output_grid": manifest["grid"]})
    else:
        manifest["resampling"]["method"] = "none (native 10 m)"
    manifest["sar"] = {}
    for tag in ("t0", "t1"):
        slices = {it.id: download.fetch(it, cfg.cache_dir / "grd", cfg.s3_region) for it in items[tag]}
        db, manifest["sar"][tag] = georef.load_pass(slices, cfg.aoi_bbox, native, factor)
        # Change detection is pixel to pixel: both dates must be on the one output grid.
        assert db.shape == (len(CHANNELS), grid.height, grid.width), (tag, db.shape, grid)
        path = cfg.out_dir / f"sigma0_db_{tag}.tif"
        georef.write_raster(path, db, grid, nodata=float("nan"), names=CHANNELS)
        log.info("%s: %s (valid %.1f%%, median dB %s)", tag.upper(), path,
                 100 * manifest["sar"][tag]["valid_fraction"],
                 {c: v[1] for c, v in manifest["sar"][tag]["db_p2_p50_p98"].items()})

    manifest["preprocessing"] = {
        "channels": list(CHANNELS),
        "calibration": "sigma0 = max(DN^2 - N, 0) / K^2, K = sigmaNought LUT (calibration-iw-{pol}.xml), "
                       "N = thermal noise LUT (noise-iw-{pol}.xml), both bilinear in (line, pixel); "
                       "dB = 10 log10(max(sigma0, 1e-10))",
        "border_noise": f"per line, run of DN < {BORDER_DN} ({BORDER_SMOOTH_PX} px median) from the swath edge -> NaN",
        "nodata": "DN == 0, border noise and pixels outside the slices -> NaN; valid = both channels finite",
        "georeference": "AOI window in radar geometry from GCPs, then warp with GCP thin-plate spline, "
                        "bilinear on linear sigma0, to the common grid; slices mosaicked first-valid",
        "not_applied": ["terrain correction / RTC", "speckle filtering"],
        "standardization": "(dB - mean) / std per channel; invalid (non-finite) pixels -> 0",
        "norm_constants_db": {c: {"mean": m, "std": s} for c, (m, s) in load_norm_constants().items()},
    }

    # Stage 4, inference: patch inference on each stack -> water probability and water mask
    manifest["inference"] = {"patch_contract": {
        "patch_size": cfg.patch_size, "overlap": cfg.overlap, "stride": cfg.patch_size - cfg.overlap,
        "batch_size": cfg.batch_size, "device": cfg.device,
        "tiling": "regular grid; last row/column of windows shifted back to end on the raster edge; "
                  "NaN padding only if the raster is smaller than one patch",
        "input": "patches read from sigma0_db_{t0,t1}.tif and standardized on the fly",
        "stitching": "weighted mean of softmax water probability; weight ramps linearly from the "
                     "patch edge to 1 at `overlap` px inside (cross-fade between neighbours)",
        "threshold": f"water = probability > {cfg.water_threshold}",
        "nodata": f"{inference.NODATA} in water_t*.tif (NaN in water_prob_t*.tif) where the input is invalid"}}
    masks = {}
    for tag in ("t0", "t1"):
        start = time.perf_counter()
        ds = inference.PatchDataset(cfg.out_dir / f"sigma0_db_{tag}.tif", cfg.patch_size, cfg.overlap)
        prob = inference.predict(detector, ds, cfg.batch_size)
        georef.write_raster(cfg.out_dir / f"water_prob_{tag}.tif", prob, grid, nodata=float("nan"))
        mask = masks[tag] = inference.to_mask(prob, cfg.water_threshold)
        path = cfg.out_dir / f"water_{tag}.tif"
        georef.write_raster(path, mask, grid, nodata=inference.NODATA)
        seconds = time.perf_counter() - start
        water_fraction = float((mask == inference.WATER).sum() / (mask != inference.NODATA).sum())
        manifest["inference"][tag] = {"patches": len(ds), "seconds": round(seconds, 1),
                                      "water_fraction_of_valid": round(water_fraction, 4)}
        log.info("%s: %s (%d patches in %.0f s, water %.1f%% of valid pixels)",
                 tag.upper(), path, len(ds), seconds, 100 * water_fraction)

    # Stage 5, change detection: change detection -> PWB and flood polygons
    change_masks, counts = change.change_masks(masks["t0"], masks["t1"])
    pixel_km2 = cfg.resolution_m ** 2 / 1e6
    manifest["change_detection"] = {
        "logic": change.LOGIC,
        "nodata_policy": "a pixel nodata on either date is excluded from both outputs",
        "area_km2_by_class": {k: round(n * pixel_km2, 3) for k, n in counts.items()},
        "vectorization": {"connectivity": 4, "min_area_m2": cfg.min_area_m2,
                          "min_area_px": round(cfg.min_area_m2 / cfg.resolution_m ** 2, 2),
                          "min_area_rule": "polygons below min_area dropped; holes below min_area filled",
                          "area_crs": grid.crs, "output_crs": "EPSG:4326 (RFC 7946), 6 decimals"}}
    # Optional: DEM filter, the same exclusion for both outputs
    if cfg.dem_filter:
        dem_paths, manifest["dem_filter"] = dem.fetch(cfg)
        slope = dem.slope_on_grid(dem_paths, grid)
        georef.write_raster(cfg.out_dir / "slope_deg.tif", slope, grid, nodata=float("nan"))
        steep = slope > cfg.max_slope_deg  # NaN (no DEM) -> False: nothing removed
        manifest["dem_filter"].update({
            "rule": f"pwb, flood &= ~(slope > {cfg.max_slope_deg} deg)",
            "slope": f"central differences on the DEM resampled bilinear to the {cfg.resolution_m:g} m output grid "
                     f"(spacing {cfg.resolution_m:g} m)",
            "steep_km2": round(int(steep.sum()) * pixel_km2, 3),
            "removed_km2": {name: round(int((m & steep).sum()) * pixel_km2, 3)
                            for name, m in change_masks.items()}})
        for m in change_masks.values():
            m &= ~steep
        log.info("DEM filter: removed %s km2 (slope > %.1f deg)",
                 manifest["dem_filter"]["removed_km2"], cfg.max_slope_deg)

    for name, filename in (("pwb", "pwb.geojson"), ("flood", "flood_extent.geojson")):
        features, stats = change.to_features(change_masks[name], grid, cfg.min_area_m2)
        path = cfg.out_dir / filename
        change.write_geojson(path, features)
        manifest["change_detection"][name] = {"file": filename, **stats}
        log.info("%s: %s (%d polygons, %.2f km2; %d below min area dropped)",
                 name.upper(), path, stats["polygons"], stats["area_km2"], stats["dropped_polygons"])

    manifest["run"]["seconds"] = round(time.perf_counter() - run_start, 1)
    manifest["run"]["outputs"] = sorted({p.name for p in cfg.out_dir.iterdir() if p.is_file()}
                                        | {"run_manifest.json"})
    path = cfg.out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote %s", path)
