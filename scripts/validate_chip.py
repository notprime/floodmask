"""
Offline check of the model input contract on a Sen1Floods11 hand-labelled chip.

Sen1Floods11 S1Hand chips are already sigma0 dB (GEE S1_GRD), so this proves
that standardize() + FloodDetector reproduce the published Bolivia accuracy
(test IoU 0.65 for the region).

With --ours, it also checks our own calibration: our T1 stack (pipeline output,
same acquisition) is resampled onto the chip grid, compared to the GEE dB, and
run through the same model against the same hand label.

The chip is only used here. It is never an input to the Bolivia pipeline run.

    python scripts/validate_chip.py                    # Bolivia_103757, CPU
    python scripts/validate_chip.py --device cuda
    floodmask --bbox -65.28 -14.27 -65.20 -14.20 --out outputs/chipcheck   # chip is north of the sub-AOI
    python scripts/validate_chip.py --ours outputs/chipcheck/sigma0_db_t1.tif
"""

import argparse
import json
import logging
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.warp import Resampling, reproject, transform_bounds

from floodmask.config import Config
from floodmask.model import FloodDetector
from floodmask.preprocess import CHANNELS, load_norm_constants, standardize

BASE = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled"
URLS = {"s1": BASE + "/S1Hand/{chip}_S1Hand.tif", "label": BASE + "/LabelHand/{chip}_LabelHand.tif"}

log = logging.getLogger("validate_chip")


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s", url)
        urllib.request.urlretrieve(url, dest)
    return dest


def describe(src) -> dict:
    """The metadata the pipeline relies on, checked before processing."""
    return {"crs": str(src.crs), "transform": list(src.transform)[:6], "width": src.width,
            "height": src.height, "count": src.count, "dtype": src.dtypes[0], "nodata": str(src.nodata),
            "bounds_wgs84": list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))}


def db_stats(db: np.ndarray, valid: np.ndarray, label: np.ndarray) -> dict:
    """Per-channel dB distribution, overall and split by hand label (1 = water, 0 = dry)."""
    out = {}
    for i, c in enumerate(CHANNELS):
        x = db[i][valid]
        out[c] = {"mean": float(x.mean()), "std": float(x.std()),
                  "p2_p50_p98": [float(v) for v in np.percentile(x, [2, 50, 98])],
                  "median_water": float(np.median(db[i][valid & (label == 1)])),
                  "median_dry": float(np.median(db[i][valid & (label == 0)]))}
    return out


def scores(pred: np.ndarray, label: np.ndarray, valid: np.ndarray) -> dict:
    m = valid & (label != -1)  # -1 = not labelled
    tp = int((pred & (label == 1) & m).sum())
    fp = int((pred & (label == 0) & m).sum())
    fn = int((~pred & (label == 1) & m).sum())
    return {"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1), "tp": tp, "fp": fp, "fn": fn,
            "water_fraction_pred": float(pred[m].mean()), "water_fraction_label": float((label[m] == 1).mean())}


def compare_ours(path: Path, chip_profile: dict, db: np.ndarray, valid: np.ndarray,
                 label: np.ndarray, detector: FloodDetector) -> dict:
    """Our calibrated T1 dB vs the GEE chip dB, and the model result on each."""
    ours = np.full(db.shape, np.nan, np.float32)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, [1, 2]), ours, dst_transform=chip_profile["transform"],
                  dst_crs=chip_profile["crs"], dst_nodata=np.nan, resampling=Resampling.bilinear)
        both = valid & np.isfinite(ours).all(axis=0)
    coverage = both.sum() / valid.sum()  # share of the chip's valid pixels we also have
    if coverage < 0.5:
        raise RuntimeError(f"{path} covers only {100 * coverage:.0f}% of the chip's valid pixels; "
                           "run floodmask with a --bbox around the chip")

    x, ours_valid = standardize(ours, load_norm_constants())
    prob = detector(torch.from_numpy(x)[None])[0].cpu().numpy()
    out = {"overlap_fraction": float(both.mean()),
           "scores": scores((prob > 0.5) & ours_valid, label, ours_valid), "db_vs_gee": {}}
    for i, c in enumerate(CHANNELS):
        d = ours[i][both] - db[i][both]
        out["db_vs_gee"][c] = {"median_offset": float(np.median(d)),  # ours - GEE
                               "mad": float(np.median(np.abs(d - np.median(d)))),
                               "pearson_r": float(np.corrcoef(ours[i][both], db[i][both])[0, 1]),
                               "median_water_ours": float(np.median(ours[i][both & (label == 1)])),
                               "median_dry_ours": float(np.median(ours[i][both & (label == 0)]))}
        log.info("%s ours - GEE: median %+.2f dB (MAD %.2f), r = %.2f", c, out["db_vs_gee"][c]["median_offset"],
                 out["db_vs_gee"][c]["mad"], out["db_vs_gee"][c]["pearson_r"])
    log.info("Model on our calibration: water IoU %.3f, precision %.3f, recall %.3f",
             out["scores"]["iou"], out["scores"]["precision"], out["scores"]["recall"])
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chip", default="Bolivia_103757")
    p.add_argument("--cache", type=Path, default=cfg.cache_dir)
    p.add_argument("--out", type=Path, default=cfg.out_dir / "validation")
    p.add_argument("--device", default=cfg.device)
    p.add_argument("--ours", type=Path, help="our T1 stack (sigma0_db_t1.tif) to compare on the chip grid")
    a = p.parse_args()

    s1_path = fetch(URLS["s1"].format(chip=a.chip), a.cache / "sen1floods11" / f"{a.chip}_S1Hand.tif")
    lb_path = fetch(URLS["label"].format(chip=a.chip), a.cache / "sen1floods11" / f"{a.chip}_LabelHand.tif")

    with rasterio.open(s1_path) as src:
        meta = describe(src)
        db = src.read().astype(np.float64)
        profile = src.profile
        if src.nodata is not None and np.isfinite(src.nodata):
            db[db == src.nodata] = np.nan
    with rasterio.open(lb_path) as src:
        label = src.read(1).astype(np.int64)
    log.info("Chip metadata: %s", meta)
    if db.shape[1:] != label.shape:
        raise RuntimeError(f"S1 {db.shape} and label {label.shape} grids differ")

    x, valid = standardize(db, load_norm_constants())
    log.info("Valid pixels: %.1f%%; standardized mean %s, std %s (training set: 0 and 1)",
             100 * valid.mean(), x[:, valid].mean(axis=1).round(2), x[:, valid].std(axis=1).round(2))

    detector = FloodDetector(cfg.hf_repo, a.cache / "model", a.device)
    prob = detector(torch.from_numpy(x)[None])[0].cpu().numpy()
    pred = (prob > 0.5) & valid

    s = scores(pred, label, valid)
    log.info("%s: water IoU %.3f, precision %.3f, recall %.3f (published Bolivia test IoU 0.65)",
             a.chip, s["iou"], s["precision"], s["recall"])

    a.out.mkdir(parents=True, exist_ok=True)
    aoi = cfg.aoi_bbox
    b = meta["bounds_wgs84"]
    report = {"chip": a.chip, "metadata": meta, "valid_fraction": float(valid.mean()),
              "inside_sub_aoi": aoi[0] <= b[0] and aoi[1] <= b[1] and b[2] <= aoi[2] and b[3] <= aoi[3],
              "reference_db": db_stats(db, valid, label), "scores": s, "model": detector.info}
    if a.ours:
        report["ours"] = compare_ours(a.ours, profile, db, valid, label, detector)
    (a.out / f"{a.chip}.json").write_text(json.dumps(report, indent=2))
    # Probability raster on the chip's own grid, to inspect next to the label in QGIS.
    for k in ("blockxsize", "blockysize", "tiled", "interleave"):
        profile.pop(k, None)
    profile.update(count=1, dtype="float32", nodata=-1.0, compress="deflate")
    with rasterio.open(a.out / f"{a.chip}_prob.tif", "w", **profile) as dst:
        dst.write(np.where(valid, prob, -1.0).astype(np.float32), 1)
    log.info("Wrote %s", a.out / f"{a.chip}.json")


if __name__ == "__main__":
    main()
