# floodmask — SAR flood mapping from open Sentinel-1 data

CPU-only, inference-only pipeline that finds Sentinel-1 GRD pre/post (T0/T1) acquisitions
over the Beni floodplain (Bolivia, Feb 2018 flood), downloads them from a public STAC
catalog, calibrates DN → sigma0 dB the way a pretrained UNet-ResNet34 flood model expects,
runs it patch-by-patch on both dates, and derives permanent water bodies (PWB) and flood
extent from the T0/T1 change. Outputs are GIS-ready (GeoTIFF + GeoJSON) with a full
provenance manifest. This is a personal project; you (Claude) act as pair programmer
when extending it.

## Setup and run

```powershell
conda activate floodmask         # env already exists locally with everything installed
python -m pytest tests -q        # 5 offline tests: stitching, change detection, DEM slope
floodmask --help
floodmask --out outputs          # full run, ~51 s on CPU once scenes are cached
```

If rebuilding from scratch: `pip install torch torchvision` first (PyPI CPU wheel is
fine), then `pip install -e ".[dev]"`. Do not mix conda-forge rasterio/GDAL with pip
torch on Windows — two OpenMP runtimes abort with `OMP: Error #15`.

**Caches** (reused across runs, never re-downloaded if present):
- `cache/grd/<STAC item id>/` — VV/VH COGs + calibration/noise XML (anonymous S3, `sentinel-s1-l1c`)
- `cache/model/` — HF checkpoint (`model.pt`) via `hf_hub_download`
- `cache/dem/` — Copernicus GLO-30 DEM tile (only with `--dem-filter`)
- `cache/sen1floods11/` — S1Hand chip + label, used only by `scripts/validate_chip.py` (never a pipeline input)

**Fast / offline path:** `cache/grd/` already has all 4 slices (T0+T1, 2
slices each) and `cache/model/` has the weights, so `floodmask --out outputs` needs no
network. For a quick small-AOI run: `floodmask --bbox -65.28 -14.27 -65.20 -14.20 --out
outputs/chipcheck` (already cached too, ~seconds, same code path as the full run).

## Module map (`src/floodmask/`)

| File | Responsibility | Key functions/classes |
|---|---|---|
| `cli.py` | argparse → `Config`, calls `pipeline.run` | `parse_args`, `main` |
| `config.py` | one dataclass holding every run parameter (defaults = Bolivia event) | `Config` |
| `pipeline.py` | orchestrates all stages in order, writes `run_manifest.json` | `run(cfg)` |
| `stac.py` | Earth Search discovery of T0/T1 items | `discover(cfg)` |
| `download.py` | anonymous S3 fetch of VV/VH COGs + calibration/noise XML, cached | `fetch(item, cache_dir, region)` |
| `preprocess.py` | LUT parsing/interpolation, radiometric calibration, border-noise mask, dB, z-score | `calibrate`, `border_noise`, `to_db`, `standardize`, `load_norm_constants` |
| `georef.py` | radar-geometry AOI window, GCP thin-plate-spline warp to UTM, slice mosaic, raster I/O | `target_grid`, `_aoi_window`, `_slice_sigma0`, `load_pass`, `write_raster` |
| `model.py` | frozen pretrained UNet-ResNet34 wrapper | `FloodDetector` (forward → water probability) |
| `inference.py` | `torch.utils.data.Dataset` over the GeoTIFF, patch tiling, blended stitching | `PatchDataset`, `predict`, `to_mask`, `_blend_weights` |
| `change.py` | T0/T1 pixel logic, vectorization, GeoJSON | `change_masks`, `to_features`, `write_geojson` |
| `dem.py` | optional: Copernicus DEM slope filter | `fetch`, `slope_on_grid` |
| `scripts/validate_chip.py` | offline-only check against Sen1Floods11 S1Hand chip (never a pipeline input) | `main`, `compare_ours` |

## Data flow

```
stac.discover (T0/T1 item IDs, 2 slices/date)
  -> download.fetch (S3 COGs + calibration/noise XML, cached)
  -> preprocess.calibrate: DN -> sigma0 = max(DN^2 - N, 0) / K^2  (noise-removed, linear power)
  -> georef._slice_sigma0: GCP thin-plate-spline warp to common 10 m UTM grid, bilinear on LINEAR sigma0
  -> georef.load_pass: mosaic the 2 slices (first valid pixel wins) -> preprocess.to_db -> sigma0_db_t{0,1}.tif
  -> inference.PatchDataset: per-patch standardize (z-score, vendored Sen1Floods11 constants)
  -> model.FloodDetector (frozen UNet-ResNet34): softmax water probability
  -> inference.predict: weighted-average stitching -> water_prob_t{0,1}.tif -> threshold -> water_t{0,1}.tif
  -> change.change_masks: PWB = water_t1 & water_t0, flood = water_t1 & ~water_t0 (nodata on either date -> excluded from both)
  -> [optional dem.slope_on_grid: pwb, flood &= ~(slope > max_slope_deg)]
  -> change.to_features: polygonize, drop <25px polygons, fill small holes, reproject to WGS84
  -> pwb.geojson, flood_extent.geojson + run_manifest.json (every stage appends its section)
```

T0 and T1 are produced on the **identical grid** (one `Grid` from `georef.target_grid`),
so change detection is a plain pixel-to-pixel comparison — this is the one design choice
that makes everything downstream simple.

## Key design decisions (as implemented)

- **AOI**: Beni sub-AOI `[-65.35, -14.65, -65.05, -14.35]`, 3268×3352 px @ 10 m. Only the
  AOI window (+256 px margin, `MARGIN_PX` in `georef.py`) is read from each slice.
- **T0**: latest pre-event pass on the same relative orbit 156 (`stac.discover`), here
  2018-02-03 — same geometry as T1, but already rainy season (T0 is 22.5% water).
- **Patch contract**: 512×512 (multiple of 32), 64 px overlap, stride 448. Last
  row/column window shifted to end flush with the raster edge (no padding unless the
  raster is smaller than one patch). Stitching = weighted mean of **softmax probability**
  (not labels), linear ramp weight 0→1 over the overlap. Threshold 0.5 (`--threshold`).
- **Nodata policy**: NaN in float rasters, 255 in `water_t*.tif`. Carried explicitly from
  DN=0 / border-noise / swath edge all the way to the GeoJSON; a pixel nodata on *either*
  date is excluded from both PWB and flood.
- **Calibration order**: calibrate (noise removal) in radar geometry (where the LUTs
  live) → warp in **linear** sigma0 (never resample logarithms) → dB once on the final
  grid → z-score per patch at inference time. Thermal noise removal (`N` from
  `noise-iw-{pol}.xml`) is essential, not optional — without it, chip recall was 0.17; with
  it, 0.73 (see `docs/design-notes.md`).
- **CRS/grid alignment**: both dates warped to the same `Grid` (EPSG:32720, 10 m,
  AOI-snapped). GeoJSON output reprojected once (batched) to EPSG:4326, 6 decimals.
- **Minimum mapping unit**: 25 px (0.25 ha) polygons dropped, holes <25 px filled;
  rasters (`water_t*.tif`) stay raw/unfiltered.
- **DEM filter** (`--dem-filter`, off by default): Copernicus GLO-30 slope > 5°
  (`--max-slope-deg`) removed from **both** PWB and flood (one shared exclusion mask, so
  a pixel never moves between classes). On this flat floodplain it mostly trims real
  flood at riverbanks (GLO-30 is a surface model, sees canopy top) — documented as a
  negative/mixed result, kept off by default.

## Coding principles

Simplicity over complexity. Clear, readable, flat modules (no framework beyond argparse +
a dataclass `Config`). Only write code the pipeline needs or that would otherwise make an
output wrong — no speculative abstractions, no unused flexibility. No new dependencies
unless there's no reasonable way around it (current deps: numpy, torch,
segmentation-models-pytorch, huggingface_hub, pystac-client, rasterio, shapely, boto3).
CPU-only is the target and the reported runtime; `--device cuda` exists for debugging only.

## Working rules for you (Claude)

- Propose a short plan before any non-trivial change (new feature, refactor, extending a
  stage) — think out loud briefly, then act.
- Keep diffs small and scoped to the change at hand; don't touch unrelated modules.
- After every change, run `python -m pytest tests -q`; for anything touching
  preprocessing/inference, prefer a quick run on the cached small bbox
  (`--bbox -65.28 -14.27 -65.20 -14.20 --out outputs/chipcheck`) over the full AOI to keep
  iteration fast.
- Preserve the manifest contract: if a stage's behavior changes, update what it writes
  into `run_manifest.json` too — the manifest is the run's documentation-as-code.
- Don't reach for network calls mid-task unless necessary; everything needed for the
  full Bolivia run is already cached (see Setup above).

## Details

See `docs/design-notes.md` for the full rationale behind each design choice, the
calibration details and dB-offset-vs-GEE investigation, known limitations, and
scalability notes.
