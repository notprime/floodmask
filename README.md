# floodmask: SAR flood mapping from open Sentinel-1 data

**Table of Contents**
- [Overview](#overview)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
  - [Patch inference contract](#patch-inference-contract)
  - [Change detection](#change-detection)
  - [Coarser resolution](#coarser-resolution---resolution-30)
- [Preprocessing](#preprocessing)
- [Design choices](#design-choices)
  - [Optional Copernicus DEM filter](#optional-copernicus-dem-false-positive-filter)
- [Scalability notes](#scalability-notes)
- [Limitations & next steps](#limitations--next-steps)
  - [Sentinel-1 vs very-high-resolution tasked SAR](#sentinel-1-vs-very-high-resolution-tasked-sar)
- [Acknowledgements](#acknowledgements)

---
<br>

## Overview

floodmask is a personal project to see how fast an end-to-end flood map can be
produced from open data alone. Sentinel-1 GRD scenes are discovered over a public STAC
catalog, calibrated, run through a pretrained UNet-ResNet34 patch by patch, and a
pre/post (T0/T1) comparison yields flood extent and permanent water bodies (PWB) as
GIS-ready GeoJSON and GeoTIFF, with a provenance manifest for every run.

The case study is the Beni department, Bolivia, February 2018, a well-known
[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) event. It also gives me a
labelled chip to sanity-check the preprocessing against.

```
STAC search -> download GRD -> calibrate + denoise -> warp to UTM grid -> patch inference -> change detection
```

## Quick start

```powershell
conda create -n floodmask python=3.12 -y
conda activate floodmask
# torch first, from the PyTorch index (cu126 for GPU debugging; the CPU wheel is enough for the run)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"     # dev = pytest
python -m pytest tests -q   # offline tests: stitching, change detection, DEM slope
floodmask --help
floodmask --out outputs
```

Outputs in `outputs/`:

| File | Content |
|---|---|
| `water_t0.tif`, `water_t1.tif` | water masks, uint8: 0 land, 1 water, 255 nodata |
| `pwb.geojson` | stable water, `water_t1 & water_t0` |
| `flood_extent.geojson` | flood, `water_t1 & ~water_t0` |
| `run_manifest.json` | provenance, STAC IDs, preprocessing, patch contract, flood logic, areas |
| `sigma0_db_t*.tif`, `water_prob_t*.tif` | intermediate: model input in dB, blended water probability |

Everything comes from pip; conda only provides Python. Mixing pip torch with
conda-forge rasterio/GDAL loads two OpenMP runtimes on Windows and aborts
(`OMP: Error #15`). Install torch before `pip install -e .`, or pip pulls the
CPU-only wheel from PyPI.

GRD assets are downloaded anonymously from the `sentinel-s1-l1c` S3 bucket
(AWS Open Data, eu-central-1): no AWS account needed. Full scenes for T0 + T1
(2 slices x VV/VH each, a few GB) are cached under `cache/grd/` and reused.

`floodmask --help` lists all options. `--device cuda` exists for debugging only;
the reported runtime is CPU (see [Design choices](#design-choices)). `--dem-filter` adds the
optional DEM filter ([details](#optional-copernicus-dem-false-positive-filter)); it
downloads one Copernicus DEM tile into `cache/dem/`.

## How it works

```
pipeline.py -> stac.py -> download.py -> preprocess.py  -> georef.py -> inference.py (-> model.py) -> change.py
```

`pipeline.py` runs the stages in order; every stage adds its section to `run_manifest.json`.

1. **STAC discovery** (`stac.py`). Earth Search `sentinel-1-grd`, AOI bbox,
   relative orbit 156, descending. T1 = the pass on 2018-02-15; T0 = the latest
   earlier pass on the same track (2018-02-03, 12 days before). Each pass covers
   the AOI with two adjacent slices, so each date is a list of items. Item IDs,
   candidate dates and the selection rule go into the manifest.
2. **Download** (`download.py`). VV/VH COGs plus calibration and noise XML,
   anonymous S3, cached in `cache/grd/`.
3. **Preprocessing** (`preprocess.py`, `georef.py`). DN -> thermal-noise-corrected
   sigma0 -> common 10 m UTM grid -> dB. Details in [Preprocessing](#preprocessing).
   Output: `sigma0_db_{t0,t1}.tif`.
4. **Patch inference** (`inference.py`, `model.py`). `FloodDetector` wraps the
   frozen UNet-ResNet34; `PatchDataset` serves standardized 512 px windows;
   overlapping predictions are blended. Output: `water_{t0,t1}.tif`.
5. **Change detection** (`change.py`). PWB and flood masks, vectorized to
   `pwb.geojson` and `flood_extent.geojson`.

**Run on the Beni sub-AOI** (3268 × 3352 px, 1095 km²): 51 s on CPU with the
scenes already cached. The first run also downloads the scenes (a few GB).

| Class (valid on both dates) | Area |
|---|---|
| Flood (new water at T1) | 130.9 km² raw, 125.2 km² in `flood_extent.geojson` |
| Receded (water at T0 only) | 12.5 km² |
| Nodata (swath edge) | 8.8% of the AOI |

Flood is ten times receded water: the change is mostly real, not classification noise,
which would add to both about equally.

### Patch inference contract

`inference.py`, also recorded in `run_manifest.json` under `inference.patch_contract`.

| | Choice | Why |
|---|---|---|
| Input | `PatchDataset` reads windows from `sigma0_db_{t0,t1}.tif` and standardizes each patch on the fly | the GeoTIFF is the contract between preprocessing and inference; memory per patch, not per scene |
| Patch | 512 × 512 (`--patch-size`, multiple of 32) | model-native size |
| Overlap | 64 px (`--overlap`), stride 448 | the UNet sees less context near a patch border, so borders are never used alone |
| Edges | last window shifted back to end on the raster edge | every patch is full of real data; no padding artefacts. NaN padding only if the raster is smaller than one patch |
| Stitching | weighted mean of the softmax water probability; weight ramps linearly from the patch edge to 1 at 64 px inside | neighbours cross-fade (weights sum to 1 on the overlap), no visible seams; averaging probabilities, not labels |
| Decision | water = probability > 0.5 (`--threshold`) | same as argmax for two classes |
| Nodata | invalid input -> 0 for the model (as in training), then masked: 255 in `water_t*.tif`, NaN in `water_prob_t*.tif` | swath edge and DN 0 never become land or water |

Outputs on the common grid: `water_t{0,1}.tif` (uint8: 0 land, 1 water, 255 nodata)
and `water_prob_t{0,1}.tif` (float32 blended probability, kept to inspect the
threshold). The stitching is checked against a pointwise stand-in model: the
tiled result equals the full-image result to float precision for any patch size
and overlap.

### Change detection

`change.py`, also recorded in `run_manifest.json` under `change_detection`.

| | Choice | Why |
|---|---|---|
| PWB | `water_t1 & water_t0` | stable water derived from the SAR pair itself (no JRC layer), so the method works wherever there are two passes |
| Flood | `water_t1 & ~water_t0` | new water at T1 |
| Nodata | a pixel nodata on **either** date is in neither output | no flood claimed where T0 is unknown (swath edge) |
| Threshold | the same 0.5 water probability on both dates | PWB and flood come from one decision; a stricter PWB threshold would push uncertain stable water into "flood" |
| Polygons | `rasterio.features.shapes`, 4-connected, on the UTM grid | exact pixel boundaries; 4-connectivity avoids polygons touching at a single corner (invalid geometry) |
| Minimum mapping unit | polygons < 2500 m² (0.25 ha = 25 px at 10 m, `--min-area-m2`) dropped; holes < 2500 m² filled | ~5 GRD resolution cells (~20 × 22 m each): smaller blobs are speckle-level. Filters the vectors only; `water_t*.tif` stay raw |
| Attributes | `id`, `area_m2` (computed in UTM, after hole filling) | area is what a dashboard sums; computing it in lon/lat would be wrong |
| CRS | WGS84 lon/lat, 6 decimals (~0.1 m) | GeoJSON (RFC 7946) mandates WGS84 |

The manifest also reports the full T0/T1 breakdown in km² (`pwb`, `flood`,
`receded`, `dry`, `nodata`) and, per output, polygons kept / dropped by the
minimum-area filter, to judge the threshold. `tests/test_change.py` checks the
logic, nodata exclusion, the area filter and the reprojection on a hand-made pair.

### Coarser resolution (`--resolution 30`)

`--resolution` (metres, a multiple of 10, default 10 = no resampling, bit-identical
to before) block-averages the **linear sigma0** on the projected 10 m UTM grid, after
the GCP warp and slice mosaic and before dB. 3×3 = 9 looks at 30 m. NaN pixels are
left out of the mean; an output pixel is nodata below 50 % valid inputs
(`georef.MIN_VALID_FRACTION`). Edges are padded with NaN, not cropped. Outputs go to
`outputs_30m/` unless `--out` is given; the manifest `resampling` block records the
method, nodata rule, edge handling, grids and patch footprint.

The model, z-score constants and the 512/64 patch contract are unchanged, so at 30 m:
a 512 patch covers ~15.4 km (vs 5.1 km), features look 3× smaller and speckle is
reduced. This is a domain shift from the 10 m Sen1Floods11 training data: expect
some degradation, most on narrow rivers. The minimum mapping unit is in m², and the
DEM slope is computed on the 30 m output grid.

## Preprocessing

Model input contract (from the training repo): channel 0 = VV, channel 1 = VH,
sigma0 in dB as in Sen1Floods11 (GEE `COPERNICUS/S1_GRD`), z-scored per channel.

| Step | Where | What |
|---|---|---|
| Download | `download.py` | full `vv`, `vh` COGs + `schema-calibration-{vv,vh}` and `schema-noise-{vv,vh}` XML per STAC item |
| Metadata check | `georef._check` | 1 band uint16 DN, GCPs in EPSG:4326 (GRD is in radar geometry, no CRS transform) |
| Crop | `georef._aoi_window` | AOI corners -> radar rows/cols through the GCPs, +256 px margin; only that window is read |
| Border noise | `preprocess.border_noise` | per line, the run of DN < 20 (5 px median) from the swath edge -> nodata |
| Calibration + thermal noise removal | `preprocess.calibrate` | `sigma0 = max(DN^2 - N, 0) / K^2`, K = `sigmaNought` LUT, N = `noiseLut`, both bilinearly interpolated in (line, pixel); DN 0 -> nodata |
| Georeference | `georef._slice_sigma0` | warp to UTM 20S (EPSG:32720), 10 m, AOI-snapped grid; thin-plate spline through the GCPs; bilinear on **linear** sigma0 |
| Mosaic | `georef.load_pass` | the two slices of each pass, first valid pixel wins |
| dB | `preprocess.to_db` | `10 log10(max(sigma0, 1e-10))`, written to `sigma0_db_{t0,t1}.tif` |
| Standardize | `preprocess.standardize` | `(dB - mean) / std` with the vendored Sen1Floods11 constants; invalid -> 0, as in training |

T0 and T1 are produced on the **same grid**, so change detection is pixel to pixel.

Assumptions and known gaps vs. GEE:
- **Thermal noise removal** follows ESA (Miranda & Meadows, *Radiometric Calibration of S-1 Level-1 Products*, ESA-EOPG-CSCOP-TN-0002; [Piantanida et al., Thermal denoising of products generated by the S-1 IPF](https://sentinel.esa.int/documents/247904/2142675/Thermal-Denoising-of-Products-Generated-by-Sentinel-1-IPF)), as in SNAP and GEE: the noise power N (DN^2 units) is subtracted before calibration, negatives clipped to 0. Our scenes (Feb 2018) use the pre-IPF 2.90 noise format (range vectors only); the later range x azimuth format is rejected with a clear error.
- **Border noise.** Before IPF 2.90 (13 March 2018) the swath edges carry a strip of DN 1-5 (not 0), about 130 px wide in our scenes, vs DN >= ~45 for real data. Without masking it reads as -44 dB and the model calls it water on both dates (3.4% of the AOI, all on the east swath edge): a fake permanent-water strip. GEE removes it too.
- **No terrain correction.** Geolocation comes from the GRD GCPs only. The Beni floodplain is flat lowland, so the error is small, and T0/T1 share orbit 156 and geometry, so it largely cancels in the change detection.
- **No speckle filtering** (GEE applies none either).
- **Swath edge.** The AOI's east edge (−65.05) touches the edge of the orbit-156 swath, so ~5.5% of the AOI is not imaged on either date, and the border-noise strip next to it (~3.3%) is masked: 8.8% of the AOI is nodata (NaN) and excluded from all outputs.

Check against the reference. The S1Hand chip Bolivia_103757 (same T1 acquisition)
lies just north of the sub-AOI, so the pipeline is run on a small bbox around it
(same code path, cached scenes), then compared:

```bash
floodmask --bbox -65.28 -14.27 -65.20 -14.20 --out outputs/chipcheck
python scripts/validate_chip.py --ours outputs/chipcheck/sigma0_db_t1.tif
```

The script downloads the chip and its hand label from the public Sen1Floods11
bucket into `cache/` (offline check only, never a pipeline input).

| Input to the model | Water IoU | Precision | Recall |
|---|---|---|---|
| GEE S1Hand chip (reference) | 0.81 | 0.93 | 0.86 |
| Ours, calibration only | 0.16 | 0.86 | 0.17 |
| **Ours, + thermal noise removal** (final) | **0.66** | **0.87** | **0.73** |

Median dB by hand label, ours vs GEE (before -> after noise removal):

| | Dry land | Water |
|---|---|---|
| VV | −8.77 -> −8.82 vs −8.65 | −17.27 -> **−17.69** vs −18.05 |
| VH | −14.33 -> −14.54 vs −14.38 | −22.88 -> **−24.67** vs −24.86 |

Without noise removal our calibration already matched GEE on land, but water
was brighter, by 2 dB in VH: VH water sat on the instrument noise floor
(about −23 dB), so the channel that separates water best carried no
information there, and the model found 17% of the water. Subtracting the
noise LUT moves water to within 0.2-0.4 dB of GEE and leaves land unchanged,
as expected (noise is ~10% of land power, ~50% of calm-water power). Recall
goes from 0.17 to 0.73; IoU 0.66 is in line with the model's published
Bolivia test IoU (0.65).

The remaining gap to the GEE chip is per pixel, not a bias (median offset
< 0.05 dB, but MAD ~2.4 dB, r = 0.77): our pixels are a different
resampling of the same scene (GCP thin-plate spline to UTM, then bilinear
again onto the chip grid for this check), while GEE terrain-corrects with a
DEM. Different speckle realisations and sub-pixel shifts mostly hit water
edges (precision stays 0.87).

## Design choices

**CPU-only.** The pipeline is built to run on a laptop CPU, with no GPU or cloud
account: anyone can reproduce it, and the reported runtime (51 s for two dates of 11M px)
is the one a reader will actually get. `--device cuda` exists for debugging only.

**AOI crop.** I mapped a Beni sub-AOI inside the Sen1Floods11 Bolivia event area,
not the ~200k km² event polygon. Only the AOI window (+256 px margin) of each slice
is read and warped, so memory and time scale with the AOI, not the scene, which is
what makes the CPU-only run practical. It is also next to a hand-labelled S1Hand chip,
so the preprocessing can be checked against a reference
([Preprocessing](#preprocessing)). The cost: the AOI's east edge touches the swath
edge, so 8.8% of it is nodata on both dates. A bbox shifted west would be fully imaged,
but I kept the AOI aligned with the Sen1Floods11 Beni reference area.

**SAR-derived PWB.** Permanent water comes from the T0/T1 pair itself, not from an
external layer such as JRC Global Surface Water. The method is self-contained and
works wherever two same-geometry passes exist. The price is the T0 choice below.

**T0 choice.** Latest pass on the same orbit (2018-02-03): same geometry and
incidence angle as T1, so differences are water, not viewing angle. The cost:
the Beni was already in the rainy season, and T0 is 22.5% water. So
`pwb.geojson` is "water on both dates", which includes water from before
3 February, and `flood_extent.geojson` is only the water that appeared between
3 and 15 February. An earlier, dry-season T0 would give a cleaner PWB, but
with a two-month gap more unrelated change (crops, soil moisture) would be
flagged as flood.

**Nodata policy.** Nodata is carried explicitly at every stage: NaN in the sigma0
and probability rasters, 255 in `water_t*.tif`. A pixel nodata on either date is
in neither GeoJSON. The model itself sees invalid pixels as 0 (as in training)
and its output there is discarded. The alternative, treating nodata as land,
would turn every T1 water pixel at the T0 swath edge into flood.

**PWB threshold.** The same 0.5 probability on both dates, so PWB and flood
partition the T1 water exactly. A stricter PWB threshold (e.g. 0.8 at T0) would
make PWB more "certain", but the uncertain stable water would then be reported
as flood, the more costly error on an emergency dashboard. `water_prob_t*.tif`
are kept to revisit this.

**Minimum mapping unit.** 25 px (0.25 ha) on the polygons only. It removes 89%
of the flood polygons but 5% of the flood area: the dashboard gets fewer, cleaner
polygons, and the rasters stay raw for anyone who wants the speckle-level detail.

**Full download vs windowed reads.** Full COGs are downloaded and cached:
simple, and the cache makes re-runs instant. Reading only the AOI window via
HTTP range requests would avoid most of the transfer
([Scalability notes](#scalability-notes)).

**Simplicity over throughput.** One process, whole-AOI arrays in memory
(11M px: a few hundred MB), CPU by default. Right for one AOI; the next section
is what changes at scale.

### Optional Copernicus DEM false-positive filter

`dem.py`, optional: `floodmask --dem-filter [--max-slope-deg 5]`. Recorded in
`run_manifest.json` under `dem_filter`; the slope is written to `slope_deg.tif`.

**Method.** SAR calls a pixel water when it is dark. Radar shadow behind steep
slopes is dark too, and still water cannot stand on a slope. So:

| | Choice | Why |
|---|---|---|
| DEM | Copernicus GLO-30 (`cop-dem-glo-30` on Earth Search, one tile for the AOI) | same STAC catalog, no account |
| Slope | DEM resampled bilinear to the 10 m UTM grid, then central differences, in degrees | slope in metres on our grid; the vertical datum does not matter for a slope |
| Rule | `pwb, flood &= ~(slope > max_slope_deg)`, default 5° | a common slope threshold for flood mapping |
| Where | after change detection, before vectorization; `water_t*.tif` stay raw | the minimum-area filter then runs on the filtered masks |
| Both outputs | one exclusion mask for PWB **and** flood | a filtered pixel leaves both outputs, never moves from one to the other. Radar shadow is the same on both dates (same orbit), so on hilly ground it would land in PWB, not flood |
| No DEM | slope NaN, nothing removed | no evidence, no change |

**Impact on the Beni AOI.**

| | 5° | 10° |
|---|---|---|
| Flood pixels removed | 1.66 km² (1.3%) | 0.46 km² (0.35%) |
| PWB pixels removed | 4.59 km² (2.2%) | 2.03 km² (1.0%) |
| `flood_extent.geojson` area | 125.2 -> 124.1 km² (−0.9%) | 125.2 -> 125.1 km² (−0.1%) |

At 5°, 66.6 km² of the AOI (6%) is above the threshold, yet water is 3-5×
less likely than average to be steep: the water map is not concentrated on
terrain artefacts.

**Interpretation.** On this floodplain the filter removes real water, not
artefacts:
- **There is no terrain.** Radar shadow, the false positive the filter targets,
  cannot occur. The "steep" pixels follow river meander belts, forest edges and
  canopy texture: GLO-30 is a surface model (TanDEM-X), so it sees the top of the
  gallery forest, and its water bodies are flattened, which leaves a step at every
  bank.
- **The slopes are gentle.** Median 7.3° (p90 14.6°, p99 24.8°): most steep
  pixels sit just above 5°, at the noise level of a surface model over forest.
  At 10° most of the removed flood disappears (0.46 km² instead of 1.66 km²),
  and only a third of it reaches the GeoJSON (−0.15 km²): the rest are isolated
  pixels that the minimum mapping unit already drops or fills as holes.
  The pixels above 10° are DEM spikes, not coherent areas.
- **The removed flood is at the water's edge.** 61% of it lies within 30 m of
  PWB, against 16% of all steep pixels. The bank of a river or lake is exactly
  where a flood starts: these are most likely true flood pixels.
- **It also removes speckle.** At 5° the minimum-area filter drops ~2,000 fewer
  flood polygons, because many small specks on steep pixels were already removed.

**Decision.** Off by default. The filter is cheap, generalizes to hilly AOIs,
and on this one shows that the flood map is not a terrain artefact, but here it
mostly trims real flood at the riverbanks. Better options for forested
floodplains: HAND (height above nearest drainage) instead of slope, FABDEM
(Copernicus DEM with forests and buildings removed), or slope computed on a
smoothed (~90 m) DEM, which ignores canopy texture and keeps real terrain.

## Scalability notes

At ~1,000 scenes/day (full IW GRD scenes, ~250 × 170 km, ~40× our AOI):

- **Data transfer.** ~1-2 GB of COGs per scene (VV + VH): 1-2 TB/day of full
  downloads, most of it never used if only part of each scene is of interest.
- **Memory.** `predict` holds two full-size arrays (the probability and weight sums for the whole
  raster), and the warp holds the whole sigma0 stack: several GB per scene,
  more than a typical worker.
- **Compute.** Our run takes 51 s on CPU for two dates of 11M px. At ~40× per
  scene that is ~30 min per scene pair on one CPU: several hundred CPU-hours
  per day.
- **Orchestration.** A single sequential CLI run: no retries, no
  parallelism, no record of what has already been processed.
- **Vectors.** One GeoJSON per run grows to millions of polygons per scene
  and is slow to load in a dashboard.
- **Redundant work.** Each T1 is the next pass's T0, but the pipeline runs
  inference on it twice.

**Mitigation: process fixed tiles, once per acquisition.** Split every scene
onto a fixed tile grid (e.g. 10 × 10 km) and make (acquisition, tile)
the unit of work in a job queue. Each job reads only its
window from the COG, runs preprocessing and inference, and stores the water
mask keyed by STAC item ID. Change detection becomes a cheap second job that
pairs a tile's latest mask with its previous one. Memory per job is bounded by
the tile size, workers scale horizontally (GPU for inference), failed tiles are
retried alone, and every acquisition is inferred once instead of twice.

## Limitations & next steps

- **IPF >= 2.90 noise format**: after March 2018, ESA changed how the noise tables are stored, so the pipeline rejects later scenes. Fix: multiply the range and azimuth noise vectors.
- **Terrain correction (RTC) with the Copernicus DEM**. GEE terrain-corrects, we don't in this repository. Small differences in flat areas, needed in hilly areas.
- **Better PWB**. Our T0 (3 Feb) is already rainy season, so our PWB includes early flood water. A dry-season T0 would bring better PWB.
- **Multi-date baseline**. Use the median of several pre-event passes as T0,
  so that a speckled T0 pixel is not read as change.
- **Windowed COG reads**: instead of full downloads, only read the AOI bytes ([Scalability notes](#scalability-notes)).
- **Threshold calibration.** Choose the water threshold on the Bolivia S1Hand
  chips (offline) from `water_prob_t*.tif`, instead of the default 0.5.
- **Flooded vegetation.** Water under forest canopy raises backscatter
  (double bounce) instead of lowering it, so the model misses it as it looks for dark water. The Beni has
  gallery forest along the rivers; VV/VH ratio or a change-in-backscatter rule
  could flag it.
- **Other resolutions and sensors**: see below.

### Sentinel-1 vs very-high-resolution tasked SAR

Tasked commercial SAR constellations (X-band, sub-metre Spot imagery, VV only, scenes of
a few km) resolve streets and buildings and revisit within hours, so they can catch the
flood peak and follow how the flood evolves; Sentinel-1 GRD is ~20 m, with a fixed 6-12 day repeat, but covers 250 km swaths systematically, so it suits regional
extent, and a same-geometry pre-event T0 almost always exists, which tasked sensors have only where they were tasked in advance. The choice is detail and timing vs coverage and
baseline, depending on whether the dashboard needs a town or a whole basin.

This model would not transfer as is: such sensors are X-band and VV only (Sentinel-1 is
C-band, VV + VH), and 1 m pixels change speckle, texture and the scale a 512 px
patch sees, so it would need fine-tuning on that data, or training from scratch.

## Acknowledgements

- **Sen1Floods11**: Bonafilia, Tellman, Anderson, Issenberg, *Sen1Floods11: A Georeferenced
  Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1*, CVPR Workshops 2020
  ([repository](https://github.com/cloudtostreet/Sen1Floods11)). The Bolivia event metadata
  (`Sen1Floods11_Metadata.geojson`), the normalization constants (`sen1floods11_norm_constants.json`)
  and the S1Hand chip + hand label used by `scripts/validate_chip.py` come from this dataset.
- **Sentinel-1 data**: contains modified Copernicus Sentinel data (2018), provided by the
  European Union / ESA under the free, full and open Copernicus data policy. GRD scenes are read
  from the AWS Open Data `sentinel-s1-l1c` bucket.
- **Earth Search STAC** by [Element 84](https://element84.com/earth-search/)
  (`sentinel-1-grd` and `cop-dem-glo-30` collections), queried with `pystac-client`.
- **Copernicus DEM GLO-30**: produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and
  © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA;
  all rights reserved. Used only with `--dem-filter`.
- **Pretrained model**: [`Governor6191/sar-flood-extent-unet-resnet34`](https://huggingface.co/Governor6191/sar-flood-extent-unet-resnet34)
  on Hugging Face (code: [Governor6191/sar-flood-extent](https://github.com/Governor6191/sar-flood-extent)),
  a UNet with a ResNet34 encoder trained on Sen1Floods11, released under the **MIT license**. The weights are
  downloaded at run time and never modified; the architecture is built with
  [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).
- **Thermal-noise and calibration references**: ESA documents cited in [Preprocessing](#preprocessing).

**AI-assisted development.** Parts of this project were written with Claude (Anthropic) as a
pair-programming assistant: discussing design options, drafting and reviewing code and documentation,
and interpreting diagnostics. I set the design, ran every stage on my machine, validated the results
(run manifests, QGIS inspection, the Sen1Floods11 chip check) and reviewed all code before keeping it.
Every number in this README comes from my own runs.
