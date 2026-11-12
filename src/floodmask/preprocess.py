"""
SAR preprocessing shared by the pipeline and the offline chip check.

Model input contract (from the training repo, see README):
    channel 0 = VV, channel 1 = VH, sigma0 in dB,
    z-scored per channel with the Sen1Floods11 constants below,
    invalid pixels filled with 0 (= the per-channel mean), as in training.

Two steps:
    border_noise(), calibrate(), to_db():  uint16 DN -> sigma0 (noise removed) -> dB   (radar geometry)
    standardize():                         dB -> model input                          (output grid)
"""

import json
import xml.etree.ElementTree as ET
from importlib import resources
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

CHANNELS = ("VV", "VH")  # model channel order
BORDER_DN = 20          # border noise DN is 1-5, real data at the swath edge >= ~45 (5th percentile)
BORDER_SMOOTH_PX = 5    # median along range before thresholding

Lut = tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]  # (vector lines, [(pixels, values)])


def read_lut(xml_path: Path, pol: str, vectors_path: str, value_tag: str) -> Lut:
    """Sparse LUT from a GRD annotation XML (calibration or noise).

    Returns the radar-geometry lines of the vectors, and for each vector its
    (pixels, values) samples.
    """
    root = ET.parse(xml_path).getroot()
    found = root.findtext("adsHeader/polarisation")
    if found != pol.upper():
        raise ValueError(f"{xml_path}: polarisation {found}, expected {pol.upper()}")
    vectors = root.findall(vectors_path)
    if not vectors:
        raise ValueError(f"{xml_path}: no {vectors_path} (IPF >= 2.90 noise format is not supported)")

    def floats(v, tag):
        return np.array(v.findtext(tag).split(), dtype=np.float64)

    lines = np.array([int(v.findtext("line")) for v in vectors])
    return lines, [(floats(v, "pixel"), floats(v, value_tag)) for v in vectors]


def read_sigma_lut(xml_path: Path, pol: str) -> Lut:
    """sigmaNought calibration LUT (calibration-iw-{pol}.xml), K in sigma0 = DN^2 / K^2."""
    return read_lut(xml_path, pol, "calibrationVectorList/calibrationVector", "sigmaNought")


def read_noise_lut(xml_path: Path, pol: str) -> Lut:
    """Thermal noise LUT (noise-iw-{pol}.xml), N in DN^2 units.

    IPF < 2.90 format (our 2018 scenes): one noiseVectorList, range dependent.
    From IPF 2.90 (13 March 2018) the noise is split into range x azimuth
    vectors, which is not handled here.
    """
    return read_lut(xml_path, pol, "noiseVectorList/noiseVector", "noiseLut")


def _interp(lut: Lut, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a sparse LUT at every (row, col) of a window."""
    lines, vectors = lut
    v_cols = np.stack([np.interp(cols, px, v) for px, v in vectors])  # along range, per vector
    j = np.clip(np.searchsorted(lines, rows, side="right") - 1, 0, len(lines) - 2)
    t = np.clip((rows - lines[j]) / (lines[j + 1] - lines[j]), 0, 1)[:, None]
    return (1 - t) * v_cols[j] + t * v_cols[j + 1]  # then along azimuth


def calibrate(dn: np.ndarray, sigma_lut: Lut, noise_lut: Lut, row_off: int = 0, col_off: int = 0) -> np.ndarray:
    """uint16 DN window -> thermal-noise-removed sigma0 (linear, float32). DN 0 is GRD nodata -> NaN.

    sigma0 = max(DN^2 - N, 0) / K^2 (ESA S1 radiometric calibration and
    thermal denoising, as in SNAP and GEE S1_GRD), with K and N bilinearly
    interpolated from their sparse LUTs at each pixel. (row_off, col_off)
    place the window in the full scene, where the LUT coordinates are defined.
    """
    rows = np.arange(row_off, row_off + dn.shape[0])
    cols = np.arange(col_off, col_off + dn.shape[1])
    k = _interp(sigma_lut, rows, cols)
    noise = _interp(noise_lut, rows, cols)

    power = np.maximum(dn.astype(np.float64) ** 2 - noise, 0)
    sigma0 = (power / k ** 2).astype(np.float32)
    sigma0[dn == 0] = np.nan
    return sigma0


def border_noise(dn: np.ndarray, near: bool, far: bool) -> np.ndarray:
    """Mask of GRD border noise in a DN window.

    Before IPF 2.90 the swath edges hold a strip of very low, non-zero DN
    (1-5 in our scenes, vs >= ~45 for real data) instead of 0. Per line, the
    border is the run of low pixels from the swath edge inward; a 5 px median
    along range keeps one speckle pixel from ending or extending it.
    near / far: the window touches the first / last column of the scene.
    """
    pad = BORDER_SMOOTH_PX // 2
    padded = np.pad(dn, ((0, 0), (pad, pad)), mode="edge")
    low = np.median(sliding_window_view(padded, BORDER_SMOOTH_PX, axis=1), axis=-1) < BORDER_DN
    cols = np.arange(dn.shape[1])
    mask = np.zeros(dn.shape, bool)
    if near:
        mask |= cols < np.cumprod(low, axis=1).sum(axis=1)[:, None]
    if far:
        mask |= cols >= dn.shape[1] - np.cumprod(low[:, ::-1], axis=1).sum(axis=1)[:, None]
    return mask


def to_db(sigma0: np.ndarray) -> np.ndarray:
    """Linear sigma0 -> dB, floored at -100 dB. NaN (nodata) stays NaN."""
    return 10 * np.log10(np.maximum(sigma0, 1e-10))


def load_norm_constants() -> dict[str, tuple[float, float]]:
    """Per-channel (mean, std) in dB.

    Recovered by the model author by fitting (dB - mean) / std between the
    original Sen1Floods11 S1Hand dB tiles and the preprocessed-dl tiles the
    model was trained on (R^2 = 1.0). Vendored because the HF repo ships only
    the weights.
    """
    text = resources.files("floodmask").joinpath("sen1floods11_norm_constants.json").read_text()
    ch = json.loads(text)["channels"]
    return {c: (ch[c]["mean"], ch[c]["std"]) for c in CHANNELS}


def standardize(db: np.ndarray, constants: dict) -> tuple[np.ndarray, np.ndarray]:
    """(2, H, W) sigma0 dB -> (z-scored float32 (2, H, W), valid mask (H, W)).

    A pixel is valid only if both channels are finite. Invalid pixels are set
    to 0, the same fill the model saw during training; callers must still
    mask them out of the outputs using the returned `valid` mask.
    """
    if db.shape[0] != 2:
        raise ValueError(f"expected (2, H, W) VV/VH, got {db.shape}")
    valid = np.isfinite(db).all(axis=0)
    out = np.empty(db.shape, dtype=np.float32)
    for i, c in enumerate(CHANNELS):
        mean, std = constants[c]
        out[i] = (db[i] - mean) / std
    out[:, ~valid] = 0.0
    return out, valid
