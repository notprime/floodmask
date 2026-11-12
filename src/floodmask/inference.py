"""
Patch-based inference: sigma0 dB GeoTIFF -> water probability on the same grid.

Patch contract:
  - Tiling: patch_size x patch_size windows, stride = patch_size - overlap.
    The last row/column of windows is shifted back to end on the raster edge,
    so every patch is full-size, real data. Only a raster smaller than one
    patch is padded (NaN, i.e. nodata).
  - Input: each patch is read from the GeoTIFF and standardized on the fly
    (preprocess.standardize). Invalid pixels -> 0, as in training.
  - Stitching: the model gives a softmax water probability per patch.
    Overlapping patches are blended with a weighted average. The weight ramps
    linearly from the patch edge to 1 at `overlap` pixels inside it, so two
    neighbours cross-fade and the patch borders (less context) count less.
  - Output: water = blended probability > threshold; nodata where the input
    is invalid (outside the swath, or DN 0).
"""

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset

from floodmask.model import FloodDetector
from floodmask.preprocess import CHANNELS, load_norm_constants, standardize

LAND, WATER, NODATA = 0, 1, 255  # values in water_t{0,1}.tif


def _starts(size: int, patch: int, stride: int) -> list[int]:
    """Window offsets along one axis: regular stride, last one flush with the edge."""
    last = max(size - patch, 0)
    return list(range(0, last, stride)) + [last]


def _blend_weights(patch: int, overlap: int) -> np.ndarray:
    """(patch, patch) weights: linear ramp over the `overlap` border, 1 inside."""
    i = np.arange(patch) + 0.5
    ramp = np.minimum(1.0, np.minimum(i, patch - i) / max(overlap, 1))
    return np.outer(ramp, ramp).astype(np.float32)


class PatchDataset(Dataset):
    """Standardized (VV, VH) patches of a sigma0 dB GeoTIFF written by georef."""

    def __init__(self, path, patch_size: int, overlap: int):
        if patch_size % 32:
            raise ValueError(f"patch_size {patch_size}: must be a multiple of 32 (UNet downsampling)")
        if not 0 <= overlap < patch_size:
            raise ValueError(f"overlap {overlap}: must be in [0, patch_size)")
        with rasterio.open(path) as src:
            if src.descriptions != CHANNELS:
                raise ValueError(f"{path}: bands {src.descriptions}, expected {CHANNELS}")
            self.height, self.width = src.height, src.width
        self.path, self.patch_size, self.overlap = path, patch_size, overlap
        self.constants = load_norm_constants()
        stride = patch_size - overlap
        self.windows = [(r, c) for r in _starts(self.height, patch_size, stride)
                        for c in _starts(self.width, patch_size, stride)]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i):
        """-> (standardized (2, P, P) float32, (row, col) of the patch in the raster)."""
        r, c = self.windows[i]
        # Opened per patch: cheap next to the forward pass, and safe with DataLoader workers.
        # boundless only pads when the raster is smaller than a patch.
        with rasterio.open(self.path) as src:
            db = src.read(window=Window(c, r, self.patch_size, self.patch_size),
                          boundless=True, fill_value=np.nan)
        x, _ = standardize(db, self.constants)
        return torch.from_numpy(x), torch.tensor([r, c])


def predict(detector: FloodDetector, ds: PatchDataset, batch_size: int) -> np.ndarray:
    """Blended water probability (H, W) float32; NaN where the input is invalid."""
    p = ds.patch_size
    weight = _blend_weights(p, ds.overlap)
    acc = np.zeros((ds.height, ds.width), np.float32)
    wsum = np.zeros_like(acc)
    for x, offsets in DataLoader(ds, batch_size=batch_size):
        prob = detector(x).cpu().numpy()
        for pr, (r, c) in zip(prob, offsets.tolist()):
            h, w = min(p, ds.height - r), min(p, ds.width - c)  # < p only for a tiny raster
            acc[r:r + h, c:c + w] += pr[:h, :w] * weight[:h, :w]
            wsum[r:r + h, c:c + w] += weight[:h, :w]
    prob = acc / wsum  # every pixel is covered by at least one window

    with rasterio.open(ds.path) as src:
        valid = np.isfinite(src.read()).all(axis=0)
    prob[~valid] = np.nan
    return prob


def to_mask(prob: np.ndarray, threshold: float) -> np.ndarray:
    """Probability -> uint8 LAND / WATER / NODATA."""
    mask = np.where(prob > threshold, WATER, LAND).astype(np.uint8)
    mask[np.isnan(prob)] = NODATA
    return mask
