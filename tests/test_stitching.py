"""
Patch tiling + stitching must not change the result of a pointwise model:
the tiled prediction equals the full-image one for any patch size / overlap,
and nodata stays nodata. Runs offline, no weights needed.

    python -m pytest tests      (or: python tests/test_stitching.py)
"""

import numpy as np
import torch
from rasterio.transform import Affine

from floodmask import georef, inference
from floodmask.preprocess import CHANNELS, load_norm_constants, standardize


def pointwise_model(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x[:, 0] - x[:, 1])


def check(tmp_dir, h, w, patch, overlap):
    db = np.random.default_rng(0).normal(-15, 5, (2, h, w)).astype(np.float32)
    db[:, :, -20:] = np.nan  # swath edge
    db[1, 5, 5] = np.nan     # one channel invalid
    grid = georef.Grid("EPSG:32720", Affine(10, 0, 500000, 0, -10, 8400000), w, h)
    path = tmp_dir / f"stack_{h}x{w}.tif"
    georef.write_raster(path, db, grid, nodata=float("nan"), names=CHANNELS)

    prob = inference.predict(pointwise_model, inference.PatchDataset(path, patch, overlap), batch_size=3)

    x, valid = standardize(db, load_norm_constants())
    expected = pointwise_model(torch.from_numpy(x)[None])[0].numpy()
    assert np.array_equal(np.isnan(prob), ~valid)
    np.testing.assert_allclose(prob[valid], expected[valid], atol=1e-5)
    assert ((inference.to_mask(prob, 0.5) == inference.NODATA) == ~valid).all()


def test_stitching(tmp_path):
    for h, w, patch, overlap in [(1100, 1300, 512, 64),  # regular grid + shifted last window
                                 (1100, 1300, 256, 128),
                                 (1100, 1300, 512, 0),
                                 (300, 700, 512, 64)]:   # smaller than one patch: padded
        check(tmp_path, h, w, patch, overlap)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        test_stitching(Path(d))
    print("OK")
