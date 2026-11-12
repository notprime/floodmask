"""
Multilook (10 m -> coarser) on tiny arrays: linear-power mean, nodata rule,
edge padding, grid scaling. Offline.

    python -m pytest tests
"""

import numpy as np
from rasterio.transform import Affine

from floodmask import georef
from floodmask.preprocess import to_db


def test_mean_is_linear_not_db():
    block = np.ones((1, 3, 3), np.float32)
    block[0, 0, 0] = 100.0
    out = georef.multilook(block, 3)
    assert out.shape == (1, 1, 1)
    assert np.isclose(out[0, 0, 0], 108 / 9)
    # dB of the linear mean is higher than the mean of the dBs (geometric mean biases low)
    assert to_db(out)[0, 0, 0] > to_db(block).mean()


def test_nodata_excluded_and_valid_fraction():
    a = np.full((1, 6, 3), np.nan, np.float32)
    a[0, :3] = 2.0
    a[0, 0, :2] = np.nan                      # 7 of 9 valid -> mean of the valid ones
    a[0, 3, :2] = 4.0                         # 2 of 9 valid (< MIN_VALID_FRACTION) -> NaN
    out = georef.multilook(a, 3)
    assert out.shape == (1, 2, 1)
    assert out[0, 0, 0] == 2.0
    assert np.isnan(out[0, 1, 0])


def test_edges_padded_not_cropped():
    a = np.full((1, 4, 5), 3.0, np.float32)   # 4 x 5 is not a multiple of 3
    out = georef.multilook(a, 3)
    assert out.shape == (1, 2, 2)             # ceil
    assert out[0, 0, 0] == 3.0                # full block
    assert out[0, 0, 1] == 3.0                # 3 x 2 = 6 of 9 real pixels >= 0.5: mean of real ones only
    assert np.isnan(out[0, 1, 0]) and np.isnan(out[0, 1, 1])   # 3 and 2 of 9 real pixels: too few


def test_coarsen_grid():
    g = georef.Grid("EPSG:32720", Affine(10, 0, 260000, 0, -10, 8400000), 100, 100)
    c = georef.coarsen_grid(g, 3)
    assert (c.transform.a, c.transform.e) == (30, -30)
    assert (c.transform.c, c.transform.f) == (260000, 8400000)
    assert (c.width, c.height) == (34, 34)
    assert c.crs == g.crs
