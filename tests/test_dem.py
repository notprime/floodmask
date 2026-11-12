"""
DEM filter on a synthetic DEM in EPSG:4326 (like GLO-30): a plane rising
0.1 m per metre northward (5.71 deg) in the west half, flat in the east half.
Checks the slope after resampling to the UTM grid. Offline.
"""

import numpy as np
import rasterio
from rasterio.transform import Affine, from_bounds
from rasterio.warp import transform, transform_bounds

from floodmask import dem, georef

GRID = georef.Grid("EPSG:32720", Affine(10, 0, 260000, 0, -10, 8400000), 300, 300)


def write_dem(path):
    left, bottom, right, top = transform_bounds(GRID.crs, "EPSG:4326", 259000, 8396000, 264000, 8401000)
    w = h = 200  # ~1 arcsec, as GLO-30
    tf = from_bounds(left, bottom, right, top, w, h)
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    lon, lat = tf.c + cols * tf.a, tf.f + rows * tf.e
    east, north = transform("EPSG:4326", GRID.crs, lon.ravel(), lat.ravel())
    east, north = np.reshape(east, (h, w)), np.reshape(north, (h, w))
    z = np.where(east < 261500, 0.1 * (north - 8396000), 50.0).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32",
                       crs="EPSG:4326", transform=tf) as dst:
        dst.write(z, 1)


def test_slope(tmp_path):
    write_dem(tmp_path / "dem.tif")
    slope = dem.slope_on_grid([tmp_path / "dem.tif"], GRID)
    assert slope.shape == (GRID.height, GRID.width)
    np.testing.assert_allclose(np.median(slope[20:280, 20:120]), np.degrees(np.arctan(0.1)), atol=0.05)
    assert np.nanmax(slope[20:280, 180:280]) < 0.01


def test_no_dem_removes_nothing(tmp_path):
    write_dem(tmp_path / "dem.tif")
    far = georef.Grid(GRID.crs, Affine(10, 0, 400000, 0, -10, 8400000), 50, 50)  # outside the DEM
    slope = dem.slope_on_grid([tmp_path / "dem.tif"], far)
    assert np.isnan(slope).all()
    assert not (slope > 5).any()
