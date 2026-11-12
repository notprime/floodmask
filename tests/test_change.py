"""
Change detection on a hand-made T0/T1 pair: flood / PWB logic, nodata
exclusion, minimum-area filter, hole filling, WGS84 output. Offline.

    python -m pytest tests
"""

import numpy as np
from rasterio.transform import Affine

from floodmask import change, georef
from floodmask.inference import LAND, NODATA, WATER

GRID = georef.Grid("EPSG:32720", Affine(10, 0, 260000, 0, -10, 8400000), 100, 100)


def pair():
    t0 = np.full((100, 100), LAND, np.uint8)
    t1 = t0.copy()
    t0[10:30, 10:30] = t1[10:30, 10:30] = WATER  # lake on both dates -> PWB (400 px)
    t1[50:90, 50:90] = WATER                     # new water -> flood (1600 px)
    t1[60:62, 60:62] = LAND                      # 4 px dry hole inside the flood
    t1[5:7, 80:82] = WATER                       # 4 px speckle -> flood, below min area
    t0[70:80, 10:20] = WATER                     # receded water: in neither output
    t1[50:90, 85:90] = NODATA                    # swath edge at T1 cuts the flood
    return t0, t1


def test_logic_and_nodata():
    masks, counts = change.change_masks(*pair())
    assert counts["pwb"] == 400
    assert counts["flood"] == 40 * 35 - 4 + 4
    assert counts["receded"] == 100
    assert counts["nodata"] == 200
    assert not (masks["flood"] & masks["pwb"]).any()


def test_vectorize():
    masks, _ = change.change_masks(*pair())
    flood, stats = change.to_features(masks["flood"], GRID, min_area=2500)
    assert stats["polygons"] == 1 and stats["dropped_polygons"] == 1    # speckle dropped
    f = flood[0]
    assert f["properties"]["area_m2"] == 40 * 35 * 100                   # hole filled
    assert f["properties"]["area_km2"] == 0.14
    assert len(f["geometry"]["coordinates"]) == 1                        # exterior ring only
    lon, lat = np.array(f["geometry"]["coordinates"][0]).T
    assert (-66 < lon).all() and (lon < -64).all() and (-16 < lat).all() and (lat < -14).all()

    flood, stats = change.to_features(masks["flood"], GRID, min_area=0)
    assert stats["polygons"] == 2 and len(flood[0]["geometry"]["coordinates"]) + \
        len(flood[1]["geometry"]["coordinates"]) == 3                    # hole kept


if __name__ == "__main__":
    test_logic_and_nodata()
    test_vectorize()
    print("OK")
