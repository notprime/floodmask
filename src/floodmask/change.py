"""
Change detection: water_t0 + water_t1 -> PWB and flood polygons (GeoJSON).

Logic (pixel to pixel, T0 and T1 share one grid):
  valid = both dates valid (not nodata)
  pwb   = water_t1 & water_t0    stable water: rivers, lakes, oxbows
  flood = water_t1 & ~water_t0   new water at T1
A pixel nodata on either date is in neither output.

Vector outputs:
  - polygonize on the output grid (4-connected: no polygons touching only at a corner)
  - minimum mapping unit: drop polygons smaller than min_area (m2), and fill
    holes smaller than min_area (speckle-size islands inside water)
  - area_m2 computed in the grid CRS (UTM), then geometry reprojected to
    WGS84 lon/lat, as GeoJSON requires (RFC 7946), 6 decimals (~0.1 m)
"""

import json
import logging
from pathlib import Path

import numpy as np
from rasterio.features import shapes
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, mapping, shape

from floodmask.georef import Grid
from floodmask.inference import NODATA, WATER

log = logging.getLogger(__name__)

LOGIC = {"valid": "water_t0 != nodata & water_t1 != nodata",
         "pwb": "water_t1 & water_t0",
         "flood": "water_t1 & ~water_t0"}


def change_masks(water_t0: np.ndarray, water_t1: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """uint8 masks -> {"pwb", "flood"} bool masks, and a T0/T1 area breakdown in km2."""
    valid = (water_t0 != NODATA) & (water_t1 != NODATA)
    w0, w1 = valid & (water_t0 == WATER), valid & (water_t1 == WATER)
    masks = {"pwb": w1 & w0, "flood": w1 & ~w0}
    # The other two transitions are not outputs, but tell whether the change is plausible.
    classes = {**masks, "receded": w0 & ~w1, "dry": valid & ~w0 & ~w1, "nodata": ~valid}
    return masks, {k: int(v.sum()) for k, v in classes.items()}


def _fill_small_holes(poly: Polygon, min_area: float) -> Polygon:
    return Polygon(poly.exterior, [r for r in poly.interiors if Polygon(r).area >= min_area])


def _round(coords, ndigits: int = 6):
    if isinstance(coords[0], (int, float)):
        return [round(c, ndigits) for c in coords]
    return [_round(c, ndigits) for c in coords]


def to_features(mask: np.ndarray, grid: Grid, min_area: float) -> tuple[list[dict], dict]:
    """Bool mask -> GeoJSON features in WGS84, and polygon stats for the manifest.
    min_area is in m2 (grid CRS units), so it does not depend on the pixel size."""
    polys, dropped, dropped_area = [], 0, 0.0
    for geom, _ in shapes(mask.astype(np.uint8), mask=mask, connectivity=4, transform=grid.transform):
        poly = shape(geom)
        if poly.area < min_area:
            dropped, dropped_area = dropped + 1, dropped_area + poly.area
            continue
        polys.append(_fill_small_holes(poly, min_area))
    # One call for all polygons: rasterio sets up its GDAL environment on every call.
    lonlat = transform_geom(grid.crs, "EPSG:4326", [mapping(p) for p in polys]) if polys else []
    features = [{"type": "Feature",
                 "properties": {"id": i, "area_m2": round(p.area, 1), "area_km2": round(p.area / 1e6, 4)},
                 "geometry": {"type": "Polygon", "coordinates": _round(g["coordinates"])}}
                for i, (p, g) in enumerate(zip(polys, lonlat))]
    kept_area = sum(f["properties"]["area_m2"] for f in features)
    stats = {"pixels": int(mask.sum()), "polygons": len(features), "area_km2": round(kept_area / 1e6, 3),
             "dropped_polygons": dropped, "dropped_area_km2": round(dropped_area / 1e6, 3)}
    return features, stats


def write_geojson(path: Path, features: list[dict]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
