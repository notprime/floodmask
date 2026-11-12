"""
STAC discovery: pick the T0 (pre) and T1 (post) Sentinel-1 GRD scenes.

T1 = the pass on the event date; 
T0 = the latest earlier pass on the same relative orbit and orbit direction. 

Over the Beni sub-AOI each pass is cut into two adjacent GRD slices, 
so each date maps to a *list* of items.
"""

import logging
from datetime import date, timedelta

from pystac_client import Client

from floodmask.config import Config

log = logging.getLogger(__name__)


def discover(cfg: Config) -> tuple[dict, dict]:
    """Return {"t0": [items], "t1": [items]} and a JSON summary for the manifest."""
    t1 = date.fromisoformat(cfg.t1_date)
    start = t1 - timedelta(days=cfg.t0_search_days)
    search = Client.open(cfg.stac_url).search(
        collections=[cfg.collection], bbox=cfg.aoi_bbox,
        datetime=f"{start}/{t1}T23:59:59Z")

    # Keep our track only; each date's pass may come as several slices.
    by_date: dict[date, list] = {}
    for it in search.items():
        p = it.properties
        if p["sat:relative_orbit"] == cfg.relative_orbit and p["sat:orbit_state"] == cfg.orbit_state:
            by_date.setdefault(it.datetime.date(), []).append(it)

    if t1 not in by_date:
        raise RuntimeError(f"No scene on orbit {cfg.relative_orbit} on {t1}")
    earlier = [d for d in by_date if d < t1]
    if not earlier:
        raise RuntimeError(f"No pre-event scene in the {cfg.t0_search_days} days before {t1}")
    t0 = max(earlier)  # latest pre-event pass

    items = {"t0": by_date[t0], "t1": by_date[t1]}
    summary = {k: {"date": v[0].datetime.date().isoformat(),
                   "item_ids": sorted(i.id for i in v)} for k, v in items.items()}
    for k, s in summary.items():
        log.info("%s: %s, %d slice(s)", k.upper(), s["date"], len(s["item_ids"]))
    summary["search"] = {"stac_url": cfg.stac_url, "collection": cfg.collection,
                         "bbox": list(cfg.aoi_bbox), "window": [str(start), str(t1)],
                         "orbit": cfg.relative_orbit, "orbit_state": cfg.orbit_state,
                         "candidate_dates": sorted(str(d) for d in by_date),
                         "rule": "T1 = pass on event date; T0 = latest earlier pass, same orbit"}
    return items, summary
