"""
Download the full GRD assets each STAC item needs: the VV/VH measurement COGs
and their calibration and thermal noise XML.

Earth Search links these on the sentinel-s1-l1c S3 bucket, which allows
anonymous access (AWS Open Data; verified, no account needed), although the
STAC items still carry a stale requester-pays flag.

Files land in <cache>/grd/<item_id>/ and are reused on later runs.
"""

import logging
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig

log = logging.getLogger(__name__)

ASSETS = ("vv", "vh", "schema-calibration-vv", "schema-calibration-vh", "schema-noise-vv", "schema-noise-vh")


def fetch(item, cache_dir: Path, region: str) -> dict[str, Path]:
    """Return {asset key: local path} for one STAC item, downloading what is missing."""
    s3 = boto3.client("s3", region_name=region, config=BotoConfig(signature_version=UNSIGNED))
    paths = {}
    for key in ASSETS:
        href = item.assets[key].href
        if not href.startswith("s3://"):
            raise ValueError(f"{item.id}/{key}: expected an s3:// href, got {href}")
        bucket, _, obj = href.removeprefix("s3://").partition("/")
        dest = cache_dir / item.id / Path(obj).name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            log.info("Downloading %s", href)
            part = dest.with_name(dest.name + ".part")  # no half-written file on interruption
            s3.download_file(bucket, obj, str(part))
            part.rename(dest)
        paths[key] = dest
    return paths
