"""
freshness.py
------------
Guard against forecasting on stale imagery.

Helioviewer's getJP2Image returns the *nearest available* frame to the
requested timestamp, with no error and no indication of how far off it is.
During an SDO/HMI outage it therefore keeps serving the last good frame.
Between 2026-04-19 04:00 and 2026-04-20 04:00 this produced 25 byte-identical
files, which the pipeline logged as 25 independent hourly forecasts, all with
probability 0.197937.

Duplicates like that inflate the apparent sample size and bias verification
scores. Detect them on ingest and record the hour as *missing* rather than as
a repeated observation.

Long term the stronger fix is to ask the API which frame it actually returned
(getClosestImage reports the real observation time) and reject anything
further from the requested hour than the tolerance. Content hashing needs no
extra request and catches the same failure, so it runs first.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

STALE_LOG = Path("stale_frames.csv")


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_of_file(path: Path) -> str | None:
    try:
        return content_hash(Path(path).read_bytes())
    except OSError:
        return None


def latest_saved_hash(basedir: Path, pattern: str = "*.jp2") -> str | None:
    """Hash of the most recent file already on disk, or None if there is none."""
    files = sorted(Path(basedir).rglob(pattern))
    return hash_of_file(files[-1]) if files else None


def record_stale(requested: datetime, digest: str, note: str = "duplicate of previous frame") -> None:
    """Append one row so outages stay auditable instead of silently vanishing."""
    new = not STALE_LOG.exists()
    with STALE_LOG.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("detected_at,requested_time,sha256,note\n")
        fh.write(
            f"{datetime.now(timezone.utc).isoformat()},"
            f"{requested.isoformat()},{digest},{json.dumps(note)}\n"
        )


def is_stale(content: bytes, previous_hash: str | None) -> bool:
    """True when this download is byte-identical to the last one kept."""
    return previous_hash is not None and content_hash(content) == previous_hash
