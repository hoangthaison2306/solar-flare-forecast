"""
solar_pipeline.py
-----------------
Downloads HMI magnetogram JP2 images from Helioviewer API,
then converts them to JPG.
"""

import requests
import datetime
from pathlib import Path
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START_DATE  = '2026-02-01 00:00:00'
END_DATE    = None
CADENCE_MIN = 60
TOLERANCE   = datetime.timedelta(minutes=60)

SOURCE_ID   = 19
BASEDIR_JP2 = './data/hmi_jp2/'
BASEDIR_JPG = './data/hmi_jpg/'

RESIZE     = True
IMG_WIDTH  = 512
IMG_HEIGHT = 512

# Each entry is (label, base_url) — tried in order, first reachable wins.
# The IAS mirror serves its API at the root (no /api/ prefix).
API_URLS = [
    ("IAS mirror", "https://helioviewer.ias.u-psud.fr"),
    ("official",   "https://api.helioviewer.org"),
]


def get_working_api_base(timeout: int = 10) -> str | None:
    """Try each base URL in order and return the first one that responds."""
    probe_date = "2026-01-01T00:00:00.000Z"
    for label, base in API_URLS:
        probe = f"{base}/v2/getJP2Image/?date={probe_date}&sourceId=19&jpip=true"
        try:
            r = requests.get(probe, timeout=timeout)
            if r.status_code < 500:
                print(f"  [API]   Using: {label} ({base})")
                return base
        except Exception:
            pass
        print(f"  [API]   Unreachable: {label} ({base})")
    return None


# ---------------------------------------------------------------------------
# Step 1 — Download JP2 images
# ---------------------------------------------------------------------------

def download_from_helioviewer(
    start_date: str               = START_DATE,
    end_date: str | None          = END_DATE,
    cadence_min: int              = CADENCE_MIN,
    basedir: str                  = BASEDIR_JP2,
    source_id: int                = SOURCE_ID,
    tolerance: datetime.timedelta = TOLERANCE,
) -> int:
    api_base = get_working_api_base()
    if api_base is None:
        print("  [ERROR] All API mirrors are unreachable. Aborting.")
        return 0

    dt = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    dt_end = (
        datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        if end_date is None
        else datetime.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
    )
    step    = datetime.timedelta(minutes=cadence_min)
    counter = 0

    end_label = (
        dt_end.strftime("%Y-%m-%d %H:%M:%S") + " (UTC now)"
        if end_date is None else end_date
    )

    print(f"{'='*60}")
    print(f"  Helioviewer Download — HMI Magnetogram (sourceId={source_id})")
    print(f"  Period  : {start_date}  ->  {end_label}")
    print(f"  Cadence : {cadence_min} min  |  Tolerance : {tolerance}")
    print(f"  Save to : {basedir}")
    print(f"{'='*60}\n")

    current = dt
    while current < dt_end:

        final_date = current.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        save_dir = (
            Path(basedir)
            / f"{current.year}"
            / f"{current.month:02d}"
            / f"{current.day:02d}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = (
            f"HMI.m{current.year}.{current.month:02d}.{current.day:02d}"
            f"_{current.hour:02d}.{current.minute:02d}.{current.second:02d}.jp2"
        )
        file_path = save_dir / filename

        if file_path.exists():
            print(f"  [SKIP]  {filename}  (already exists)")
            current += step
            continue

        jpip_url = f"{api_base}/v2/getJP2Image/?date={final_date}&sourceId={source_id}&jpip=true"

        try:
            response      = requests.get(jpip_url, timeout=30)
            url           = str(response.content)
            url_tail      = url.rsplit('/', 1)[-1]
            date_str      = url_tail.rsplit('__', 1)[0][:-4]
            date_received = datetime.datetime.strptime(date_str, "%Y_%m_%d__%H_%M_%S")
            delta         = abs(current - date_received)
        except Exception as e:
            print(f"  [ERROR] Could not parse JPIP response for {final_date}: {e}")
            current += step
            continue

        if delta <= tolerance:
            jp2_url = f"{api_base}/v2/getJP2Image/?date={final_date}&sourceId={source_id}"
            try:
                hmi_data = requests.get(jp2_url, timeout=60)
                hmi_data.raise_for_status()
                file_path.write_bytes(hmi_data.content)
                counter += 1
                print(f"  [OK]    {filename}  |  closest: {date_received}  |  delta={delta}")
            except Exception as e:
                print(f"  [ERROR] Download failed for {final_date}: {e}")
        else:
            print(f"  [SKIP]  {current}  |  closest={date_received}  delta={delta} > tolerance")

        current += step

    print(f"\n  Total downloaded: {counter} files\n")
    return counter


# ---------------------------------------------------------------------------
# Step 2 — Convert JP2 -> JPG
# ---------------------------------------------------------------------------

def jp2_to_jpg_conversion(
    source:      str  = BASEDIR_JP2,
    destination: str  = BASEDIR_JPG,
    resize:      bool = RESIZE,
    width:       int  = IMG_WIDTH,
    height:      int  = IMG_HEIGHT,
) -> None:
    source      = Path(source)
    destination = Path(destination)
    jp2_files   = sorted(source.rglob("*.jp2"))

    if not jp2_files:
        print(f"  [WARNING] No JP2 files found under {source}")
        return

    print(f"{'='*60}")
    print(f"  JP2 -> JPG Conversion  |  {len(jp2_files)} files")
    print(f"{'='*60}\n")

    errors = []

    for i, jp2_path in enumerate(jp2_files):
        stem = jp2_path.stem
        try:
            ts_part = stem.split('HMI.m')[1]
            dt = datetime.datetime.strptime(ts_part, "%Y.%m.%d_%H.%M.%S")
        except (IndexError, ValueError) as e:
            print(f"  [ERROR] Cannot parse timestamp from {jp2_path.name}: {e}")
            errors.append(jp2_path)
            continue

        out_dir = destination / f"{dt.year}" / f"{dt.month:02d}" / f"{dt.day:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        jpg_name = (
            f"HMI.m{dt.year}.{dt.month:02d}.{dt.day:02d}"
            f"_{dt.hour:02d}.{dt.minute:02d}.{dt.second:02d}.jpg"
        )
        jpg_path = out_dir / jpg_name

        if jpg_path.exists():
            print(f"  [{i+1:03d}] SKIP  {jpg_name}")
            continue

        try:
            image = cv2.imread(str(jp2_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("cv2.imread returned None")
            if resize:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            if image.dtype == np.uint16:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(str(jpg_path), image)
            print(f"  [{i+1:03d}] OK    {jpg_name}")
        except Exception as e:
            print(f"  [{i+1:03d}] ERROR {jp2_path.name}: {e}")
            errors.append(jp2_path)

    print(f"\n  Successful: {len(jp2_files) - len(errors)} / {len(jp2_files)}")
    if errors:
        print(f"  Failed:     {len(errors)}")
        for p in errors:
            print(f"      {p}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    download_from_helioviewer()
    jp2_to_jpg_conversion()