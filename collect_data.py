"""
solar_pipeline.py
-----------------
Adapted from professor's code.
Downloads HMI magnetogram JP2 images from Helioviewer API for 2026-02-01
at a 1-hour cadence, then converts them to JPG.

Steps:
  1. download_from_helioviewer() — fetches JP2s from Helioviewer API
  2. jp2_to_jpg_conversion()    — converts JP2s to resized JPGs via OpenCV
"""

import requests
import datetime
from pathlib import Path
import pandas as pd
import os
import csv
import cv2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START_DATE  = '2026-02-01 00:00:00'   # start of collection
END_DATE    = None                     # None = current UTC time at runtime
CADENCE_MIN = 60                       # 1 image per hour
TOLERANCE   = datetime.timedelta(minutes=60)   # acceptance window for closest image

SOURCE_ID   = 19                       # HMI Line-of-sight magnetogram (4K)
BASEDIR_JP2 = './data/hmi_jp2/'        # where JP2 files are saved
BASEDIR_JPG = './data/hmi_jpg/'        # where JPG files are saved

RESIZE      = True
IMG_WIDTH   = 512
IMG_HEIGHT  = 512


# ---------------------------------------------------------------------------
# Step 1 — Download JP2 images
# ---------------------------------------------------------------------------

def download_from_helioviewer(
    start_date: str            = START_DATE,
    end_date: str | None       = END_DATE,   # None = current UTC time
    cadence_min: int = CADENCE_MIN,
    basedir: str     = BASEDIR_JP2,
    source_id: int   = SOURCE_ID,
    tolerance: datetime.timedelta = TOLERANCE,
) -> int:
    """
    Download HMI magnetogram JP2 images from the Helioviewer API.

    Iterates from start_date to end_date in cadence_min steps.
    If end_date is None, collects up to the current UTC time.
    falls within the tolerance window before downloading.

    Directory structure created:
        basedir/
            year/
                month/
                    day/
                        HMI.m<year>.<month>.<day>_<hour>.<min>.<sec>.jp2

    Returns the number of successfully downloaded files.
    """
    dt     = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.UTC)
    dt_end = datetime.datetime.now(datetime.UTC) if end_date is None else datetime.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.UTC)
    step     = datetime.timedelta(minutes=cadence_min)
    counter  = 0

    end_label = dt_end.strftime("%Y-%m-%d %H:%M:%S") + " (UTC now)" if end_date is None \
                else end_date

    print(f"{'='*60}")
    print(f"  Helioviewer Download — HMI Magnetogram (sourceId={source_id})")
    print(f"  Period  : {start_date}  →  {end_label}")
    print(f"  Cadence : {cadence_min} min  |  Tolerance : {tolerance}")
    print(f"  Save to : {basedir}")
    print(f"{'='*60}\n")

    # Step starts at start_date and walks forward
    current = dt
    while current < dt_end:

        # ---- Build query timestamp string ----
        final_date = current.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        # ---- Create save directory ----
        save_dir = Path(basedir) / f"{current.year}" \
                                  / f"{current.month:02d}" \
                                  / f"{current.day:02d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # ---- Build filename ----
        filename = (
            f"HMI.m{current.year}.{current.month:02d}.{current.day:02d}"
            f"_{current.hour:02d}.{current.minute:02d}.{current.second:02d}.jp2"
        )
        file_path = save_dir / filename

        # Skip if already downloaded
        if file_path.exists():
            print(f"  [SKIP]  {filename}  (already exists)")
            current += step
            continue

        # ---- Step 1a: JPIP request to get timestamp of closest image ----
        jpip_url = (
            f"https://api.helioviewer.org/v2/getJP2Image/"
            f"?date={final_date}&sourceId={source_id}&jpip=true"
        )

        try:
            response  = requests.get(jpip_url, timeout=30)
            url       = str(response.content)

            # Parse received timestamp from URI  e.g. "...2026_02_01__00_05_12..."
            url_tail      = url.rsplit('/', 1)[-1]
            date_str      = url_tail.rsplit('__', 1)[0][:-4]   # strip trailing chars
            date_received = datetime.datetime.strptime(date_str, "%Y_%m_%d__%H_%M_%S")

            delta = abs(current - date_received)
        except Exception as e:
            print(f"  [ERROR] Could not parse JPIP response for {final_date}: {e}")
            current += step
            continue

        # ---- Step 1b: Check tolerance, then download actual JP2 ----
        if delta <= tolerance:
            jp2_url = (
                f"https://api.helioviewer.org/v2/getJP2Image/"
                f"?date={final_date}&sourceId={source_id}"
            )
            try:
                hmi_data = requests.get(jp2_url, timeout=60)
                hmi_data.raise_for_status()
                file_path.write_bytes(hmi_data.content)
                counter += 1
                print(f"  [OK]    {filename}  |  closest: {date_received}  |  Δ={delta}")
            except Exception as e:
                print(f"  [ERROR] Download failed for {final_date}: {e}")
        else:
            print(f"  [SKIP]  {current}  |  closest={date_received}  Δ={delta} > tolerance")

        current += step

    print(f"\n  ✓ Total downloaded: {counter} files\n")
    return counter


# ---------------------------------------------------------------------------
# Step 2 — Convert JP2 → JPG
# ---------------------------------------------------------------------------

def jp2_to_jpg_conversion(
    source:  str  = BASEDIR_JP2,
    destination: str = BASEDIR_JPG,
    resize:  bool = RESIZE,
    width:   int  = IMG_WIDTH,
    height:  int  = IMG_HEIGHT,
) -> None:
    """
    Walk the JP2 source directory, convert every .jp2 file to .jpg,
    and save it mirroring the same year/month/day hierarchy.

    OpenCV is used for both reading (IMREAD_UNCHANGED) and writing.
    Relies on OpenCV being built with OpenJPEG / Jasper support.

    If resize=True, images are resized to (width × height) using INTER_AREA
    (best for downscaling).
    """
    source      = Path(source)
    destination = Path(destination)

    # ---- Collect all JP2 files ----
    jp2_files = sorted(source.rglob("*.jp2"))

    if not jp2_files:
        print(f"  [WARNING] No JP2 files found under {source}")
        return

    print(f"{'='*60}")
    print(f"  JP2 → JPG Conversion")
    print(f"  Source      : {source}")
    print(f"  Destination : {destination}")
    print(f"  Resize      : {resize}  ({width}×{height} px)")
    print(f"  Files found : {len(jp2_files)}")
    print(f"{'='*60}\n")

    errors = []

    for i, jp2_path in enumerate(jp2_files):
        # Reconstruct timestamp from filename
        # Pattern: HMI.m<year>.<month>.<day>_<hour>.<min>.<sec>.jp2
        stem = jp2_path.stem   # e.g. HMI.m2026.02.01_00.00.00
        try:
            ts_part = stem.split('HMI.m')[1]           # 2026.02.01_00.00.00
            dt = datetime.datetime.strptime(ts_part, "%Y.%m.%d_%H.%M.%S")
        except (IndexError, ValueError) as e:
            print(f"  [ERROR] Cannot parse timestamp from {jp2_path.name}: {e}")
            errors.append(jp2_path)
            continue

        # Build destination path
        out_dir = destination / f"{dt.year}" / f"{dt.month:02d}" / f"{dt.day:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        jpg_name = (
            f"HMI.m{dt.year}.{dt.month:02d}.{dt.day:02d}"
            f"_{dt.hour:02d}.{dt.minute:02d}.{dt.second:02d}.jpg"
        )
        jpg_path = out_dir / jpg_name

        if jpg_path.exists():
            print(f"  [{i+1:03d}] SKIP  {jpg_name}  (already converted)")
            continue

        try:
            image = cv2.imread(str(jp2_path), cv2.IMREAD_UNCHANGED)

            if image is None:
                raise ValueError("cv2.imread returned None — check OpenJPEG support in OpenCV")

            if resize:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

            # Normalise 16-bit to 8-bit if needed (HMI raw data can be 16-bit)
            if image.dtype == np.uint16:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            cv2.imwrite(str(jpg_path), image)
            print(f"  [{i+1:03d}] OK    {jpg_name}")

        except Exception as e:
            print(f"  [{i+1:03d}] ERROR {jp2_path.name}: {e}")
            errors.append(jp2_path)

    print(f"\n  ✓ Conversion complete.")
    print(f"  ✓ Successful : {len(jp2_files) - len(errors)} / {len(jp2_files)}")
    if errors:
        print(f"  ✗ Failed     : {len(errors)}")
        for p in errors:
            print(f"      {p}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np   # needed inside jp2_to_jpg_conversion for 16-bit norm

    # Step 1: Download JP2 images
    download_from_helioviewer()

    # Step 2: Convert JP2 → JPG
    jp2_to_jpg_conversion()