import re
import requests
import datetime
from pathlib import Path
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START_DATE  = '2026-02-01 00:00:00'   # fallback if no files exist yet
END_DATE    = None
CADENCE_MIN = 60
TOLERANCE   = datetime.timedelta(minutes=60)

SOURCE_ID   = 19
BASEDIR_JP2 = './data/hmi_jp2/'
BASEDIR_JPG = './data/hmi_jpg/'

RESIZE     = True
IMG_WIDTH  = 512
IMG_HEIGHT = 512

API_URLS = [
    ("IAS mirror", "https://api.gs671-suske.ndc.nasa.gov/"),
    ("official",   "https://api.helioviewer.org"),
]


def get_working_api_base(timeout: int = 10) -> str | None:
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


def parse_hmi_time_from_filename(path: Path) -> datetime.datetime | None:
    """
    Parse time from:
    HMI.m2026.03.08_05.00.00.jp2
    HMI.m2026.03.08_05.00.00.jpg
    """
    match = re.search(r"HMI\.m(\d{4})\.(\d{2})\.(\d{2})_(\d{2})\.(\d{2})\.(\d{2})", path.name)
    if not match:
        return None
    y, mo, d, h, mi, s = map(int, match.groups())
    return datetime.datetime(y, mo, d, h, mi, s)


def get_resume_start_time(
    basedir_jp2: str = BASEDIR_JP2,
    fallback_start: str = START_DATE,
    cadence_min: int = CADENCE_MIN,
) -> datetime.datetime:
    """
    Find the newest existing JP2 file and resume from the next cadence step.
    If no JP2 exists yet, use START_DATE.
    """
    jp2_dir = Path(basedir_jp2)
    jp2_files = list(jp2_dir.rglob("*.jp2"))

    latest_dt = None
    for p in jp2_files:
        dt = parse_hmi_time_from_filename(p)
        if dt is not None and (latest_dt is None or dt > latest_dt):
            latest_dt = dt

    if latest_dt is None:
        start_dt = datetime.datetime.strptime(fallback_start, "%Y-%m-%d %H:%M:%S")
        print(f"  [RESUME] No existing JP2 files found. Starting from fallback: {start_dt}")
        return start_dt

    resume_dt = latest_dt + datetime.timedelta(minutes=cadence_min)
    print(f"  [RESUME] Latest existing JP2: {latest_dt}")
    print(f"  [RESUME] Resuming from:       {resume_dt}")
    return resume_dt


def download_from_helioviewer(
    start_date: str | None        = None,
    end_date: str | None          = END_DATE,
    cadence_min: int              = CADENCE_MIN,
    basedir: str                  = BASEDIR_JP2,
    source_id: int                = SOURCE_ID,
    tolerance: datetime.timedelta = TOLERANCE,  # kept only so your function signature stays similar
) -> int:
    api_base = get_working_api_base()
    if api_base is None:
        print("  [ERROR] All API mirrors are unreachable. Aborting.")
        return 0

    if start_date is None:
        dt = get_resume_start_time(
            basedir_jp2=basedir,
            fallback_start=START_DATE,
            cadence_min=cadence_min,
        )
    else:
        dt = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")

    dt_end = (
        datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        if end_date is None
        else datetime.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
    )

    step = datetime.timedelta(minutes=cadence_min)
    counter = 0

    if dt >= dt_end:
        print("  [INFO] No new timestamps to download.")
        return 0

    end_label = (
        dt_end.strftime("%Y-%m-%d %H:%M:%S") + " (UTC now)"
        if end_date is None else end_date
    )

    print(f"{'='*60}")
    print(f"  Helioviewer Download — HMI Magnetogram (sourceId={source_id})")
    print(f"  Period  : {dt.strftime('%Y-%m-%d %H:%M:%S')}  ->  {end_label}")
    print(f"  Cadence : {cadence_min} min")
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

        jp2_url = f"{api_base}/v2/getJP2Image/?date={final_date}&sourceId={source_id}"

        try:
            response = requests.get(jp2_url, timeout=60)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()

            # reject obvious HTML/error pages
            if "text/html" in content_type or response.text.lstrip().startswith("<!DOCTYPE html") or response.text.lstrip().startswith("<html"):
                print(f"  [SKIP]  {filename}  | server returned HTML instead of JP2")
                current += step
                continue

            # optional small-size guard for bad responses
            if len(response.content) < 1000:
                print(f"  [SKIP]  {filename}  | response too small to be a valid JP2")
                current += step
                continue

            file_path.write_bytes(response.content)
            counter += 1
            print(f"  [OK]    {filename}")

        except Exception as e:
            print(f"  [ERROR] Download failed for {final_date}: {e}")

        current += step

    print(f"\n  Total downloaded: {counter} files\n")
    return counter


def jp2_to_jpg_conversion(
    source: str = BASEDIR_JP2,
    destination: str = BASEDIR_JPG,
    resize: bool = RESIZE,
    width: int = IMG_WIDTH,
    height: int = IMG_HEIGHT,
) -> None:
    source = Path(source)
    destination = Path(destination)
    jp2_files = sorted(source.rglob("*.jp2"))

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


if __name__ == '__main__':
    download_from_helioviewer()
    jp2_to_jpg_conversion()