import requests
import datetime
from pathlib import Path
import cv2
import numpy as np
import time

SOURCE_ID = 19
JP2_DIR = Path("./data/latest_jp2")
JPG_DIR = Path("./data/latest_jpg")

JP2_DIR.mkdir(parents=True, exist_ok=True)
JPG_DIR.mkdir(parents=True, exist_ok=True)

# Each entry is (label, base_url) — tried in order, first reachable wins.
# The IAS mirror serves its API at the root (no /api/ prefix).
API_URLS = [
    ("IAS mirror",  "https://helioviewer.ias.u-psud.fr"),
    ("official",    "https://api.helioviewer.org"),
]


def get_working_api_base(timeout: int = 10) -> str | None:
    """Try each base URL in order and return the first one that responds."""
    probe_date = "2026-01-01T00:00:00.000Z"
    for label, base in API_URLS:
        probe = f"{base}/v2/getJP2Image/?date={probe_date}&sourceId=19&jpip=true"
        try:
            r = requests.get(probe, timeout=timeout)
            if r.status_code < 500:
                print(f"Using API: {label} ({base})")
                return base
        except Exception as e:
            pass
        print(f"Unreachable: {label} ({base})")
    return None


def download_latest_image():
    api_base = get_working_api_base()
    if api_base is None:
        print("All API mirrors are unreachable. Skipping this cycle.")
        return None, None

    now_utc = datetime.datetime.now(datetime.UTC).replace(minute=0, second=0, microsecond=0)
    final_date = now_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    jp2_url = f"{api_base}/v2/getJP2Image/?date={final_date}&sourceId={SOURCE_ID}"

    jp2_name = f"latest_{now_utc.strftime('%Y%m%d_%H%M%S')}.jp2"
    jp2_path = JP2_DIR / jp2_name

    max_retries = 3

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt+1}: downloading from {api_base}...")
            response = requests.get(jp2_url, timeout=60)
            response.raise_for_status()

            jp2_path.write_bytes(response.content)
            print(f"Downloaded: {jp2_path}")
            return jp2_path, now_utc

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(5)

    print("API unavailable after all retries.")
    return None, None


def convert_jp2_to_jpg(jp2_path: Path, timestamp: datetime.datetime):
    image = cv2.imread(str(jp2_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError("cv2.imread returned None - check OpenJPEG support in OpenCV")

    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

    if image.dtype == np.uint16:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    jpg_name = f"latest_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
    jpg_path = JPG_DIR / jpg_name
    cv2.imwrite(str(jpg_path), image)

    stable_latest = JPG_DIR / "latest_image.jpg"
    cv2.imwrite(str(stable_latest), image)

    print(f"Converted: {jpg_path}")
    print(f"Updated stable file: {stable_latest}")

    return jpg_path, stable_latest


if __name__ == "__main__":
    jp2_path, timestamp = download_latest_image()
    if jp2_path is None:
        print("Skipping this cycle - no data available")
    else:
        convert_jp2_to_jpg(jp2_path, timestamp)