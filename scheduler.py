"""
scheduler.py
------------
Run this once in a separate terminal alongside your Streamlit app.
It executes scrape_ssw.py immediately on startup, then repeats every hour.

Usage:
    python scheduler.py

Keep this terminal open while you want scraping to continue.
"""

import schedule
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime


SCRIPT = Path(__file__).parent / "scrape_ssw.py"


def run_scraper():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] Running scrape_ssw.py ...")
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(result.stdout)
            print(f"[{now}] ✓ Scrape complete.")
        else:
            print(f"[{now}] ✗ Scraper exited with code {result.returncode}")
            if result.stderr:
                print("STDERR:", result.stderr)
    except Exception as e:
        print(f"[{now}] ✗ Failed to run scraper: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("  Solar Flare Scraper — Hourly Scheduler")
    print("  scrape_ssw.py will run every 60 minutes.")
    print("  Press Ctrl+C to stop.")
    print("=" * 50)

    # Run immediately on startup so you don't wait an hour for first data
    run_scraper()

    # Then schedule every hour
    schedule.every(60).minutes.do(run_scraper)

    while True:
        schedule.run_pending()
        time.sleep(30)   # check every 30 s — low CPU cost