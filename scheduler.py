"""
scheduler.py
------------
Run this once in a separate terminal alongside your Streamlit app.

It executes the main pipeline immediately on startup, then repeats every hour:
1. collect_data.py
2. predict.py
3. scrape_ssw.py

Usage:
    python scheduler.py

Keep this terminal open while you want the pipeline to continue.
"""

import schedule
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent

COLLECT_SCRIPT = BASE_DIR / "collect_data.py"
PREDICT_SCRIPT = BASE_DIR / "predict.py"
SCRAPE_SCRIPT  = BASE_DIR / "scrape_ssw.py"


def run_script(script_path: Path) -> bool:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] Running {script_path.name} ...")

    if not script_path.exists():
        print(f"[{now}] ✗ File not found: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
        )

        if result.stdout:
            print(result.stdout.strip())

        if result.returncode == 0:
            print(f"[{now}] ✓ {script_path.name} complete.")
            return True
        else:
            print(f"[{now}] ✗ {script_path.name} exited with code {result.returncode}")
            if result.stderr:
                print("STDERR:")
                print(result.stderr.strip())
            return False

    except Exception as e:
        print(f"[{now}] ✗ Failed to run {script_path.name}: {e}")
        return False


def run_pipeline():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 60)
    print(f"[{now}] Starting hourly pipeline")
    print("=" * 60)

    ok_collect = run_script(COLLECT_SCRIPT)
    if not ok_collect:
        print(f"[{now}] Pipeline stopped: collect_data.py failed.")
        return

    ok_predict = run_script(PREDICT_SCRIPT)
    if not ok_predict:
        print(f"[{now}] Pipeline stopped: predict.py failed.")
        return

    ok_scrape = run_script(SCRAPE_SCRIPT)
    if not ok_scrape:
        print(f"[{now}] Pipeline stopped: scrape_ssw.py failed.")
        return

    done = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{done}] ✓ Hourly pipeline complete.")


if __name__ == "__main__":
    print("=" * 60)
    print("  Solar Flare Pipeline — Hourly Scheduler")
    print("  Runs: collect_data.py -> predict.py -> scrape_ssw.py")
    print("  First run starts immediately, then repeats every 60 minutes.")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    # Run immediately on startup
    run_pipeline()

    # Then run every hour
    schedule.every(60).minutes.do(run_pipeline)

    while True:
        schedule.run_pending()
        time.sleep(30)