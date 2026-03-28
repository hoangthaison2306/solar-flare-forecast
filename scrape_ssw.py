import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "https://www.lmsal.com/solarsoft/ssw/last_events-2026/last_events_{}/index.html"
OUTPUT_FILE = Path("lmsal_all_2026_clean.csv")

COLUMNS = [
    "Event#",
    "EName",
    "Start",
    "Stop",
    "Peak",
    "GOES Class",
    "Derived Position",
]


def crawl_single_page(url: str) -> list[list[str]]:
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        tables = soup.find_all("table")

        rows = []

        for table in tables:
            for tr in table.find_all("tr"):
                cells = tr.find_all("td")
                if not cells:
                    continue

                row = [td.get_text(" ", strip=True) for td in cells]

                # keep only rows that look like actual event rows
                if len(row) >= 7 and row[0].strip().isdigit():
                    rows.append(row[:7])

        return rows

    except Exception as e:
        print(f"Failed to crawl {url}: {e}")
        return []


def load_existing_data(file_path: Path) -> tuple[pd.DataFrame, set, datetime]:
    if file_path.exists():
        df_existing = pd.read_csv(file_path)

        if not df_existing.empty:
            df_existing["Start"] = pd.to_datetime(df_existing["Start"], errors="coerce")

            # remove bad rows just in case
            df_existing = df_existing.dropna(subset=["Start"]).copy()

            seen = set(
                zip(
                    df_existing["Start"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                    df_existing["GOES Class"].astype(str),
                )
            )

            last_time = df_existing["Start"].max()

            # go back 1 hour for safety
            start_time = last_time - timedelta(hours=1)

            print(f"Existing file found: {file_path}")
            print(f"Loaded {len(df_existing)} existing rows")
            print(f"Resuming from: {start_time}")

            return df_existing, seen, start_time

    print("No existing file found. Starting from 2026-01-01 00:01")
    empty_df = pd.DataFrame(columns=["Start", "GOES Class", "Class_Type"])
    return empty_df, set(), datetime(2026, 1, 1, 0, 1)


def main():
    df_existing, seen, current = load_existing_data(OUTPUT_FILE)

    end = datetime.now()
    step = timedelta(hours=1)

    all_new_rows = []

    while current <= end:
        timestamp = current.strftime("%Y%m%d_%H%M")
        url = BASE_URL.format(timestamp)

        print(f"Fetching: {timestamp}")

        rows = crawl_single_page(url)

        for row in rows:
            start_value = row[2].strip()
            goes_value = row[5].strip()

            key = (start_value, goes_value)

            if key not in seen:
                seen.add(key)
                all_new_rows.append(row)

        current += step

    if all_new_rows:
        df_new = pd.DataFrame(all_new_rows, columns=COLUMNS)

        df_new["Start"] = pd.to_datetime(df_new["Start"], errors="coerce")
        df_new["Class_Type"] = df_new["GOES Class"].astype(str).str[0]

        df_new = df_new[["Start", "GOES Class", "Class_Type"]]
        df_new = df_new.dropna(subset=["Start"]).copy()

        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        print("No new rows found.")
        df_final = df_existing.copy()

    # final cleanup
    if not df_final.empty:
        df_final["Start"] = pd.to_datetime(df_final["Start"], errors="coerce")
        df_final = df_final.dropna(subset=["Start"]).copy()
        df_final = df_final.drop_duplicates(subset=["Start", "GOES Class"])
        df_final = df_final.sort_values("Start").reset_index(drop=True)

    df_final.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"Total rows in dataset: {len(df_final)}")
    print(df_final.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()