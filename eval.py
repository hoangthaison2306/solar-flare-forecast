"""
evaluate.py
-----------
Connects prediction_history.csv with lmsal_all_2026_clean.csv
and prints TSS / HSS for the last 1 week and last 1 month.

TP: prob >= 0.5 AND M/X flare occurred in forecast window
FP: prob >= 0.5 AND no M/X flare in window
FN: prob <  0.5 AND M/X flare occurred in forecast window
TN: prob <  0.5 AND no M/X flare in window
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── config ───────────────────────────────────────────────────────────────────
PREDICTION_CSV = Path("prediction_history.csv")
LMSAL_CSV      = Path("lmsal_all_2026_clean.csv")
MIN_CLASS      = "M"

CLASS_ORDER = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}

# ── helpers ───────────────────────────────────────────────────────────────────

def class_meets_minimum(goes_class, min_class=MIN_CLASS):
    letter = str(goes_class).strip()[0].upper() if goes_class else "?"
    return CLASS_ORDER.get(letter, -1) >= CLASS_ORDER.get(min_class, 0)

def parse_image_time(image_path):
    match = re.search(
        r"HMI\.m(\d{4})\.(\d{2})\.(\d{2})_(\d{2})\.(\d{2})\.(\d{2})",
        str(image_path)
    )
    if not match:
        return None
    y, mo, d, h, mi, s = map(int, match.groups())
    return datetime(y, mo, d, h, mi, s)

# ── load ──────────────────────────────────────────────────────────────────────

def load_predictions():
    df = pd.read_csv(PREDICTION_CSV)
    if "image_time" not in df.columns:
        df["image_time"] = df["image_path"].apply(parse_image_time)
    else:
        df["image_time"] = pd.to_datetime(df["image_time"], errors="coerce")
        mask = df["image_time"].isna()
        df.loc[mask, "image_time"] = df.loc[mask, "image_path"].apply(parse_image_time)
    df["forecast_end"] = pd.to_datetime(df["forecast_end"], errors="coerce")
    df["probability"]  = pd.to_numeric(df["probability"], errors="coerce")
    return df.dropna(subset=["image_time", "forecast_end"]).drop_duplicates("image_time")

def load_lmsal():
    df = pd.read_csv(LMSAL_CSV)
    df["Start"] = pd.to_datetime(df["Start"], errors="coerce")
    df = df.dropna(subset=["Start"])
    df = df[df["GOES Class"].astype(str).str[0].str.upper()
              .apply(class_meets_minimum)]
    return df

# ── ground truth ──────────────────────────────────────────────────────────────

def assign_ground_truth(pred_df, lmsal_df):
    flare_starts = lmsal_df["Start"].values
    gt = []
    for _, row in pred_df.iterrows():
        win_s = np.datetime64(row["image_time"])
        win_e = np.datetime64(row["forecast_end"])
        gt.append(1 if ((flare_starts >= win_s) & (flare_starts <= win_e)).any() else 0)
    pred_df = pred_df.copy()
    pred_df["gt_label"] = gt
    return pred_df

# ── metrics ───────────────────────────────────────────────────────────────────

def compute_tss_hss(df):
    high_prob = df["probability"] >= 0.5
    mx_flare  = df["gt_label"] == 1

    TP = int(( high_prob &  mx_flare).sum())
    FP = int(( high_prob & ~mx_flare).sum())
    FN = int((~high_prob &  mx_flare).sum())
    TN = int((~high_prob & ~mx_flare).sum())

    pod = TP / (TP + FN) if (TP + FN) else 0
    far = FP / (FP + TN) if (FP + TN) else 0
    tss = pod - far

    P = TP + FN
    N = TN + FP
    denom = (P * (FN + TN)) + ((TP + FP) * N)
    hss = (2 * (TP * TN - FN * FP) / denom) if denom else 0

    return round(tss, 4), round(hss, 4)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    pred_df  = load_predictions()
    lmsal_df = load_lmsal()
    df       = assign_ground_truth(pred_df, lmsal_df)

    # anchor to last prediction date so windows stay within actual data
    now = df['image_time'].max()

    windows = [
        ("Last 1 Week",  now - pd.Timedelta(weeks=1)),
        ("Last 1 Month", now - pd.Timedelta(days=30)),
        ("Last 2 Months", now - pd.Timedelta(days=60))
    ]

    rows = []
    for label, since in windows:
        subset = df[df["image_time"] >= since]
        if subset.empty:
            rows.append((label, "N/A", "N/A"))
        else:
            tss, hss = compute_tss_hss(subset)
            rows.append((label, f"{tss:+.4f}", f"{hss:+.4f}"))

    print()
    print(f"{'Period':<14} {'TSS':>8} {'HSS':>8}")
    print("-" * 32)
    for label, tss, hss in rows:
        print(f"{label:<14} {tss:>8} {hss:>8}")
    print()

if __name__ == "__main__":
    main()