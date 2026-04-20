import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import time as _time
from textwrap import dedent

# ── Must be first ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TCU Solar Flare Forecast",
    page_icon="eclipse_icon_128.png",
    layout="wide",
)

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

# ── Load external CSS ──────────────────────────────────────────────────────
_LIGHT_ROOT_VARS = """
--bg:#edf4fb;
--bg2:#f8fbff;
--bg3:rgba(255,255,255,0.96);
--text:#081a2f;
--text-dim:rgba(8,26,47,0.82);
--text-faint:rgba(8,26,47,0.58);
--border:rgba(8,26,47,0.14);
--border-dim:rgba(8,26,47,0.08);
--card-bg:rgba(255,255,255,0.98);
--banner-border:rgba(11,91,160,0.20);
--meta-bg:rgba(244,248,253,0.98);
--meta-border:rgba(8,26,47,0.10);
--hover-bg:rgba(11,91,160,0.06);
--skill-bg:rgba(230,241,252,0.96);
--skill-border:rgba(11,91,160,0.18);
--bar-track:rgba(8,26,47,0.10);
--placeholder-bg:#e7f0f9;
--placeholder-border:rgba(8,26,47,0.10);
--metric-bg:rgba(245,249,255,0.98);
--metric-border:rgba(8,26,47,0.10);
--metric-div:rgba(8,26,47,0.07);
--footer-color:rgba(8,26,47,0.42);
--shadow-soft:0 8px 24px rgba(18,52,86,0.08);
--shadow-card:0 12px 30px rgba(18,52,86,0.10);
--glow:0 0 0 1px rgba(11,91,160,0.05);
"""

_DARK_ROOT_VARS = """
--bg:#05050f;
--bg2:#0b0b1e;
--bg3:rgba(255,255,255,0.02);
--text:#f4f7fb;
--text-dim:rgba(244,247,251,0.82);
--text-faint:rgba(244,247,251,0.58);
--border:rgba(255,255,255,0.10);
--border-dim:rgba(255,255,255,0.06);
--card-bg:rgba(255,255,255,0.03);
--banner-border:rgba(198,123,255,0.20);
--meta-bg:rgba(255,255,255,0.03);
--meta-border:rgba(255,255,255,0.07);
--hover-bg:rgba(255,255,255,0.03);
--skill-bg:rgba(198,123,255,0.05);
--skill-border:rgba(198,123,255,0.15);
--bar-track:rgba(255,255,255,0.07);
--placeholder-bg:#000;
--placeholder-border:rgba(255,255,255,0.07);
--metric-bg:rgba(255,255,255,0.02);
--metric-border:rgba(255,255,255,0.06);
--metric-div:rgba(255,255,255,0.05);
--footer-color:rgba(244,247,251,0.42);
--shadow-soft:0 10px 30px rgba(0,0,0,0.22);
--shadow-card:0 12px 36px rgba(0,0,0,0.18);
--glow:0 0 0 1px rgba(198,123,255,0.08);
"""

def load_css(file_name: str = "style.css") -> None:
    css_path = Path(file_name)
    if not css_path.exists():
        st.warning(f"{file_name} not found. UI will render without custom styling.")
        return

    css = css_path.read_text(encoding="utf-8")
    theme = st.session_state.get("theme", "dark")
    root_vars = _LIGHT_ROOT_VARS if theme == "light" else _DARK_ROOT_VARS

    st.markdown(
        dedent(f"""
        <style>
        {css}

        :root {{
            {root_vars}
        }}
        </style>
        """),
        unsafe_allow_html=True,
    )

load_css()
# ── Top bar ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tcu-banner">
    <div class="tcu-left">
        <span class="tcu-badge">TCU</span>
        <span class="tcu-name">Texas Christian University &nbsp;·&nbsp; Solar Flare Forecast</span>
    </div>
    <div class="tcu-right">
        <div class="live-badge"><span class="live-dot"></span>LIVE</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="page-content">', unsafe_allow_html=True)

if st.button("🌙 / ☀️", key="theme_toggle"):
    st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
    st.rerun()

# ── Load data ──────────────────────────────────────────────────────────────
HISTORY_FILE = Path("prediction_history.csv")
LMSAL_FILE = Path("lmsal_all_2026_clean.csv")
SSW_FILE = LMSAL_FILE

if not HISTORY_FILE.exists():
    st.warning("No prediction history found yet.")
    st.stop()

df = pd.read_csv(HISTORY_FILE)

df["prediction_time"] = pd.to_datetime(df.get("prediction_time"), errors="coerce")

if "image_time" in df.columns:
    df["image_time"] = pd.to_datetime(df["image_time"], errors="coerce")
else:
    st.error("CSV is missing image_time. Run the prediction pipeline first.")
    st.stop()

# Force true 24-hour forecast window in the app display/evaluation
df["forecast_end"] = df["image_time"] + pd.Timedelta(hours=24)
df["probability"] = pd.to_numeric(df.get("probability"), errors="coerce")

df = (
    df.dropna(subset=["image_time"])
      .sort_values("image_time", ascending=False)
      .reset_index(drop=True)
)

if df.empty:
    st.warning("Prediction history is empty.")
    st.stop()

latest = df.iloc[0]

# one row per image hour, newest 12
board_df = (
    df.drop_duplicates(subset="image_time", keep="first")
      .head(12)
      .reset_index(drop=True)
)

# ── Helpers ────────────────────────────────────────────────────────────────
def is_flare(label: str) -> bool:
    return str(label).strip().lower() in ["flare", "yes flare", "yes"]

def fmt_prob(x) -> float:
    if pd.isna(x):
        return 0.0
    return x * 100 if x <= 1.0 else x

def _class_is_mx(goes_class: str) -> bool:
    letter = str(goes_class).strip()[:1].upper() if goes_class else "?"
    return letter in ("M", "X")

@st.cache_data
def load_lmsal():
    if not LMSAL_FILE.exists():
        return None
    lmsal = pd.read_csv(LMSAL_FILE)
    lmsal["Start"] = pd.to_datetime(lmsal["Start"], errors="coerce")
    lmsal = lmsal.dropna(subset=["Start"])
    lmsal = lmsal[
        lmsal["GOES Class"].astype(str).str[0].str.upper().apply(_class_is_mx)
    ]
    return lmsal

def assign_gt(pred_df: pd.DataFrame, lmsal_df: pd.DataFrame) -> pd.DataFrame:
    flare_starts = lmsal_df["Start"].values
    gt = []
    for _, row in pred_df.iterrows():
        ws = np.datetime64(row["image_time"])
        we = np.datetime64(row["forecast_end"])
        gt.append(1 if ((flare_starts >= ws) & (flare_starts <= we)).any() else 0)
    pred_df = pred_df.copy()
    pred_df["gt_label"] = gt
    return pred_df

def compute_skill(subset: pd.DataFrame):
    hp = subset["probability"] >= 0.5
    mx = subset["gt_label"] == 1

    TP = int(( hp &  mx).sum())
    FP = int(( hp & ~mx).sum())
    FN = int((~hp &  mx).sum())
    TN = int((~hp & ~mx).sum())

    pod = TP / (TP + FN) if (TP + FN) else 0
    far = FP / (FP + TN) if (FP + TN) else 0
    tss = pod - far

    P = TP + FN
    N = TN + FP
    denom = (P * (FN + TN)) + ((TP + FP) * N)
    hss = (2 * (TP * TN - FN * FP) / denom) if denom else 0

    return round(tss, 4), round(hss, 4)

def goes_badge_html(cls: str) -> str:
    cls = (cls or "").strip()
    if not cls:
        return '<span class="goes-none">—</span>'
    letter = cls[0].upper()
    css = {
        "X": "goes-X",
        "M": "goes-M",
        "C": "goes-C",
        "B": "goes-B",
        "A": "goes-A",
    }.get(letter, "goes-B")
    return f'<span class="goes-badge {css}">{cls}</span>'

@st.cache_data(ttl=3600)
def load_ssw_flares(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["Start", "GOES Class", "Class_Type"])
    df_ssw = pd.read_csv(path)
    df_ssw["Start"] = pd.to_datetime(df_ssw["Start"], errors="coerce")
    df_ssw = df_ssw.dropna(subset=["Start"]).sort_values("Start").reset_index(drop=True)
    return df_ssw

# ── Two-column layout ──────────────────────────────────────────────────────
left, right = st.columns([1.3, 1], gap="large")

# ════════════════════════════════════════════════════════════════
# LEFT — Latest prediction + solar image
# ════════════════════════════════════════════════════════════════
with left:
    st.markdown('<div class="sec-label">Latest Prediction</div>', unsafe_allow_html=True)

    flare = is_flare(latest["prediction_label"])
    pct = fmt_prob(latest["probability"])
    card_cls = "alert-card" if flare else "alert-card no-flare"
    eyebrow = "PROBABILITY OF FLARE PREDICTION"
    title_txt = "Flare Detected" if flare else "No Flare Detected"
    bar_cls = "bar-flare" if flare else "bar-noflare"
    conf_cls = "conf-val-flare" if flare else "conf-val-noflare"
    pred_time_s = latest["image_time"].strftime("%Y-%m-%d %H:%M")
    end_time_s = latest["forecast_end"].strftime("%Y-%m-%d %H:%M")

    st.markdown(f"""
    <div class="{card_cls}">
        <div class="alert-eyebrow">{eyebrow}</div>
        <div class="alert-title">{title_txt}</div>
        <div class="conf-row">
            <span class="conf-label">Model Confidence</span>
            <span class="{conf_cls}">{pct:.2f}%</span>
        </div>
        <div class="bar-track">
            <div class="{bar_cls}" style="width:{min(pct, 100):.1f}%"></div>
        </div>
        <div class="meta-grid">
            <div class="meta-cell">
                <div class="meta-cell-label">Forecast Start</div>
                <div class="meta-cell-val">{pred_time_s}</div>
            </div>
            <div class="meta-cell">
                <div class="meta-cell-label">forecast End</div>
                <div class="meta-cell-val">→ {end_time_s}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label" style="margin-top:16px;">Latest Solar Image</div>', unsafe_allow_html=True)
    image_path = latest.get("image_path", "")
    if image_path and Path(str(image_path)).exists():
        image = Image.open(str(image_path))
        st.image(image, width="content")
    else:
        st.markdown("""
        <div class="img-placeholder">
            HMI MAGNETOGRAM · NOT FOUND
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# RIGHT — 12-hour forecast board
# ════════════════════════════════════════════════════════════════
with right:
    st.markdown('<div class="sec-label">Last 12-Hour Prediction Board</div>', unsafe_allow_html=True)

    items_html = []
    for _, row in board_df.iterrows():
        f = is_flare(row["prediction_label"])
        icon_cls = "fi-icon f" if f else "fi-icon nf"
        lbl_cls = "fi-label-f" if f else "fi-label-nf"
        prb_cls = "fi-prob-f" if f else "fi-prob-nf"
        icon = '<img src="app/static/supernova.png" width="28" style="vertical-align:middle;">'
        text = "Flare Detected" if f else "No Flare Detected"
        pct_r = fmt_prob(row["probability"])
        t_start = row["image_time"].strftime("%Y-%m-%d %H:%M")
        t_end = row["forecast_end"].strftime("%Y-%m-%d %H:%M")

        items_html.append(f"""
        <div class="forecast-item">     
            <div class="{icon_cls}">{icon}</div>
            <div class="fi-body">
                <div class="{lbl_cls}">{text}</div>
                <div class="fi-time">{t_start} · until {t_end}</div>
            </div>
            <div class="{prb_cls}">{pct_r:.2f}%</div>
        </div>
        """)

    st.markdown("".join(items_html), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# SKILL SCORES
# ════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="sec-label" style="margin-top:28px;">Model Skill Scores &nbsp;·&nbsp; M/X class &nbsp;·&nbsp; prob &ge; 0.5</div>',
    unsafe_allow_html=True
)

_lmsal = load_lmsal()
if _lmsal is None:
    st.markdown(
        '<p class="info-text">lmsal_all_2026_clean.csv not found.</p>',
        unsafe_allow_html=True,
    )
else:
    _edf = df.drop_duplicates(subset="image_time").copy()
    _edf["probability"] = pd.to_numeric(_edf["probability"], errors="coerce")
    _edf = assign_gt(_edf, _lmsal)
    _anchor = _edf["image_time"].max()

    _wins = [
        ("Last 1 Week", _anchor - pd.Timedelta(weeks=1)),
        ("Last 1 Month", _anchor - pd.Timedelta(days=30)),
        ("From 01 Feb 26", pd.Timestamp("2026-01-02")),
    ]

    _skill_rows = []
    for _lbl, _since in _wins:
        _sub = _edf[_edf["image_time"] >= _since]
        if _sub.empty or _sub["gt_label"].sum() == 0:
            _skill_rows.append((_lbl, "N/A", "N/A"))
        else:
            _t, _h = compute_skill(_sub)
            _skill_rows.append((_lbl, f"{_t:+.4f}", f"{_h:+.4f}"))

    _tbody = "".join(
        f'<tr><td>{r[0]}</td><td class="skill-val">{r[1]}</td><td class="skill-val">{r[2]}</td></tr>'
        for r in _skill_rows
    )

    st.markdown(f"""
    <div class="skill-card">
        <table class="skill-table">
            <thead>
                <tr>
                    <th>Period</th>
                    <th>TSS</th>
                    <th>HSS</th>
                </tr>
            </thead>
            <tbody>
                {_tbody}
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# Confusion Matrix
# ════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="sec-label" style="margin-top:28px;">Confusion Matrix &nbsp;·&nbsp; M/X class &nbsp;·&nbsp; prob &ge; 0.5 &nbsp;·&nbsp; all data</div>',
    unsafe_allow_html=True
)
if _lmsal is None:
    st.markdown(
        '<p style="font-size:11px;color:rgba(232,228,216,0.3);'
        'font-family:Space Mono,monospace">lmsal_all_2026_clean.csv not found.</p>',
        unsafe_allow_html=True
    )
else:
    _cm_df = df.drop_duplicates(subset="image_time").copy()
    _cm_df["probability"] = pd.to_numeric(_cm_df["probability"], errors="coerce")
    _cm_df = assign_gt(_cm_df, _lmsal)

    _hp = _cm_df["probability"] >= 0.5
    _mx = _cm_df["gt_label"] == 1
    _TP = int((_hp &  _mx).sum())
    _FP = int((_hp & ~_mx).sum())
    _FN = int((~_hp &  _mx).sum())
    _TN = int((~_hp & ~_mx).sum())
    _N  = _TP + _FP + _FN + _TN

    _pod  = _TP / (_TP + _FN) if (_TP + _FN) else 0
    _far  = _FP / (_FP + _TN) if (_FP + _TN) else 0
    _tss  = _pod - _far
    _P    = _TP + _FN
    _Nn   = _TN + _FP
    _den  = (_P * (_FN + _TN)) + ((_TP + _FP) * _Nn)
    _hss  = (2 * (_TP * _TN - _FN * _FP) / _den) if _den else 0

    _cell_base = (
        "text-align:center;padding:16px 10px;border-radius:8px;"
        "font-family:'Space Mono',monospace;"
    )
    _tp_style = _cell_base + "background:rgba(30,180,80,0.12);border:1px solid rgba(30,180,80,0.30);"
    _tn_style = _cell_base + "background:rgba(30,120,255,0.10);border:1px solid rgba(30,120,255,0.25);"
    _fp_style = _cell_base + "background:rgba(255,60,60,0.10);border:1px solid rgba(255,60,60,0.25);"
    _fn_style = _cell_base + "background:rgba(255,160,30,0.10);border:1px solid rgba(255,160,30,0.25);"

    _lbl_style = "font-size:9px;letter-spacing:.14em;color:rgba(232,228,216,.28);text-transform:uppercase;margin-bottom:5px;"
    _num_tp = "font-size:30px;font-weight:700;color:#44ee88;"
    _num_tn = "font-size:30px;font-weight:700;color:#44aaff;"
    _num_fp = "font-size:30px;font-weight:700;color:#ff5555;"
    _num_fn = "font-size:30px;font-weight:700;color:#ffaa44;"
    _pct_s  = "font-size:10px;color:rgba(232,228,216,.3);margin-top:3px;"

    _axis_th = (
        "font-family:'Space Mono',monospace;font-size:9px;letter-spacing:.14em;"
        "text-transform:uppercase;color:rgba(232,228,216,.22);font-weight:400;"
        "text-align:center;padding:0 0 8px 0;"
    )
    _axis_td_v = (
        "font-family:'Space Mono',monospace;font-size:9px;letter-spacing:.12em;"
        "text-transform:uppercase;color:rgba(232,228,216,.22);font-weight:400;"
        "text-align:center;writing-mode:vertical-rl;transform:rotate(180deg);"
        "padding-right:8px;white-space:nowrap;"
    )
    _metric_bar_style = (
        "display:flex;gap:0;margin-top:14px;border-radius:8px;overflow:hidden;"
        "border:1px solid rgba(255,255,255,0.06);"
    )
    _metric_item = (
        "flex:1;text-align:center;padding:10px 6px;"
        "background:rgba(255,255,255,0.02);font-family:'Space Mono',monospace;"
        "border-right:1px solid rgba(255,255,255,0.05);"
    )
    _metric_last = (
        "flex:1;text-align:center;padding:10px 6px;"
        "background:rgba(255,255,255,0.02);font-family:'Space Mono',monospace;"
    )
    _metric_lbl = "font-size:8px;letter-spacing:.14em;color:rgba(232,228,216,.24);text-transform:uppercase;margin-bottom:4px;"
    _metric_val = "font-size:13px;font-weight:700;color:#c67bff;"

    st.markdown(f"""
<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
border-radius:10px;padding:20px 22px 16px;margin-bottom:20px;">
<table style="width:100%;border-collapse:separate;border-spacing:6px;">
  <thead>
    <tr>
      <th style="width:60px;"></th>
      <th style="{_axis_th}">Predicted Flare</th>
      <th style="{_axis_th}">Predicted No Flare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="{_axis_td_v}">Actual Flare</td>
      <td style="{_tp_style}">
        <div style="{_lbl_style}">True Positive</div>
        <div style="{_num_tp}">{_TP}</div>
        <div style="{_pct_s}">{_TP/_N*100:.1f}%</div>
      </td>
      <td style="{_fn_style}">
        <div style="{_lbl_style}">False Negative</div>
        <div style="{_num_fn}">{_FN}</div>
        <div style="{_pct_s}">{_FN/_N*100:.1f}%</div>
      </td>
    </tr>
    <tr>
      <td style="{_axis_td_v}">Actual No Flare</td>
      <td style="{_fp_style}">
        <div style="{_lbl_style}">False Positive</div>
        <div style="{_num_fp}">{_FP}</div>
        <div style="{_pct_s}">{_FP/_N*100:.1f}%</div>
      </td>
      <td style="{_tn_style}">
        <div style="{_lbl_style}">True Negative</div>
        <div style="{_num_tn}">{_TN}</div>
        <div style="{_pct_s}">{_TN/_N*100:.1f}%</div>
      </td>
    </tr>
  </tbody>
</table>
<div style="{_metric_bar_style}">
  <div style="{_metric_item}">
    <div style="{_metric_lbl}">POD</div>
    <div style="{_metric_val}">{_pod:.4f}</div>
  </div>
  <div style="{_metric_item}">
    <div style="{_metric_lbl}">FAR</div>
    <div style="{_metric_val}">{_far:.4f}</div>
  </div>
  <div style="{_metric_item}">
    <div style="{_metric_lbl}">TSS</div>
    <div style="{_metric_val}">{_tss:+.4f}</div>
  </div>
  <div style="{_metric_last}">
    <div style="{_metric_lbl}">HSS</div>
    <div style="{_metric_val}">{_hss:+.4f}</div>
  </div>
</div>
<p style="font-size:9px;color:rgba(232,228,216,.18);font-family:'Space Mono',monospace;margin-top:10px;margin-bottom:0;">
  n={_N} predictions &nbsp;·&nbsp; {_TP+_FN} M/X flare windows &nbsp;·&nbsp; threshold prob &ge; 0.5
</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# MERGED TABLE — Predictions aligned with actual GOES results
# ════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="sec-label" style="margin-top:28px;">Prediction vs Actual GOES Results</div>',
    unsafe_allow_html=True
)

flare_df = load_ssw_flares(SSW_FILE)

merged_df = df.copy()
merged_df["probability"] = pd.to_numeric(merged_df["probability"], errors="coerce")
merged_df = merged_df.dropna(subset=["image_time", "forecast_end"]).copy()
merged_df = merged_df.sort_values("image_time", ascending=False).reset_index(drop=True)

def format_actual_goes_result(start_time, end_time, flare_df):
    if flare_df.empty:
        return "<span class='hist-empty'>No Flare Detected</span>"

    matches = flare_df[
        (flare_df["Start"] >= start_time) &
        (flare_df["Start"] <= end_time)
    ].sort_values("Start")

    if matches.empty:
        return "<span class='hist-empty'>No Flare Detected</span>"

    badges = [
        goes_badge_html(str(ev.get("GOES Class", "")).strip())
        for _, ev in matches.iterrows()
    ]
    return "<div class='actual-goes-inline'>" + "".join(badges) + "</div>"

def actual_goes_text(start_time, end_time, flare_df):
    if flare_df.empty:
        return "No Flare Detected"

    matches = flare_df[
        (flare_df["Start"] >= start_time) &
        (flare_df["Start"] <= end_time)
    ].sort_values("Start")

    if matches.empty:
        return "No Flare Detected"

    return ", ".join(
        str(ev.get("GOES Class", "")).strip()
        for _, ev in matches.iterrows()
    )

# ---------- Download dataframe ----------
download_df = merged_df.copy()
download_df["Forecast Start"] = download_df["image_time"].dt.strftime("%Y-%m-%d %H:%M")
download_df["Forecast End"] = download_df["forecast_end"].dt.strftime("%Y-%m-%d %H:%M")
download_df["Prediction"] = download_df["prediction_label"]
download_df["Probability"] = download_df["probability"].apply(lambda x: f"{fmt_prob(x):.2f}%")
download_df["Actual GOES Result"] = download_df.apply(
    lambda row: actual_goes_text(row["image_time"], row["forecast_end"], flare_df),
    axis=1
)
download_df = download_df[
    ["Forecast Start", "Forecast End", "Prediction", "Probability", "Actual GOES Result"]
]

st.download_button(
    label="Download CSV",
    data=download_df.to_csv(index=False).encode("utf-8"),
    file_name="prediction_vs_actual_results.csv",
    mime="text/csv"
)

# ---------- Pagination ----------
ROWS_PER_PAGE = 20

if "merged_table_page" not in st.session_state:
    st.session_state["merged_table_page"] = 1

total_rows = len(merged_df)
total_pages = max(1, (total_rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)

st.session_state["merged_table_page"] = min(
    max(1, st.session_state["merged_table_page"]),
    total_pages
)

start_idx = (st.session_state["merged_table_page"] - 1) * ROWS_PER_PAGE
end_idx = start_idx + ROWS_PER_PAGE
page_df = merged_df.iloc[start_idx:end_idx].copy()

rows_html = []
for _, row in page_df.iterrows():
    f = is_flare(row["prediction_label"])
    lc = "lf" if f else "lnf"
    pc = "pf" if f else "pnf"
    pct_h = fmt_prob(row["probability"])
    start_s = row["image_time"].strftime("%Y-%m-%d %H:%M")
    end_s = row["forecast_end"].strftime("%Y-%m-%d %H:%M")
    pred_txt = row["prediction_label"]
    actual_goes_html = format_actual_goes_result(row["image_time"], row["forecast_end"], flare_df)

    rows_html.append(
        f"<tr>"
        f"<td><span class='forecast-time'>{start_s}</span></td>"
        f"<td><span class='forecast-time'>{end_s}</span></td>"
        f"<td><span class='{lc}'>{pred_txt}</span></td>"
        f"<td><span class='{pc}'>{pct_h:.2f}%</span></td>"
        f"<td>{actual_goes_html}</td>"
        f"</tr>"
    )

table_html = (
    "<div class='hist-wrap'>"
    "<table class='hist-table'>"
    "<thead>"
    "<tr>"
    "<th>Forecast Start</th>"
    "<th>Forecast End</th>"
    "<th>Prediction</th>"
    "<th>Probability</th>"
    "<th>Actual GOES Result</th>"
    "</tr>"
    "</thead>"
    "<tbody>"
    + "".join(rows_html) +
    "</tbody>"
    "</table>"
    "</div>"
)

st.markdown(table_html, unsafe_allow_html=True)

# ---------- Left / Right buttons ----------
prev_col, info_col, next_col = st.columns([1, 2, 1])

with prev_col:
    if st.button("← Previous", disabled=(st.session_state["merged_table_page"] <= 1), key="prev_page_btn"):
        st.session_state["merged_table_page"] -= 1
        st.rerun()

with info_col:
    st.markdown(
        f"<div class='page-info'>Page {st.session_state['merged_table_page']} of {total_pages} "
        f"· Rows {start_idx + 1}-{min(end_idx, total_rows)} of {total_rows}</div>",
        unsafe_allow_html=True
    )

with next_col:
    if st.button("Next →", disabled=(st.session_state["merged_table_page"] >= total_pages), key="next_page_btn"):
        st.session_state["merged_table_page"] += 1
        st.rerun()

st.markdown(
    "<p class='source-note'>Prediction rows aligned with GOES events occurring within each forecast window</p>",
    unsafe_allow_html=True
)

# ── Auto-refresh every 60 minutes ─────────────────────────────────────────
REFRESH_INTERVAL = 3600  # seconds

if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = _time.time()

elapsed = _time.time() - st.session_state["last_refresh"]
remaining = max(0, REFRESH_INTERVAL - int(elapsed))

if elapsed >= REFRESH_INTERVAL:
    st.session_state["last_refresh"] = _time.time()
    st.cache_data.clear()
    st.rerun()

mins, secs = divmod(remaining, 60)
st.markdown(
    f'<p class="refresh-footer">next refresh in {mins:02d}:{secs:02d}</p>',
    unsafe_allow_html=True,
)