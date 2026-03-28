import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
from datetime import timedelta

# ── Must be first ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TCU Solar Flare Forecast",
    page_icon="☀️",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [data-testid="stApp"] {
    background-color: #05050f !important;
    color: #e8e4d8 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stHeader"]    { background: transparent !important; }
[data-testid="stDecoration"]{ display: none !important; }
#MainMenu, footer           { visibility: hidden; }

.block-container {
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}
[data-testid="stAppViewContainer"] > .main {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

.tcu-banner {
    width: 100%;
    height: 56px;
    background: #0b0b1e;
    border-bottom: 1px solid rgba(255,120,50,0.22);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 32px;
    box-sizing: border-box;
    margin-top: 58px;
}
.tcu-left  { display:flex; align-items:center; gap:14px; }
.tcu-badge {
    font-family: 'Space Mono', monospace;
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.14em; color: #c67bff;
    border: 1px solid rgba(198,123,255,0.4);
    padding: 3px 9px; border-radius: 2px;
}
.tcu-name  { font-size:13px; color:rgba(232,228,216,0.38); letter-spacing:.07em; font-weight:300; }
.live-badge {
    display:flex; align-items:center; gap:7px;
    font-family:'Space Mono',monospace;
    font-size:10px; color:#ff7a32; letter-spacing:.1em;
}
.live-dot {
    width:7px; height:7px; border-radius:50%;
    background:#ff7a32;
    animation: livepulse 1.4s ease-in-out infinite;
}
@keyframes livepulse {
    0%,100%{ opacity:1; transform:scale(1); }
    50%    { opacity:.3; transform:scale(.6); }
}

.page-content { padding: 1.5rem 2rem 0 2rem; }

.sec-label {
    font-family:'Space Mono',monospace;
    font-size:9px; letter-spacing:.2em;
    color:rgba(232,228,216,.28);
    text-transform:uppercase; margin-bottom:10px;
}

.alert-card {
    background: linear-gradient(135deg,rgba(255,80,30,.11) 0%,rgba(255,140,60,.04) 100%);
    border: 1px solid rgba(255,90,30,.32);
    border-radius: 10px;
    padding: 20px 22px 18px;
    position: relative; overflow: hidden;
    margin-bottom: 14px;
}
.alert-card::before {
    content:''; position:absolute;
    left:0; top:0; bottom:0; width:3px;
    background:linear-gradient(180deg,#ff5a1e,#ffaa44);
}
.alert-card.no-flare {
    background:linear-gradient(135deg,rgba(30,160,255,.09) 0%,rgba(60,200,255,.03) 100%);
    border-color:rgba(30,140,255,.28);
}
.alert-card.no-flare::before { background:linear-gradient(180deg,#1e8aff,#44ccff); }
.alert-eyebrow {
    font-family:'Space Mono',monospace;
    font-size:9px; letter-spacing:.22em; color:#ff7a40; margin-bottom:5px;
}
.alert-card.no-flare .alert-eyebrow { color:#44aaff; }
.alert-title {
    font-size:26px; font-weight:600;
    color:#fff5ee; line-height:1.15; margin-bottom:14px;
}
.conf-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:5px; }
.conf-label { font-size:11px; color:rgba(232,228,216,.42); font-weight:300; }
.conf-val-flare   { font-family:'Space Mono',monospace; font-size:14px; font-weight:700; color:#ffaa44; }
.conf-val-noflare { font-family:'Space Mono',monospace; font-size:14px; font-weight:700; color:#44aaff; }
.bar-track { height:4px; background:rgba(255,255,255,.07); border-radius:2px; overflow:hidden; margin-bottom:14px; }
.bar-flare   { height:100%; background:linear-gradient(90deg,#ff5a1e,#ffcc44); border-radius:2px; }
.bar-noflare { height:100%; background:linear-gradient(90deg,#1e6aff,#44ddff); border-radius:2px; }
.meta-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.meta-cell {
    background:rgba(255,255,255,.03);
    border:1px solid rgba(255,255,255,.06);
    border-radius:6px; padding:9px 12px;
}
.meta-cell-label { font-size:9px; letter-spacing:.12em; color:rgba(232,228,216,.26); text-transform:uppercase; margin-bottom:3px; }
.meta-cell-val   { font-family:'Space Mono',monospace; font-size:11px; color:rgba(232,228,216,.7); line-height:1.5; }

.forecast-item {
    display:flex; align-items:center; gap:13px;
    padding:10px 14px;
    border-bottom:1px solid rgba(255,255,255,.04);
    border-radius:6px; margin-bottom:3px;
    transition:background .15s;
}
.forecast-item:hover { background:rgba(255,255,255,.03); }
.fi-icon {
    width:34px; height:34px; border-radius:7px;
    display:flex; align-items:center; justify-content:center;
    flex-shrink:0; font-size:14px;
}
.fi-icon.f  { background:rgba(255,80,30,.12);  border:1px solid rgba(255,80,30,.28); }
.fi-icon.nf { background:rgba(30,140,255,.10); border:1px solid rgba(30,140,255,.24); }
.fi-body    { flex:1; min-width:0; }
.fi-label-f  { font-size:13px; font-weight:500; color:#ffcba0; margin-bottom:2px; }
.fi-label-nf { font-size:13px; font-weight:500; color:#aaddff; margin-bottom:2px; }
.fi-time     { font-family:'Space Mono',monospace; font-size:9px; color:rgba(232,228,216,.26); letter-spacing:.05em; }
.fi-prob-f   { font-family:'Space Mono',monospace; font-size:12px; color:#ff8844; font-weight:700; white-space:nowrap; }
.fi-prob-nf  { font-family:'Space Mono',monospace; font-size:12px; color:#44aaff; font-weight:700; white-space:nowrap; }

.hist-wrap { overflow-x:auto; margin-top:8px; }
.hist-table { width:100%; border-collapse:collapse; font-size:11px; }
.hist-table th {
    font-family:'Space Mono',monospace; font-size:8px;
    letter-spacing:.14em; text-transform:uppercase;
    color:rgba(232,228,216,.24); font-weight:400;
    padding:0 8px 8px 0; text-align:left;
    border-bottom:1px solid rgba(255,255,255,.06);
}
.hist-table td {
    padding:7px 8px 7px 0;
    font-family:'Space Mono',monospace; font-size:10px;
    color:rgba(232,228,216,.52);
    border-bottom:1px solid rgba(255,255,255,.03);
}
.lf  { color:#ffcba0 !important; }
.lnf { color:#aaddff !important; }
.pf  { color:#ff8844 !important; font-weight:700; }
.pnf { color:#44aaff !important; font-weight:700; }

.goes-wrap { overflow-x:auto; margin-top:8px; }
.goes-table { width:100%; border-collapse:collapse; font-size:11px; }
.goes-table th {
    font-family:'Space Mono',monospace; font-size:8px;
    letter-spacing:.14em; text-transform:uppercase;
    color:rgba(232,228,216,.24); font-weight:400;
    padding:0 8px 8px 0; text-align:left;
    border-bottom:1px solid rgba(255,255,255,.06);
}
.goes-table td {
    padding:7px 8px 7px 0;
    font-family:'Space Mono',monospace; font-size:10px;
    color:rgba(232,228,216,.52);
    border-bottom:1px solid rgba(255,255,255,.03);
}
.goes-badge {
    display:inline-block; padding:2px 7px; border-radius:3px;
    font-family:'Space Mono',monospace; font-size:10px; font-weight:700;
    letter-spacing:.04em;
}
.goes-X  { background:rgba(255,40,40,.18);  color:#ff6060; border:1px solid rgba(255,60,60,.35); }
.goes-M  { background:rgba(255,130,30,.18); color:#ffaa44; border:1px solid rgba(255,130,30,.35); }
.goes-C  { background:rgba(255,220,60,.14); color:#ffe066; border:1px solid rgba(255,220,60,.30); }
.goes-B  { background:rgba(80,180,255,.12); color:#66ccff; border:1px solid rgba(80,180,255,.28); }
.goes-A  { background:rgba(150,255,150,.10);color:#88ee88; border:1px solid rgba(150,255,150,.24); }
.goes-none{ color:rgba(232,228,216,.18); font-style:italic; }

h2, h3 { color:#e8e4d8 !important; font-family:'DM Sans',sans-serif !important; }
[data-testid="stDataFrame"] { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ── Top bar ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tcu-banner">
    <div class="tcu-left">
        <span class="tcu-badge">TCU</span>
        <span class="tcu-name">Texas Christian University &nbsp;·&nbsp; Solar Flare Forecast</span>
    </div>
    <div class="live-badge"><span class="live-dot"></span>LIVE</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="page-content">', unsafe_allow_html=True)

# ── Load prediction history ────────────────────────────────────────────────
HISTORY_FILE = Path("prediction_history.csv")

if not HISTORY_FILE.exists():
    st.warning("No prediction history found yet.")
    st.stop()

df = pd.read_csv(HISTORY_FILE)
df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")

if "image_time" in df.columns:
    df["image_time"] = pd.to_datetime(df["image_time"], errors="coerce")
else:
    st.error("CSV is missing image_time. Run the new prediction pipeline first.")
    st.stop()

df["forecast_end"] = pd.to_datetime(df["forecast_end"], errors="coerce")
df = df.dropna(subset=["image_time"]).sort_values("image_time", ascending=False).reset_index(drop=True)

latest = df.iloc[0]

board_df = (
    df.drop_duplicates(subset="image_time", keep="first")
      .head(12)
      .reset_index(drop=True)
)

# ── Helpers ────────────────────────────────────────────────────────────────
def is_flare(label):
    return str(label).strip().lower() in ["flare", "yes flare", "yes"]

def fmt_prob(x):
    return x * 100 if x <= 1.0 else x

def goes_badge_html(cls: str) -> str:
    cls = (cls or "").strip()
    if not cls:
        return '<span class="goes-none">—</span>'
    letter = cls[0].upper()
    css = {"X": "goes-X", "M": "goes-M", "C": "goes-C", "B": "goes-B", "A": "goes-A"}.get(letter, "goes-B")
    return f'<span class="goes-badge {css}">{cls}</span>'

# ── Two-column layout ──────────────────────────────────────────────────────
left, right = st.columns([1.3, 1], gap="large")

# ════════════════════════════════════════════════════════════════
# LEFT — Latest prediction + solar image
# ════════════════════════════════════════════════════════════════
with left:
    st.markdown('<div class="sec-label">Latest Prediction</div>', unsafe_allow_html=True)

    flare       = is_flare(latest["prediction_label"])
    pct         = fmt_prob(latest["probability"])
    card_cls    = "alert-card" if flare else "alert-card no-flare"
    eyebrow     = "FLARE ALERT" if flare else "NO FLARE DETECTED"
    title_txt   = "Solar Flare Predicted" if flare else "No Solar Flare"
    bar_cls     = "bar-flare" if flare else "bar-noflare"
    conf_cls    = "conf-val-flare" if flare else "conf-val-noflare"
    pred_time_s = latest["image_time"].strftime("%Y-%m-%d %H:%M")
    end_time_s  = latest["forecast_end"].strftime("%Y-%m-%d %H:%M")

    st.markdown(f"""
    <div class="{card_cls}">
        <div class="alert-eyebrow">{eyebrow}</div>
        <div class="alert-title">{title_txt}</div>
        <div class="conf-row">
            <span class="conf-label">Model Confidence</span>
            <span class="{conf_cls}">{pct:.2f}%</span>
        </div>
        <div class="bar-track">
            <div class="{bar_cls}" style="width:{min(pct,100):.1f}%"></div>
        </div>
        <div class="meta-grid">
            <div class="meta-cell">
                <div class="meta-cell-label">Image Time</div>
                <div class="meta-cell-val">{pred_time_s}</div>
            </div>
            <div class="meta-cell">
                <div class="meta-cell-label">Forecast Window</div>
                <div class="meta-cell-val">→ {end_time_s}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label" style="margin-top:16px;">Latest Solar Image</div>',
                unsafe_allow_html=True)
    image_path = latest.get("image_path", "")
    if image_path and Path(str(image_path)).exists():
        image = Image.open(str(image_path))
        st.image(image, width="stretch")
    else:
        st.markdown("""
        <div style="background:#000;border-radius:10px;
                    border:1px solid rgba(255,255,255,0.07);
                    height:280px;display:flex;align-items:center;
                    justify-content:center;
                    color:rgba(232,228,216,0.18);
                    font-family:'Space Mono',monospace;font-size:11px;">
            HMI MAGNETOGRAM · NOT FOUND
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# RIGHT — 12-hour forecast board
# ════════════════════════════════════════════════════════════════
with right:
    st.markdown('<div class="sec-label">Last 12 Forecast Board</div>', unsafe_allow_html=True)

    items_html = []
    for _, row in board_df.iterrows():
        f        = is_flare(row["prediction_label"])
        icon_cls = "fi-icon f"   if f else "fi-icon nf"
        lbl_cls  = "fi-label-f"  if f else "fi-label-nf"
        prb_cls  = "fi-prob-f"   if f else "fi-prob-nf"
        icon     = "🔥" if f else "✅"
        text     = "Yes Flare" if f else "No Flare"
        pct_r    = fmt_prob(row["probability"])
        t_start  = row["image_time"].strftime("%Y-%m-%d %H:%M")
        t_end    = row["forecast_end"].strftime("%Y-%m-%d %H:%M")

        items_html.append(f"""
        <div class="forecast-item">
            <div class="{icon_cls}">{icon}</div>
            <div class="fi-body">
                <div class="{lbl_cls}">{text}</div>
                <div class="fi-time">{t_start} · until {t_end}</div>
            </div>
            <div class="{prb_cls}">{pct_r:.2f}%</div>
        </div>""")

    st.markdown("".join(items_html), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# BOTTOM — Recent prediction history
# ════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-label" style="margin-top:28px;">Recent Prediction History</div>',
            unsafe_allow_html=True)

history_df = df.head(20).copy()
rows_html  = []
for _, row in history_df.iterrows():
    f       = is_flare(row["prediction_label"])
    lc      = "lf"  if f else "lnf"
    pc      = "pf"  if f else "pnf"
    pct_h   = fmt_prob(row["probability"])
    t_s     = row["image_time"].strftime("%Y-%m-%d %H:%M")
    e_s     = row["forecast_end"].strftime("%Y-%m-%d %H:%M")
    lbl_txt = row["prediction_label"]

    rows_html.append(f"""
    <tr>
        <td>{t_s}</td>
        <td>{e_s}</td>
        <td class="{lc}">{lbl_txt}</td>
        <td class="{pc}">{pct_h:.2f}%</td>
    </tr>""")

st.markdown(f"""
<div class="hist-wrap">
<table class="hist-table">
  <thead>
    <tr>
      <th>Image Time</th>
      <th>Forecast End</th>
      <th>Label</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>{"".join(rows_html)}</tbody>
</table>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# GOES FLARE DATA — from lmsal_all_2026_clean.csv (scrape_ssw.py)
# ════════════════════════════════════════════════════════════════

SSW_FILE = Path("lmsal_all_2026_clean.csv")


@st.cache_data(ttl=3600)
def load_ssw_flares(path: Path) -> pd.DataFrame:
    """Load CSV written by scrape_ssw.py — columns: Start, GOES Class, Class_Type"""
    if not path.exists():
        return pd.DataFrame(columns=["Start", "GOES Class", "Class_Type"])
    df_ssw = pd.read_csv(path)
    df_ssw["Start"] = pd.to_datetime(df_ssw["Start"], errors="coerce")
    df_ssw = df_ssw.dropna(subset=["Start"]).sort_values("Start").reset_index(drop=True)
    return df_ssw


flare_df = load_ssw_flares(SSW_FILE)

SSW_NOTE = "Source: lmsal_all_2026_clean.csv (scrape_ssw.py)"

# ── Full event log ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="sec-label" style="margin-top:36px;">'
    'All GOES Flare Events (2026-01-01 → Today)</div>',
    unsafe_allow_html=True,
)

if flare_df.empty:
    st.markdown(
        '<p style="color:rgba(232,228,216,.3);font-size:11px;">'
        'No flare data found — run scrape_ssw.py to generate lmsal_all_2026_clean.csv.</p>',
        unsafe_allow_html=True,
    )
else:
    ev_rows = []
    for _, row in flare_df.sort_values("Start", ascending=False).iterrows():
        st_str = row["Start"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["Start"]) else "—"
        badge  = goes_badge_html(str(row.get("GOES Class", "")))
        ctype  = str(row.get("Class_Type", "")).strip() or "—"
        ev_rows.append(f"""
        <tr>
            <td>{st_str}</td>
            <td>{badge}</td>
            <td>{ctype}</td>
        </tr>""")

    st.markdown(f"""
<div class="goes-wrap" style="max-height:420px;overflow-y:auto;">
<table class="goes-table">
  <thead>
    <tr>
      <th>Start (UTC)</th>
      <th>GOES Class</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>{"".join(ev_rows)}</tbody>
</table>
</div>
<p style="font-size:9px;color:rgba(232,228,216,.22);font-family:'Space Mono',monospace;margin-top:6px;">
  {len(flare_df)} events · {SSW_NOTE}
</p>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close .page-content

# ── Auto-refresh every 60 minutes ─────────────────────────────────────────
# Reloads the page so the app picks up the latest lmsal_all_2026_clean.csv
import time as _time
REFRESH_INTERVAL = 3600   # seconds

if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = _time.time()

elapsed = _time.time() - st.session_state["last_refresh"]
remaining = max(0, REFRESH_INTERVAL - int(elapsed))

if elapsed >= REFRESH_INTERVAL:
    st.session_state["last_refresh"] = _time.time()
    st.cache_data.clear()
    st.rerun()

# Small footer showing next refresh countdown
mins, secs = divmod(remaining, 60)
st.markdown(
    f'<p style="font-size:9px;color:rgba(232,228,216,.14);'
    f'font-family:\'Space Mono\',monospace;text-align:right;'
    f'padding:0 2rem 1rem 0;">next refresh in {mins:02d}:{secs:02d}</p>',
    unsafe_allow_html=True,
)