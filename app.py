import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
from datetime import timedelta

st.set_page_config(
    page_title="Solar Flare Forecast Dashboard",
    page_icon="☀️",
    layout="wide"
)

HISTORY_FILE = Path("prediction_history.csv")

st.title("☀️ Solar Flare Forecast Dashboard")

if not HISTORY_FILE.exists():
    st.warning("No prediction history found yet.")
    st.stop()

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(HISTORY_FILE)
df["prediction_time"] = pd.to_datetime(df["prediction_time"])
df["forecast_end"] = pd.to_datetime(df["forecast_end"])

# Sort newest first
df = df.sort_values("prediction_time", ascending=False).reset_index(drop=True)

# Latest prediction
latest = df.iloc[0]

# Get recent predictions to show in the board
# Assumption: each row is a forecast made at a different time, each covering the next 12 hours
board_df = df.head(12).copy()

# ----------------------------
# Styling helpers
# ----------------------------
def label_badge(label):
    if str(label).strip().lower() in ["flare", "yes flare", "yes"]:
        return "🔥 Yes Flare"
    return "✅ No Flare"

def probability_percent(x):
    return f"{x * 100:.2f}%"

# ----------------------------
# Layout
# ----------------------------
left, right = st.columns([1.3, 1])

# ============================
# LEFT: LATEST PREDICTION
# ============================
with left:
    st.subheader("Latest Prediction")

    pred_label = latest["prediction_label"]
    pred_prob = latest["probability"]
    pred_time = latest["prediction_time"]
    forecast_end = latest["forecast_end"]
    image_path = latest.get("image_path", "")

    card_color = "#ffdddd" if str(pred_label).lower() in ["flare", "yes flare", "yes"] else "#ddffdd"
    border_color = "#ff4b4b" if str(pred_label).lower() in ["flare", "yes flare", "yes"] else "#2e8b57"

    st.markdown(
        f"""
        <div style="
            background-color: {card_color};
            border-left: 8px solid {border_color};
            padding: 18px;
            border-radius: 10px;
            margin-bottom: 15px;
        ">
            <h3 style="margin-bottom: 10px;">{label_badge(pred_label)}</h3>
            <p><b>Confidence:</b> {probability_percent(pred_prob)}</p>
            <p><b>Prediction Time:</b> {pred_time}</p>
            <p><b>Forecast Window:</b> {pred_time} → {forecast_end}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(float(pred_prob))

    if image_path and Path(image_path).exists():
        image = Image.open(image_path)
        st.image(image, caption="Latest Solar Image", width='stretch')
    else:
        st.info("Latest image not found.")

# ============================
# RIGHT: 12-HOUR BOARD
# ============================
with right:
    st.subheader("12-Hour Forecast Board")

    for _, row in board_df.iterrows():
        label = row["prediction_label"]
        start_time = row["prediction_time"]
        end_time = row["forecast_end"]

        is_flare = str(label).strip().lower() in ["flare", "yes flare", "yes"]
        bg = "#ffe5e5" if is_flare else "#e8f8e8"
        border = "#ff4b4b" if is_flare else "#2e8b57"
        icon = "🔥" if is_flare else "✅"
        text = "Yes Flare" if is_flare else "No Flare"

        st.markdown(
            f"""
            <div style="
                background-color: {bg};
                border-left: 6px solid {border};
                padding: 10px 12px;
                border-radius: 8px;
                margin-bottom: 10px;
            ">
                <div style="font-size: 16px; font-weight: 600;">{icon} {text}</div>
                <div style="font-size: 13px;">{start_time.strftime('%Y-%m-%d %H:%M')}</div>
                <div style="font-size: 12px; color: #555;">
                    Valid for next 12h until {end_time.strftime('%Y-%m-%d %H:%M')}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================
# OPTIONAL TABLE AT BOTTOM
# ============================
st.subheader("Recent Prediction History")

display_df = df.copy()
display_df["prediction_time"] = display_df["prediction_time"].dt.strftime("%Y-%m-%d %H:%M")
display_df["forecast_end"] = display_df["forecast_end"].dt.strftime("%Y-%m-%d %H:%M")
display_df["probability"] = display_df["probability"].apply(probability_percent)

st.dataframe(
    display_df[["prediction_time", "forecast_end", "prediction_label", "probability"]],
    width='stretch',
    hide_index=True
)