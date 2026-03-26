import re
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH   = Path("new-fold1.pth")
IMAGE_DIR    = Path("data/hmi_jpg")
HISTORY_FILE = Path("prediction_history.csv")

CLASS_NAMES = ["No Flare", "Flare"]

# -------------------------
# LOAD MODEL
# -------------------------
from explainingFullDisk.modeling.model import Custom_AlexNet

model = Custom_AlexNet(train=False)
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# -------------------------
# HELPERS
# -------------------------
def parse_image_time_from_name(path: Path) -> datetime | None:
    """
    Example filename:
    HMI.m2026.03.08_05.00.00_TAI.jpg
    """
    match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})_(\d{2})\.(\d{2})\.(\d{2})", path.name)
    if not match:
        return None
    y, mo, d, h, mi, s = map(int, match.groups())
    return datetime(y, mo, d, h, mi, s)

def get_latest_image_path(image_dir: Path) -> Path | None:
    jpg_files = list(image_dir.rglob("*.jpg"))
    if not jpg_files:
        return None

    valid = []
    for p in jpg_files:
        image_time = parse_image_time_from_name(p)
        if image_time is not None:
            valid.append((image_time, p))

    if not valid:
        return None

    valid.sort(key=lambda x: x[0], reverse=True)
    return valid[0][1]

def predict_image(image_path: Path) -> tuple[str, float]:
    img = Image.open(image_path).convert("L")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)
        pred_idx   = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    label = CLASS_NAMES[pred_idx]
    return label, confidence

def load_history(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        if "image_time" in df.columns:
            df["image_time"] = pd.to_datetime(df["image_time"], errors="coerce")
        return df
    return pd.DataFrame(columns=[
        "prediction_time",
        "image_time",
        "forecast_end",
        "prediction_label",
        "probability",
        "image_path",
    ])

def already_logged_for_image(history_df: pd.DataFrame, image_path: Path) -> bool:
    if history_df.empty or "image_path" not in history_df.columns:
        return False
    return str(image_path) in history_df["image_path"].astype(str).values

# -------------------------
# MAIN
# -------------------------
latest_image = get_latest_image_path(IMAGE_DIR)

if latest_image is None:
    print("No JPG images found. Skipping.")
    raise SystemExit

image_time = parse_image_time_from_name(latest_image)
if image_time is None:
    print(f"Could not parse image time from filename: {latest_image.name}")
    raise SystemExit

history_df = load_history(HISTORY_FILE)

if already_logged_for_image(history_df, latest_image):
    print(f"Latest image already predicted: {latest_image}")
    raise SystemExit

label, confidence = predict_image(latest_image)

new_row = pd.DataFrame([{
    "prediction_time":  datetime.now().isoformat(),
    "image_time":       image_time.isoformat(),
    "forecast_end":     (image_time + timedelta(hours=12)).isoformat(),
    "prediction_label": label,
    "probability":      confidence,
    "image_path":       str(latest_image),
}])

history_df = pd.concat([history_df, new_row], ignore_index=True)
history_df = history_df.sort_values("image_time", ascending=False).head(500)
history_df.to_csv(HISTORY_FILE, index=False)

print("Saved latest prediction:")
print(f"  image_time:       {image_time}")
print(f"  prediction_label: {label}")
print(f"  probability:      {confidence:.4f}")
print(f"  image_path:       {latest_image}")