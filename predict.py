import re
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from datetime import datetime, timedelta
import torchvision.transforms as transforms
from explainingFullDisk.modeling.model import Custom_AlexNet

OUTPUT_CSV = Path("prediction_history.csv")
IMAGE_DIR = Path("./data/hmi_jpg/")

# -------------------------
# create model
# -------------------------
model = Custom_AlexNet(train=False)

# load weights
checkpoint = torch.load("new-fold1.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# -------------------------
# preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# -------------------------
# helpers
# -------------------------
def parse_image_time_from_name(path: Path) -> datetime | None:
    """
    Parse image timestamp from filename.
    Supports:
      HMI.m2026.03.08_05.00.00.jpg
      HMI.m2026.03.08_05.00.00_TAI.jpg
    """
    match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})_(\d{2})\.(\d{2})\.(\d{2})", path.name)
    if not match:
        return None
    y, mo, d, h, mi, s = map(int, match.groups())
    return datetime(y, mo, d, h, mi, s)


def load_existing_predictions(csv_path: Path) -> tuple[pd.DataFrame, set[str]]:
    """
    Load existing prediction history and return:
    - dataframe
    - set of existing image_time strings for fast duplicate checking
    """
    if csv_path.exists():
        df = pd.read_csv(csv_path)

        if "image_time" in df.columns:
            df["image_time"] = pd.to_datetime(df["image_time"], errors="coerce")
        else:
            df["image_time"] = pd.NaT

        existing_times = set(
            df["image_time"]
            .dropna()
            .dt.strftime("%Y-%m-%dT%H:%M:%S")
            .tolist()
        )

        return df, existing_times

    empty_df = pd.DataFrame(columns=[
        "prediction_time",
        "image_time",
        "forecast_end",
        "prediction_label",
        "probability",
        "image_path",
    ])
    return empty_df, set()


# -------------------------
# load existing history
# -------------------------
existing_df, existing_times = load_existing_predictions(OUTPUT_CSV)

results = []

# -------------------------
# predict only new timestamps
# -------------------------
for jpg_path in IMAGE_DIR.rglob("*.jpg"):
    try:
        # Parse image_time first
        image_time = parse_image_time_from_name(jpg_path)
        if image_time is None:
            image_time = datetime.fromtimestamp(jpg_path.stat().st_mtime)

        image_time_str = image_time.strftime("%Y-%m-%dT%H:%M:%S")

        # Skip if this timestamp already exists
        if image_time_str in existing_times:
            print(f"SKIP {jpg_path.name} (timestamp already predicted)")
            continue

        img = Image.open(jpg_path).convert("L")
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        # Probability of Flare class
        probability = probs[0, 1].item()
        prediction_label = "Flare" if pred == 1 else "No Flare"

        prediction_time = datetime.now()
        forecast_end = image_time + timedelta(hours=24)

        results.append({
            "prediction_time": prediction_time.isoformat(),
            "image_time": image_time.isoformat(),
            "forecast_end": forecast_end.isoformat(),
            "prediction_label": prediction_label,
            "probability": probability,
            "image_path": str(jpg_path),
        })

        # Add immediately so duplicates in same run are also skipped
        existing_times.add(image_time_str)

        print(f"{jpg_path.name} -> {prediction_label} ({probability:.4f})")

    except Exception as e:
        print(f"Failed on {jpg_path}: {e}")

# -------------------------
# save
# -------------------------
if results:
    new_df = pd.DataFrame(results)

    if not existing_df.empty:
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = new_df

    df["image_time"] = pd.to_datetime(df["image_time"], errors="coerce")
    df = df.dropna(subset=["image_time"])
    df = df.sort_values("image_time", ascending=False)
    df = df.drop_duplicates(subset="image_time", keep="first")

    # convert back to iso strings for consistency
    df["image_time"] = df["image_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    if "forecast_end" in df.columns:
        df["forecast_end"] = pd.to_datetime(df["forecast_end"], errors="coerce")
        df["forecast_end"] = df["forecast_end"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    if "prediction_time" in df.columns:
        df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")
        df["prediction_time"] = df["prediction_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(new_df)} new predictions to {OUTPUT_CSV}")
    print(f"Total rows in file: {len(df)}")
else:
    print("No new predictions were generated.")