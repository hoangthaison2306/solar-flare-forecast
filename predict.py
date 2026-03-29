import re
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from datetime import datetime, timedelta
import torchvision.transforms as transforms
from explainingFullDisk.modeling.model import Custom_AlexNet

OUTPUT_CSV = Path("prediction_history.csv")

# create model
model = Custom_AlexNet(train=False)

# load weights
checkpoint = torch.load("new-fold1.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


def parse_image_time_from_name(path: Path) -> datetime | None:
    """
    Parse image timestamp from filename.
    Supports: HMI.m2026.03.08_05.00.00.jpg  or  HMI.m2026.03.08_05.00.00_TAI.jpg
    fix: use filename-based time instead of unreliable st_mtime.
    fix: adds image_time column required by app.py.
    """
    match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})_(\d{2})\.(\d{2})\.(\d{2})", path.name)
    if not match:
        return None
    y, mo, d, h, mi, s = map(int, match.groups())
    return datetime(y, mo, d, h, mi, s)


results = []

for jpg_path in Path("./data/hmi_jpg/").rglob("*.jpg"):
    try:
        img = Image.open(jpg_path).convert("L")
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        probability = probs[0, 1].item()
        prediction_label = "Flare" if pred == 1 else "No Flare"

        # fix: parse time from filename; fall back to mtime only if parse fails
        image_time = parse_image_time_from_name(jpg_path)
        if image_time is None:
            image_time = datetime.fromtimestamp(jpg_path.stat().st_mtime)

        prediction_time = datetime.now()
        forecast_end    = image_time + timedelta(hours=12)

        results.append({
            "prediction_time":  prediction_time.isoformat(),
            "image_time":       image_time.isoformat(),   # fix: required by app.py
            "forecast_end":     forecast_end.isoformat(),
            "prediction_label": prediction_label,
            "probability":      probability,
            "image_path":       str(jpg_path),
        })

        print(f"{jpg_path.name} -> {prediction_label} ({probability:.4f})")

    except Exception as e:
        print(f"Failed on {jpg_path}: {e}")

if results:
    df = pd.DataFrame(results)
    df = df.sort_values("image_time", ascending=False)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} predictions to {OUTPUT_CSV}")
else:
    print("No predictions were generated.")