import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import torch
from PIL import Image
import torchvision.transforms as transforms

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = Path("model/your_model.pth")
IMAGE_PATH = Path("data/latest_image.jpg")
HISTORY_FILE = Path("prediction_history.csv")

CLASS_NAMES = ["No Flare", "Flare"]

# -------------------------
# LOAD MODEL
# -------------------------
model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
model.eval()

# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # remove if model uses RGB
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# LOAD IMAGE
# -------------------------
img = Image.open(IMAGE_PATH)
input_tensor = transform(img).unsqueeze(0)

# -------------------------
# RUN PREDICTION
# -------------------------
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    pred_prob = probs[0, pred_idx].item()

prediction_time = datetime.now()
forecast_end = prediction_time + timedelta(hours=12)

new_row = {
    "prediction_time": prediction_time.strftime("%Y-%m-%d %H:%M:%S"),
    "forecast_end": forecast_end.strftime("%Y-%m-%d %H:%M:%S"),
    "prediction_label": CLASS_NAMES[pred_idx],
    "probability": pred_prob,
    "image_path": str(IMAGE_PATH)
}

# -------------------------
# SAVE / APPEND
# -------------------------
if HISTORY_FILE.exists():
    df = pd.read_csv(HISTORY_FILE)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
else:
    df = pd.DataFrame([new_row])

df.to_csv(HISTORY_FILE, index=False)

print("Prediction appended to prediction_history.csv")
print(new_row)