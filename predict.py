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
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

results = []

# load image
for jpg_path in Path("./data/hmi_jpg/").rglob("*.jpg"):
    try: 
        img = Image.open(jpg_path).convert("L")
        tensor = transform(img).unsqueeze(0)

# inference
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        probability = probs[0, pred].item()          
        prediction_label = "Flare" if pred == 1 else "No Flare" 

        prediction_time = datetime.fromtimestamp(jpg_path.stat().st_mtime)
        forecast_end = prediction_time + timedelta(hours=12)               

        results.append({                             
            "prediction_time": prediction_time,
            "forecast_end": forecast_end,
            "prediction_label": prediction_label,
            "probability": probability,
            "image_path": str(jpg_path)
        })
        #print("Prediction:", pred.item())
        #print("Probabilities:", probs)

        print(f"{jpg_path.name} -> {prediction_label} ({probability:.4f})")  

    except Exception as e:                            
        print(f"Failed on {jpg_path}: {e}")          

if results:                                           
    df = pd.DataFrame(results)
    df = df.sort_values("prediction_time", ascending=False)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} predictions to {OUTPUT_CSV}")
else:
    print("No predictions were generated.")
