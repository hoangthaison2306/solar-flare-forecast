import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from explainingFullDisk.modeling.model import Custom_AlexNet


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

# load image
for jpg_path in Path("./data/hmi_jpg/").rglob("*.jpg"):
    img = Image.open(jpg_path).convert("L")
    tensor = transform(img).unsqueeze(0)

# inference
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)

    print("Prediction:", pred.item())
    print("Probabilities:", probs)
