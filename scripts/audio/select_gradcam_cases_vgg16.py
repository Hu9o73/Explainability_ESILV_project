import os
import sys
import json
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.audio.vgg16.model import load_vgg16_audio

DATA_DIR = "data/audio/spectrograms"
CLASSES = ["fake", "real"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = load_vgg16_audio(device=DEVICE)
model.eval()

cases = {"TP": [], "FP": [], "FN": [], "TN": []}

for gt_idx, gt_name in enumerate(CLASSES):
    folder = os.path.join(DATA_DIR, gt_name)

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".png", ".jpg")):
            continue

        path = os.path.join(folder, fname)
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = torch.argmax(model(x), dim=1).item()

        if gt_idx == 1 and pred == 1:
            cases["TP"].append(path)
        elif gt_idx == 0 and pred == 1:
            cases["FP"].append(path)
        elif gt_idx == 1 and pred == 0:
            cases["FN"].append(path)
        elif gt_idx == 0 and pred == 0:
            cases["TN"].append(path)

print("\nConfusion cases:")
for k, v in cases.items():
    print(f"{k}: {len(v)}")

selected = {k: v[0] for k, v in cases.items() if v}

OUT = "data/samples/gradcam_cases_vgg16_audio.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(OUT, "w") as f:
    json.dump(selected, f, indent=2)

print(f"\nSaved -> {OUT}")
