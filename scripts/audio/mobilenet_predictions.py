import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.audio.mobilenet.model import load_mobilenet_audio

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

model = load_mobilenet_audio(device=DEVICE)
model.eval()

y_true, y_pred = [], []

print("\nRunning MobileNet AUDIO predictions...\n")

for gt_idx, gt_name in enumerate(CLASSES):
    folder = os.path.join(DATA_DIR, gt_name)
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".png", ".jpg")):
            continue

        path = os.path.join(folder, fname)
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        y_true.append(gt_idx)
        y_pred.append(pred)

        print(f"{fname:30s} | GT: {gt_name} | PRED: {CLASSES[pred]}")

print(f"\nAccuracy: {(np.array(y_true)==np.array(y_pred)).mean():.3f}")
