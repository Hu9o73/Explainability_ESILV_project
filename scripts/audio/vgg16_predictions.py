import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# -------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.audio.vgg16.model import load_vgg16_audio

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_DIR = "data/audio/spectrograms"
CLASSES = ["fake", "real"]
IMG_SIZE = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# TRANSFORM
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = load_vgg16_audio(device=DEVICE)
model.eval()

# -------------------------------------------------
# PREDICTION LOOP
# -------------------------------------------------
y_true, y_pred = [], []

print("\nRunning VGG16 AUDIO predictions...\n")

for gt_idx, gt_name in enumerate(CLASSES):
    folder = os.path.join(DATA_DIR, gt_name)
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg")))

    print(f"Processing {gt_name.upper()} ({len(files)} spectrograms)\n")

    for filename in files:
        path = os.path.join(folder, filename)
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred].item()

        y_true.append(gt_idx)
        y_pred.append(pred)

        print(
            f"{filename:30s} | "
            f"GT: {gt_name:5s} | "
            f"PRED: {CLASSES[pred]:5s} | "
            f"CONF: {conf:.3f}"
        )

    print("-" * 70)

# -------------------------------------------------
# METRICS (indicatives)
# -------------------------------------------------
y_true, y_pred = np.array(y_true), np.array(y_pred)
acc = (y_true == y_pred).mean()

print("\nSummary")
print("-------")
print(f"Samples  : {len(y_true)}")
print(f"Accuracy : {acc:.3f}")
