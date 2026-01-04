import os
import sys
import json
import torch
import shap
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.audio.mobilenet.model import load_mobilenet_audio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "data/samples/shap_mobilenet_audio"
os.makedirs(OUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

with open("data/samples/gradcam_cases_mobilenet_audio.json") as f:
    cases = json.load(f)

model = load_mobilenet_audio(device=DEVICE)
model.eval()

def model_forward(x):
    with torch.no_grad():
        logits = model(x.to(DEVICE))
        return torch.softmax(logits, dim=1)

background = torch.zeros((1, 3, 224, 224)).to(DEVICE)
explainer = shap.GradientExplainer(model_forward, background)

for name, path in cases.items():
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0)

    shap_values = explainer.shap_values(x.to(DEVICE))

    shap.image_plot(
        shap_values,
        x.numpy(),
        show=False
    )

    out_path = os.path.join(OUT_DIR, f"{name}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"SHAP {name} saved -> {out_path}")
