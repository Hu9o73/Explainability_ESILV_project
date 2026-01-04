import os
import sys
import json
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.audio.vgg16.model import load_vgg16_audio
from xai.gradcam import gradcam

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "data/samples/gradcam_vgg16_audio"
os.makedirs(OUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

with open("data/samples/gradcam_cases_vgg16_audio.json") as f:
    cases = json.load(f)

model = load_vgg16_audio(device=DEVICE)
model.eval()

for name, path in cases.items():
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = torch.argmax(model(x), dim=1).item()

    overlay = gradcam(
        model=model,
        input_tensor=x,
        target_layer=model.features[-1],
        class_idx=pred,
        original_image=img
    )

    out = os.path.join(OUT_DIR, f"{name}.jpg")
    overlay.save(out)
    print(f"{name} saved -> {out}")
