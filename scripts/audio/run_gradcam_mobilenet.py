import os
import sys
import json
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.audio.mobilenet.model import load_mobilenet_audio
from xai.gradcam import gradcam

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "data/samples/gradcam_mobilenet_audio"
os.makedirs(OUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

with open("data/samples/gradcam_cases_mobilenet_audio.json") as f:
    cases = json.load(f)

model = load_mobilenet_audio(device=DEVICE)
model.eval()

for case_name, img_path in cases.items():
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1))

    overlay = gradcam(
        model=model,
        input_tensor=x,
        target_layer=model.features[-1],
        class_idx=pred,
        original_image=img
    )

    out_path = os.path.join(OUT_DIR, f"{case_name}.jpg")
    overlay.save(out_path)
    print(f"{case_name} saved -> {out_path}")
