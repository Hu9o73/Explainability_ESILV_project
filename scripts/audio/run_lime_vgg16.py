import os
import sys
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.audio.vgg16.model import load_vgg16_audio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "data/samples/lime_vgg16_audio"
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

def predict_fn(images):
    tensors = []
    for img in images:
        pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        tensors.append(transform(pil))
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()

explainer = lime_image.LimeImageExplainer()

for name, path in cases.items():
    image = np.array(Image.open(path).convert("RGB"))

    explanation = explainer.explain_instance(
        image=image,
        classifier_fn=predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=1000
    )

    label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        label,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    fig, ax = plt.subplots()
    ax.imshow(mark_boundaries(temp / 255.0, mask))
    ax.axis("off")

    out_path = os.path.join(OUT_DIR, f"{name}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"LIME {name} saved -> {out_path}")
