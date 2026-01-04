import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


RAW_IMAGE_DIR = "data/images/raw"
OUTPUT_DIR = "data/images/processed"

IMG_SIZE = (224, 224)
CLASSES = ["fake", "real"]  

# ImageNet normalization (compatible AlexNet / DenseNet)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_xray(input_path, output_path):
    """
    - Load grayscale X-ray
    - Convert to 3-channel (RGB-like)
    - Resize
    - Normalize (ImageNet)
    """
    # Load grayscale
    img_gray = Image.open(input_path).convert("L")

    # Convert grayscale -> RGB (3 identical channels)
    img_rgb = Image.merge("RGB", (img_gray, img_gray, img_gray))

    tensor = transform(img_rgb)

    # Convert back to image for saving
    tensor = tensor.permute(1, 2, 0).numpy()

    # De-normalize for visualization/storage
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = (tensor * 255).astype(np.uint8)

    cv2.imwrite(
        output_path,
        cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
    )


def process_all():
    print("Starting X-ray preprocessing (CheXpert)...")

    for label in CLASSES:
        input_dir = os.path.join(RAW_IMAGE_DIR, label)
        output_dir = os.path.join(OUTPUT_DIR, label)

        ensure_dir(output_dir)

        files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        print(f"Processing {label}: {len(files)} images")

        for filename in files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                process_xray(input_path, output_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print("X-ray preprocessing finished successfully.")


if __name__ == "__main__":
    process_all()
