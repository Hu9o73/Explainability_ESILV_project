# models/audio/vgg16/model.py

import os
import torch
import torch.nn as nn
from torchvision import models

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class VGG16Audio(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Replace classifier
        self.backbone.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.backbone(x)


def load_model(device="cpu"):
    model = VGG16Audio(num_classes=2)

    weights_path = os.path.join(THIS_DIR, "weights.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights.pth not found in {THIS_DIR}")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model
