# models/audio/mobilenet/model.py

import os
import torch
import torch.nn as nn
from torchvision import models

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class MobileNetAudio(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        self.backbone.classifier[1] = nn.Linear(
            self.backbone.last_channel,
            num_classes
        )

    def forward(self, x):
        return self.backbone(x)


def load_model(device="cpu"):
    model = MobileNetAudio(num_classes=2)

    weights_path = os.path.join(THIS_DIR, "weights.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights.pth not found in {THIS_DIR}")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model
