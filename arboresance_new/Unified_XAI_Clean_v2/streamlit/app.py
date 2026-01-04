# ==================================================
# FORCE PROJECT ROOT (CRITICAL FOR STREAMLIT)
# ==================================================
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==================================================
# Imports
# ==================================================
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2

# Models
from models.image.alexnet import load_model as load_alexnet
from models.image.densenet import load_model as load_densenet

# XAI (TES CLASSES)
from xai.image.gradcam import GradCAM
from xai.image.lime import LimeExplainer
from xai.image.shap import ShapExplainer, shap_to_heatmap

# ==================================================
# Page config
# ==================================================
st.set_page_config(
    page_title="Unified XAI Interface",
    layout="wide"
)

st.title("Unified Explainable AI Interface")
st.write("Image and audio classification with explainability methods.")

# ==================================================
# Sidebar - Input
# ==================================================
uploaded_file = st.sidebar.file_uploader(
    "Upload an image (.png, .jpg)",
    type=["png", "jpg", "jpeg"]
)

# ==================================================
# Model registry
# ==================================================
IMAGE_MODELS = {
    "alexnet": ("AlexNet", load_alexnet),
    "densenet": ("DenseNet", load_densenet),
}

XAI_METHODS = ["gradcam", "lime", "shap"]

# ==================================================
# Main logic
# ==================================================
if uploaded_file is not None:

    # --------------------------------------------------
    # Model selection
    # --------------------------------------------------
    model_key = st.sidebar.selectbox(
        "Select model",
        list(IMAGE_MODELS.keys()),
        format_func=lambda k: IMAGE_MODELS[k][0]
    )

    xai_method = st.sidebar.selectbox(
        "Select XAI method",
        XAI_METHODS
    )

    # --------------------------------------------------
    # Load image
    # --------------------------------------------------
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=300)

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    x = transform(image).unsqueeze(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.to(device)

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    with st.spinner("Loading model..."):
        model = IMAGE_MODELS[model_key][1](device=device)

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = probs[0, pred_idx].item()

    LABELS = ["fake", "real"]

    st.success(
        f"Prediction: **{LABELS[pred_idx]}** "
        f"(confidence: {confidence:.3f})"
    )

    st.divider()
    st.subheader("Explainability")

    # ==================================================
    # GRAD-CAM
    # ==================================================
    if xai_method == "gradcam":
        if st.button("Run Grad-CAM"):
            with st.spinner("Computing Grad-CAM..."):
                target_layer = model.features[-1]
                cam = GradCAM(model, target_layer)
                heatmap = cam.generate(x, class_idx=pred_idx)
                overlay = cam.overlay_on_image(image, heatmap)

            st.image(overlay, caption="Grad-CAM", use_column_width=True)

    # ==================================================
    # LIME
    # ==================================================
    elif xai_method == "lime":
        if st.button("Run LIME"):
            with st.spinner("Computing LIME..."):
                explainer = LimeExplainer(
                    model=model,
                    device=device,
                    transform=transform
                )

                image_np = np.array(image)
                lime_vis = explainer.explain(
                    image_np=image_np,
                    class_idx=pred_idx
                )

            st.image(lime_vis, caption="LIME", use_column_width=True)

    # ==================================================
    # SHAP
    # ==================================================
    elif xai_method == "shap":
        if st.button("Run SHAP (slow)"):
            with st.spinner("Computing SHAP (this may take time)..."):
                background = torch.zeros((1, 3, 224, 224)).to(device)
                explainer = ShapExplainer(model, background)

                shap_values = explainer.explain(x)
                heatmap = shap_to_heatmap(shap_values[pred_idx][0])

                heatmap = cv2.resize(heatmap, (224, 224))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                img_np = np.array(image.resize((224, 224)))
                overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

            st.image(overlay, caption="SHAP", use_column_width=True)
