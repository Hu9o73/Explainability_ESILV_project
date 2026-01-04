#!/bin/bash

# Stop immediately on error
set -e
set -o pipefail

# Error handler
trap 'echo ""; echo "❌ ERROR in AUDIO PIPELINE"; echo "➡️ Command failed at line $LINENO"; exit 1' ERR

echo "======================================"
echo " RUNNING AUDIO PIPELINE (VGG16 + MobileNet)"
echo "======================================"

# -----------------------------
# VGG16 AUDIO
# -----------------------------
echo ""
echo ">>> VGG16 Audio - Predictions"
python scripts/audio/vgg16_predictions.py

echo ""
echo ">>> VGG16 Audio - Select GradCAM cases"
python scripts/audio/select_gradcam_vgg16_audio.py

echo ""
echo ">>> VGG16 Audio - GradCAM"
python scripts/audio/run_gradcam_vgg16_audio.py

echo ""
echo ">>> VGG16 Audio - LIME"
python scripts/audio/run_lime_vgg16.py

echo ""
echo ">>> VGG16 Audio - SHAP"
python scripts/audio/run_shap_vgg16.py


# -----------------------------
# MOBILENET AUDIO
# -----------------------------
echo ""
echo ">>> MobileNet Audio - Predictions"
python scripts/audio/mobilenet_predictions.py

echo ""
echo ">>> MobileNet Audio - Select GradCAM cases"
python scripts/audio/select_gradcam_mobilenet.py

echo ""
echo ">>> MobileNet Audio - GradCAM"
python scripts/audio/run_gradcam_mobilenet.py

echo ""
echo ">>> MobileNet Audio - LIME"
python scripts/audio/run_lime_mobilenet.py

echo ""
echo ">>> MobileNet Audio - SHAP"
python scripts/audio/run_shap_mobilenet.py


echo ""
echo "======================================"
echo " AUDIO PIPELINE FINISHED SUCCESSFULLY "
echo "======================================"
