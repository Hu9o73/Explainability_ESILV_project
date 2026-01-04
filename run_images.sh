#!/bin/bash

# Stop immediately on error
set -e
set -o pipefail

# Error handler
trap 'echo ""; echo "❌ ERROR in IMAGE PIPELINE"; echo "➡️ Command failed at line $LINENO"; exit 1' ERR

echo "======================================"
echo " RUNNING IMAGE PIPELINE (AlexNet + DenseNet)"
echo "======================================"

# -----------------------------
# ALEXNET
# -----------------------------
echo ""
echo ">>> AlexNet - Predictions"
python scripts/alexnet/alexnet_predictions.py

echo ""
echo ">>> AlexNet - Select GradCAM cases"
python scripts/alexnet/select_cases_alexnet.py

echo ""
echo ">>> AlexNet - GradCAM"
python scripts/alexnet/run_gradcam_alexnet.py

echo ""
echo ">>> AlexNet - LIME"
python scripts/alexnet/run_lime_alexnet.py

echo ""
echo ">>> AlexNet - SHAP"
python scripts/alexnet/run_shap_alexnet.py


# -----------------------------
# DENSENET
# -----------------------------
echo ""
echo ">>> DenseNet - Predictions"
python scripts/densenet/densenet_predictions.py

echo ""
echo ">>> DenseNet - Select GradCAM cases"
python scripts/densenet/select_gradcam_cases_densenet.py

echo ""
echo ">>> DenseNet - GradCAM"
python scripts/densenet/run_gradcam_densenet.py

echo ""
echo ">>> DenseNet - LIME"
python scripts/densenet/run_lime_densenet.py

echo ""
echo ">>> DenseNet - SHAP"
python scripts/densenet/run_shap_densenet.py


echo ""
echo "======================================"
echo " IMAGE PIPELINE FINISHED SUCCESSFULLY "
echo "======================================"
