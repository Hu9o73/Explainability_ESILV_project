#!/bin/bash

# =========================================
# Create Unified XAI Clean Architecture
# Root = directory where this script lives
# =========================================

set -e

# 📍 Se placer dans le dossier du script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/Unified_XAI_Clean"

echo "📁 Création du projet dans : $PROJECT_ROOT"

# Création du dossier racine
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

# -------------------------
# Dossiers racine
# -------------------------
mkdir -p \
  data \
  preprocessing \
  models \
  inference \
  streamlit \
  docs \
  notebooks \
  assets/figures

# -------------------------
# Configs (sélection TP/FP/FN/TN)
# -------------------------
mkdir -p \
  configs/image/alexnet \
  configs/image/densenet \
  configs/audio/vgg16 \
  configs/audio/mobilenet

# -------------------------
# Scripts (exécution)
# -------------------------
mkdir -p \
  scripts/image/alexnet \
  scripts/image/densenet \
  scripts/audio/vgg16 \
  scripts/audio/mobilenet

# -------------------------
# XAI (implémentations)
# -------------------------
mkdir -p \
  xai/image \
  xai/audio

# -------------------------
# Outputs (résultats XAI)
# -------------------------
mkdir -p \
  outputs/image/alexnet/gradcam/{TP,FP,FN,TN} \
  outputs/image/alexnet/lime/{TP,FP,FN,TN} \
  outputs/image/alexnet/shap/{TP,FP,FN,TN} \
  outputs/image/densenet \
  outputs/audio/vgg16 \
  outputs/audio/mobilenet

# -------------------------
# Fichiers de base
# -------------------------
touch README.md requirements.txt .gitignore

# -------------------------
# .gitkeep pour versionner les dossiers vides
# -------------------------
find . -type d -empty -exec touch {}/.gitkeep \;

echo "✅ Architecture Unified_XAI_Clean créée avec succès."
