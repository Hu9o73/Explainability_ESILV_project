#!/bin/bash

echo "######################################"
echo "# EXPLAINABILITY AI - FULL PIPELINE   #"
echo "######################################"

echo ""
echo ">>> STEP 1: IMAGES"
bash run_images.sh

echo ""
echo ">>> STEP 2: AUDIOS"
bash run_audio.sh

echo ""
echo "######################################"
echo "# ALL PIPELINES FINISHED SUCCESSFULLY #"
echo "######################################"
