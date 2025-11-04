#!/bin/bash
# Source conda's shell functions (update the path to your conda installation if necessary)
source /home/sabdelazim/miniconda3/etc/profile.d/conda.sh

# Paths (adjust as necessary)
IMAGE_FOLDER="/home/sabdelazim/Desktop/Kaitlyn_Catalyst/ct_classifier/datasets/CaltechCT/eccv_18_all_images_sm"
DETECTOR_JSON="/home/sabdelazim/Desktop/Kaitlyn_Catalyst/ct_classifier/datasets/CaltechCT/detector_output_file.json"
CLASSIFIER_JSON="/home/sabdelazim/Desktop/Kaitlyn_Catalyst/ct_classifier/datasets/CaltechCT/classifier_output_file.json"
PREDICTIONS_JSON="/home/sabdelazim/Desktop/Kaitlyn_Catalyst/ct_classifier/datasets/CaltechCT/prediction_output_file.json"

# -------------------
# Step 1: Run the detector in the "speciesnet" environment
# -------------------
echo "Activating speciesnet environment for detection..."
conda activate speciesnet
echo "Running detector..."
python -m speciesnet.scripts.run_model --detector_only --folders "$IMAGE_FOLDER" --predictions_json "$DETECTOR_JSON"
echo "Detector finished. Deactivating environment."
conda deactivate

# -------------------
# Step 2: Run the classifier in the "speciesnet-tf" environment
# -------------------
echo "Activating speciesnet-tf environment for classification..."
conda activate speciesnet-tf
echo "Running classifier..."
python -m speciesnet.scripts.run_model --classifier_only --folders "$IMAGE_FOLDER" --predictions_json "$CLASSIFIER_JSON" --detections_json "$DETECTOR_JSON"
echo "Classifier finished. Deactivating environment."
conda deactivate

echo "Ensemble run complete."
