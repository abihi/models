#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
#python "${WORK_DIR}"/model_test.py -v

# Go to datasets folder and convert Bontouch segmentation dataset.
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
#sh download_and_convert_bontouch.sh

# Go back to original directory.
cd "${CURRENT_DIR}"

BONTOUCH_DIR="${WORK_DIR}/${DATASET_DIR}/Bontouch"
AULA_DATASET="${BONTOUCH_DIR}/aula_dataset/"
HALLWAY_DATASET="${BONTOUCH_DIR}/hallway_dataset/"
KITCHEN_DATASET="${BONTOUCH_DIR}/kitchen_dataset/"
LIVINGROOM_DATASET="${BONTOUCH_DIR}/livingroom_dataset/"
DININGROOM_DATASET="${BONTOUCH_DIR}/diningroom_dataset/"


cd "${WORK_DIR}"

echo "Running prediction on Bontouch dataset (Hallway)"
python prediction_bontouch_dataset.py \
--path=${HALLWAY_DATASET} \
--filetype="png"
# echo "Calculating mean intersection over union (Hallway)"
# python eval_bontouch_dataset.py

# echo "Running prediction on Bontouch dataset (Aula)"
# python prediction_bontouch_dataset.py \
#   --path=${AULA_DATASET} \
#   --filetype="png"
#
# echo "Running prediction on Bontouch dataset (Kitchen)"
# python prediction_bontouch_dataset.py \
#   --path=${KITCHEN_DATASET} \
#   --filetype="png"

echo "Running prediction on Bontouch dataset (LivingRoom)"
python prediction_bontouch_dataset.py \
  --path=${LIVINGROOM_DATASET} \
  --filetype="png"

# echo "Running prediction on Bontouch dataset (DiningRoom)"
# python prediction_bontouch_dataset.py \
#   --path=${DININGROOM_DATASET} \
#   --filetype="png"
