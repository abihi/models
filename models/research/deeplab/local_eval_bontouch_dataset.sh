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
python "${WORK_DIR}"/model_test.py -v

# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
sh download_and_convert_bontouch.sh

# Go back to original directory.
cd "${CURRENT_DIR}"

BONTOUCH_DIR = "${WORK_DIR}/${DATASET_DIR}/Bontouch"
HALLWAY_PRED_LOGDIR = "${BONTOUCH_DIR}/hallway_dataset_voc/predictions"
HALLWAY_DATASET = "${BONTOUCH_DIR}/hallway_dataset_voc/tfrecord"
mkdir -p "${HALLWAY_PRED_LOGDIR}"

echo "Running prediction on hallway segment"
python "${WORK_DIR}"/prediction_bontouch_dataset.py
