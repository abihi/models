# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./Bontouch"
#mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

cd "${CURRENT_DIR}"

# Root path for Bontouch dataset.
BONTOUCH_ROOT="${CURRENT_DIR}/Bontouch"

echo "Converting Hallway dataset..."
OUTPUT_DIR="${WORK_DIR}/hallway_dataset/tfrecord"
mkdir -p "${OUTPUT_DIR}"
#python ./relabel_bontouch_dataset.py
python ./build_bontouch_data.py  \
  --bontouch_image_folder="${BONTOUCH_ROOT}/hallway_dataset/images" \
  --bontouch_label_folder="${BONTOUCH_ROOT}/hallway_dataset/raw_segmentation" \
  --output_dir="${OUTPUT_DIR}"
