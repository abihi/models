# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./SUN_RGBD"

cd "${CURRENT_DIR}"

OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

echo "Converting SUN-RGBD dataset..."
python ./build_ade20k_data.py  \
  --train_image_folder="${WORK_DIR}/images/train/" \
  --train_image_label_folder="${WORK_DIR}/annotations/train/" \
  --val_image_folder="${WORK_DIR}/images/test/" \
  --val_image_label_folder="${WORK_DIR}/annotations/test/" \
  --output_dir="${OUTPUT_DIR}"
