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
python "${WORK_DIR}"/model_test.py -v

# Go to datasets folder and download SUNRGBD segmentation dataset.
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
sh download_and_convert_sun_rgbd.sh

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
SUNRGBD_FOLDER="SUN_RGBD"
EXP_FOLDER="exp/train_on_trainval_set_mobilenetv2"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${SUNRGBD_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${SUNRGBD_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${SUNRGBD_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${SUNRGBD_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${SUNRGBD_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="https://storage.googleapis.com/mobilenet_v2/checkpoints"
CKPT_NAME="mobilenet_v2_1.0_192"
TF_INIT_CKPT="mobilenet_v2_1.0_192.tgz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

SUNRGBD_DATASET="${WORK_DIR}/${DATASET_DIR}/${SUNRGBD_FOLDER}/tfrecord"

TRAIN_CROP_SIZE=257
EVIS_CROP_SIZE_X=737
EVIS_CROP_SIZE_Y=737
NUM_ITERATIONS=100
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="mobilenet_v2" \
  --dataset="sun_rgbd" \
  --output_stride=16 \
  --train_crop_size="${TRAIN_CROP_SIZE}" \
  --train_crop_size="${TRAIN_CROP_SIZE}" \
  --train_batch_size=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --initialize_last_layer=false \
  --last_layers_contain_logits_only=true \
  --fine_tune_batch_norm=false \
  --tf_initial_checkpoint="${INIT_FOLDER}/${CKPT_NAME}/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${SUNRGBD_DATASET}"

  # Run evaluation. This performs eval over the full test set (5050 images)
  python "${WORK_DIR}"/eval_old.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="mobilenet_v2" \
    --dataset="sun_rgbd" \
    --eval_crop_size="${EVIS_CROP_SIZE_X}" \
    --eval_crop_size="${EVIS_CROP_SIZE_Y}" \
    --checkpoint_dir="${TRAIN_LOGDIR}" \
    --eval_logdir="${EVAL_LOGDIR}" \
    --dataset_dir="${SUNRGBD_DATASET}" \
    --max_number_of_evaluations=1

  # Visualize the results.
  python "${WORK_DIR}"/vis.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="mobilenet_v2" \
    --dataset="sun_rgbd" \
    --vis_crop_size="${EVIS_CROP_SIZE_X}" \
    --vis_crop_size="${EVIS_CROP_SIZE_Y}" \
    --checkpoint_dir="${TRAIN_LOGDIR}" \
    --vis_logdir="${VIS_LOGDIR}" \
    --dataset_dir="${SUNRGBD_DATASET}" \
    --max_number_of_iterations=1

  # Export the trained checkpoint.
  CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
  EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

  python "${WORK_DIR}"/export_model.py \
    --logtostderr \
    --checkpoint_path="${CKPT_PATH}" \
    --export_path="${EXPORT_PATH}" \
    --model_variant="mobilenet_v2" \
    --num_classes=14 \
    --crop_size="${TRAIN_CROP_SIZE}" \
    --crop_size="${TRAIN_CROP_SIZE}" \
    --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
