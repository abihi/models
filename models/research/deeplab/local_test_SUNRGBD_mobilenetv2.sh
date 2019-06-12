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

# Go to datasets folder and download SUNRGBD segmentation dataset.
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
#sh download_and_convert_sun_rgbd.sh

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
TRAINVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${SUNRGBD_FOLDER}/${EXP_FOLDER}/train/val"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${TRAINVAL_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"


#Large: https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz
#Medium-large: mobilenet_v2_1.0_192
#Medium: mobilenet_v2_1.0_160
#Medium-small: mobilenet_v2_1.0_128
#Small: mobilenet_v2_1.0_96
#ADE20K pretrained: http://download.tensorflow.org/models/deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz
TF_INIT_ROOT="https://storage.googleapis.com/mobilenet_v2/checkpoints"
CKPT_NAME="mobilenet_v2_1.0_224"
TF_INIT_CKPT="mobilenet_v2_1.0_224.tgz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

SUNRGBD_DATASET="${WORK_DIR}/${DATASET_DIR}/${SUNRGBD_FOLDER}/tfrecord"

TRAIN_CROP_SIZE=256
EVIS_CROP_SIZE_X=737
EVIS_CROP_SIZE_Y=737
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --trainval_split="trainval" \
  --model_variant="mobilenet_v2" \
  --dataset="sun_rgbd" \
  --output_stride=16 \
  --depth_multiplier=1.0 \
  --train_crop_size="${TRAIN_CROP_SIZE}" \
  --train_crop_size="${TRAIN_CROP_SIZE}" \
  --train_batch_size=16 \
  --trainval_batch_size=16 \
  --initialize_last_layer=false \
  --last_layers_contain_logits_only=true \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/${CKPT_NAME}.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${SUNRGBD_DATASET}"

  #Run evaluation. This performs eval over the full test set (5050 images)
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
  # python "${WORK_DIR}"/vis.py \
  #   --logtostderr \
  #   --vis_split="val" \
  #   --model_variant="mobilenet_v2" \
  #   --dataset="sun_rgbd_relabeled" \
  #   --vis_crop_size="${EVIS_CROP_SIZE_X}" \
  #   --vis_crop_size="${EVIS_CROP_SIZE_Y}" \
  #   --checkpoint_dir="${TRAIN_LOGDIR}" \
  #   --vis_logdir="${VIS_LOGDIR}" \
  #   --dataset_dir="${SUNRGBD_DATASET}" \
  #   --max_number_of_iterations=1

  # Export the trained checkpoint.
  CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-20000"
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
