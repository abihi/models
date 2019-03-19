#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local test on ADE20K VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

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

# Go to datasets folder and download ADE20K segmentation dataset.
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
sh download_and_convert_ade20k.sh

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
ADE20K_FOLDER="ADE20K"
EXP_FOLDER="exp/train_on_trainval_set_mobilenetv2"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${ADE20K_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ADE20K_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ADE20K_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ADE20K_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${ADE20K_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
CKPT_NAME="deeplabv3_mnv2_ade20k_train_2018_12_03"
TF_INIT_CKPT="deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

ADE20K_DATASET="${WORK_DIR}/${DATASET_DIR}/${ADE20K_FOLDER}/tfrecord"

TRAIN_CROP_SIZE=257
EVIS_CROP_SIZE=1601
NUM_ITERATIONS=150000
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="mobilenet_v2" \
  --dataset="ade20k" \
  --output_stride=16 \
  --train_crop_size="${TRAIN_CROP_SIZE}" \
  --train_crop_size="${TRAIN_CROP_SIZE}" \
  --train_batch_size=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=false \
  --tf_initial_checkpoint="${INIT_FOLDER}/${CKPT_NAME}/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${ADE20K_DATASET}"

  # Run evaluation. This performs eval over the full val split (2000 images) and
  # will take a while.
  # Using the provided checkpoint, one should expect mIOU=32.04%.
  python "${WORK_DIR}"/eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="mobilenet_v2" \
    --dataset="ade20k" \
    --eval_crop_size="${EVIS_CROP_SIZE}" \
    --eval_crop_size="${EVIS_CROP_SIZE}" \
    --checkpoint_dir="${TRAIN_LOGDIR}" \
    --eval_logdir="${EVAL_LOGDIR}" \
    --dataset_dir="${ADE20K_DATASET}" \
    --max_number_of_evaluations=1

  # Visualize the results.
  python "${WORK_DIR}"/vis.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="mobilenet_v2" \
    --dataset="ade20k" \
    --vis_crop_size="${EVIS_CROP_SIZE}" \
    --vis_crop_size="${EVIS_CROP_SIZE}" \
    --checkpoint_dir="${TRAIN_LOGDIR}" \
    --vis_logdir="${VIS_LOGDIR}" \
    --dataset_dir="${ADE20K_DATASET}" \
    --max_number_of_iterations=1

  # Export the trained checkpoint.
  CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
  EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

  python "${WORK_DIR}"/export_model.py \
    --logtostderr \
    --checkpoint_path="${CKPT_PATH}" \
    --export_path="${EXPORT_PATH}" \
    --model_variant="mobilenet_v2" \
    --num_classes=151 \
    --crop_size="${TRAIN_CROP_SIZE}" \
    --crop_size="${TRAIN_CROP_SIZE}" \
    --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.