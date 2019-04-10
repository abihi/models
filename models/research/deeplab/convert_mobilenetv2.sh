#!/bin/sh

PB_FILE=trained_models/sunrgbd_relabel_mobilenet/frozen_inference_graph.pb
TF_FILE=relabel_sunrgbd.tflite
DIMENSION=257

echo "# clear"
rm -rf tensorboard/*

echo "# import to tensorboard"
python import_pb_to_tensorboard.py \
  --model_dir=$PB_FILE \
  --log_dir="tensorboard/"

echo "# clear"
rm -rf ${TF_FILE}

echo "# convert pb to tflite"
# tflite_convert \
#     --output_file=$TF_FILE \
#     --graph_def_file=$PB_FILE \
#     --output_format=TFLITE \
#     --input_arrays=MobilenetV2/MobilenetV2/input \
#     --output_arrays=ArgMax \
#     --input_shapes=1,${DIMENSION},${DIMENSION},3 \
#     --inference_input_type=QUANTIZED_UINT8 \
#     --inference_type=FLOAT \
#     --mean_values=128 \
#     --std_dev_values=127 \
#     --post_training_quantize \

python convert_to_tflite.py

cp relabel_sunrgbd.tflite /home/abihi/JejuNet/android/app/src/main/assets/new_relabel_sunrgbd.tflite
