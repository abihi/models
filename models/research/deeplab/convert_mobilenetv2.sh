#!/bin/sh

PB_FILE=good_relabel_sunrgbd_mobilenet/frozen_inference_graph.pb
TF_FILE=relabel_sunrgbd.tflite
DIMENSION=257

echo "# import to tensorboard"
python import_pb_to_tensorboard.py \
  --model_dir=$PB_FILE \
  --log_dir="tensorboard/"

echo "# clear"

rm -rf *.tflite

echo "# convert"

tflite_convert \
    --output_file=$TF_FILE \
    --graph_def_file=$PB_FILE \
    --output_format=TFLITE \
    --input_arrays=ImageTensor \
    --output_arrays=RawSemanticPredictions \
    --input_shapes=1,${DIMENSION},${DIMENSION},3 \
    --inference_input_type=QUANTIZED_UINT8 \
    --inference_type=FLOAT \
    --allow_custom_ops \
    --mean_values=128 \
    --std_dev_values=127 \
    --default_ranges_min=-1 \
    --default_ranges_max=1 \
