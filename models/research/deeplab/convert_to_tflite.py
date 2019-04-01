import tensorflow as tf

graph_def_file = "/home/abihi/tf/models/research/deeplab/trained_models/sunrgbd_relabel_mobilenet/frozen_inference_graph.pb"
input_arrays = ["MobilenetV2/MobilenetV2/input"]
output_arrays = ["ArgMax"]
shapes = {
  "ImageTensor": [1,257,257,3]
}

converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays, input_shapes=shapes)
tflite_model = converter.convert()
open("relabel_sunrgbd_mobilenet.tflite", "wb").write(tflite_model)
