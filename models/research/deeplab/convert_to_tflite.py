import tensorflow as tf

graph_def_file = "/home/abihi/tf/models/research/deeplab/good_pascal_mobilenet/frozen_inference_graph.pb"
input_arrays = ["ImageTensor"]
output_arrays = ["SemanticPredictions"]
shapes = {
  "ImageTensor": [1,513,513,3]
}

converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays, input_shapes=shapes)
tflite_model = converter.convert()
open("mobilenetv2_pascalVOC.tflite", "wb").write(tflite_model)
