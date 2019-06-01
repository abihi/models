import tensorflow as tf

#graph_def_file = "/home/abihi/tf/models/research/deeplab/trained_models/sunrgbd_relabel_mobilenet/frozen_inference_graph.pb"
graph_def_file = "/home/abihi/mobileNetsV2Models/32x32frozen_inference_graph.pb"
input_arrays = ["MobilenetV2/MobilenetV2/input"]
output_arrays = ["ArgMax"]
shapes = {
  "ImageTensor": [1,257,257,3]
}

converter = tf.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)#, input_shapes=shapes)
#converter.post_training_quantize=True
#converter.default_ranges_stats=(0, 6)
converter.inference_type=tf.float32
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
#converter.inference_input_type=tf.uint8
#converter.quantized_input_stats={"MobilenetV2/MobilenetV2/input": (128,127)}
tflite_model = converter.convert()
open("/home/abihi/mobileNetsV2Models/32x32frozen_inference_graph.tflite", "wb").write(tflite_model)
