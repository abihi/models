import os
import sys
import numpy as np
import tensorflow as tf

from PIL import Image
from datasets import visualize_data

# Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_path="mobilenet_v2_deeplab_v3_256_myquant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tflite_model(image):
    input_shape = input_details[0]['shape']
    input = image[None, :, :, :]
    #print "input data shape: ", input.shape

    input_test = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    #print "input test shape: ", input_test.shape

    input_data = np.array(input_test, dtype=np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #output_data = output_data[0, :, :, 0]
    print "output data shape: ", output_data.shape
    print output_details[0]['name']

    return output_data.astype(np.uint8)

def resize(image, input_size):
    width, height = image.size
    resize_ratio = 1.0 * input_size / max(width, height)
    target_size = (input_size, input_size)#(int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    return np.asarray(resized_image)

def predictions(files):
    count = 0
    for filename in files:
        im = Image.open(filename)
        count += 1
        resized_im = resize(im, 257)
        seg_map = tflite_model(resized_im)

        seg_image = visualize_data.label_to_color_image(seg_map).astype(np.uint8)
        visualize_data.vis_segmentation(im, resized_im, seg_map, 1)

        filename_preds = filename.replace("images", "predictions", 1)
        filename_preds = filename_preds.replace("jpg", "png", 1)
        filename_vis = filename.replace("images", "predictions_vis", 1)

        img_seg=Image.fromarray(seg_map.astype(np.uint8), mode='L')
        img_seg.save(filename_preds)
        img_seg.close()

        img_vis=Image.fromarray(seg_image)
        img_vis.save(filename_vis)
        img_vis.close()
        im.close()

        sys.stdout.write('\r>> Running prediction on image %d/%d' % (
            count, len(files)))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()

import glob

preds_dir = "datasets/Bontouch/hallway_dataset_voc/predictions"
preds_vis = "datasets/Bontouch/hallway_dataset_voc/predictions_vis"
hallway_files = glob.glob("/home/abihi/tf/models/research/deeplab/datasets/Bontouch/hallway_dataset_voc/images/*.jpg")
if not os.path.isdir(preds_dir):
    os.mkdir(preds_dir)

if not os.path.isdir(preds_vis):
    os.mkdir(preds_vis)

print "Running predictions on hallway segment"
predictions(hallway_files)
