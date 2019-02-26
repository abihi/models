import tarfile
with tarfile.open('deeplab_model.tar.gz', 'w:gz') as tar:
  tar.add('datasets/pascal_voc_seg/exp/train_on_trainval_set_mobilenetv2/export/frozen_inference_graph.pb', arcname="frozen_inference_graph.pb")

import os
import StringIO
import tarfile
import tempfile
import urllib
import cv2

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf

## Helper methods ##
class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

#_TARBALL_NAME = 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz'
_TARBALL_NAME = 'deeplab_model.tar.gz'

MODEL = DeepLabModel(_TARBALL_NAME)
print('model loaded successfully!')

## Change this part to openCV camera
from os import path
import time

file_name = 'datasets/ADE20K/ADEChallengeData2016/images/validation/ADE_val_00000026.jpg'

#orignal_im = Image.open(file_name)
#print("Original im type: ",orignal_im.tpye())

#print 'running deeplab on image frame'
#resized_im, seg_map = MODEL.run(orignal_im)
#vis_segmentation(resized_im, seg_map)

#OpenCV live test
def live_test():
    cv2.namedWindow("mobilenet_v2 Deeplab (PascalVOC)")
    cap = cv2.VideoCapture(0)
    rval = True

    while rval:
    	start = time.time()
    	rval, frame = cap.read()
        if rval == False:
            break
        im_frame = Image.fromarray(frame)
    	end = time.time()
    	print '%30s' % 'Grabbed camera frame in ', str((end - start)*1000), 'ms'

    	start = time.time()
        print 'running deeplab on image frame'
        resized_im, seg_map = MODEL.run(im_frame)
        end = time.time()
    	print '%30s' % 'Model inference done in ', str((end - start)*1000), 'ms'

        start = time.time()
        seg_image = label_to_color_image(seg_map).astype(np.uint8)
        resized_im = np.asarray(resized_im)
        # apply the overlay
        alpha = 0.8
        cv2.addWeighted(seg_image, alpha, resized_im, 1 - alpha, 0, resized_im)
    	end = time.time()
    	print '%30s' % 'Segmentation map overlay in ', str((end - start)*1000), 'ms\n'

    	cv2.imshow("mobilenet_v2 Deeplab (PascalVOC)", resized_im)

        print rval

    	key = cv2.waitKey(1)
    	if key == 27: # exit on ESC
    	    break
    cap.release()
    cv2.destroyAllWindows()

#OpenCV test
live_test()
