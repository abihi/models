import tarfile
with tarfile.open('deeplab_model.tar.gz', 'w:gz') as tar:
  tar.add('good_relabel_mobilenet/frozen_inference_graph.pb', arcname="frozen_inference_graph.pb")

import os
import StringIO
import tarfile
import tempfile
import urllib

from datasets import visualize_data
import numpy as np
from PIL import Image

import tensorflow as tf
import cv2

## Helper methods ##
class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 257
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

_TARBALL_NAME = 'deeplab_model.tar.gz'

MODEL = DeepLabModel(_TARBALL_NAME)
print('model loaded successfully!')

import time

#OpenCV live test
def live_test():
    cv2.namedWindow("mobilenet_v2 Deeplab (ADE20K relabled)")
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
        seg_image = visualize_data.label_to_color_image(seg_map).astype(np.uint8)
        resized_im = np.asarray(resized_im)
        # apply the overlay
        alpha = 0.8
        cv2.addWeighted(seg_image, alpha, resized_im, 1 - alpha, 0, resized_im)
    	end = time.time()
    	print '%30s' % 'Segmentation map overlay in ', str((end - start)*1000), 'ms\n'

        final_image = cv2.resize(resized_im, None, fx=7, fy=5)
    	cv2.imshow("mobilenet_v2 Deeplab (ADE20K relabled)", final_image)

        print rval

    	key = cv2.waitKey(1)
    	if key == 27: # exit on ESC
    	    break
    cap.release()
    cv2.destroyAllWindows()

#OpenCV test
live_test()
