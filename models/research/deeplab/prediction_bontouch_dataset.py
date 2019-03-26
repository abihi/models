import tarfile
with tarfile.open('deeplab_model.tar.gz', 'w:gz') as tar:
  tar.add('good_relabel_sunrgbd_mobilenet/frozen_inference_graph.pb', arcname="frozen_inference_graph.pb")

import os
import sys
import tarfile

from datasets import visualize_data
from PIL import Image
import tensorflow as tf
import numpy as np

class DeepLabModel(object):
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 257
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
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

def predictions(files):
    count = 0
    for filename in files:
        im = Image.open(filename)
        count += 1
        resized_im, seg_map = MODEL.run(im)

        seg_image = visualize_data.label_to_color_image(seg_map).astype(np.uint8)
        #visualize_data.vis_segmentation(resized_im, seg_map, 1)

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
