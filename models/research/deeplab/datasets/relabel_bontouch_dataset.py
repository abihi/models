from PIL import Image
import numpy as np
import os
import glob
from matplotlib import gridspec
from matplotlib import pyplot as plt

def create_pascal_label_colormap():
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = seg_map#label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('relabeled image')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.5)
  plt.axis('off')
  plt.title('relabel overlay')

  plt.grid('off')
  plt.show()

LABEL_NAMES = np.asarray([
  'background','wall','floor'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

def relabel_images(files):
    count = 0
    for filename in files:
        #Converts img to grayscale
        im = Image.open(filename).convert('L')
        count += 1

        im_mat=np.asarray(im.getdata(),dtype=np.uint8).reshape((im.size[1],im.size[0]))
        #im_vis = label_to_color_image(im_mat)

        im_mat[im_mat==38] = 2
        im_mat[im_mat==75] = 1
        im_mat[im_mat>2] = 0

        im_mat=np.asarray(im_mat,dtype=np.uint8)
        img=Image.fromarray(im_mat,mode='L')

        #vis_segmentation(im, im_mat)

        im.close()

        if count % 5 == 0:
            print "Relabeling file ", count, "of", len(files)

        filename = filename.replace("SegmentationClassPNG", "raw_segmentation", 1)
        img.save(filename)
        img.close()

# Values after converting rgb segmap to grayscale
# 1 (wall) <- 75
# 2 (floor) <- 38

cwd = os.getcwd()

hallway_dir = cwd + "/Bontouch/hallway_dataset_voc/raw_segmentation"
if(not os.path.isdir(hallway_dir)):
    os.mkdir(hallway_dir)
hallway_files = glob.glob("Bontouch/hallway_dataset_voc/SegmentationClassPNG/*.png")

relabel_images(hallway_files)
