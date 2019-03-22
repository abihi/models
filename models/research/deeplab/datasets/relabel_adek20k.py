from PIL import Image
from os import listdir
from os.path import isfile, join
import glob
import numpy as np
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

# 1 (wall) <- 9(window), 15(door), 23(painting), 33(fence), 43(pillar), 44(sign board), 145(bulletin board)
wall = np.array([1, 9, 15, 23, 33, 43, 44, 145])
# 2 (floor) <- 7(road), 14(ground), 30(field), 53(path), 55(runway), 29(rug, carpet, carpeting)
floor = np.array([4, 7, 14, 30, 53, 55, 29])
#Rest of classes labeled as 0 (background)

trainingfiles = glob.glob("ADE20K/ADEChallengeData2016/annotations/training/*.png")
validationfiles = glob.glob("ADE20K/ADEChallengeData2016/annotations/validation/*.png")

def relabel_images(files):
    count = 0
    for filename in files:
        im = Image.open(filename)

        im_mat=np.asarray(im.getdata(),dtype=np.uint8).reshape((im.size[1],im.size[0]))

        #im_vis = label_to_color_image(im_mat)

        mask_floor = reduce( lambda m, n: m|n ,[ im_mat == i for i in floor])
        im_mat[mask_floor] = 2
        mask_wall = reduce( lambda m, n: m|n ,[ im_mat == i for i in wall])
        im_mat[mask_wall] = 1
        im_mat[im_mat > 2] = 0

        im_mat=np.asarray(im_mat,dtype=np.uint8)
        img=Image.fromarray(im_mat,mode='L')

        #vis_segmentation(im_vis, im_mat)

        im.close()
        filename = filename.replace("ADE20K", "ADE20K_relabeled", 1)
        count += 1
        if count % 250 == 0:
            print "Ade20k relabeling iteration: ", count
        img.save(filename)
        img.close()

relabel_images(trainingfiles)
relabel_images(validationfiles)
