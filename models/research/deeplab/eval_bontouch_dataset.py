from __future__ import division
from PIL import Image
import numpy as np
import glob
import itertools

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
    seg_image = seg_map  # label_to_color_image(seg_map).astype(np.uint8)
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
    'background', 'wall', 'floor'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


def evaluate(predictions, groundtruth, input_size):
    iou_scores = []
    count = 0
    for prediction, groundtruth in itertools.izip(predictions, groundtruth):
        # Converts img to grayscale
        im_pred = Image.open(prediction)
        im_gt = Image.open(groundtruth)
        count += 1

        # Resize groundtruth image
        width, height = im_gt.size
        resize_ratio = input_size / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_im_gt = im_gt.resize(target_size, Image.ANTIALIAS)

        pred_mat = np.asarray(im_pred.getdata(), dtype=np.uint8).reshape((im_pred.size[1], im_pred.size[0]))
        gt_mat = np.asarray(resized_im_gt.getdata(), dtype=np.uint8).reshape(
            (resized_im_gt.size[1], resized_im_gt.size[0]))

        intersection = np.logical_and(gt_mat, pred_mat)
        union = np.logical_or(gt_mat, pred_mat)
        iou_score = np.sum(intersection) / np.sum(union)

        iou_scores.append(iou_score)

        if count % 25 == 0:
            print "Calculating IoU(", iou_score, ")", count, "of", len(predictions)
            # visualize comparison
            pred_mat[pred_mat == 1] = 128
            pred_mat[pred_mat == 2] = 235
            gt_mat[gt_mat == 1] = 128
            gt_mat[gt_mat == 2] = 235
            img_pred = Image.fromarray(pred_mat, mode='L')
            img_gt = Image.fromarray(gt_mat, mode='L')
            vis_segmentation(img_pred, img_gt)

    return sum(iou_scores) / len(iou_scores)


hallway_predictions = sorted(glob.glob("datasets/Bontouch/hallway_dataset_voc/predictions/*.png"))
hallway_groundtruth = sorted(glob.glob("datasets/Bontouch/hallway_dataset_voc/raw_segmentation/*.png"))
hallway_mIoU = evaluate(hallway_predictions, hallway_groundtruth, 257)
print "mIoU =", hallway_mIoU
