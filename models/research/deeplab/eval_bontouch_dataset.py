from __future__ import division
from PIL import Image
import numpy as np
import sys
import glob
import itertools

from datasets import visualize_data

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

        sys.stdout.write('\r>> Calculating IoU of image %d/%d' % (
            count, len(predictions)))
        sys.stdout.flush()

        #if count % 25 == 0:
            # visualize comparison
            # pred_mat[pred_mat == 1] = 128
            # pred_mat[pred_mat == 2] = 235
            # gt_mat[gt_mat == 1] = 128
            # gt_mat[gt_mat == 2] = 235
            # img_pred = Image.fromarray(pred_mat, mode='L')
            # img_gt = Image.fromarray(gt_mat, mode='L')
            # visualize_data.vis_segmentation(img_pred, img_gt, 1)
    sys.stdout.write('\n')
    sys.stdout.flush()
    return sum(iou_scores) / len(iou_scores)

hallway_predictions = sorted(glob.glob("datasets/Bontouch/hallway_dataset_voc/predictions/*.png"))
hallway_groundtruth = sorted(glob.glob("datasets/Bontouch/hallway_dataset_voc/raw_segmentation/*.png"))
hallway_mIoU = evaluate(hallway_predictions, hallway_groundtruth, 257)
print "mIoU =", hallway_mIoU
