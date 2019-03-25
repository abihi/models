from __future__ import division
from PIL import Image
import numpy as np
import sys
import glob
import itertools
import os

from datasets import visualize_data

cwd = os.getcwd()

def evaluate(predictions, groundtruth, input_size):
    iou_scores = []
    count = 0
    for prediction, groundtruth in itertools.izip(predictions, groundtruth):
        org_file  = groundtruth.replace("raw_segmentation", "images", 1)
        org_file  = org_file.replace("png", "jpg", 1)
        org_image = Image.open(cwd+"/"+org_file)
        # Converts img to grayscale
        im_pred = Image.open(prediction)
        im_gt = Image.open(groundtruth)
        count += 1

        # Resize groundtruth image
        width, height = im_gt.size
        resize_ratio = input_size / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_im_gt = im_gt.resize(target_size, Image.ANTIALIAS)

        pred_mat_floor = np.asarray(im_pred.getdata(), dtype=np.uint8).reshape((im_pred.size[1], im_pred.size[0]))
        gt_mat_floor = np.asarray(resized_im_gt.getdata(), dtype=np.uint8).reshape(
            (resized_im_gt.size[1], resized_im_gt.size[0]))
        pred_mat_wall = np.asarray(im_pred.getdata(), dtype=np.uint8).reshape((im_pred.size[1], im_pred.size[0]))
        gt_mat_wall = np.asarray(resized_im_gt.getdata(), dtype=np.uint8).reshape(
            (resized_im_gt.size[1], resized_im_gt.size[0]))

        #Floor and wall indexes
        w_indx = 1
        f_indx = 2

        pred_mat_floor[np.logical_not(pred_mat_floor == f_indx)] = 0
        gt_mat_floor[np.logical_not(gt_mat_floor == 2)] = 0

        intersection_floor = np.logical_and(gt_mat_floor, pred_mat_floor)
        union_floor = np.logical_or(gt_mat_floor, pred_mat_floor)
        iou_score_floor = np.sum(intersection_floor) / np.sum(union_floor)

        pred_mat_wall[np.logical_not(pred_mat_wall == w_indx)] = 0
        gt_mat_wall[np.logical_not(gt_mat_wall == 1)] = 0

        intersection_wall = np.logical_and(gt_mat_wall, pred_mat_wall)
        union_wall = np.logical_or(gt_mat_wall, pred_mat_wall)
        iou_score_wall = np.sum(intersection_wall) / np.sum(union_wall)

        iou_score = (iou_score_wall + iou_score_floor) / 2

        iou_scores.append(iou_score)

        sys.stdout.write('\r>> Calculating IoU of image %d/%d' % (
            count, len(predictions)))
        sys.stdout.flush()

        # if count % 25 == 0:
        #     # Visualize comparison
        #     pred_mat_floor[pred_mat_floor == f_indx] = 128
        #     pred_mat_wall[pred_mat_wall == w_indx] = 235
        #     gt_mat_floor[gt_mat_floor == 2] = 128
        #     gt_mat_wall[gt_mat_wall == 1] = 235
        #     img_pred_wall = Image.fromarray(pred_mat_wall, mode='L')
        #     img_pred_floor = Image.fromarray(pred_mat_floor, mode='L')
        #     img_gt_wall = Image.fromarray(gt_mat_wall, mode='L')
        #     img_gt_floor = Image.fromarray(gt_mat_floor, mode='L')
        #     visualize_data.vis_segmentation(org_image, img_pred_wall, img_gt_wall, 1)
        #     visualize_data.vis_segmentation(org_image, img_pred_floor, img_gt_floor, 1)
    sys.stdout.write('\n')
    sys.stdout.flush()
    return sum(iou_scores) / len(iou_scores)

hallway_predictions = sorted(glob.glob("datasets/Bontouch/hallway_dataset_voc/predictions/*.png"))
hallway_groundtruth = sorted(glob.glob("datasets/Bontouch/hallway_dataset_voc/raw_segmentation/*.png"))
hallway_mIoU = evaluate(hallway_predictions, hallway_groundtruth, 257)
print "mIoU =", hallway_mIoU
