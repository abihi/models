from __future__ import division
from PIL import Image
import numpy as np
import sys
import glob
import itertools
import os
import matplotlib as mpl

from datasets import visualize_data
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('path', 'datasets/Bontouch/hallway_dataset', 'Dataset folder')
cwd = os.getcwd()

def evaluate(predictions, groundtruth, input_size, vis):
    iou_scores_wall = []
    iou_scores_floor = []
    iou_scores = []
    count = 0
    for prediction, groundtruth in itertools.izip(predictions, groundtruth):
        org_file  = groundtruth.replace("raw_segmentation", "images", 1)
        org_file  = org_file.replace("png", "jpg", 1)
        org_image = Image.open(cwd+"/"+org_file)

        im_pred = Image.open(prediction)
        im_gt = Image.open(groundtruth)
        count += 1

        # Resize groundtruth image, change target_size to (input_size, input_size) for tflite model
        width, height = im_gt.size
        resize_ratio = 1.0 * 1920 / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height)) #(input_size, input_size)
        im_pred = im_pred.resize(target_size, Image.ANTIALIAS)
        resized_im_gt = im_gt.resize(target_size, Image.ANTIALIAS)
        org_image = org_image.resize(target_size, Image.ANTIALIAS)

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

        iou_scores_wall.append(iou_score_wall)
        iou_scores_floor.append(iou_score_floor)
        iou_scores.append(iou_score)

        sys.stdout.write('\r>> Calculating IoU of image %d/%d' % (
            count, len(predictions)))
        sys.stdout.flush()

        #Visualize IoU evaluation
        # if count % 25 == 0:
        cm_hot = mpl.cm.get_cmap('inferno')
        pred_mat_floor[pred_mat_floor == f_indx] = 235
        pred_mat_wall[pred_mat_wall == w_indx] = 128
        gt_mat_floor[gt_mat_floor == 2] = 235
        gt_mat_wall[gt_mat_wall == 1] = 128
        img_pred_wall = Image.fromarray(pred_mat_wall, mode='P')
        img_pred_floor = Image.fromarray(pred_mat_floor, mode='P')
        img_gt_wall = Image.fromarray(gt_mat_wall, mode='P')
        img_gt_floor = Image.fromarray(gt_mat_floor, mode='P')
        img_gt_wall = cm_hot(np.array(img_gt_wall))
        img_pred_wall = cm_hot(np.array(img_pred_wall))
        img_pred_floor = cm_hot(np.array(img_pred_floor))
        img_gt_wall = np.uint8(img_gt_wall * 255)
        img_pred_wall = np.uint8(img_pred_wall * 255)
        img_pred_floor = np.uint8(img_pred_floor * 255)
        img_pred_floor = Image.fromarray(img_pred_floor)
        img_gt_wall = Image.fromarray(img_gt_wall)
        img_pred_wall = Image.fromarray(img_pred_wall)

        width, height = org_image.size
        resize_ratio_vis = 1.0 * 1920 / max(width, height)
        target_size_vis = (int(resize_ratio_vis * width), int(resize_ratio_vis * height))
        img_pred_floor = img_pred_floor.convert('RGB')
        img_vis = Image.blend(org_image, img_pred_floor, 0.5).resize(target_size_vis, Image.ANTIALIAS)

        # visualize_data.vis_segmentation(img_vis, org_image, img_pred_floor, 1)
        # visualize_data.vis_segmentation(img_gt_wall, org_image, img_pred_wall, 1)
        # visualize_data.vis_segmentation(img_pred_wall, org_image, img_gt_wall, 1)

        filename_vis = cwd+"/"+org_file
        filename_vis = filename_vis.replace("images", "vis", 1)
        img_vis.save(filename_vis)
        img_vis.close()


    sys.stdout.write('\n')
    sys.stdout.flush()
    mIoU_wall = sum(iou_scores_wall) / len(iou_scores_wall)
    mIoU_floor= sum(iou_scores_floor) / len(iou_scores_floor)
    return mIoU_wall, mIoU_floor

predictions = sorted(glob.glob(FLAGS.path + "/predictions/*.png"))
groundtruth = sorted(glob.glob(FLAGS.path + "/raw_segmentation/*.png"))
mIoU_wall, mIoU_floor = evaluate(predictions, groundtruth, 257, False)

print "mIoU(wall)=", mIoU_wall
print "mIoU(floor)=", mIoU_floor
