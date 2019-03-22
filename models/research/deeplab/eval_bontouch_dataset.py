from PIL import Image
import numpy as np
import glob
import itertools

# Values after converting rgb segmap to grayscale
# 1 (wall) <- 75
# 2 (floor) <- 38

hallway_predictions = sorted(glob.glob("datasets/Bontouch/hallway_dataset_voc/predictions/*.png"))
hallway_groundtruth = sorted(glob.glob("datasets/Bontouch/hallway_dataset_voc/raw_segmentation/*.png"))
hallway_iou_scores = [None] * len(hallway_predictions)

count = 0
for prediction, groundtruth in itertools.izip(hallway_predictions, hallway_groundtruth):
    #Converts img to grayscale
    im_pred = Image.open(prediction)
    im_gt = Image.open(groundtruth)
    count += 1

    pred_mat = np.asarray(im_pred.getdata(),dtype=np.uint8).reshape((im_pred.size[1],im_pred.size[0]))
    gt_mat = np.asarray(im_gt.getdata(),dtype=np.uint8).reshape((im_gt.size[1],im_gt.size[0]))

    intersection = np.logical_and(gt_mat, pred_mat)
    union = np.logical_or(gt_mat, pred_mat)
    iou_score = np.sum(intersection) / np.sum(union)

    print iou_score

    hallway_iou_scores.append(iou_score)

    if count % 5 == 0:
        print "Calculating IoU ", count, "of 305"

hallway_mIoU = sum(hallway_iou_scores) / len(hallway_iou_scores)
print "Hallway mIoU=", hallway_mIoU
