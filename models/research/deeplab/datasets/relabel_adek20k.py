from PIL import Image
from os import listdir
from os.path import isfile, join
import glob
import sys
import os
import numpy as np
import visualize_data

# 1 (wall) <- 9(window), 15(door), 23(painting), 33(fence), 43(pillar), 44(sign board), 145(bulletin board)
wall = np.array([1, 9, 15, 23, 33, 43, 44, 145])
# 2 (floor) <- 7(road), 14(ground), 30(field), 53(path), 55(runway), 29(rug, carpet, carpeting)
floor = np.array([4, 7, 14, 30, 53, 55, 29])
#Rest of classes labeled as 0 (background)
cwd = os.getcwd()

def relabel_images(files):
    count = 0
    for filename in files:
        org_file  = filename.replace("annotations", "images", 1)
        org_file  = org_file.replace("png", "jpg", 1)
        print cwd+"/"+org_file
        org_image = Image.open(cwd+"/"+org_file)
        im = Image.open(filename)

        im_mat=np.asarray(im.getdata(),dtype=np.uint8).reshape((im.size[1],im.size[0]))

        im_vis = visualize_data.label_to_color_image(im_mat)

        mask_floor = reduce( lambda m, n: m|n ,[ im_mat == i for i in floor])
        im_mat[mask_floor] = 2
        mask_wall = reduce( lambda m, n: m|n ,[ im_mat == i for i in wall])
        im_mat[mask_wall] = 1
        im_mat[im_mat > 2] = 0

        im_mat=np.asarray(im_mat,dtype=np.uint8)
        img=Image.fromarray(im_mat,mode='L')

        visualize_data.vis_segmentation(org_image, im_vis, im_mat, 1)

        im.close()
        filename = filename.replace("ADE20K", "ADE20K_relabeled", 1)
        count += 1

        sys.stdout.write('\r>> Relabeling image %d/%d' % (count, len(files)))
        sys.stdout.flush()

        img.save(filename)
        img.close()
    sys.stdout.write('\n')
    sys.stdout.flush()

trainingfiles = glob.glob("ADE20K/ADEChallengeData2016/annotations/training/*.png")
validationfiles = glob.glob("ADE20K/ADEChallengeData2016/annotations/validation/*.png")
relabel_images(trainingfiles)
relabel_images(validationfiles)
