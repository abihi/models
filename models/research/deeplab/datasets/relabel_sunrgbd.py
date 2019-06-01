from PIL import Image
import numpy as np
import os
import glob
import sys
import visualize_data

cwd = os.getcwd()

def relabel_images(files, type):
    count = 0
    for filename in files:
        org_file  = filename.replace("annotations", "images", 1)
        org_file  = org_file.replace("png", "jpg", 1)
        org_image = Image.open(cwd+"/"+org_file)

        im = Image.open(filename)
        count += 1

        im_mat=np.asarray(im.getdata(),dtype=np.uint8).reshape((im.size[1],im.size[0]))
        im_vis = np.asarray(im.getdata(),dtype=np.uint8).reshape((im.size[1],im.size[0])) #visualize_data.label_to_color_image(im_mat)

        mask = np.logical_and(im_mat!=0, im_mat!=5)
        im_mat[mask] = 0
        #im_mat[im_mat==12] = 1
        im_mat[im_mat==5] = 1

        im_mat=np.asarray(im_mat,dtype=np.uint8)
        img=Image.fromarray(im_mat,mode='L')

        #visualize_data.vis_segmentation(org_image, im_vis, im_mat, 1)

        im.close()
        org_image.close()

        sys.stdout.write('\r>> Relabeling %s image %d/%d' % (type, count, len(files)))
        sys.stdout.flush()

        filename = filename.replace(type, type+"_relabel", 1)
        img.save(filename)
        img.close()
    sys.stdout.write('\n')
    sys.stdout.flush()

# Values after converting rgb segmap to grayscale
# 2 (wall) <- 12
# 1 (floor) <- 5

sun_train_dir = cwd + "/SUN_RGBD/annotations/train_relabel"
sun_test_dir = cwd + "/SUN_RGBD/annotations/test_relabel"
sun_trainval_dir = cwd + "/SUN_RGBD/annotations/trainval_relabel"
if(not os.path.isdir(sun_train_dir)):
    os.mkdir(sun_train_dir)
if(not os.path.isdir(sun_test_dir)):
    os.mkdir(sun_test_dir)
if(not os.path.isdir(sun_trainval_dir)):
    os.mkdir(sun_trainval_dir)
sun_train_files = glob.glob("SUN_RGBD/annotations/train/*.png")
sun_test_files  = glob.glob("SUN_RGBD/annotations/test/*.png")
sun_trainval_files  = glob.glob("SUN_RGBD/annotations/trainval/*.png")

relabel_images(sun_train_files, "train")
relabel_images(sun_test_files, "test")
relabel_images(sun_trainval_files, "trainval")
