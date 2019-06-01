from PIL import Image
import numpy as np
import os
import glob
import sys
import visualize_data

cwd = os.getcwd()

def relabel_images(files):
    count = 0
    for filename in files:
        org_file  = filename.replace("SegmentationClassPNG", "images", 1)
        org_file  = org_file.replace("png", "jpg", 1)
        org_image = Image.open(cwd+"/"+org_file)
        #Converts img to grayscale
        im = Image.open(filename).convert('L')
        count += 1

        im_mat=np.asarray(im.getdata(),dtype=np.uint8).reshape((im.size[1],im.size[0]))
        im_vis = visualize_data.label_to_color_image(im_mat)

        im_mat[im_mat==38] = 1
        im_mat[im_mat==75] = 2
        im_mat[im_mat>2] = 0

        im_mat=np.asarray(im_mat,dtype=np.uint8)
        img=Image.fromarray(im_mat,mode='L')

        #visualize_data.vis_segmentation(im_vis, im_mat, im, 1)

        im.close()

        sys.stdout.write('\r>> Relabeling image %d/%d' % (count, len(files)))
        sys.stdout.flush()

        filename = filename.replace("SegmentationClassPNG", "raw_segmentation", 1)
        img.save(filename)
        img.close()
    sys.stdout.write('\n')
    sys.stdout.flush()

# Values after converting rgb segmap to grayscale
# 1 (floor) <- 38
# 2 (wall) <- 75

hallway_dir = cwd + "/Bontouch/hallway_dataset/raw_segmentation"
if(not os.path.isdir(hallway_dir)):
    os.mkdir(hallway_dir)
hallway_files = glob.glob("Bontouch/hallway_dataset/SegmentationClassPNG/*.png")

aula_dir = cwd + "/Bontouch/aula_dataset/raw_segmentation"
if(not os.path.isdir(aula_dir)):
    os.mkdir(aula_dir)
aula_files = glob.glob("Bontouch/aula_dataset/SegmentationClassPNG/*.png")

diningroom_dir = cwd + "/Bontouch/diningroom_dataset/raw_segmentation"
if(not os.path.isdir(diningroom_dir)):
    os.mkdir(diningroom_dir)
diningroom_files = glob.glob("Bontouch/diningroom_dataset/SegmentationClassPNG/*.png")

livingroom_dir = cwd + "/Bontouch/livingroom_dataset/raw_segmentation"
if(not os.path.isdir(livingroom_dir)):
    os.mkdir(livingroom_dir)
livingroom_files = glob.glob("Bontouch/livingroom_dataset/SegmentationClassPNG/*.png")

kitchen_dir = cwd + "/Bontouch/kitchen_dataset/raw_segmentation"
if(not os.path.isdir(kitchen_dir)):
    os.mkdir(kitchen_dir)
kitchen_files = glob.glob("Bontouch/kitchen_dataset/SegmentationClassPNG/*.png")

print "relabeling hallway segment"
relabel_images(hallway_files)
print "relabeling aula segment"
relabel_images(aula_files)
print "relabeling dining room segment"
relabel_images(diningroom_files)
print "relabeling living room segment"
relabel_images(livingroom_files)
print "relabeling kitchen segment"
relabel_images(kitchen_files)
