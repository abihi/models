from PIL import Image
import numpy as np
import os
import glob
import visualize_data

def relabel_images(files):
    count = 0
    for filename in files:
        #Converts img to grayscale
        im = Image.open(filename).convert('L')
        count += 1

        im_mat=np.asarray(im.getdata(),dtype=np.uint8).reshape((im.size[1],im.size[0]))
        im_vis = visualize_data.label_to_color_image(im_mat)

        im_mat[im_mat==38] = 2
        im_mat[im_mat==75] = 1
        im_mat[im_mat>2] = 0

        im_mat=np.asarray(im_mat,dtype=np.uint8)
        img=Image.fromarray(im_mat,mode='L')

        visualize_data.vis_segmentation(im_vis, im_mat, 1)

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
