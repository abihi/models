from PIL import Image
import numpy as np
import os
import glob

# Values after converting rgb segmap to grayscale
# 1 (wall) <- 75
# 2 (floor) <- 38

cwd = os.getcwd()

hallway_dir = cwd + "/Bontouch/hallway_dataset_voc/raw_segmentation"
if(not os.path.isdir(hallway_dir)):
    os.mkdir(hallway_dir)
hallway_files = glob.glob("Bontouch/hallway_dataset_voc/SegmentationClassPNG/*.png")

count = 0
for filename in hallway_files:
    #Converts img to grayscale
    im = Image.open(filename).convert('L')
    count += 1

    im_mat=np.asarray(im.getdata(),dtype=np.uint8).reshape((im.size[1],im.size[0]))

    for x in range(im.size[1]):
        for y in range(im.size[0]):
            if im_mat[x, y] == 75:
                im_mat[x,y] = 1
            elif im_mat[x,y] == 38:
                im_mat[x,y] = 2
            else:
                im_mat[x,y] = 0

    im_mat=np.asarray(im_mat,dtype=np.uint8) #if values still in range 0-255!
    img=Image.fromarray(im_mat,mode='L')
    im.close()

    if count % 5 == 0:
        print "Converting file ", count, "of 305"

    filename = filename.replace("SegmentationClassPNG", "raw_segmentation", 1)
    img.save(filename)
    img.close()
