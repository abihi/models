from PIL import Image
from os import listdir
from os.path import isfile, join
import glob

# 1 (wall) <- 9(window), 15(door), 23(painting), 33(fence), 43(pillar), 44(sign board), 145(bulletin board)
wall = [1, 9, 15, 23, 33, 43, 44, 145]
# 2 (floor) <- 7(road), 14(ground), 30(field), 53(path), 55(runway), 29(rug, carpet, carpeting)
floor = [4, 7, 14, 30, 53, 55, 29]
#Rest of classes labeled as 0 (background)

trainingfiles = glob.glob("/home/abihi/annotations_relabeled/training/*.png")
validationfiles = glob.glob("/home/abihi/annotations_relabeled/validation/*.png")
count = 0
for filename in trainingfiles:
    im = Image.open(filename)
    pixelMap = im.load()

    img = Image.new( im.mode, im.size)
    pixelsNew = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if pixelMap[i,j] in wall:
                pixelsNew[i,j] = 1
            elif pixelMap[i,j] in floor:
                pixelsNew[i,j] = 2
            else:
                pixelsNew[i,j] = 0
    im.close()
    filename = filename.replace("training", "training_relabel", 1)
    count += 1
    if count % 500 == 0:
        print "Training set iteration: ", count
    img.save(filename)
    img.close()

count = 0
for filename in validationfiles:
    im = Image.open(filename)
    pixelMap = im.load()

    img = Image.new( im.mode, im.size)
    pixelsNew = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if pixelMap[i,j] in wall:
                pixelsNew[i,j] = 1
            elif pixelMap[i,j] in floor:
                pixelsNew[i,j] = 2
            else:
                pixelsNew[i,j] = 0
    im.close()
    filename = filename.replace("validation", "validation_relabel", 1)
    count += 1
    if count % 500 == 0:
        print "Validation set iteration: ", count
    img.save(filename)
    img.close()
