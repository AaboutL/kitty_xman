from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from utilities.data_preparation import data_server
import numpy as np
import cv2

# imageDirs = ["../data/images/lfpw/trainset/", "../data/images/helen/trainset/", "../data/images/afw/"]
# boundingBoxFiles = ["../data/boxesLFPWTrain.pkl", "../data/boxesHelenTrain.pkl", "../data/boxesAFW.pkl"]

imageDirs = ["/home/public/nfs132_0/landmark/dataset/untouch/untouch_labeled/total/data/"]
bbox_file = '/home/public/nfs132_0/landmark/dataset/untouch/untouch_labeled/total/images_bbox_list.txt'
datasetDir = "../data/"

img = np.zeros([224, 224])
meanShape = np.load("../data/meanFaceShape.npz")["meanShape"]
for pt in meanShape:
    cv2.circle(img, (int(pt[0]), int(pt[1])), 1, 255)


trainSet = data_server.DataServer(initialization='rect')
trainSet.collect_data(imageDirs, bbox_file, meanShape, 0, 100000, True) # take 0 ~ 100 as validation set
# trainSet.collect_data(imageDirs, bbox_file, meanShape, 0, 2, True)
trainSet.load_image()
trainSet.gen_perturbations(10, [0.1, 0.1, 10, 0.25])
trainSet.NormalizeImages()
# trainSet.Save(datasetDir)
print("here")
trainSet.save_tfrecord("/home/slam/workspace/DL/alignment_method/align_untouch/temp/test.record")