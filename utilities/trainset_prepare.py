from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from utilities.data_preparation import data_server
import numpy as np
import cv2
import gc

# meanShape = np.load("../data/meanFaceShape.npz")["meanShape"] + np.array([56, 56])
meanShape = np.genfromtxt('/home/slam/workspace/DL/alignment_method/align_untouch/meanshape_untouch.txt')
meanShape = np.reshape(meanShape, [82, 2]) * 116

imageDirs = ["/home/slam/nfs132_0/landmark/dataset/untouch/untouch_labeled/total/data/"]
bbox_file = '/home/slam/nfs132_0/landmark/dataset/untouch/untouch_labeled/total/images_bbox_list.txt'

trainSet = data_server.DataServer(initialization='bbox', imgsize=[116, 116])
trainSet.collect_data(imageDirs, bbox_file, meanShape, 0, 100000, True) # take 0 ~ 100 as validation set
# trainSet.collect_data(imageDirs, bbox_file, meanShape, 0, 2, True)
trainSet.load_image()
trainSet.gen_perturbations(10, [0.1, 0.1, 10, 0.25])
trainSet.NormalizeImages()
# trainSet.Save(datasetDir)
print("here")
trainSet.save_tfrecord("/home/slam/nfs132_0/landmark/dataset/untouch/untouch_labeled/total/train_116.record", pts_num=82)
del trainSet
gc.collect()

imageDirs_test = ["/home/slam/nfs132_0/landmark/dataset/untouch/untouch_labeled/testset/"]
bbox_file_test = '/home/slam/nfs132_0/landmark/dataset/untouch/untouch_labeled/total/images_bbox_list_testset.txt'

testSet = data_server.DataServer(initialization='bbox', imgsize=[116, 116])
testSet.collect_data(imageDirs_test, bbox_file_test, meanShape, 0, 1000, False) # take 0 ~ 100 as validation set
testSet.load_image()
testSet.gen_perturbations(10, [0.1, 0.1, 10, 0.25])
# testSet.NormalizeImages()
print("here")
testSet.save_tfrecord("/home/slam/nfs132_0/landmark/dataset/untouch/untouch_labeled/total/test_116.record", pts_num=82)
del testSet
gc.collect()
