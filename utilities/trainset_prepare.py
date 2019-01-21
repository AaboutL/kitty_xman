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

if False:
    print('prepare trainset')
    imageDirs_train = ["/home/slam/nfs132_0/landmark/dataset/untouch/total/"]
    bbox_file_train = '/home/slam/nfs132_0/landmark/dataset/untouch/untouch_labeled/total/images_bbox_list.txt'
    trainSet = data_server.DataServer(initialization='rect', imgsize=[116, 116])
    trainSet.collect_data(imageDirs_train, None, meanShape, 0, 100000, False) # take 0 ~ 100 as validation set
    # trainSet.collect_data(imageDirs, None, meanShape, 0, 2, False)
    trainSet.load_image()
    trainSet.gen_perturbations(5, [0.1, 0.1, 10, 0.25])
    trainSet.NormalizeImages()
    # trainSet.Save(datasetDir)
    trainSet.save_tfrecord("/home/slam/nfs132_0/landmark/dataset/untouch/train_116.record", pts_num=82)
    del trainSet
    gc.collect()
    # while True:
    #     print('test')

if True:
    print('prepare testset')
    imageDirs_test = ["/home/slam/nfs132_0/landmark/dataset/untouch/untouch_labeled/testset/"]
    bbox_file_test = '/home/slam/nfs132_0/landmark/dataset/untouch/untouch_labeled/total/images_bbox_list_testset.txt'
    testSet = data_server.DataServer(initialization='rect', imgsize=[116, 116])
    # testSet.collect_data(imageDirs_test, bbox_file_test, meanShape, 0, 1000, False) # take 0 ~ 100 as validation set
    testSet.collect_data(imageDirs_test, None, meanShape, 0, 1000, False) # take 0 ~ 100 as validation set
    testSet.load_image()
    testSet.CropResizeRotateAll()

    # testSet.NormalizeImages()
    testSet.save_tfrecord("/home/slam/nfs132_0/landmark/dataset/untouch/test_116.record", pts_num=82)
    del testSet
    gc.collect()
