from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from utilities.data_preparation import data_server
import numpy as np

# imageDirs = ["../data/images/lfpw/trainset/", "../data/images/helen/trainset/", "../data/images/afw/"]
# boundingBoxFiles = ["../data/boxesLFPWTrain.pkl", "../data/boxesHelenTrain.pkl", "../data/boxesAFW.pkl"]

imageDirs = ["/home/public/nfs72/face/ibugs/300W/01_Indoor/",
             # "/home/public/nfs72/face/ibugs/lfpw/trainset/",
             # "/home/public/nfs72/face/ibugs/helen/trainset/",
             "/home/public/nfs72/face/ibugs/afw/"]
boundingBoxFiles = ["../data/py3boxes300WIndoor.pkl",
                    "../data/py3boxesLFPWTrain.pkl",
                    "../data/py3boxesHelenTrain.pkl",
                    "../data/py3boxesAFW.pkl"]
bbox_file = '/home/public/nfs132_0/landmark/dataset/ibugs/images_bbox_train_ibugs.txt'
datasetDir = "../data/"

meanShape = np.load("../data/meanFaceShape.npz")["meanShape"]

trainSet = data_server.DataServer(initialization='rect')
# trainSet.PrepareData(imageDirs, None, meanShape, 100, 100000, True) # take 0 ~ 100 as validation set
trainSet.collect_data(imageDirs, bbox_file, meanShape, 0, 2, True)
trainSet.load_image()
trainSet.gen_perturbations(10, [0.2, 0.2, 20, 0.25])
# trainSet.NormalizeImages()
# trainSet.Save(datasetDir)
print("here")
trainSet.save_tfrecord("/home/hanfy/workspace/DL/alignment/align_untouch/temp/test.record")