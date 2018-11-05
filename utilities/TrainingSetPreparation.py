from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from utilities.ImageServer import ImageServer
import numpy as np

# imageDirs = ["../data/images/lfpw/trainset/", "../data/images/helen/trainset/", "../data/images/afw/"]
# boundingBoxFiles = ["../data/boxesLFPWTrain.pkl", "../data/boxesHelenTrain.pkl", "../data/boxesAFW.pkl"]

imageDirs = ["/home/public/nfs72/face/ibugs/300W/01_Indoor/",
             "/home/public/nfs72/face/ibugs/lfpw/trainset/",
             "/home/public/nfs72/face/ibugs/helen/trainset/",
             "/home/public/nfs72/face/ibugs/afw/"]
boundingBoxFiles = ["../data/py3boxes300WIndoor.pkl",
                    "../data/py3boxesLFPWTrain.pkl",
                    "../data/py3boxesHelenTrain.pkl",
                    "../data/py3boxesAFW.pkl"]
datasetDir = "/home/public/nfs72/hanfy/datasets/DAN/"

meanShape = np.load("../data/meanFaceShape.npz")["meanShape"]

trainSet = ImageServer(initialization='rect')
# trainSet.PrepareData(imageDirs, None, meanShape, 100, 100000, True)
trainSet.PrepareData(imageDirs, None, meanShape, 10, 100, True)
trainSet.LoadImages()
trainSet.GeneratePerturbations(10, [0.2, 0.2, 20, 0.25])
trainSet.NormalizeImages()
trainSet.Save(datasetDir)

validationSet = ImageServer(initialization='box')
validationSet.PrepareData(imageDirs, boundingBoxFiles, meanShape, 0, 100, False)
validationSet.LoadImages()
validationSet.CropResizeRotateAll()
validationSet.imgs = validationSet.imgs.astype(np.float32)
validationSet.NormalizeImages(trainSet)
validationSet.Save(datasetDir)