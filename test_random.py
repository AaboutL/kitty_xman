from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
import numpy as np
import argparse
import time

from utilities import dataset
from net_archs import AlexNet
from evaluate import landmark_eval
from train import loss_func


# input_file = '/home/public/nfs132_1/hanfy/align/ibugs/trainset.hdf5'
# dset = dataset.Dataset()
# image_set, shape_set= dset.read_hdf5(input_file)
# RandomIdx = np.random.choice(image_set.shape[0], 64, False)
test = open('/home/public/nfs132_1/hanfy/results/alex_l1_bbox_flip/mid_result.txt', 'w+')
for i in range(10000000):
    # start = time.time()
    # imgs = image_set[RandomIdx]
    # shapes = shape_set[RandomIdx]
    # duration = time.time() - start
    # print('time ', duration)
    print(i)
    test.write(str(i)+'\n')
test.close()

