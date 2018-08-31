from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
import cv2

from utilities import tfrecords_util
from utilities import dataset
from utilities import preprocess
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def convert_to_tfrecord(image_set, points_set, output_file):
    assert len(image_set)==len(points_set)
    print('writing')
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for i in range(len(image_set)):
            image = image_set[i]
            image_raw = image.tostring()
            pts = points_set[i]
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': tfrecords_util.bytes_feature(image_raw),
                    'label': tfrecords_util.float_list_feature(pts)
                }
            ))
            record_writer.write(example.SerializeToString())

# generate trainset
# tmp_dir = '/home/public/nfs132_1/hanfy/align/ibugs/tmp'
train_dir = '/home/public/nfs72/face/ibugs'
trainset_tfrecord = '/home/public/nfs132_1/hanfy/align/ibugs/trainset_bbox_flip_norm.record'
dset = dataset.Dataset()
dset.get_datalist(train_dir, ['png', 'jpg'])
image_set, _, points_set = dset.gether_data()
mean, std = dset.normalize_pts(points_set)
print('mean:', mean)
size_factor = 1.0 / 224.0
# points_set_norm = np.divide(np.subtract(points_set, mean), std) * size_factor
points_set_norm = np.divide(np.subtract(np.multiply(points_set , size_factor), mean), std)
points_set_norm_flatten = [sum(pts.tolist(), []) for pts in points_set_norm]
convert_to_tfrecord(image_set, points_set_norm_flatten , trainset_tfrecord)

# generate testset
# root_dir = '/home/public/nfs132_1/hanfy/align/ibugs/testset'
# validationset_tfrecord = '/home/public/nfs132_1/hanfy/align/ibugs/validationset_bbox_norm.record'
# dset = dataset.Dataset()
# dset.get_datalist(train_dir, ['png', 'jpg'])
# image_set, _, points_set = dset.gether_data(is_flip=False)
# # mean, std = dset.normalize_pts(points_set)
# points_set_norm = np.divide(np.subtract(np.multiply(points_set , size_factor), mean), std)
# points_set_norm_flatten = [sum(pts.tolist(), []) for pts in points_set_norm]
# convert_to_tfrecord(image_set, points_set_norm_flatten, validationset_tfrecord)
