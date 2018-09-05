from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
import cv2

from utilities import tfrecords_util
from utilities import dataset
from utilities import preprocess
from utilities import visualize
from utilities import model_tool
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
# train_dir = '/home/public/nfs132_1/hanfy/align/ibugs/tmp'
train_dir = '/home/public/nfs72/face/ibugs'
trainset_tfrecord = '/home/public/nfs132_1/hanfy/align/ibugs/trainset_bbox_flip.record'
trainset_tfrecord_norm = '/home/public/nfs132_1/hanfy/align/ibugs/trainset_bbox_flip_norm.record'
dset = dataset.Dataset()
dset.get_datalist(train_dir, ['png', 'jpg'])
image_set, _, shapes = dset.gether_data()
mean, std, normed_shapes = dset.normalize_pts(shapes, 1.0/224.0)
np.savetxt('/home/hanfy/workspace/DL/alignment/align_untouch/shape_mean.txt', mean, header='mean')
np.savetxt('/home/hanfy/workspace/DL/alignment/align_untouch/shape_std.txt', mean, header='std')
# for i in range(len(image_set)):
#     img = image_set[i].copy()
#     res = shapes[i].copy()
    # res = np.multiply(np.add(np.multiply(normed_shapes[i], std), mean) + 0.5, 224.0)
    # print('norm', normed_shapes[i])
    # print('re', res)
    # print('ori', shapes[i])
    # visualize.show_points(img, res)
    # visualize.show_points(img, shapes[i], color=(0, 0, 255))
    # visualize.show_image(img, 'img', 0)
# exit(0)
normed_shapes_flatten = [sum(pts.tolist(), []) for pts in normed_shapes]
convert_to_tfrecord(image_set, normed_shapes_flatten , trainset_tfrecord_norm)
normed_shapes_flatten = [sum(pts.tolist(), []) for pts in shapes]
convert_to_tfrecord(image_set, normed_shapes_flatten , trainset_tfrecord)

# generate testset
test_dir = '/home/public/nfs132_1/hanfy/align/ibugs/testset'
validationset_tfrecord = '/home/public/nfs132_1/hanfy/align/ibugs/validationset_bbox.record'
# test_dir = '/home/public/nfs132_1/hanfy/align/ibugs/tmp'
# validationset_tfrecord = '/home/public/nfs132_1/hanfy/align/ibugs/tmpset_bbox.record'
dset = dataset.Dataset()
dset.get_datalist(test_dir, ['png', 'jpg'])
image_set, _, points_set = dset.gether_data(is_flip=False)
points_set_norm_flatten = [sum(pts.tolist(), []) for pts in points_set]
convert_to_tfrecord(image_set, points_set_norm_flatten, validationset_tfrecord)
