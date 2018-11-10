from utilities import dataset
from utilities import preprocess
import tensorflow as tf
import numpy as np
import cv2
import os

from utilities.tfrecord import read_tfrecord
from utilities import visualize
from evaluate import landmark_eval

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# tmp_dir = '/home/public/nfs132_1/hanfy/align/ibugs/tmp'
# tmp_dir = '/home/public/nfs132_1/hanfy/align_data/ibugs/common'
tmp_dir = '/home/public/nfs132_1/hanfy/align_data/ibugs/challenge'
# root_dir = '/home/public/nfs72/face/ibugs'
# root_dir = '/home/public/nfs132_1/hanfy/align/ibugs/testset'
train_items = []
validate_items = []
dset = dataset.Dataset()
dset.get_datalist(tmp_dir, ['png', 'jpg'])
# dset.save_hdf5('/home/public/nfs132_1/hanfy/align_data/ibugs/common.hdf5', True, False)
dset.save_hdf5('/home/public/nfs132_1/hanfy/align_data/ibugs/challenge.hdf5', True, False)

# tmp_dir = '/home/public/nfs132_1/hanfy/align/ibugs/tmp'
# dset = dataset.Dataset()
# dset.get_datalist(tmp_dir, ['png', 'jpg'])
# total_image, a, points_set = dset.gether_data()
# mean, std = dset.normalize_pts(points_set)

# output_tfrecords = '/home/public/nfs132_1/hanfy/align/ibugs/validationset_bbox.record'
# filename_queue = tf.train.string_input_producer([output_tfrecords], num_epochs=2)
# images, labels = read_tfrecord.read_and_decode(filename_queue, is_shuffle=True)
# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# with tf.Session() as sess:
#     sess.run(init_op)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     for i in range(1000):
#         imgs, labs = sess.run([images, labels])
#         for j in range(30):
#             img = imgs[j].copy()
#             print(np.shape(img))
#             points = labs[j]
#             print(points)
#             visualize.show_points(img, points, dim=1)
#             pts1 = np.reshape(points, [68,2])
#             pts2 = np.reshape(points, [68,2])+2
#             print('pts1', pts1)
#             print('pts2', pts2)
#             error = np.mean(np.sqrt(np.sum((pts1 - pts2)**2, axis=1)))
#             print(error)
#             cv2.imshow('img', img)
#             cv2.waitKey(0)
#     coord.request_stop()
#     coord.join(threads=threads)





