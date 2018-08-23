from utilities import dataset
from utilities import preprocess
import tensorflow as tf
import numpy as np
import cv2
import os

from utilities.tfrecord import read_tfrecord
from utilities import visualize

os.environ['CUDA_VISIBLE_DEVICES'] = ''

root_dir = '/home/public/nfs72/face/ibugs'
train_items = []
validate_items = []
trainset_tfrecords = '/home/public/nfs132_1/hanfy/align/ibugs/trainset.record'
validationset_tfrecords = '/home/public/nfs132_1/hanfy/align/ibugs/validationset.record'

dset = dataset.Dataset()
dset.get_datalist(root_dir, ['png', 'jpg'])
dset.gether_data(True)
# dset.save(trainset_tfrecords, format='tfrecords')
# dset.save(validationset_tfrecords, format='tfrecords')
print('finished!')


# filename_queue = tf.train.string_input_producer([output_tfrecords], num_epochs=3)
# images, labels = read_tfrecord.read_and_decode(filename_queue)
# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# with tf.Session() as sess:
#     sess.run(init_op)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     for i in range(1000):
#         imgs, labs = sess.run([images, labels])
#         for j in range(4):
#             img = imgs[j]
#             points = labs[j]
#             visualize.show_points(img, points, dim=1)
#             cv2.imshow('img', img)
#             cv2.waitKey(0)
#     coord.request_stop()
#     coord.join(threads=threads)

