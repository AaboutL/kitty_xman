# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
import cv2
import time
import numpy as np
import random

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def resize_image(image, img_w, img_h):
    return cv2.resize(image, (img_w, img_h))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def generate_file_list(data_dir):
    sub_dirs = os.listdir(data_dir)
    img_paths = []
    labels = []
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(data_dir, sub_dir)
        items = os.listdir(sub_dir_path)
        num_imgs = len(items)
        labels.extend([int(sub_dir)]*num_imgs)
        for item in items:
            img_path = os.path.join(sub_dir_path, item)
            img_paths.append(img_path)
    return img_paths, labels

def generate_tfrecord(data_dir, output_file, img_w, img_h):
    print('generating %s' %output_file)
    img_paths, labels = generate_file_list(data_dir)
    indices = np.arange(0, len(img_paths))
    random.shuffle(indices)
    print('indices:', indices)
    print('img num: ', len(img_paths))
    print('label num: ', len(labels))
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for i in indices:
            print('path:', img_paths[i])
            # image = tf.gfile.FastGFile(img_paths[i], 'rb').read()
            # image = tf.image.decode_png(image)
            # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # image = tf.py_func(resize_image, [image, img_w, img_h], [tf.float32])
            # image = tf.image.rgb_to_grayscale(image)
            # start = time.time()
            # image = sess.run(image)
            # print('time:', time.time() - start)
            # image_raw = image.tostring()
            image = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (img_w, img_h))
            image_raw = image.tostring()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': _bytes_feature(image_raw),
                    'label': _int64_feature(labels[i])
                }
            ))
            record_writer.write(example.SerializeToString())

def read_tfrecord(record_file):
    reader = tf.TFRecordReader()
    _, example = reader.read(record_file)
    features = tf.parse_single_example(example, features={
        'image':tf.FixedLenFeature([], tf.string)
    })


if __name__ == "__main__":
    generate_tfrecord('/home/public/nfs72/hanfy/datasets/testdata', "data/hand_test.record", 60, 60)

