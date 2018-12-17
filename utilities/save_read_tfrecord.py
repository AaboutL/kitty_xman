from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
import cv2

from utilities import tfrecords_util

def save_tfrecord(image_set, points_set, output_file):
    assert len(image_set)==len(points_set)
    print('writing')
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        print("num samples: ", len(image_set))
        for i in range(len(image_set)):
            image = image_set[i]
            image_raw = image.tostring()
            pts = points_set[i]
            print("img type: ", image.dtype)
            print("img shape: ", image.shape)
            print("pts shp: ", pts.dtype)
            for j in range(len(pts)//2):
                cv2.circle(image, (int(pts[j*2]), int(pts[j*2+1])), 2, 255)
            cv2.imshow("img", image)
            cv2.waitKey(1)
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    # 'image': tfrecords_util.bytes_feature(tf.compat.as_bytes(image_raw)),
                    'image': tfrecords_util.bytes_feature(image_raw),
                    'label': tfrecords_util.float_list_feature(pts)
                }
            ))
            record_writer.write(example.SerializeToString())

def read_tfrecord1(rec_file, is_shuffle=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(rec_file)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string, default_value=""),
            'label': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }
    )
    image = tf.decode_raw(features['image'], tf.uint8)
    # image = tf.decode_raw(features['image'], tf.float32)
    image = tf.reshape(image, [112, 112, 1])
    label = features['label']
    label = tf.reshape(label, [136])

    if is_shuffle:
        print("test")
        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=2,
                                                capacity=40,
                                                num_threads=1,
                                                min_after_dequeue=5)
    else:
        images, labels = tf.train.batch([image, label],
                                        batch_size=4,
                                        capacity=20,
                                        num_threads=4
                                        )

    return images, labels

def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            # 'label': tf.FixedLenFeature([], tf.float32),
            'label': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        }
    )
    image = tf.decode_raw(features['image'], tf.uint8)
    # image.set_shape((224, 224, 3))
    image = tf.reshape(image, [224, 224, 3])
    points = features['label']
    return image, points

def convert_from_tfrecord(input_file, batch_size=0, num_epochs=1, is_shuffle=True):
    if not num_epochs:
        num_epochs = None

    num_samples = sum(1 for _ in tf.python_io.tf_record_iterator(input_file))
    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(input_file)

        dataset = dataset.map(decode)

        if is_shuffle is True:
            dataset = dataset.shuffle(num_samples) # buffer_size should be the data set size
        if num_epochs>1:
            dataset = dataset.repeat(num_epochs)
        if batch_size>0:
            dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()