from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import argparse
import os

from utilities import tfrecords_util
from utilities import dataset

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'

def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.float32),
        }
    )
    image = tf.decode_raw(features['image'], tf.uint8)
    # image.set_shape((224, 224, 3))
    image = tf.reshape(image, [224, 224, 3])
    points = features['label']
    return image, points

def normalize(image, points):
    image = tf.cast(image, tf.float32) * (1./ 255) - 0.5
    return image, points

def convert_from_tfrecord(batch_size, num_epochs, input_file):
    if not num_epochs:
        num_epochs = None

    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(input_file)

        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)

        dataset = dataset.shuffle(3883) # buffer_size should be the data set size
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()
