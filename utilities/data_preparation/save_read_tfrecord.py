from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utilities.data_preparation import tfrecords_util


def save_tfrecord(image_set, points_set, output_file):
    assert len(image_set)==len(points_set)
    print('writing')
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        print("num samples: ", len(image_set))
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

def load_tfrecord(rec_file, pts_num=68, img_shape=[112,112], is_shuffle=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(rec_file)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string, default_value=""),
            'label': tf.FixedLenFeature([pts_num * 2], tf.float32)
        }
    )
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, img_shape)
    label = features['label']
    label = tf.reshape(label, [pts_num * 2])

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

def decode(serialized_example, pts_num=68, img_shape=[224,224,3]):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.float32),
        }
    )
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, img_shape)
    points = features['label']
    points = tf.reshape(points, [pts_num * 2])
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