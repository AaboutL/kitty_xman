from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
import os
import tensorflow as tf
import cv2
from utilities import tfrecords_util

def read_text(dataset_dir, dataset_txt):
    img_paths = []
    labels = []
    with open(dataset_txt, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            items = line.strip('\n').split(' ')
            img_path = os.path.join(dataset_dir, items[0][16:])
            label = items[1]
            img_paths.append(img_path)
            labels.append(label)
        return img_paths, labels

img_paths, labels = read_text('/home/public/nfs70/hao/cifar10/train', '/home/public/nfs70/hao/cifar10/train/train.txt')

def save_tfrecord(image_paths, labels, output_file):
    print('writing tfrecord...')
    with tf.python_io.TFRecordWriter(output_file) as writer:
        print(len(img_paths))
        for i in range(len(image_paths)):
            img = cv2.imread(image_paths[i])
            img_raw = img.tostring()
            label = labels[i]
            example = tf.train.Example(features=tf.train.Features(
                feature = {
                    'image' : tfrecords_util.bytes_feature(img_raw),
                    'label' : tfrecords_util.int64_feature(int(label)),
                }
            ))
            writer.write(example.SerializeToString())
    print('finished!')

def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        }
    )
    image = tf.decode_raw(features['image'], tf.uint8)
    # image.set_shape((224, 224, 3))
    image = tf.reshape(image, [32, 32, 3])
    label = features['label']
    return image, label

def normalize(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label

def read_tfrecord(input_file, batch_size=0, num_epochs=1, is_preprocess=True, is_shuffle=True):
    if not num_epochs:
        num_epochs = None

    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(input_file)

        dataset = dataset.map(decode)
        if is_preprocess is True:
            dataset = dataset.map(normalize)

        if is_shuffle is True:
            dataset = dataset.shuffle(1000+3*batch_size) # buffer_size should be the data set size
        if num_epochs>1:
            dataset = dataset.repeat(num_epochs)
        if batch_size>0:
            dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()

# save_tfrecord(img_paths, labels, '/home/public/nfs132_1/hanfy/cifar10/data/cifar10.record')

with tf.Session() as sess:
    image_batch, label_batch = read_tfrecord('/home/public/nfs132_1/hanfy/cifar10/data/cifar10.record', 1000)
    imgs, labels = sess.run([image_batch, label_batch])
    for i in range(len(imgs)):
        print('show')
        cv2.imshow('dd', imgs[i])
        cv2.waitKey(0)