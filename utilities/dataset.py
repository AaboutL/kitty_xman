from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import cv2
import numpy as np
import h5py
import tensorflow as tf
import random

from utilities import preprocess
from utilities import visualize
from utilities import tfrecords_util

show = False

class Dataset(object):
    def __init__(self):
        '''
        :param root_dir: root_dir is the directory contains the datasets. Suppose the dataset is arranged as a.png together with a.pts, so
                         the image and the points have the same prefix
        :param datalist: contains the whole image paths
        '''
        # self.root_dir = root_dir
        self.datalist = []
        self.preprocess = preprocess.Preprocess(None, None, [224, 224])

    def get_datalist(self, root_dir, format):
        items = os.listdir(root_dir)
        items.sort()
        for item in items:
            # if item == 'testset': continue
            if item in ['testset', 'ibug']:continue
            path = os.path.join(root_dir, item)
            if not os.path.isfile(path):
                self.get_datalist(path, format)
            else:
                if item.split('.')[-1] not in format:
                    continue
                img_path = os.path.join(root_dir, item)
                self.datalist.append(img_path)

    def read_pts(self, pts_path):
        points = []
        with open(pts_path, 'r') as pts_f:
            lines = pts_f.readlines()
            used_lines = lines[3: -1]
            for line in used_lines:
                point = line.strip('\n').split(' ')
                point = [float(point[0]), float(point[1])]
                points.append(point)
        return points

    def gether_data(self, is_bbox_aug=True, is_flip=True):
        total_image = []
        total_pts_flatten = []
        total_pts = []
        for item in self.datalist:
            img_path = item
            pts_path = item[0:-3] + 'pts'
            img = cv2.imread(img_path)
            pts = self.read_pts(pts_path)
            self.preprocess.set_img(img)
            self.preprocess.set_pts(pts)

            resized_img, resized_pts = self.preprocess.resize_data(is_bbox_aug)
            if resized_img is None:
                continue
            resized_flatten_pts = sum(resized_pts.tolist(), [])
            total_image.append(resized_img)
            total_pts_flatten.append(resized_flatten_pts)
            total_pts.append(resized_pts)
            if is_flip is True:
                mirrored_img, mirrored_pts = self.preprocess.flip_left_right(resized_img, resized_pts)
                total_image.append(mirrored_img)
                mirrored_flatten_pts = sum(mirrored_pts.tolist(), [])
                total_pts_flatten.append(mirrored_flatten_pts)
                total_pts.append(mirrored_pts)
        return total_image, total_pts_flatten, total_pts

    def normalize_pts(self, points_set, scale_factor):
        scaled_points = np.asarray(points_set) * scale_factor - 0.5
        mean = np.mean(scaled_points, axis=0)
        std = np.std(scaled_points, axis=0)
        # print('mean', mean)
        # print('std', std)
        normed_points = np.divide(np.subtract(scaled_points, mean), std)
        return mean, std, normed_points

    def save(self, output_file, format):
        if format=='hdf5':
            self.save_hdf5(output_file)
        if format=='tfrecords':
            self.save_tfrecords(output_file)

    def read(self, input_file, format):
        if format=='hdf5':
            self.read_hdf5(input_file)
        if format=='tfrecords':
            self.read_tfrecords(input_file)

    def save_hdf5(self, output_file, is_bbox_aug=True, is_flip=True):
        with h5py.File(output_file, 'w') as output_f:
            total_image, _, total_pts = self.gether_data(is_bbox_aug, is_flip)
            img_set = output_f.create_dataset('image_dset', np.shape(total_image), dtype='i8', data=total_image)
            pts_set = output_f.create_dataset('points_dset', np.shape(total_pts), dtype='f', data=total_pts) # [68, 2]

    def read_hdf5(self, input_file):
        with h5py.File(input_file, 'r') as input_f:
            image_set = input_f['/image_dset'].value
            points_set = input_f['/points_dset'].value # [-1, 68, 2]
            return image_set, points_set

    def save_tfrecords(self, output_file):
        print('generating %s' %output_file)
        total_image, total_pts, _ = self.gether_data()
        print('total sampes:', len(total_image))
        indices = np.arange(0, len(total_image))
        random.shuffle(indices)
        with tf.python_io.TFRecordWriter(output_file) as record_writer:
            for i in indices:
                image = total_image[i]
                image_raw = image.tostring()
                pts = total_pts[i]
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': tfrecords_util.bytes_feature(image_raw),
                        'label': tfrecords_util.float_list_feature(pts),
                    }
                ))
                record_writer.write(example.SerializeToString())




