from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import cv2
import numpy as np
import h5py

from utilities import preprocess
from utilities import visualize

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
            if item == 'testset': continue
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

    def gether_data(self):
        total_image = []
        total_pts = []
        for item in self.datalist:
            img_path = item
            pts_path = item[0:-3] + 'pts'
            img = cv2.imread(img_path)
            pts = self.read_pts(pts_path)
            self.preprocess.set_img(img)
            self.preprocess.set_pts(pts)
            self.preprocess.get_shape_gt()
            if show:
                visualize.show_points(self.preprocess.image, self.preprocess.points)
                visualize.show_rect(self.preprocess.image, self.preprocess.ori_bbox)
                visualize.show_image(self.preprocess.image, 'ori', 0)

            norm_img, norm_pts = self.preprocess.normalize_data()
            pts_float = np.asarray(norm_pts, np.float32)
            norm_bbox = cv2.boundingRect(pts_float)
            if show:
                visualize.show_points(norm_img, norm_pts)
                visualize.show_rect(norm_img, norm_bbox)
                visualize.show_image(norm_img, 'norm', 0)
            total_image.append(norm_img)
            total_pts.append(norm_pts)
        return total_image, total_pts

    def save(self, output_file):
        with h5py.File(output_file, 'w') as output_f:
            total_image, total_pts = self.gether_data()
            img_set = output_f.create_dataset('image_dset', np.shape(total_image), dtype='i8', data=total_image)
            pts_set = output_f.create_dataset('points_dset', np.shape(total_pts), dtype='f', data=total_pts)

    def read(self, input_file):
        with h5py.File(input_file, 'r') as input_f:
            image_set = input_f['/image_dset'].value
            points_set = input_f['/points_dset'].value

            return image_set, points_set




