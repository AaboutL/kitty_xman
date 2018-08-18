from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import cv2
import random
import numpy as np
import h5py

class Preprocess(object):
    def __init__(self, image, points, target_size, scale=[1.0, 1.2]):
        '''
        :param image: origin image
        :param points: origin points, [[x,y]...]
        :param target_size: to which size do we resize, [width, height]
        :param scale: upper and lower bound of the random scale
        '''
        self.image = image
        self.points = points
        self.target_size = target_size
        self.scale = scale
        self.rand_scale = random.Random()

    def set_img(self, image):
        self.image = image

    def set_pts(self, points):
        self.points = points

    def set_target_size(self, target_size):
        self.target_size = target_size

    def set_scale(self, scale):
        self.scale = scale

    def get_shape_gt(self):
        print(self.points)
        points = [np.asarray(point, np.float32) for point in self.points]
        points =np.asarray(points)
        print(points)
        print(type(points))
        self.ori_bbox = cv2.boundingRect(points)
        print(self.ori_bbox)
        self.ori_face = self.image[self.ori_bbox[1]: self.ori_bbox[1] + self.ori_bbox[3], self.ori_bbox[0]: self.ori_bbox[0] + self.ori_bbox[2]]
        self.shape_gt = self.points - [self.ori_bbox[0], self.ori_bbox[1]]

    def normalize_data(self):
        normalized_gts = np.multiply(np.divide(self.shape_gt, [self.ori_bbox.width, self.ori_bbox.height]), self.target_size)
        normalized_img = cv2.resize(self.ori_face, (self.target_size[0], self.target_size[1]))
        return normalized_img, normalized_gts

    def bbox_aug(self):
        # TODO
        ori_bbox = cv2.boundingRect(self.points)
        ori_bbox_w = ori_bbox.width
        ori_bbox_h = ori_bbox.height
        ori_head = self.image[ori_bbox.y: ori_bbox.y + ori_bbox.height, ori_bbox.x: ori_bbox.x + ori_bbox.width]

        new_width = self.rand_scale.uniform(self.scale[0], self.scale[1]) * ori_bbox_w
        new_height = self.rand_scale.uniform(self.scale[0], self.scale[1]) * ori_bbox_h






