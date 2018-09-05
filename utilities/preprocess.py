from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import cv2
import random
import numpy as np
import h5py

from utilities import visualize

class Preprocess(object):
    def __init__(self, image, points, target_size, scale=[0.1, 0.2]):
        '''
        :param image: origin image
        :param points: origin points, [[x,y]...]
        :param target_size: to which size do we resize, [width, height]
        :param scale: upper and lower bound of the random scale
        '''
        self.image = image
        self.points_ori = points
        self.target_size = target_size
        self.scale = scale
        self.rand_scale = random.Random()

    def set_img(self, image):
        self.image = image

    def set_pts(self, points):
        self.points_ori = points

    def set_target_size(self, target_size):
        self.target_size = target_size

    def set_scale(self, scale):
        self.scale = scale

    def get_crop_shape(self, is_bbox_aug=True):
        if is_bbox_aug is False:
            bbox = self.ori_bbox
        else:
            bbox = self.auged_bbox
        # print(bbox[0], bbox[1], bbox[2], bbox[3])
        self.crop_face = self.image[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
        self.crop_shape = np.subtract(self.points_ori, [bbox[0], bbox[1]])

    def resize_data(self, is_bbox_aug=True):
        self.ori_bbox = cv2.boundingRect(np.asarray(self.points_ori, np.float32))
        if is_bbox_aug is True:
            bbox = self.bbox_aug1()
            if bbox is None:
                return None, None
        else:
            bbox = self.ori_bbox

        self.get_crop_shape(is_bbox_aug)
        resized_gts = np.multiply(np.divide(self.crop_shape, [bbox[2], bbox[3]]), self.target_size)
        resized_img = cv2.resize(self.crop_face, (self.target_size[0], self.target_size[1]))

        return resized_img, resized_gts

    def bbox_aug(self):
        height = self.image.shape[0]
        width = self.image.shape[1]

        ori_bbox = self.ori_bbox
        ori_bbox_w = ori_bbox[2]
        ori_bbox_h = ori_bbox[3]

        delta_up = self.rand_scale.uniform(self.scale[0], self.scale[1]) * ori_bbox_h
        delta_down = self.rand_scale.uniform(self.scale[0], self.scale[1]) * ori_bbox_h
        delta_left = self.rand_scale.uniform(self.scale[0], self.scale[1]) * ori_bbox_w
        delta_right = self.rand_scale.uniform(self.scale[0], self.scale[1]) * ori_bbox_w

        left = np.maximum(int(ori_bbox[0]-delta_left), 0)
        right = np.minimum(int(ori_bbox[0]+ori_bbox[2]+delta_right), width)
        up = np.maximum(int(ori_bbox[1] - delta_up), 0)
        down = np.minimum(int(ori_bbox[1]+ori_bbox[3]+delta_down), height)
        self.auged_bbox = [left, up, right-left, down-up]
        return [left, up, right-left, down-up]

    def bbox_aug1(self):
        height = self.image.shape[0]
        width = self.image.shape[1]

        ori_bbox = self.ori_bbox
        ori_bbox_x = ori_bbox[0]
        ori_bbox_y = ori_bbox[1]
        ori_bbox_w = ori_bbox[2]
        ori_bbox_h = ori_bbox[3]

        center_x = ori_bbox_x + int(ori_bbox_w/2)
        center_y = ori_bbox_y + int(ori_bbox_h/2)
        big_size = np.maximum(ori_bbox_w, ori_bbox_h)
        delta_s = int(np.ceil(self.rand_scale.uniform(self.scale[0], self.scale[1]) * big_size))
        bbox_s = big_size + delta_s
        half_s = bbox_s // 2
        if bbox_s > np.minimum(height, width):
            self.auged_bbox = None
            return None
        left = center_x - half_s
        up = center_y - half_s
        right = center_x + half_s
        down = center_y + half_s
        if left < 0:
            left = 0
        if up < 0:
            up = 0
        if right > width:
            left = width - bbox_s
        if down > height:
            up = height - bbox_s
        self.auged_bbox = [left, up, bbox_s, bbox_s]
        return [left, up, bbox_s, bbox_s]


    def flip_left_right(self, image, pts):
        pts_cp = pts.copy()
        flipped_img = cv2.flip(image, 1)
        img_w = image.shape[1]
        pts_cp = np.abs(np.subtract(pts_cp, [img_w, 0]))
        mirrored_pts = self.mirrorShape(pts_cp)
        return flipped_img, mirrored_pts

    def mirrorShape(self, facialshape):
        fs = facialshape.copy()
        lEyeIndU = list(range(36, 40))
        lEyeIndD = [40, 41]
        rEyeIndU = list(range(42, 46))
        rEyeIndD = [46, 47]
        lBrowInd = list(range(17, 22))
        rBrowInd = list(range(22, 27))

        uMouthInd = list(range(48, 55))
        dMouthInd = list(range(55, 60))
        uInnMouthInd = list(range(60, 65))
        dInnMouthInd = list(range(65, 68))
        noseInd = list(range(31, 36))
        beardInd = list(range(17))

        lEyeU = fs[lEyeIndU].copy()
        lEyeD = fs[lEyeIndD].copy()
        rEyeU = fs[rEyeIndU].copy()
        rEyeD = fs[rEyeIndD].copy()
        lBrow = fs[lBrowInd].copy()
        rBrow = fs[rBrowInd].copy()
        uMouth = fs[uMouthInd].copy()
        dMouth = fs[dMouthInd].copy()
        uInnMouth = fs[uInnMouthInd].copy()
        dInnMouth = fs[dInnMouthInd].copy()
        nose = fs[noseInd].copy()
        beard = fs[beardInd].copy()

        lEyeIndU.reverse()
        lEyeIndD.reverse()
        rEyeIndU.reverse()
        rEyeIndD.reverse()
        lBrowInd.reverse()
        rBrowInd.reverse()

        uMouthInd.reverse()
        dMouthInd.reverse()
        uInnMouthInd.reverse()
        dInnMouthInd.reverse()
        beardInd.reverse()
        noseInd.reverse()

        fs[rEyeIndU] = lEyeU
        fs[rEyeIndD] = lEyeD
        fs[lEyeIndU] = rEyeU
        fs[lEyeIndD] = rEyeD
        fs[rBrowInd] = lBrow
        fs[lBrowInd] = rBrow

        fs[uMouthInd] = uMouth
        fs[dMouthInd] = dMouth
        fs[uInnMouthInd] = uInnMouth
        fs[dInnMouthInd] = dInnMouth
        fs[noseInd] = nose
        fs[beardInd] = beard
        return fs

