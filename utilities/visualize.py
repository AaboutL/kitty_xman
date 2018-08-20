from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import cv2
import os
import numpy as np

def show_image(image, name, waitkey=0):
    cv2.imshow(name, image)
    cv2.waitKey(waitkey)

def show_points(image, points):
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)

def show_rect(image, rect):
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0]+rect[2]-1, rect[1]+rect[3]-1), (0, 0, 255))