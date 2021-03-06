from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import cv2
import os
import numpy as np

def show_image(image, name, waitkey=0):
    cv2.imshow(name, image)
    cv2.waitKey(waitkey)

def show_points(image, facialshape, dim=2, color=(0, 255, 0), pointsize=2, show_idx=False):
    if dim == 2:
        for i in range(len(facialshape)):
            cv2.circle(image, (int(facialshape[i][0]), int(facialshape[i][1])), pointsize, color, -1)
            if show_idx:
                cv2.putText(image, str(i), (int(facialshape[i][0]), int(facialshape[i][1])), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (255, 0, 0))
    if dim == 1:
        for i in range(len(facialshape)//2):
            cv2.circle(image, (int(facialshape[i*2]), int(facialshape[i*2+1])), pointsize, color, -1)
            if show_idx:
                cv2.putText(image, str(i), (int(facialshape[i*2]), int(facialshape[i*2+1])), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (255, 0, 0))

def show_rect(image, rect):
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0]+rect[2]-1, rect[1]+rect[3]-1), (0, 0, 255))


