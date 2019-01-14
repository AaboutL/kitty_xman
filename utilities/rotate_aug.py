import numpy as np
import argparse
import os
import copy
import time
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import pdb
import cv2
import random


def get_files(rootDir):
    list_dirs = os.walk(rootDir)
    file_lists = []

    for root, dirs, files in list_dirs:
        for f in files:
            file_lists.append(os.path.join(root, f))
    return file_lists


pts_num = 68

if __name__ == '__main__':
    lists = get_files("train")
    save_path = "train_rotate/"
    img_list = [name for name in lists
                if name.endswith('.jpg') or name.endswith('.png')]

    count = 0
    for im_name in img_list:
        print(im_name)
        name = im_name.split("/")[-1]
        name = name.split(".")[0]

        src_img = cv2.imread(im_name)
        rows, cols, channel = src_img.shape
        fp = open(im_name[0:-3] + "pts", "r")
        points = []
        for i in range(pts_num):
            s_line = fp.readline()
            sub_str = s_line.split()
            pts = np.array([float(x) for x in sub_str])
            points.append(pts)
        fp.close()
        fp = open(im_name[0:-3] + "rect", "r")
        s_line = fp.readline()
        s_line = fp.readline()
        sub_str = s_line.split()
        bbox = np.array([float(x) for x in sub_str])
        fp.close()

        # rotate img

        for i in range(5):
            angle = random.randint(5, 30)  # 60
            sign = random.randrange(-1, 2, 2)
            angle = angle * sign
            M = cv2.getRotationMatrix2D((bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]), angle, 1)
            dst = cv2.warpAffine(src_img, M, (cols, rows))
            save_img_path = os.path.join(save_path, name + '_' + str(count) + '_' + str(abs(angle)) + '.jpg')
            cv2.imwrite(save_img_path, dst)

            fp = open(save_img_path[0:-3] + 'pts', 'w')
            LT_x = cols
            LT_y = rows
            RB_x = 0
            RB_y = 0
            for i in range(pts_num):
                pts = points[i]
                vec = np.array([[pts[0]], [pts[1]], [1]])
                res = np.dot(M, vec)
                fp.write("%f %f\n" % (res[0], res[1]))
                if LT_x > res[0]:
                    LT_x = res[0]
                if RB_x < res[0]:
                    RB_x = res[0]
                if LT_y > res[1]:
                    LT_y = res[1]
                if RB_y < res[1]:
                    RB_y = res[1]
            fp.close()

            roi = [int(LT_x), int(LT_y), int(RB_x), int(RB_y)]
            if roi[0] < 0:
                roi[0] = 0
            if roi[1] < 0:
                roi[1] = 0
            if roi[2] > cols - 1:
                roi[2] = cols - 1
            if roi[3] > rows - 1:
                roi[3] = rows - 1

            fp = open(save_img_path[0:-3] + 'rect', 'w')
            fp.write("1\n")
            fp.write("%d %d %d %d\n" % (roi[0], roi[1], roi[2] - roi[0] + 1, roi[3] - roi[1] + 1))
            fp.close()
            count = count + 1
            # pdb.set_trace()
            # print(res)

