import numpy as np
from utilities.data_preparation import utils
import os
import sys
import shutil
import argparse

def main(args):
    with open(args.input, 'r') as in_f:
        lines = in_f.readlines()
    for line in lines:
        items = line.strip('\n').split(' ')
        img_path = items[-1]
        img_name = img_path.split('/')[-1]
        name = img_name[:-3]
        img_path_new = os.path.join(args.save_folder, img_name)
        pts_path = os.path.join(args.save_folder, name+'txt')
        pts = items[:-1]
        print(len(pts))
        pts = np.asarray([float(pt) for pt in pts]).reshape(len(pts)//2, 2)
        utils.saveToPts(pts_path, pts)
        # shutil.copyfile(img_path, img_path_new)


def parse_augments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/home/slam/nfs132_0/landmark/dataset/untouch/png_jpg/jpg/landmarks_list.txt')
    parser.add_argument('--save_folder', type=str, default='/home/slam/nfs132_0/landmark/dataset/untouch/png_jpg/jpg')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_augments(sys.argv[1:]))
