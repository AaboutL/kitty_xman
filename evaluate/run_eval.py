from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
from evaluate import eval_tools

def get_landmark(lines):
    landmarks = []
    for line in lines:
        line = line.strip('\n').split(' ')[0:-1]
        pts = np.asarray([float(pt) for pt in line]).reshape(len(line)//2, 2)
        landmarks.append(pts)
    return landmarks

def main(args):
    with open(args.gtlandmarkfile, 'r') as gt_f:
        gtlandmarks = gt_f.readlines()
        gtlandmarks = get_landmark(gtlandmarks)
        print(len(gtlandmarks))
    with open(args.predictlandmarkfile, 'r') as pre_f:
        predictlandmarks = pre_f.readlines()
        predictlandmarks = get_landmark(predictlandmarks)
        print(len(predictlandmarks))
    errors = eval_tools.landmark_error(gtlandmarks, predictlandmarks, args.dist_type)
    eval_tools.auc_error(errors=errors, failure_threshold=args.failure_threshold, showCurve=args.show_curve)

def parser_augments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gtlandmarkfile', type=str, help='path to the groundtruth landmarks file',
                        default='/home/slam/nfs132_0/landmark/dataset/untouch/png_jpg/jpg/landmarks_list.txt')
    parser.add_argument('--predictlandmarkfile', type=str, help='path to the predict landmark file',
                        default='/home/slam/nfs132_0/landmark/dataset/untouch/png_jpg/png/landmarks_list.txt')
    parser.add_argument('--failure_threshold', type=float, help='The predict landmark treat as failure if the score is above the threshold',
                        default=0.08)
    parser.add_argument('--step', type=float, help='the threshold growing step',
                        default=0.0001)
    parser.add_argument('--dist_type', type=str, help='The type of distance',
                        default='centers')
    parser.add_argument('--show_curve', type=bool, help='whether to show the ROC curve',
                        default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parser_augments(sys.argv[1:]))
