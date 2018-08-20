from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
from evaluate.facial_landmark import landmark_eval

def main(args):
    with open(args.gtlandmarkfile, 'r') as gt_f:
        gtlandmarks = gt_f.readlines()
    with open(args.predictlandmarkfile, 'r') as pre_f:
        predictlandmarks = pre_f.readlines()
    errors = landmark_eval.landmark_error(gtlandmarks, predictlandmarks, args.dist_type)
    landmark_eval.auc_error(errors, args.failure_threshold, args.show_curve)

def parser_augments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gtlandmarkfile', type=str, help='path to the groundtruth landmarks file',
                        default='')
    parser.add_argument('--predictlandmarkfile', type=str, help='path to the predict landmark file',
                        default='')
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
