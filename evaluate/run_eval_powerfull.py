import xml.etree.ElementTree as ET
import argparse
import sys
import os
import numpy as np
from evaluate import eval_tools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def evaluate_subset(pixel_errors, norm_errors):
    avg_pixel_error = np.mean(pixel_errors)
    avg_norm_error = np.mean(norm_errors)
    print("Average pixel error: {0}".format(avg_pixel_error))
    print("Average norm error: {0}".format(avg_norm_error))
    return avg_pixel_error, avg_norm_error

def evaluate(angle, pixel_errors, norm_errors, split_type='uniform', split_num=3, split_ids=None, show_curve = False):
    error_num = len(pixel_errors)
    pixel_errors = np.asarray(pixel_errors)
    norm_errors = np.asarray(norm_errors)
    sorted_angle = np.sort(angle) # sort the angles
    sorted_index = np.argsort(angle) # get original ids of the sorted angles
    pixel_errors_parts = [] # split pixel_errors into several parts
    norm_errors_parts = [] # split norm_errors into several parts
    avg_pixel_errors = [] # each part's average pixel error
    avg_norm_errors = [] # each part's average norm error
    angle_index_parts = [] # 每一部分对应的角度的原始id

    if split_type=='uniform':
        angle_index_parts = np.array_split(sorted_index, split_num)
        for i in range(len(angle_index_parts)):
            pixel_errors_parts.append(pixel_errors[angle_index_parts[i]])
            norm_errors_parts.append(norm_errors[angle_index_parts[i]])
        # evaluate each part of errors
        for i in range(split_num):
            avg_pixel_error, avg_norm_error = evaluate_subset(pixel_errors_parts[i], norm_errors_parts[i])
            avg_pixel_errors.append(avg_pixel_error)
            avg_norm_errors.append(avg_norm_error)
    elif split_type=='manual':
        for i in range(len(split_ids) -1):
            angle_index_parts.append(sorted_index[np.where( (sorted_angle > split_ids[i]) & (sorted_angle <= split_ids[i+1]))])
        for i in range(len(angle_index_parts)):
            pixel_errors_parts.append(pixel_errors[angle_index_parts[i]])
            norm_errors_parts.append(norm_errors[angle_index_parts[i]])
        for i in range(len(pixel_errors_parts)):
            avg_pixel_error, avg_norm_error = evaluate_subset(pixel_errors_parts[i], norm_errors_parts[i])
            avg_pixel_errors.append(avg_pixel_error)
            avg_norm_errors.append(avg_norm_error)

    print(avg_pixel_errors, avg_norm_errors)

    if show_curve:
        # show errors together
        pixel_errors_sorted = []
        norm_errors_sorted = []
        for i in range(len(pixel_errors_parts)):
            pixel_errors_sorted.extend(pixel_errors_parts[i])
            norm_errors_sorted.extend(norm_errors_parts[i])
        plt.figure()
        plt.subplot(2, 1, 1)
        pixel_plt, = plt.plot(sorted_angle, pixel_errors_sorted, color='b')
        plt.subplot(2, 1, 2)
        norm_plt, = plt.plot(sorted_angle, norm_errors_sorted, color='g')
        plt.legend(handles=[pixel_plt, norm_plt], labels=['pixel_error', 'norm_error'], loc='best')
        plt.show()
        # show average errors of each parts
        plt.figure()
        plt.subplot(2, 1, 1)
        x = range(len(pixel_errors_parts))
        plt.plot(x, avg_pixel_errors)
        plt.subplot(2, 1, 2)
        plt.plot(x, avg_norm_errors)
        plt.show()

def evaluate1(angle, pixel_errors, norm_errors, split_type='uniform', split_num=3, split_ids=None, show_curve = False):
    pixel_errors = np.asarray(pixel_errors)
    norm_errors = np.asarray(norm_errors)
    sorted_angle = np.sort(angle) # sort the angles
    sorted_index = np.argsort(angle) # get original ids of the sorted angles
    pixel_errors_sorted = pixel_errors[sorted_index]
    norm_errors_sorted = norm_errors[sorted_index]
    pixel_errors_parts = [] # split pixel_errors into several parts
    norm_errors_parts = [] # split norm_errors into several parts
    avg_pixel_errors = [] # each part's average pixel error
    avg_norm_errors = [] # each part's average norm error

    if split_type=='uniform':
        pixel_errors_parts = np.array_split(pixel_errors_sorted, split_num)
        norm_errors_parts = np.array_split(norm_errors_sorted, split_num)
        # evaluate each part of errors
        for i in range(split_num):
            avg_pixel_error, avg_norm_error = evaluate_subset(pixel_errors_parts[i], norm_errors_parts[i])
            avg_pixel_errors.append(avg_pixel_error)
            avg_norm_errors.append(avg_norm_error)
    elif split_type=='manual':
        for i in range(len(split_ids) -1):
            pixel_errors_part = pixel_errors_sorted[np.where( (sorted_angle > split_ids[i]) & (sorted_angle <= split_ids[i+1]))]
            norm_errors_part = norm_errors_sorted[np.where( (sorted_angle > split_ids[i]) & (sorted_angle <= split_ids[i+1]))]
            avg_pixel_error, avg_norm_error = evaluate_subset(pixel_errors_part, norm_errors_part)
            avg_pixel_errors.append(avg_pixel_error)
            avg_norm_errors.append(avg_norm_error)

    print(avg_pixel_errors, avg_norm_errors)

    if show_curve:
        # show the whole errors
        plt.figure()
        plt.subplot(2, 1, 1)
        pixel_plt, = plt.plot(sorted_angle, pixel_errors_sorted, color='b')
        plt.ylim(0, 60)
        plt.subplot(2, 1, 2)
        norm_plt, = plt.plot(sorted_angle, norm_errors_sorted, color='g')
        plt.ylim(0, 0.2)
        plt.legend(handles=[pixel_plt, norm_plt], labels=['pixel_error', 'norm_error'], loc='best')
        plt.show()
        # show average errors of each parts
        plt.figure()
        x = range(split_num)
        plt.subplot(2, 1, 1)
        plt.bar(x, avg_pixel_errors)
        plt.xlim(0, split_num)
        plt.ylim(0., 12.)
        plt.subplot(2, 1, 2)
        plt.bar(x, avg_norm_errors)
        plt.xlim(0, split_num)
        plt.ylim(0., 0.2)
        plt.show()

def read_error_xml(input_xml, set_type, angle_type):
    tree = ET.ElementTree(file=input_xml)
    root = tree.getroot()
    images = root[1]
    pixel_errors = []
    norm_errors = []
    angle = []
    for image in images:
        sets_node = image[0]
        angles_node = image[1]
        errors_node = image[2]
        if sets_node.attrib[set_type] == 'yes':
            angle.append(float(angles_node.attrib[angle_type]))
            pixel_errors.append(float(errors_node.attrib['pixel_error']))
            norm_errors.append(float(errors_node.attrib['norm_error']))
    return angle, pixel_errors, norm_errors


def main(args):
    if args.input_xml == '':
        print('Input file should be specified!')
        exit(0)
    if args.split_type == 'manual' and args.split_ids == '':
        print('If you want manually split the testset according angle, you should specify the split ids!')
        exit(0)
    split_num = args.split_num
    if args.split_ids != '':
        split_ids = [int(id) for id in args.split_ids[1:-1].split(',')]
        split_num = len(split_ids) - 1
    input_xml = args.input_xml
    split_type = args.split_type
    angle_type = args.angle_type
    set_type = args.set_type
    angle, pixel_errors, norm_errors = read_error_xml(input_xml, set_type, angle_type)

    evaluate1(angle=angle, pixel_errors=pixel_errors, norm_errors=norm_errors, split_type=split_type, split_num=split_num, split_ids=split_ids, show_curve=True)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_xml', type=str, default='/home/slam/workspace/DL/alignment_method/align_untouch/temp/multi_models_errors_1216.xml')
    parser.add_argument('--split_num', type=int, default=3)
    parser.add_argument('--split_type', type=str, default='manual')
    parser.add_argument('--split_ids', type=str, default='[-90,-40,-20,20,40,90]')
    parser.add_argument('--angle_type', type=str, default='yaw')
    parser.add_argument('--set_type', type=str, default='total')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

