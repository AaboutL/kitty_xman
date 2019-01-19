import xml.etree.ElementTree as ET
import argparse
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.integrate import simps

'''
    How to use eval script to evaluate the landmark result.
    First: use gen_xml.py to create testset groundtruth xml. Images, points and pose data should included in one folder.
    Second: Use your algorithm to predict testset landmark, and save results to a txt.
    Third: Use calc_save_error.py to calculate error between prediction and groundtruth, and save the result to xml.
    Fourth: Use this script to evaluate two algorithms.
'''

def evaluate_subset(pixel_errors, norm_errors):
    avg_pixel_error = np.mean(pixel_errors)
    avg_norm_error = np.mean(norm_errors)
    print("Average pixel error: {0}".format(avg_pixel_error))
    print("Average norm error: {0}".format(avg_norm_error))
    return avg_pixel_error, avg_norm_error


def evaluate(angle, pixel_errors, norm_errors, split_type='uniform', split_num=3, split_ids=None, show_curve = False):
    pixel_errors = np.asarray(pixel_errors)
    norm_errors = np.asarray(norm_errors)
    sorted_angle = np.sort(angle) # sort the angles
    sorted_index = np.argsort(angle) # get original ids of the sorted angles
    pixel_errors_sorted = pixel_errors[sorted_index]
    norm_errors_sorted = norm_errors[sorted_index]
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
    return pixel_errors_sorted, norm_errors_sorted, avg_pixel_errors, avg_norm_errors

def show_error(pixel_errors_1, norm_errors_1, avg_pixel_errors_1, avg_norm_errors_1, sorted_angle_1,
         pixel_errors_2, norm_errors_2, avg_pixel_errors_2, avg_norm_errors_2, sorted_angle_2):
    plt.figure()
    # show whole errors
    plt.subplot(4, 1, 1)
    pixel_plt_1, = plt.plot(sorted_angle_1, pixel_errors_1, color='b')
    pixel_plt_2, = plt.plot(sorted_angle_2, pixel_errors_2, color='g')
    plt.ylim(0, 60)

    plt.subplot(4, 1, 2)
    norm_plt_1, = plt.plot(sorted_angle_1, norm_errors_1, color='b')
    norm_plt_2, = plt.plot(sorted_angle_2, norm_errors_2, color='g')

    # show average errors of each parts
    split_num = len(avg_pixel_errors_1)
    X = range(split_num)
    plt.subplot(4, 1, 3)
    plt.bar(X, avg_pixel_errors_1, width=-0.2, align='edge', color='b')
    plt.bar(X, avg_pixel_errors_2, width=0.2, align='edge', color='g')
    for x, y in zip(X, avg_pixel_errors_1):
        plt.text(x - 0.2, y + 0.05, '%.3f'%y, ha='center', va='bottom')
    for x, y in zip(X, avg_pixel_errors_2):
        plt.text(x + 0.2, y + 0.05, '%.3f'%y, ha='center', va='bottom')
    plt.xlim(0, split_num)
    plt.ylim(0., 12.)

    plt.subplot(4, 1, 4)
    plt.bar(X, avg_norm_errors_1, width=-0.2, align='edge', color='b')
    plt.bar(X, avg_norm_errors_2, width=0.2, align='edge', color='g')
    for x, y in zip(X, avg_norm_errors_1):
        plt.text(x - 0.2, y + 0.01, '%.3f'%y, ha='center', va='bottom')
    for x, y in zip(X, avg_norm_errors_2):
        plt.text(x + 0.2, y + 0.01, '%.3f'%y, ha='center', va='bottom')
    plt.xlim(0, split_num)
    plt.ylim(0., 0.2)

    plt.legend(handles=[pixel_plt_1, pixel_plt_2], labels=['first', 'second'], loc='best')
    plt.show()

def show_curve(norm_errors_1, norm_errors_2, failure_threshold, step=0.0001):
    nErrors_1 = len(norm_errors_1)
    xAxis_1 = list(np.arange(0., failure_threshold+step, step))
    ced_1 = [float(np.count_nonzero([norm_errors_1 <= x])) / nErrors_1 for x in xAxis_1]
    auc_1 = simps(ced_1, x=xAxis_1) / failure_threshold
    failure_rate = 1. - ced_1[-1]
    print("First AUC @ {0}: {1}".format(failure_threshold, auc_1))
    print("First Failure rate: {0}".format(failure_rate))

    nErrors_2 = len(norm_errors_2)
    xAxis_2 = list(np.arange(0., failure_threshold+step, step))
    ced_2 = [float(np.count_nonzero([norm_errors_2 <= x])) / nErrors_2 for x in xAxis_2]
    auc_2 = simps(ced_2, x=xAxis_2) / failure_threshold
    failure_rate = 1. - ced_2[-1]
    print("Second AUC @ {0}: {1}".format(failure_threshold, auc_2))
    print("Second Failure rate: {0}".format(failure_rate))

    curve_1, = plt.plot(xAxis_1, ced_1, color='b')
    curve_2, = plt.plot(xAxis_2, ced_2, color='g')
    plt.legend(handles=[curve_1, curve_2], labels=['first', 'second'], loc='best')
    plt.show()
    # plt.savefig(save_path)

    return auc_1, auc_2, failure_rate

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
    if args.input_first_xml == '' or args.input_second_xml == '':
        print('Input first and second xml files should be specified!')
        exit(0)
    if args.split_type == 'manual' and args.split_ids == '':
        print('If you want manually split the testset according angle, you should specify the split ids!')
        exit(0)
    split_num = args.split_num
    if args.split_ids != '':
        split_ids = [int(id) for id in args.split_ids[1:-1].split(',')]
        split_num = len(split_ids) - 1
    input_first_xml = args.input_first_xml
    input_second_xml = args.input_second_xml
    split_type = args.split_type
    angle_type = args.angle_type
    set_type = args.set_type
    angle_1, pixel_errors_1, norm_errors_1 = read_error_xml(input_first_xml, set_type, angle_type)
    angle_2, pixel_errors_2, norm_errors_2 = read_error_xml(input_second_xml, set_type, angle_type)
    sorted_angle_1 = np.sort(angle_1)
    sorted_angle_2 = np.sort(angle_2)

    pixel_errors_sorted_1, norm_errors_sorted_1, avg_pixel_errors_1, avg_norm_errors_1 = \
        evaluate(angle=angle_1, pixel_errors=pixel_errors_1, norm_errors=norm_errors_1, split_type=split_type, split_num=split_num, split_ids=split_ids, show_curve=True)

    pixel_errors_sorted_2, norm_errors_sorted_2, avg_pixel_errors_2, avg_norm_errors_2 = \
        evaluate(angle=angle_2, pixel_errors=pixel_errors_2, norm_errors=norm_errors_2, split_type=split_type, split_num=split_num, split_ids=split_ids, show_curve=True)

    show_error(pixel_errors_1, norm_errors_1, avg_pixel_errors_1, avg_norm_errors_1, sorted_angle_1,
         pixel_errors_2, norm_errors_2, avg_pixel_errors_2, avg_norm_errors_2, sorted_angle_2)

    show_curve(norm_errors_sorted_1, norm_errors_sorted_2, failure_threshold=0.1)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_first_xml', type=str, default='/home/slam/nfs132_0/landmark/dataset/untouch/AHD_USB/ahd_errors.xml')
    parser.add_argument('--input_second_xml', type=str, default='/home/slam/nfs132_0/landmark/dataset/untouch/AHD_USB/usb_errors.xml')
    parser.add_argument('--split_num', type=int, help="if split_type==uniform, then the testset will split to split_num parts",default=3)
    parser.add_argument('--split_type', type=str, default='manual')
    parser.add_argument('--split_ids', type=str, default='[-90,90]')
    parser.add_argument('--angle_type', type=str, default='yaw')
    parser.add_argument('--set_type', type=str, default='total')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

