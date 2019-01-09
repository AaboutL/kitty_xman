import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import pickle
import struct
import cv2

pts_num = 82


#    sys.path.insert(0, caffe_root + 'python')
def extractFeature(pts_path):
    fid = open(pts_path, 'r')
    ground_truth = []
    for line in fid:
        s_str = line.split()
        pts = [float(x) for x in s_str[0:-1]]
        if len(pts) != 164:
            print(s_str[-1])
            continue
        roi = [1000, 1000, 0, 0]
        keypoints = []
        for i in range(pts_num):
            keypoints.append([pts[2 * i], pts[2 * i + 1]])
            if pts[2 * i] < roi[0]:
                roi[0] = pts[2 * i]
            if pts[2 * i] > roi[2]:
                roi[2] = pts[2 * i]
            if pts[2 * i + 1] < roi[1]:
                roi[1] = pts[2 * i + 1]
            if pts[2 * i + 1] > roi[3]:
                roi[3] = pts[2 * i + 1]

        # pdb.set_trace()
        keypoints = np.array(keypoints)

        ctr_x = 0.5 * (roi[2] + roi[0])
        ctr_y = 0.5 * (roi[3] + roi[1])
        w = roi[2] - roi[0] + 1
        h = roi[3] - roi[1] + 1
        ex_scale = 1.1
        roi[0] = ctr_x - 0.5 * ex_scale * w
        roi[2] = ctr_x + 0.5 * ex_scale * w
        roi[1] = ctr_y - 0.5 * ex_scale * h
        roi[3] = ctr_y + 0.5 * ex_scale * h

        w = roi[2] - roi[0] + 1
        h = roi[3] - roi[1] + 1
        keypoints[:, 0] = (keypoints[:, 0] - roi[0]) / w
        keypoints[:, 1] = (keypoints[:, 1] - roi[1]) / h

        ground_truth.append(keypoints)


        # pdb.set_trace()
        # vis_detections(src_img, pre_points)
    std_point = np.mean(ground_truth, axis=0)
    np.savetxt('/home/slam/workspace/DL/alignment_method/align_untouch/meanshape_untouch.txt', std_point)
    print(std_point)
    for i in range(pts_num):
        pts = std_point[i]
        print("%f, %f, " % (pts[0], pts[1]))
    return std_point
    # with  open(fea_file,'wb') as f:
    #    for x in xrange(0, net.blobs['fc6'].data.shape[0]):
    #        for y in xrange(0, net.blobs['fc6'].data.shape[1]):
    #            f.write(struct.pack('f', net.blobs['fc6'].data[x,y]))


def readImageList(imageListFile):
    imageList = []
    with open(imageListFile, 'r') as fi:
        while (True):
            line = fi.readline().strip()
            if not line:
                break
            imageList.append(line)
    print('read imageList done image num ', len(imageList))
    return imageList


def drawROC(align_errors):
    align_errors_sort = sorted(align_errors)
    n = len(align_errors_sort)
    tmp = range(1, n + 1)
    p = [elem * 100 / n for elem in tmp]
    mean_error = np.mean(align_errors)
    print('mean_error: ', mean_error)
    plt.figure()
    plt.figure(figsize=(8, 8))
    plt.plot(align_errors_sort, p, color='darkorange',
             lw=2, label='ROC curve (mean_errors = %0.2f)' % mean_error)
    plt.xlim([0.0, 0.2])
    plt.ylim([0.0, 100])
    plt.xlabel('Error metric')
    plt.ylabel('Cumulative correct rate')
    plt.title('Performance in testing phase')
    plt.legend(loc="lower right")
    plt.savefig('CCR_curve.png')
    plt.show()


if __name__ == "__main__":
    pts_path = "/home/slam/nfs132_0/landmark/dataset/untouch/untouch_labeled/total/pts_path.txt"
    std_points = extractFeature(pts_path)