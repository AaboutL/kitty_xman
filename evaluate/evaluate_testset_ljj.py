# import _init_paths
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import pickle
import struct
import cv2
import pdb
import numpy.linalg as LA
import math
import caffe
import time

deployPrototxt =  './caffe_faster_rcnn/examples/face_align/pts82.prototxt'
modelFile = './caffe_faster_rcnn/examples/face_align2/models/face_pts__iter_80000.caffemodel'

imageListFile = '/data1/hanfy/landmark_untouch/landmark_data/used_data/test_list.txt'
#imageListFile = '/data_001/face_align/caffe_106_stn/pts106/pts_net_v1_yuv_stage2/test_list.txt'
gpuID = 1
postfix = 'pts82'
pts_num = 82
# std_points = [0.057983, 0.303535, 0.066838, 0.424183, 0.086205, 0.552676, 0.124177, 0.655471, 0.179290, 0.745902, 0.251304, 0.822962, 0.338319, 0.885529, 0.436517, 0.929080, 0.543112, 0.942096, 0.645363, 0.916335, 0.731287, 0.860275, 0.802919, 0.788652, 0.860500, 0.705016, 0.901849, 0.611423, 0.927432, 0.509442, 0.933731, 0.384452, 0.928891, 0.268416, 0.191058, 0.156854, 0.238443, 0.105142, 0.298020, 0.092419, 0.359762, 0.094397, 0.419243, 0.105939, 0.419110, 0.144415, 0.361372, 0.137393, 0.302778, 0.135242, 0.245556, 0.142299, 0.581355, 0.099612, 0.637264, 0.084262, 0.696096, 0.077351, 0.753994, 0.084552, 0.802386, 0.131150, 0.749996, 0.122185, 0.695701, 0.120375, 0.640342, 0.126795, 0.585719, 0.137378, 0.509814, 0.235850, 0.514382, 0.298202, 0.518542, 0.360915, 0.522263, 0.422651, 0.438669, 0.503414, 0.481884, 0.499996, 0.524456, 0.498163, 0.565436, 0.495948, 0.605474, 0.495287, 0.257601, 0.260211, 0.288576, 0.250474, 0.324116, 0.244464, 0.361417, 0.243852, 0.393928, 0.251123, 0.361594, 0.261542, 0.326188, 0.266633, 0.290827, 0.266141, 0.620645, 0.241007, 0.651262, 0.231492, 0.687203, 0.229084, 0.721727, 0.232266, 0.751601, 0.240247, 0.721376, 0.247909, 0.688369, 0.250796, 0.653448, 0.248336, 0.395560, 0.672012, 0.436590, 0.624340, 0.495139, 0.592582, 0.528929, 0.597802, 0.560803, 0.589899, 0.617437, 0.615908, 0.656674, 0.658904, 0.628196, 0.696055, 0.585911, 0.722771, 0.532800, 0.734366, 0.477350, 0.728519, 0.430427, 0.705950, 0.408380, 0.671168, 0.485090, 0.637914, 0.530100, 0.632537, 0.573685, 0.634255, 0.643937, 0.658899, 0.574688, 0.683699, 0.531628, 0.689720, 0.486868, 0.687963, 0.328012, 0.255520, 0.685879, 0.240310]

std_points = [0.09117715, 0.30707606, 0.09324848, 0.41954431, 0.10563363,
            0.53930686, 0.13554834, 0.63529839, 0.18063637, 0.72136517,
            0.24145734, 0.79639416, 0.31553945, 0.85885091, 0.40107435,
            0.90523391, 0.49681237, 0.92342779, 0.59257619, 0.90536403,
            0.67817381, 0.85908019, 0.75232247, 0.79670145, 0.81321151,
            0.72175094, 0.85837894, 0.63573553, 0.88837329, 0.53977695,
            0.90085993, 0.41998328, 0.90307217, 0.30751114, 0.21447486,
            0.18024831, 0.25930068, 0.13450505, 0.31400381, 0.12613003,
            0.36981687, 0.13112954, 0.42337935, 0.14421787, 0.42093509,
            0.18082662, 0.36893312, 0.17200271, 0.31626513, 0.16725429,
            0.26444659, 0.17054685, 0.57095178, 0.14429979, 0.62452464,
            0.13126451, 0.68034313, 0.12633297, 0.73503934, 0.13475825,
            0.77983972, 0.18054622, 0.72986456, 0.17079156, 0.67804174,
            0.16744044, 0.62537141, 0.17213774, 0.57335773, 0.18090177,
            0.49710396, 0.26933184, 0.49706704, 0.3289459 , 0.49703831,
            0.38882527, 0.49701083, 0.44705939, 0.41675322, 0.51578456,
            0.45662181, 0.51598197, 0.49696722, 0.5165962 , 0.53730749,
            0.51601283, 0.57717068, 0.51584427, 0.26849419, 0.28139835,
            0.29699794, 0.27403973, 0.32946369, 0.27032097, 0.36336173,
            0.27117067, 0.39292814, 0.27850187, 0.36239604, 0.28677197,
            0.32977275, 0.29013528, 0.29807281, 0.28823389, 0.60126278,
            0.27859209, 0.6308256 , 0.27129341, 0.66473061, 0.27048341,
            0.69719322, 0.27423901, 0.72571056, 0.281631  , 0.69611433,
            0.28843627, 0.66440651, 0.29029974, 0.63179582, 0.28690343,
            0.3754184 , 0.66685732, 0.41362616, 0.62694063, 0.46638467,
            0.60157902, 0.49693003, 0.60803847, 0.52748883, 0.60160067,
            0.5802302 , 0.6270102 , 0.61842825, 0.6670038 , 0.58869354,
            0.70001444, 0.54733009, 0.72260403, 0.49691238, 0.73098791,
            0.44649845, 0.72254741, 0.40513943, 0.6999065 , 0.3869719 ,
            0.66658011, 0.45679916, 0.64209252, 0.49690745, 0.63946237,
            0.53702338, 0.64213289, 0.60686957, 0.66670063, 0.53678075,
            0.68488495, 0.4969361 , 0.68833033, 0.45709294, 0.6848394 ,
            0.33176716, 0.28050377, 0.66241829, 0.28066108]

def PointMean(points, st_idx, ed_idx):
    num = ed_idx - st_idx + 1
    mean_point = [0.0,0.0]
    for idx in range(st_idx,ed_idx+1):
        mean_point = mean_point + points[idx]
    mean_point = mean_point/num
    return mean_point


def vis_detections(im, pre_points, gt_points, idx):
    """Draw detected landmarks."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(pts_num):
        ax.add_patch(
            plt.Circle(pre_points[i] ,1.0, fill=True,
                          edgecolor='red', linewidth=0.5)
            )
        ax.add_patch(
            plt.Circle(gt_points[i] ,1.0, fill=True,
                          edgecolor='blue', linewidth=0.5)
            )
       # ax.text(bbox[0], bbox[1] - 2,
       #         '{:s} {:.3f}'.format(class_name, score),
       #         bbox=dict(facecolor='blue', alpha=0.5),
       #         fontsize=14, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    rst_save_path = os.path.join('/data1/hanfy/landmark_untouch/landmark_data/eval_results/0117-1446/output/test_rst', str(idx) + '.jpg')
    plt.savefig(rst_save_path)
    plt.show()

def show(im, pre_points, gt_points):
    im = im[:, :, (2, 1, 0)]
    im = np.asarray(im, dtype=np.uint8)
    print('im shape: ',im.shape)
    cv2.imshow('im', im)
#    for i in range(pts_num):
#        print('pt: ',pre_points[i])
#        cv2.circle(im, (int(pre_points[i][0]), int(pre_points[i][1])), 2, (0, 0, 255), 2)
#        cv2.circle(im, (int(gt_points[i][0]), int(gt_points[i][1])), 2, (0, 0, 255), 2)
#    cv2.imshow("im", im)
    cv2.waitKey(0)
    

def initilize():
    print( 'initilize ... ')

#    sys.path.insert(0, caffe_root + 'python')
    caffe.set_mode_gpu()
    caffe.set_device(gpuID)
    net = caffe.Net(deployPrototxt, modelFile,caffe.TEST)
    return net  
def extractFeature(imageList, net):
    net.blobs['data'].reshape(1,1,112,112) 
    num=0
    detected_points = []
    ground_truth = []
    align_errors = []

    for imagefile in imageList:
        src_img = cv2.imread(imagefile)
        #src_img = caffe.io.load_image(imagefile_abs)
        im = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        
        [img_h,img_w,img_c] = src_img.shape
        roi = [img_w,img_h,0,0]
        keypoints = []
        fp = open(imagefile[0:-3]+"txt","r")
        for i in range(pts_num):
            s_line = fp.readline()
            sub_str = s_line.split()
            pts = np.array([float(x) for x in sub_str])
            keypoints.append(pts[0:2])
        fp.close()
        ground_truth.append(keypoints)
        # calculate inital face bounding box
        fp = open(imagefile[0:-3]+"rect","r")
        s_line = fp.readline()
        s_line = fp.readline()
        sub_str = s_line.split()
        roi = np.array([float(x) for x in sub_str])
        fp.close()
        ctr_x = 0.5*roi[2] + roi[0]
        ctr_y = 0.5*roi[3] + roi[1]
        w = roi[2]
        h = roi[3]
        ex_scale = 1.1*0.5
        roi[0] = ctr_x-ex_scale*w
        roi[2] = ctr_x+ex_scale*w
        roi[1] = ctr_y-ex_scale*h #0.6
        roi[3] = ctr_y+ex_scale*h
        h = 1.1*h
        w = 1.1*w
        roi = np.array([int(x) for x in roi])
        if roi[0]<0:
           roi[0] = 0
        if roi[1]<0:
           roi[1] = 0
        if roi[2]>img_w:
           roi[2] = img_w
        if roi[3]>img_h:
           roi[3] = img_h
        roi_im = im[roi[1]:roi[3],roi[0]:roi[2]]
        im = cv2.resize(roi_im,(112,112))
        #cv2.imwrite("roi.jpg",im)
        #pdb.set_trace()
        # extract feature
        scale_x = float(roi[2]-roi[0])
        scale_y = float(roi[3]-roi[1])
        net.blobs['data'].data[...] = im
        st_time = time.time()
        out = net.forward()
        time_elp = time.time()-st_time
        print("\nforward time cost: %fs\n"%time_elp)
        feature = out[postfix][0].tolist();
        pre_points = []
        errors = []
        for i in range(pts_num):
           #print("(%d %f %f)"%(i,feature[2*i],feature[2*i+1]))

           tmp_x = (0.5*feature[2*i]+std_points[2*i])*scale_x + roi[0]
           tmp_y = (0.5*feature[2*i+1]+std_points[2*i+1])*scale_y + roi[1]
           pre_points.append([tmp_x, tmp_y])
           errors.append(LA.norm(pre_points[i]-keypoints[i]))       
        #pdb.set_trace()
        detected_points.append(pre_points)
        # computer errors
        interocular_distance = math.sqrt(h*w)
        #interocular_distance = LA.norm(keypoints[104]-keypoints[105])
        dsum = sum(errors)
        align_errors.append(dsum/(pts_num*interocular_distance))
        #print(align_errors[num])
        num +=1
        print( 'Num ',num)
        if align_errors[num-1]>0.08:
           print(align_errors[num-1])
        #vis_detections(src_img, pre_points, keypoints, num)
        #show(src_img, pre_points, keypoints)
        #pdb.set_trace()
    return align_errors
        #with  open(fea_file,'wb') as f:
        #    for x in xrange(0, net.blobs['fc6'].data.shape[0]):
        #        for y in xrange(0, net.blobs['fc6'].data.shape[1]):
        #            f.write(struct.pack('f', net.blobs['fc6'].data[x,y]))

def readImageList(imageListFile):
    imageList = []
    with open(imageListFile,'r') as fi:
        while(True):
            line = fi.readline().strip()
            if not line:
                break
            imageList.append(line) 
    print( 'read imageList done image num ', len(imageList))
    return imageList


def drawROC( align_errors):
    nme_x = 0.08
    align_errors_sort = sorted(align_errors)
    align_errors_sort.append(1)
    num = len(align_errors_sort)
    tmp = range(1,num)
    p = [elem*100/num for elem in tmp]
    p.append(100)

    mean_error = np.mean(align_errors)
    print( 'mean_error: ',mean_error)
    ######## AUC ###########
    error_bins = []
    acc_bins = []
    for i in range(num-1):
        error_bins.append(align_errors_sort[i+1] - align_errors_sort[i])
        acc_bins.append(error_bins[i]*0.5*(p[i]+p[i+1]))
    #acc_bins = np.array(acc_bins)
    mm = abs(np.array(align_errors_sort) - nme_x)
    idx_array = np.where(mm==np.min(mm))
    idx = idx_array[0][0]
    AUC = 0
    temp_idx = idx
    if (nme_x - align_errors_sort[temp_idx])>0:
        while error_bins[temp_idx] == 0:
             temp_idx = temp_idx + 1

        AUC = np.sum(acc_bins[0:temp_idx]) + (nme_x - align_errors_sort[temp_idx])*acc_bins[temp_idx]/error_bins[temp_idx]
    else:
        while error_bins[temp_idx-1] == 0:
              temp_idx = temp_idx - 1

        AUC = np.sum(acc_bins[0:temp_idx]) + (nme_x - align_errors_sort[temp_idx])*acc_bins[temp_idx-1]/error_bins[temp_idx-1]

    #if (nme_x - align_errors_sort[idx])>0:
    #   AUC = np.sum(acc_bins[0:idx]) + (nme_x - align_errors_sort[idx])*acc_bins[idx]/error_bins[idx]
    #else:
    #   AUC = np.sum(acc_bins[0:idx]) + (nme_x - align_errors_sort[idx])*acc_bins[idx-1]/error_bins[idx-1]
    AUC = AUC/(100*nme_x)
    print("AUC(@%0.2f): %f\n Failure rate: %f" %( nme_x, AUC, 100-p[idx]))

    ####### draw roc curve#######
    plt.figure()
    plt.figure(figsize=(8,8))
    plt.plot(align_errors_sort, p, color='darkorange',
         lw=2, label='ROC curve (mean_errors = %0.4f)' % mean_error)
    plt.xlim([0.0, nme_x])
    plt.ylim([0.0, 100])
    plt.xlabel('Error metric')
    plt.ylabel('Cumulative correct rate')
    plt.grid(color='k', linestyle='--', linewidth=1)
    plt.title('Performance in testing phase')
    plt.legend(loc="lower right")
    plt.savefig('/data1/hanfy/landmark_untouch/landmark_data/eval_results/0117-1446/CCR_curve.png')
    plt.show()
    

def saveData(name,data):
    fileObject = open(name, 'w')  
    for ip in data:  
        fileObject.write('%f\n'%(ip))  
    fileObject.close()


if  __name__ == "__main__":
    net = initilize()
    imageList = readImageList(imageListFile) 
    align_errors = extractFeature(imageList, net)
    saveData("/data1/hanfy/landmark_untouch/landmark_data/eval_results/0117-1446/pts_errors.txt",align_errors)
    drawROC( align_errors)
