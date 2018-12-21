from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import argparse
import os
import numpy as np
import time

from utilities import dataset
from utilities import model_tool
from utilities import visualize
from evaluate import eval_tools
from utilities.tfrecord import read_tfrecord
import cv2

os.environ['CUDA_VISIBLE_DEVICES']=''

def main(args):
    dset = dataset.Dataset()
    dset.get_datalist(args.dataset_dir,['png', 'jpg'])
    image_set, points_set , _= dset.gether_data(is_bbox_aug=True)
    print('image_set len %d' %len(image_set))

    shape_mean = np.loadtxt('/home/hanfy/workspace/DL/alignment/align_untouch/shape_mean.txt', delimiter=' ')
    shape_std = np.loadtxt('/home/hanfy/workspace/DL/alignment/align_untouch/shape_std.txt', delimiter=' ')
    print('mean', shape_mean)
    print('std', shape_std)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            model_tool.load_model(sess, args.model)
            # model_tool.show_op_name()
            # exit(0)
            image_batch, points_batch = read_tfrecord.convert_from_tfrecord('/home/public/nfs132_1/hanfy/align/ibugs/validationset_bbox_auged.record', 64, 1, is_preprocess=False)

            # image_input = tf.get_default_graph().get_tensor_by_name('IteratorGetNext:0')
            image_input = tf.get_default_graph().get_tensor_by_name('image_input:0')
            training_placeholder = tf.get_default_graph().get_tensor_by_name('is_training:0')
            pts_pred = tf.get_default_graph().get_tensor_by_name('alexnet_v2/fc8/squeezed:0')

            start_time = time.time()
            image_set, points_set = sess.run([image_batch, points_batch])
            results = sess.run([pts_pred], feed_dict={image_input:image_set, training_placeholder:False})
            duration = time.time() - start_time
            print('%d images total cost %f, average cost %f' %(len(image_set), duration, duration/len(image_set)))

            points_set = np.reshape(points_set, [-1, 68, 2])
            results = np.reshape(results, [-1, 68, 2])
            # results = np.multiply(np.sum(np.multiply(results, model_tool.std), model_tool.mean), 224)

            # norm_errors, errors = landmark_eval.landmark_error(points_set, results, show_results=True)
            # norm_errors, errors = landmark_eval.landmark_error(points_set, points_set, show_results=True)
            # landmark_eval.auc_error(norm_errors, 0.2, showCurve=True)

            for i in range(len(image_set)):
                print('pts: ', len(results[i]))
                pred_shape = np.multiply(np.add(np.multiply(results[i], shape_std), shape_mean) + 0.5, 224.0)

                # visualize.show_points(image_set[i], results[i], dim=2)
                visualize.show_points(image_set[i], pred_shape, dim=2)
                # visualize.show_points(image_set[i], points_set[i], dim=2)
                visualize.show_points(image_set[i], points_set[i], dim=2, color=(0,0, 255))
                # cv2.putText(image_set[i], str(errors[i]), (40, 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0))
                visualize.show_image(image_set[i], 'test', 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='where is model stored',
                        # default='/home/public/nfs132_1/hanfy/models/pb_model/test.pb')
                        default='/home/public/nfs132_1/hanfy/models/align_model/model_0901_pm')
                        # default='/home/public/nfs132_1/hanfy/models/align_model/model_wingloss')
    parser.add_argument('--dataset_dir', type=str, help='dataset for test',
                        default='/home/public/nfs72/face/ibugs/lfpw/testset')

    args = parser.parse_args()
    main(args)

