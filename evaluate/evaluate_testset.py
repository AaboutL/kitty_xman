from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import argparse
import os
import numpy as np
import time
import cv2

from utilities import dataset
from utilities import model_tool
from utilities import visualize
from evaluate import landmark_eval

os.environ['CUDA_VISIBLE_DEVICES']=''

def main(args):
    os.makedirs(args.result_dir, exist_ok=True)
    img_result_dir = os.path.join(args.result_dir, 'img_result')
    os.makedirs(img_result_dir, exist_ok=True)

    dset = dataset.Dataset()
    # dset.get_datalist(args.dataset_dir,['png', 'jpg'])
    # image_set, points_set, _ = dset.gether_data(is_bbox_aug=False)
    # image_set, points_set = dset.read_hdf5(args.test_file)

    # image_set, points_set = dset.read_hdf5('/home/public/nfs132_1/hanfy/align/ibugs/trainset.hdf5')
    # points_set = points_set.reshape(len(points_set), 68*2)

    tmp_dir = '/home/public/nfs132_1/hanfy/align/ibugs/testset'
    dset = dataset.Dataset()
    dset.get_datalist(tmp_dir, ['png', 'jpg'])
    image_set, points_set, _ = dset.gether_data()

    with tf.Graph().as_default():
        with tf.Session() as sess:
            model_tool.load_model(sess, args.model)
            # model_tool.show_op_name()

            image_input = tf.get_default_graph().get_tensor_by_name('image_ph:0')
            training_placeholder = tf.get_default_graph().get_tensor_by_name('is_training:0')
            pts_pred = tf.get_default_graph().get_tensor_by_name('alexnet_v2/fc8/squeezed:0')
            print(len(image_set))

            errors = []
            for i in range(len(image_set)):
                image = []
                img = image_set[i]
                image.append(img)
                start_time = time.time()
                result = sess.run(pts_pred, feed_dict={image_input:image, training_placeholder:False})
                duration = time.time() - start_time
                print('%d images total cost %f, average cost %f' %(len(image_set), duration, duration/len(image_set)))
                res = np.reshape(result, [68, 2])
                pts = points_set[i]
                pts = np.reshape(pts, [68, 2])
                # error = np.mean(np.sum(np.sqrt(np.subtract(res, pts))))
                error = np.mean(np.sqrt(np.sum((pts - res)**2, axis=1)))
                errors.append(error)
                visualize.show_points(img, res, dim=2, color=(0,0, 255))
                visualize.show_points(img, pts, dim=2)
                cv2.putText(img, str(error), (40, 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0))
                # visualize.show_image(img, 'test', 30)
                # cv2.imwrite(args.result_dir + '/' + str(i) + '.jpg', img)
                cv2.imwrite(os.path.join(img_result_dir, str(i) + '.jpg'), img)

            mean_error = np.mean(errors)
            print('mean_error', mean_error)
            auc, failure_rate = landmark_eval.auc_error(errors, args.failure_threshold, save_path=os.path.join(args.result_dir, 'auc.jpg'))
            with open(args.result, 'w+') as fp:
                fp.write("mean error: {0}".format(mean_error) + '\n')
                fp.write("AUC @ {0}: {1}".format(args.failure_threshold, auc) + '\n')
                fp.write("Failure rate: {0}".format(failure_rate) + '\n')

            # results = np.reshape(results, [-1, 68, 2])
            # points_set = np.reshape(points_set, [-1, 68, 2])
            # norm_errors, errors = landmark_eval.landmark_error(points_set, results, show_results=True)
            # landmark_eval.auc_error(norm_errors, 1, showCurve=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='where is model stored',
                        # default='/home/public/nfs132_1/hanfy/models/pb_model/test.pb')
                        default='/home/public/nfs132_1/hanfy/models/align_model/model_0904_h5')
    parser.add_argument('--dataset_dir', type=str, help='dataset for test',
                        default='/home/public/nfs132_1/hanfy/align/ibugs/testset')
    parser.add_argument('--test_file', type=str, help='path to the validation set',
                        default='/home/public/nfs132_1/hanfy/align/ibugs/testset.hdf5')
                        # default='/home/public/nfs132_1/hanfy/align/ibugs/tmpset.hdf5')
    parser.add_argument('--result_dir', type=str, help='save image with result',
                        default='/home/public/nfs132_1/hanfy/results/0904_alex_l1_bbox_flip')
    parser.add_argument('--result', type=str, help='save test result',
                        default='/home/public/nfs132_1/hanfy/results/0904_alex_l1_bbox_flip/result.txt')
    parser.add_argument('--failure_threshold', type=float, default=10.0)

    args = parser.parse_args()
    main(args)

