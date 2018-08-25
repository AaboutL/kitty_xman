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
    dset = dataset.Dataset()
    dset.get_datalist(args.dataset_dir,['png', 'jpg'])
    image_set, points_set = dset.gether_data(is_bbox_aug=False)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            model_tool.load_model(sess, args.model)

            image_input = tf.get_default_graph().get_tensor_by_name('Placeholder_1:0')
            training_placeholder = tf.get_default_graph().get_tensor_by_name('is_training:0')
            pts_pred = tf.get_default_graph().get_tensor_by_name('alexnet_v2/fc8/squeezed:0')
            print(len(image_set))

            start_time = time.time()
            results = sess.run(pts_pred, feed_dict={image_input:image_set, training_placeholder:False})
            duration = time.time() - start_time
            print('%d images total cost %f, average cost %f' %(len(image_set), duration, duration/len(image_set)))

            results = np.reshape(results, [-1, 68, 2])
            points_set = np.reshape(points_set, [-1, 68, 2])
            norm_errors, errors = landmark_eval.landmark_error(points_set, results, show_results=True)
            # landmark_eval.auc_error(norm_errors, 1, showCurve=True)
            for i in range(len(results)):
                visualize.show_points(image_set[i], results[i], dim=2, color=(0, 0, 255))
                visualize.show_points(image_set[i], points_set[i], dim=2)
                cv2.putText(image_set[i], str(errors[i]), (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
                visualize.show_image(image_set[i], 'pts', 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='where is model stored',
                        default='/home/public/nfs132_1/hanfy/models/pb_model/test.pb')
                        # default='/home/public/nfs132_1/hanfy/models/fine_model')
    parser.add_argument('--dataset_dir', type=str, help='dataset for test',
                        default='/home/public/nfs72/face/ibugs/lfpw/testset')

    args = parser.parse_args()
    main(args)
