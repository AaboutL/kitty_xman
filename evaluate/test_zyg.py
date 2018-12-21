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
from evaluate import eval_tools

os.environ['CUDA_VISIBLE_DEVICES']=''

def main(args):
    dset = dataset.Dataset()
    dset.get_datalist(args.dataset_dir,['png', 'jpg'])
    image_set, points_set = dset.gether_data(is_bbox_aug=False)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            model_tool.load_model(sess, args.model)
            model_tool.show_op_name()

            training_placeholder = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
            image_input = tf.get_default_graph().get_tensor_by_name('Placeholder_1:0')
            pts_pred = tf.get_default_graph().get_tensor_by_name('Squeeze:0')
            print(len(image_set))

            sub_imgs = image_set[:32]
            sub_pts = points_set[:32]
            results = sess.run(pts_pred, feed_dict={image_input:sub_imgs, training_placeholder:False})
            for i in range(len(sub_imgs)):
                img = sub_imgs[i]
                res = results[i]
                gt = sub_pts[i]
                error = np.mean(np.sqrt(np.sum(np.subtract(res, gt))))
                visualize.show_points(img, res, dim=1, color=(0, 0, 255))
                visualize.show_points(img, gt, dim=1, color=(0, 255, 0))
                cv2.putText(img, str(error), (40, 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0))
                visualize.show_image(img, 'test', 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='where is model stored',
                        # default='/home/public/nfs132_1/hanfy/models/pb_model/test.pb')
                        default='/home/zhaoyg/nfs132_1/alexnet/x0828/v2')
    parser.add_argument('--dataset_dir', type=str, help='dataset for test',
                        default='/home/public/nfs132_1/hanfy/align/ibugs/testset')

    args = parser.parse_args()
    main(args)

