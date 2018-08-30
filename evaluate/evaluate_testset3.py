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
from evaluate import landmark_eval
from utilities.tfrecord import read_tfrecord
import cv2

os.environ['CUDA_VISIBLE_DEVICES']=''

def main(args):
    dset = dataset.Dataset()
    dset.get_datalist(args.dataset_dir,['png', 'jpg'])
    image_set, points_set = dset.gether_data(is_bbox_aug=True)
    print('image_set len %d' %len(image_set))

    with tf.Graph().as_default():
        with tf.Session() as sess:
            model_tool.load_model(sess, args.model)
            image_batch, points_batch = read_tfrecord.convert_from_tfrecord('/home/public/nfs132_1/hanfy/align/ibugs/validationset_bbox_auged.record', 64, 1, is_preprocess=False)

            # image_input = tf.get_default_graph().get_tensor_by_name('IteratorGetNext:0')
            image_input = tf.get_default_graph().get_tensor_by_name('image_input:0')
            training_placeholder = tf.get_default_graph().get_tensor_by_name('is_training:0')
            pts_pred = tf.get_default_graph().get_tensor_by_name('alexnet_v2/fc8/squeezed:0')

            start_time = time.time()
            for i in range(len(image_set)):
                image = image_set[i]
                pts = points_set[i]
                pts = np.reshape(pts, [68, 2])
                image_in = np.reshape(image, [1, 224, 224, 3])
                res = sess.run(pts_pred, feed_dict={image_input:image_in, training_placeholder:False})
                print(np.shape(res))
                print(res)
                res = np.reshape(res, [68, 2])
                error = np.mean(np.sqrt(np.sum(np.subtract(res, pts))))
                visualize.show_points(image, res, dim=2)
                visualize.show_points(image, pts, dim=2, color=(0,0, 255))
                cv2.putText(image_set[i], str(error), (40, 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0))
                visualize.show_image(image, 'test', 0)
            duration = time.time() - start_time
            print('%d images total cost %f, average cost %f' %(len(image_set), duration, duration/len(image_set)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='where is model stored',
                        # default='/home/public/nfs132_1/hanfy/models/pb_model/test.pb')
                        default='/home/public/nfs132_1/hanfy/models/align_model/model_wingloss_0829')
    parser.add_argument('--dataset_dir', type=str, help='dataset for test',
                        default='/home/public/nfs72/face/ibugs/lfpw/testset')

    args = parser.parse_args()
    main(args)

