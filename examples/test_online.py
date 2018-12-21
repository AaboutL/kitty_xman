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

os.environ['CUDA_VISIBLE_DEVICES']=''

def main(args):
    dset = dataset.Dataset()
    dset.get_datalist(args.dataset_dir,['png', 'jpg'])
    image_set, points_set = dset.gether_data(is_bbox_aug=False)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # saver = tf.train.Saver(max_to_keep=10)
            model_tool.load_model(sess, args.model)
            # model_tool.show_op_name()

            image_placeholder = tf.get_default_graph().get_tensor_by_name('Placeholder_1:0')
            training_placeholder = tf.get_default_graph().get_tensor_by_name('is_training:0')
            pts_pred = tf.get_default_graph().get_tensor_by_name('alexnet_v2/fc8/squeezed:0')
            print(len(image_set))
            for img in image_set:
                image = []
                image.append(img)
                start_time = time.time()
                result = sess.run(pts_pred, feed_dict={image_placeholder:image, training_placeholder:False})
                duration = time.time() - start_time
                print('%d images total cost %f, average cost %f' %(len(image_set), duration, duration/len(image_set)))

                visualize.show_points(image[0], result[0], dim=1)
                visualize.show_image(image[0], 'img')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='where is model stored',
                        # default='/home/public/nfs132_1/hanfy/models/pb_model/test.pb')
                        default='/home/public/nfs132_1/hanfy/models/fine_model')
    parser.add_argument('--dataset_dir', type=str, help='dataset for test',
                        default='/home/public/nfs72/face/ibugs/lfpw/testset')

    args = parser.parse_args()
    main(args)
