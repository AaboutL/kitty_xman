from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
import numpy as np
import argparse
import time
import sys

sys.path.append(os.path.expanduser("../utilities"))

from utilities import dataset
from net_archs import AlexNet
from net_archs import AlexNet_BN as net
from utilities.tfrecord import read_tfrecord
from utilities import model_tool
from utilities import visualize
from evaluate import landmark_eval
import cv2

from train import loss_func

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def main(args):

    shape_mean = np.loadtxt('/home/hanfy/workspace/DL/alignment/align_untouch/shape_mean.txt', delimiter=' ')
    shape_std = np.loadtxt('/home/hanfy/workspace/DL/alignment/align_untouch/shape_std.txt', delimiter=' ')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    with tf.Graph().as_default():
        val_image, val_pts = read_tfrecord.convert_from_tfrecord(args.val_file, batch_size=448, is_preprocess=False, is_shuffle=False)

        image_batch, points_batch = read_tfrecord.convert_from_tfrecord(args.input_file, args.batch_size, args.num_epochs)
        image_batch = tf.identity(image_batch, 'image_input')
        points_batch = tf.identity(points_batch, 'pts_input')
        # placeholders
        is_training = tf.placeholder(tf.bool, name='is_training')

        # construct loss
        inference, _ = net.inference(image_batch, args.num_landmarks*2, is_training, args.dropout_keep_prob)
        loss = tf.reduce_mean(loss_func.NormRmse(gtLandmarks=points_batch, predLandmarks=inference, num_points=args.num_landmarks))
        # loss = loss_func.wing_loss(gtLandmarks=points_batch, predLandmarks=inference, num_points=args.num_landmarks)
        tf.summary.scalar('norm_loss', loss)
        optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'alexnet_v2'))
        Saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
        merged = tf.summary.merge_all()
        Writer = tf.summary.FileWriter(args.log_dir, tf.get_default_graph())
        with tf.Session() as sess:
            # test
            img_val, pts_val = sess.run([val_image, val_pts])

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            step = 0
            if args.pretrained_model_dir is not None:
                step = int(model_tool.load_model(sess, model_dir=args.pretrained_model_dir))

            try:
                while True:
                    start_time = time.time()
                    _, lm_loss, summary = sess.run([optimizer, loss, merged],
                                          feed_dict={is_training : args.is_training})
                    duration = time.time() - start_time
                    print('step: [%d]\tTime %.4f\tLoss %2.4f' %(step, duration, lm_loss))
                    Writer.add_summary(summary, step)
                    step += 1
                    if step % 100 == 0:
                        pred_shapes = sess.run([inference], feed_dict={image_batch:img_val, is_training:False})
                        pred_shapes = np.reshape(pred_shapes, [len(pts_val), 68, 2])
                        pred_shapes = np.multiply(np.add(np.multiply(pred_shapes, shape_std), shape_mean) + 0.5, 224.0)
                        pts_val = np.reshape(pts_val, [len(pts_val), 68, 2])
                        # for i in range(20):
                            # img = img_val[i].copy()
                            # visualize.show_points(img, pred_pts[i], color=(0, 0, 255))
                            # visualize.show_points(img, pts_val[i], color=(0, 255, 0))
                            # visualize.show_image(img, name='slave7', waitkey=100)
                        # cv2.destroyWindow('slave7')
                        landmark_eval.landmark_error(pts_val, pred_shapes)

                        Saver.save(sess, args.model_dir + '/model', global_step=step)
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (args.num_epochs, step))
            Writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='path to the dataset',
                        default='/home/public/nfs132_1/hanfy/align/ibugs/trainset_bbox_flip_norm.record')
    parser.add_argument('--val_file', type=str, help='validation file',
                        default='/home/public/nfs132_1/hanfy/align/ibugs/validationset_bbox.record')
    parser.add_argument('--num_landmarks', type=int, help='number of landmarks on a face',
                        default=68)
    parser.add_argument('--learning_rate', type=float, help='learning rate',
                        default=0.005)
    parser.add_argument('--is_training', type=bool, help='which mode, training or inference',
                        default=True)
    parser.add_argument('--batch_size', type=int, help='size of a batch',
                        default=64)
    parser.add_argument('--num_epochs', type=int, help='how many epoches should train',
                        default=500)
    parser.add_argument('--epoch_size', type=int, help='how many batches in one epoch',
                        default=1000)
    parser.add_argument('--log_dir', type=str, help='Directory to the log file',
                        default='/home/public/nfs132_1/hanfy/logs/log_0901_pm')
    parser.add_argument('--model_dir', type=str, help='Director to the model file',
                        default='/home/public/nfs132_1/hanfy/models/align_model/model_0901_pm')
    parser.add_argument('--pretrained_model_dir', type=str, help='Directory to the pretrain model')
                        # ,default='/home/public/nfs132_1/hanfy/models/align_model/model_wingloss_0829')
    parser.add_argument('--dropout_keep_prob', type=float, help='dropout rate',
                        default=0.5)

    args = parser.parse_args()
    main(args)

