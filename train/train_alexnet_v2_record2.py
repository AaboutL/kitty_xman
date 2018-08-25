from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
import numpy as np
import argparse
import time

from utilities import dataset
from net_archs import AlexNet
from utilities.tfrecord import read_tfrecord
from train import loss_func
from evaluate import landmark_eval

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'

def main(args):
    with tf.Graph().as_default():
        # placeholders
        is_training = tf.placeholder(tf.bool, name='is_training')

        # prepare data
        val_queue = tf.train.string_input_producer([args.val_file], num_epochs=1)
        image_val, pts_val = read_tfrecord.read_and_decode(val_queue)
        train_queue = tf.train.string_input_producer([args.input_file], num_epochs=args.num_epochs)
        image_batch, pts_batch = read_tfrecord.read_and_decode(train_queue)
        # image_batch = tf.image.per_image_standardization(image_batch)
        image_batch = tf.identity(image_batch, 'image_input')
        pts_batch_ph = tf.identity(pts_batch, 'pts_input')

        # construct loss
        inference, _ = AlexNet.alexnet_v2(image_batch, args.num_landmarks*2, is_training, args.dropout_keep_prob)
        loss = tf.reduce_mean(loss_func.NormRmse(GroudTruth=pts_batch_ph, Prediction=inference, num_points=args.num_landmarks))
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'alexnet_v2'))

        Saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
        Writer = tf.summary.FileWriter(args.log_dir, tf.get_default_graph())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            try:
                while True:
                    start_time = time.time()
                    _, lm_loss = sess.run([optimizer, loss],
                                          feed_dict={is_training : args.is_training})
                    duration = time.time() - start_time
                    print('step: [%d]\tTime %.3f\tLoss %2.3f' %(step, duration, lm_loss))
                    step += 1
                    if step % 100 == 0:
                        pts_pred, pts_gt = sess.run([inference, pts_batch], feed_dict={is_training: False, image_batch:image_val})
                        landmark_eval.landmark_error(pts_gt, pts_pred)
                        Saver.save(sess, args.model_dir + '/model', global_step=step)
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (args.num_epochs, step))
            Writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='path to the dataset',
                        default='/home/public/nfs132_1/hanfy/align/ibugs/trainset.record')
    parser.add_argument('--val_file', type=str, help='validation file',
                        default='/home/public/nfs132_1/hanfy/align/ibugs/validationset.record')
    parser.add_argument('--num_landmarks', type=int, help='number of landmarks on a face',
                        default=68)
    parser.add_argument('--is_training', type=bool, help='which mode, training or inference',
                        default=True)
    parser.add_argument('--batch_size', type=int, help='size of a batch',
                        default=64)
    parser.add_argument('--num_epochs', type=int, help='how many epoches should train',
                        default=100)
    parser.add_argument('--epoch_size', type=int, help='how many batches in one epoch',
                        default=1000)
    parser.add_argument('--log_dir', type=str, help='Directory to the log file',
                        default='/home/public/nfs132_1/hanfy/logs/log_0825_pm')
    parser.add_argument('--model_dir', type=str, help='Director to the model file',
                        default='/home/public/nfs132_1/hanfy/models/align_model/model_0825_pm')
    parser.add_argument('--dropout_keep_prob', type=float, help='dropout rate',
                        default=0.5)

    args = parser.parse_args()
    main(args)

