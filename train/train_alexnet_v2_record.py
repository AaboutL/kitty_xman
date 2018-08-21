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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'

def NormRmse(GroudTruth, Prediction, num_points):
    Gt = tf.reshape(GroudTruth, [-1, num_points, 2])
    Pt = tf.reshape(Prediction, [-1, num_points, 2])
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(Gt, Pt), 2)), 1)
    norm = tf.norm(tf.reduce_mean(Gt[:, 36:42, :],1) - tf.reduce_mean(Gt[:, 42:48, :],1), axis=1)
    return loss/norm

def main(args):
    with tf.Graph().as_default():
        image_batch, points_batch = read_tfrecord.convert_from_tfrecord(args.batch_size, args.num_epochs, args.input_file)

        # placeholders
        is_training = tf.placeholder(tf.bool, name='is_training')

        # tf.summary.image('input_image', images, 10)

        # construct loss
        inference, _ = AlexNet.alexnet_v2(image_batch, args.num_landmarks*2, is_training, args.dropout_keep_prob)
        loss = tf.reduce_mean(NormRmse(GroudTruth=points_batch, Prediction=inference, num_points=args.num_landmarks))
        # tf.summary.scalar('landmark_loss', loss)
        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'alexnet_v2')):
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'alexnet_v2'))

        Saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
        # merged = tf.summary.merge_all()
        Writer = tf.summary.FileWriter(args.log_dir, tf.get_default_graph())
        # with tf.Session(config=config) as sess:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            try:
                batch_id = 0
                while True:
                    start_time = time.time()
                    _, lm_loss = sess.run([optimizer, loss],
                                          feed_dict={is_training : args.is_training})
                    duration = time.time() - start_time
                    print('step: [%d]\tTime %.3f\tLoss %2.3f' %
                          (step, duration, lm_loss))
                    batch_id += 1
                    step += 1
                    Saver.save(sess, args.model_dir + '/model', global_step=step)
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (args.num_epochs, step))
            Writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='path to the dataset',
                        default='/home/hanfy/workspace/DL/alignment/align_untouch/tmpdata/output.record')
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
                        default='/home/public/nfs132_1/hanfy/logs/align_log')
    parser.add_argument('--model_dir', type=str, help='Director to the model file',
                        default='/home/public/nfs132_1/hanfy/models/align_model')
    parser.add_argument('--dropout_keep_prob', type=float, help='dropout rate',
                        default=0.5)

    args = parser.parse_args()
    main(args)

