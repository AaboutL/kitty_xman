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

def NormRmse(GroudTruth, Prediction, num_points):
    Gt = tf.reshape(GroudTruth, [-1, num_points, 2])
    Pt = tf.reshape(Prediction, [-1, num_points, 2])
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(Gt, Pt), 2)), 1)
    # norm = tf.sqrt(tf.reduce_sum(((tf.reduce_mean(Gt[:, 36:42, :],1) - \
    #     tf.reduce_mean(Gt[:, 42:48, :],1))**2), 1))
    norm = tf.norm(tf.reduce_mean(Gt[:, 36:42, :],1) - tf.reduce_mean(Gt[:, 42:48, :],1), axis=1)
    # cost = tf.reduce_mean(loss / norm)

    return loss/norm


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='path to the dataset',
                        default='/home/hanfy/workspace/DL/alignment/align_untouch/tmpdata/output.hdf5')
    parser.add_argument('--num_landmarks', type=int, help='number of landmarks on a face',
                        default=68)
    parser.add_argument('--is_training', type=bool, help='which mode, training or inference',
                        default=True)
    parser.add_argument('--batch_size', type=int, help='size of a batch',
                        default=64)
    parser.add_argument('--epoches', type=int, help='how many epoches should train',
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

    config = tf.ConfigProto()

    dset = dataset.Dataset()
    image_set, points_set= dset.read(args.input_file)
    points_set = points_set.reshape((len(points_set), 68*2))

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        lr = tf.placeholder(tf.float32)
        images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        points_gt = tf.placeholder(tf.float32, [None, args.num_landmarks*2])
        is_training = tf.placeholder(tf.bool, name='is_training')

        # construct loss
        inference, _ = AlexNet.alexnet_v2(images, args.num_landmarks*2, is_training, args.dropout_keep_prob)
        loss = tf.reduce_mean(NormRmse(GroudTruth=points_gt, Prediction=inference, num_points=args.num_landmarks*2))
        tf.summary.scalar('landmark_loss', loss)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'alexnet_v2')):
            optimizer = tf.train.AdamOptimizer(0.001).minimize(loss,
                        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'alexnet_v2'))

        with tf.Session(config=config) as sess:
            Saver = tf.train.Saver()
            merged = tf.summary.merge_all()
            Writer = tf.summary.FileWriter(args.log_dir, sess.graph)
            tf.summary.image('input_image', images, 10)

            sess.run(tf.global_variables_initializer())
            batch_num = len(image_set) // args.batch_size
            for epoch in range(args.epoches):
                step = 0
                while step < args.epoch_size:
                    RandomIdx = np.random.choice(image_set.shape[0], args.batch_size, False)
                    start_time = time.time()
                    summary, _, lm_loss = sess.run([merged, optimizer, loss],
                                                         feed_dict={images : image_set[RandomIdx],
                                                                    points_gt : points_set[RandomIdx],
                                                                    is_training : args.is_training})
                    duration = time.time() - start_time
                    print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                          (epoch, step+1, args.epoch_size, duration, lm_loss))
                    step += 1
                Saver.save(sess, args.model_dir + '/model', global_step=step)
            Writer.close()


