#coding=utf-8
import numpy as np
import tensorflow as tf
import os
import sys
import argparse
from net_archs.tinynet_prelu import inference

def main(args):
    if args.input == '' or args.output == '':
        print('input and output file should not empty!')
        return

    meanShape = np.genfromtxt('../meanshape_untouch.txt')
    meanShape = np.reshape(meanShape, [164]).astype(np.float32) * 112
    with tf.Graph().as_default():
        imgs_ph = tf.placeholder(tf.float32, [64, 112, 112, 1], 'images_ph')
        is_train_ph = tf.placeholder(tf.bool, name='is_train')
        net = inference(imgs_ph, meanShape, pts_num=82, is_training=is_train_ph)
        saver = tf.train.Saver(tf.trainable_variables())
        with tf.Session() as sess:
            data_dict = np.load(args.input, encoding='latin1').item()
            for op_name in data_dict:
                with tf.variable_scope(op_name, reuse=True):
                    print('op_name %s'%op_name)
                    for param_name, data in data_dict[op_name].items():
                        print('param_name: %s:\n data: %r'%(param_name, data))
                        try:
                            var = tf.get_variable(param_name)
                            sess.run(var.assign(data))
                        except ValueError:
                            if not args.ignore_missing:
                                raise

            print('\n*******************************\n')
            for var in tf.trainable_variables():
                print(sess.run(var))
            saver.save(sess, args.output + '/model')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../models/landmark_82_caffe/converted_params.npy')
    parser.add_argument('--output', type=str, default='/home/slam/workspace/DL/alignment_method/align_untouch/models')
    parser.add_argument('--ignore_missing', type=float, default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))