#coding=utf-8
import numpy as np
import tensorflow as tf
import os
import sys
import argparse
from net_archs.tinynet_prelu import inference


os.environ['CUDA_VISIBLE_DEVICES'] = ''
def main(args):
    if args.input == '' or args.output == '':
        print('input and output file should not empty!')
        return
    img_size = args.img_size

    meanShape = np.genfromtxt('../meanshape_untouch.txt')
    meanShape = np.reshape(meanShape, [164]).astype(np.float32) * img_size
    with tf.Graph().as_default():
        imgs_ph = tf.placeholder(tf.float32, [None, img_size, img_size, 1], 'images_ph')
        is_train_ph = tf.placeholder(tf.bool, name='is_train')
        net = inference(imgs_ph, meanShape, pts_num=82, is_training=is_train_ph)
        saver = tf.train.Saver(tf.trainable_variables())
        vars = tf.trainable_variables()
        for var in vars:
            print(var.name[:-2])
        with tf.Session() as sess:
            data_dict = np.load(args.input, encoding='latin1').item()
            for op_name in data_dict:
                with tf.variable_scope(op_name, reuse=tf.AUTO_REUSE):
                    print('op_name %s'%op_name)
                    for param_name, data in data_dict[op_name].items():
                        print('param_name: %s:\n data: %r'%(param_name, data))
                        try:
                            if (op_name + '/' + param_name) not in [var.name[:-2] for var in vars]:
                                print(param_name)
                                continue
                            var = tf.get_variable(param_name)
                            print(var.get_shape())
                            print(data.shape)
                            sess.run(var.assign(data))
                        except ValueError:
                            if not args.ignore_missing:
                                raise

            print('\n*******************************\n')
            for var in tf.trainable_variables():
                print(sess.run(var))
            # exit(0)
            saver.save(sess, args.output + '/model')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=116)
    parser.add_argument('--input', type=str, default='../models/landmark_82_caffe/converted_params.npy')
    parser.add_argument('--output', type=str, default='../models/tmp')
    parser.add_argument('--ignore_missing', type=float, default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))