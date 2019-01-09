import tensorflow as tf
import tensorflow.contrib.slim as slim
from projects.STN.spatial_transformer import transformer
from utilities.data_preparation.save_read_tfrecord import load_tfrecord

from utilities.data_preparation.save_read_tfrecord import load_tfrecord
import argparse
import sys
import numpy as np

def inference(inputs, num_classes, is_training=True, dropout_keep_prob=0.5, scope='STN', global_pool=False):
    with tf.variable_scope(scope, 'stn', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d], outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 64, [3, 3], stride=2, padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = slim.conv2d(net, 64, [3, 3], stride=2, padding='VALID', scope='conv2')
            net = slim.flatten(net, scope='flatten')
            net = slim.fully_connected(net, 512, scope='fc1')
            net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc2')
    return net

def main(args):
    trainset = args.trainset
    validateset = args.validateset

    with tf.Graph().as_default():
        train_queue = tf.train.string_input_producer([trainset])
        validate_queue = tf.train.string_input_producer([validateset])
        images_train, thetas_train = load_tfrecord(train_queue, pts_num=3, img_shape=[224, 224, 1], batch_size=args.batch_size, is_shuffle=True)
        images_validate, thetas_validate = load_tfrecord(validate_queue, pts_num=3, img_shape=[224, 224, 1], batch_size=args.batch_size, is_shuffle=False)

        images = tf.placeholder(tf.float32, [None, 224, 224, 1], name='images_ph')
        thetas = tf.placeholder(tf.float32, [None, 6], name='thetas_ph')
        is_training = tf.placeholder(tf.bool, name='is_training')
        output_size = (224, 224)

        fc_loc = inference(images, 6, is_training)
        logits_r = thetas[:, 0:4]
        logits_t = thetas[:, 4:]
        fc_loc_r = fc_loc[:, 0:4]
        fc_loc_t = fc_loc[:, 4:]
        cross_entropy_r = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(fc_loc_r, logits_r))))
        cross_entropy_t = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(fc_loc_t, logits_t))))/112.0
        cross_entropy = cross_entropy_r + cross_entropy_t
        opt = tf.train.AdamOptimizer(learning_rate=0.01)
        optimizer = opt.minimize(cross_entropy)
        grads = opt.compute_gradients(cross_entropy)

        Saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            i = 0
            while True:
                i += 1
                if i % 100 == 0:
                    imgs_val, thetas_val = sess.run([images_validate, thetas_validate])
                    _, loss, output = sess.run([optimizer, cross_entropy, fc_loc], feed_dict={images:imgs_val,
                                                                              thetas: thetas_val,
                                                                              is_training: False})
                    print('Loss on validate set: %2.4f, output: %s'%(loss, output[0]))
                    Saver.save(sess, args.model_dir+ '/model', global_step=i)
                imgs_feed, thetas_feed = sess.run([images_train, thetas_train])
                _, loss, loss_r, loss_t = sess.run([optimizer, cross_entropy, cross_entropy_r, cross_entropy_t], feed_dict={images:imgs_feed,
                                                                          thetas: thetas_feed,
                                                                          is_training: True})
                print('steps: %d, loss: %2.4f, loss_r: %2.4f, loss_t: %2.4f'%(i, loss, loss_r, loss_t))

            coord.request_stop()
            coord.join(threads)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainset', type=str, default='/home/hanfy/dataset/STN/trans_train.record')
    parser.add_argument('--validateset', type=str, default='/home/hanfy/dataset/STN/trans_validate.record')
    parser.add_argument('--model_dir', type=str, default='/home/hanfy/models/STN')
    parser.add_argument('--batch_size', type=int, default='64')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

