from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def inference(inputs,
               num_classes=136,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2',
               global_pool=False):
    with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.batch_norm(slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1'))
            print('conv1 shape:', net.get_shape().as_list())
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
            print('pool1 shape:', net.get_shape().as_list())
            net = slim.batch_norm(slim.conv2d(net, 192, [5, 5], scope='conv2'))
            print('conv2 shape:', net.get_shape().as_list())
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            print('pool2 shape:', net.get_shape().as_list())
            net = slim.batch_norm(slim.conv2d(net, 384, [3, 3], scope='conv3'))
            print('conv3 shape:', net.get_shape().as_list())
            net = slim.batch_norm(slim.conv2d(net, 384, [3, 3], scope='conv4'))
            print('conv4 shape:', net.get_shape().as_list())
            net = slim.batch_norm(slim.conv2d(net, 256, [3, 3], scope='conv5'))
            print('conv5 shape:', net.get_shape().as_list())
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')
            print('pool5 shape:', net.get_shape().as_list())

            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                                  scope='fc6')
                print('fc6 shape:', net.get_shape().as_list())
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                print('fc7 shape:', net.get_shape().as_list())
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='dropout7')
                    net = slim.conv2d(net, num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      biases_initializer=tf.zeros_initializer(),
                                      scope='fc8')
                    print('fc8 shape:', net.get_shape().as_list())
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                        print('fc8/squeezed shape:', net.get_shape().as_list())
                    end_points[sc.name + '/fc8'] = net
            return net, end_points
inference.default_image_size = 224
