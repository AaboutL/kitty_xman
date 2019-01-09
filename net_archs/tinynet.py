import tensorflow as tf
import tensorflow.contrib.slim as slim

def inference(inputs, pts_num, is_training=True, dropout_keep_prob=0.5, scope='tiny', global_pool=False):
    with tf.variable_scope(scope, default_name='tiny', values=[inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 8, [5, 5], stride=2, padding='VALID', activation_fn=tf.nn.leaky_relu, scope='conv1_1')
            print('conv1_1: ', net.shape)
            net = slim.avg_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool1')
            print('pool1: ', net.shape)
            net = slim.conv2d(net, 16, [3, 3], stride=1, padding='VALID', activation_fn=tf.nn.leaky_relu, scope='conv2_1')
            print('conv2_1: ', net.shape)
            net = slim.conv2d(net, 16, [3, 3], stride=1, padding='VALID', activation_fn=tf.nn.leaky_relu, scope='conv2_2')
            print('conv2_2: ', net.shape)
            net = slim.avg_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool2')
            print('pool2: ', net.shape)
            net = slim.conv2d(net, 24, [3, 3], stride=1, padding='VALID', activation_fn=tf.nn.leaky_relu, scope='conv3_1')
            print('conv3_1: ', net.shape)
            net = slim.conv2d(net, 24, [3, 3], stride=1, padding='VALID', activation_fn=tf.nn.leaky_relu, scope='conv3_2')
            print('conv3_2: ', net.shape)
            net = slim.avg_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool3')
            print('pool3: ', net.shape)
            net = slim.conv2d(net, 40, [3, 3], stride=1, padding='VALID', activation_fn=tf.nn.leaky_relu, scope='conv4_1')
            print('conv4_1: ', net.shape)
            net = slim.conv2d(net, 40, [3, 3], stride=1, padding='VALID', activation_fn=tf.nn.leaky_relu, scope='conv4_2')
            print('conv4_2: ', net.shape)
            net = slim.flatten(net, scope='flatten')
            print('flatten: ', net.shape)
            net = slim.fully_connected(net, 128, activation_fn=tf.nn.leaky_relu, scope='fc1')
            print('fc1: ', net.shape)
            net = slim.fully_connected(net, 128, activation_fn=tf.nn.leaky_relu, scope='fc2')
            print('fc2: ', net.shape)
            net = slim.fully_connected(net, pts_num)
        return net


