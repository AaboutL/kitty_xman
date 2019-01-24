import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from projects.landmark_82.layers import AffineTransformLayer, TransformParamsLayer, LandmarkImageLayer, LandmarkTransformLayer

def prelu(input, name):
    with tf.variable_scope(name):
        i = input.get_shape().as_list()
        alpha = tf.get_variable('alpha', shape=(i[-1]))
        one_tensor = ops.convert_to_tensor(tf.ones(i[0:-1]+[1]))
        output = tf.nn.relu(input) + tf.multiply(one_tensor * alpha, -tf.nn.relu(-input))
    return output

tf.extract_image_patches
def inference(inputs, meanshape, pts_num, is_training=True, dropout_keep_prob=0.5, scope='tiny', global_pool=False):
    meanShape = tf.constant(meanshape)
    with tf.variable_scope('Stage_1'):
        net = slim.conv2d(inputs, 8, [5, 5], stride=2, padding='VALID', activation_fn=None, scope='conv1_1')
        # net = prelu(net, 'relu_conv1_1')
        net = tf.nn.leaky_relu(net, name='relu_conv1_1')
        print('conv1_1: ', net.shape)
        net = slim.avg_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool1')
        print('pool1: ', net.shape)
        net = slim.conv2d(net, 16, [3, 3], stride=1, padding='VALID', activation_fn=None, scope='conv2_1')
        # net = prelu(net, 'relu_conv2_1')
        net = tf.nn.leaky_relu(net, name='relu_conv2_1')
        print('conv2_1: ', net.shape)
        net = slim.conv2d(net, 16, [3, 3], stride=1, padding='VALID', activation_fn=None, scope='conv2_2')
        # net = prelu(net, 'relu_conv2_2')
        net = tf.nn.leaky_relu(net, name='relu_conv2_2')
        print('conv2_2: ', net.shape)
        net = slim.avg_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool2')
        print('pool2: ', net.shape)
        net = slim.conv2d(net, 24, [3, 3], stride=1, padding='VALID', activation_fn=None, scope='conv3_1')
        # net = prelu(net, 'relu_conv3_1')
        net = tf.nn.leaky_relu(net, name='relu_conv3_1')
        print('conv3_1: ', net.shape)
        net = slim.conv2d(net, 24, [3, 3], stride=1, padding='VALID', activation_fn=None, scope='conv3_2')
        # net = prelu(net, 'relu_conv3_2')
        net = tf.nn.leaky_relu(net, name='relu_conv3_2')
        print('conv3_2: ', net.shape)
        net = slim.avg_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool3')
        print('pool3: ', net.shape)
        net = slim.conv2d(net, 40, [3, 3], stride=1, padding='SAME', activation_fn=None, scope='conv4_1')
        # net = prelu(net, 'relu_conv4_1')
        net = tf.nn.leaky_relu(net, name='relu_conv4_1')
        print('conv4_1: ', net.shape)
        net = slim.conv2d(net, 80, [3, 3], stride=1, padding='SAME', activation_fn=None, scope='conv4_2')
        # net = prelu(net, 'relu_conv4_2')
        net = tf.nn.leaky_relu(net, name='relu_conv4_2')
        print('conv4_2: ', net.shape)
        net = slim.flatten(net, scope='flatten')
        print('flatten: ', net.shape)
        net = slim.fully_connected(net, 128, activation_fn=None, scope='mimic_ip1')
        # net = prelu(net, 'relu_ip1')
        net = tf.nn.leaky_relu(net, name='relu_ip1')
        print('fc1: ', net.shape)
        net = slim.fully_connected(net, 128, activation_fn=None, scope='mimic_ip2')
        # net = prelu(net, 'relu_ip2')
        net = tf.nn.leaky_relu(net, name='relu_ip2')
        print('fc2: ', net.shape)
        net = slim.fully_connected(net, pts_num*2, scope='pts82')
        net = net + meanShape
    return net


