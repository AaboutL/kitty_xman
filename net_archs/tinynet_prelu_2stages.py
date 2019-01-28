import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from projects.landmark_82.layers import AffineTransformLayer, TransformParamsLayer, LandmarkImageLayer, LandmarkTransformLayer, GetBBox
from train import loss_func

def prelu(input, name):
    with tf.variable_scope(name):
        i = input.get_shape().as_list()
        alpha = tf.get_variable('alpha', shape=(i[-1]))
        one_tensor = ops.convert_to_tensor(tf.ones(i[0:-1]+[1]))
        output = tf.nn.relu(input) + tf.multiply(one_tensor * alpha, -tf.nn.relu(-input))
    return output

def sub_net(inputs, name, pts_num):
    with tf.variable_scope(name):
        net = slim.conv2d(inputs, 8, [5, 5], stride=2, padding='VALID', activation_fn=None, scope='conv1')
        net = tf.nn.leaky_relu(net, name='relu_conv1')
        net = slim.conv2d(net, 16, [3, 3], stride=2, padding='VALID', activation_fn=None, scope='conv2')
        net = tf.nn.leaky_relu(net, name='relu_conv2')
        net = slim.conv2d(net, 24, [3, 3], stride=2, padding='VALID', activation_fn=None, scope='conv3')
        net = tf.nn.leaky_relu(net, name='relu_conv3')
        net = slim.conv2d(net, 48, [3, 3], stride=2, padding='VALID', activation_fn=None, scope='conv4')
        net = tf.nn.leaky_relu(net, name='relu_conv4')
        net = slim.flatten(net, scope='flatten')
        net = slim.fully_connected(net, 128, activation_fn=None, scope='mimic_ip1')
        net = tf.nn.leaky_relu(net, name='relu_ip1')
        net = slim.fully_connected(net, 128, activation_fn=None, scope='mimic_ip2')
        net = tf.nn.leaky_relu(net, name='relu_ip2')
        print('fc2: ', net.shape)
        net = slim.fully_connected(net, pts_num*2, scope='pts82')
        return net


def inference(inputs, pts_gt, learning_rate, meanshape, pts_num, is_training=True, dropout_keep_prob=0.5, scope='tiny', global_pool=False):
    batch_size = tf.shape(inputs)[0]
    img_size = tf.shape(inputs)[1]
    img_size_inv = 1 / img_size
    ret_dict = {}
    # meanShape = tf.constant(meanshape)
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
        s1_ret = net + meanshape

        s1_loss = tf.reduce_mean(loss_func.smooth_l1_loss(pts_gt, s1_ret, 82))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'Stage1')):
            s1_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(s1_loss,
                                                                                        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Stage1'))
    ret_dict['s1_ret'] = s1_ret
    ret_dict['s1_loss'] = s1_loss
    ret_dict['s1_optimizer'] = s1_optimizer


    with tf.variable_scope('Stage_2'):
        S2_AffineParam = TransformParamsLayer(s1_ret, meanshape)
        S2_InputImage = AffineTransformLayer(inputs, S2_AffineParam)
        S2_InputLandmark = LandmarkTransformLayer(s1_ret, S2_AffineParam)

        # each part keypoints
        s2_brow_l = tf.reshape(S2_InputLandmark[:, 34:52], [-1, 9, 2])
        s2_brow_r = tf.reshape(S2_InputLandmark[:, 52:70], [-1, 9, 2])
        s2_eye_l = tf.reshape(S2_InputLandmark[:, 88:104], [-1, 8, 2])
        s2_eye_r = tf.reshape(S2_InputLandmark[:, 104:120], [-1, 8, 2])
        s2_broweye_l = tf.concat([s2_brow_l, s2_eye_l], 1)
        s2_broweye_r = tf.concat([s2_brow_r, s2_eye_r], 1)
        s2_nose = tf.reshape(S2_InputLandmark[:, 70:88], [-1, 9, 2])
        s2_mouth = tf.reshape(S2_InputLandmark[:, 120:160], [-1, 20, 2])

        #crop each part
        bbox_broweye_l = GetBBox(s2_broweye_l, 0.2) * img_size_inv
        bbox_broweye_r = GetBBox(s2_broweye_r, 0.2) * img_size_inv
        bbox_nose = GetBBox(s2_nose, 0.2) * img_size_inv
        bbox_mouth = GetBBox(s2_mouth, 0.2) * img_size_inv
        broweye_l = tf.image.crop_and_resize(S2_InputImage, bbox_broweye_l, tf.range(0, batch_size, 1), [40, 40])
        broweye_r = tf.image.crop_and_resize(S2_InputImage, bbox_broweye_r, tf.range(0, batch_size, 1), [40, 40])
        nose = tf.image.crop_and_resize(S2_InputImage, bbox_nose, tf.range(0, batch_size, 1), [40, 40])
        mouth = tf.image.crop_and_resize(S2_InputImage, bbox_mouth, tf.range(0, batch_size, 1), [40, 40])

        # generate heatmap of each part
        hm_broweye_l = LandmarkImageLayer(s2_broweye_l)
        hm_broweye_r = LandmarkImageLayer(s2_broweye_r)
        hm_nose = LandmarkImageLayer(s2_nose)
        hm_mouth = LandmarkImageLayer(s2_mouth)


    return ret_dict


