import tensorflow as tf
import os

import importlib
import tensorflow.contrib.slim as slim
from tools import model_tools

model_dir = '/home/public/nfs72/hanfy/models/hand/fine_model/hand_36_no_fused_squeezenet-20180621-115258/ori'
model_file = os.path.join(model_dir, '20180621-115258-284000.data-00000-of-00001')
meta_file = os.path.join(model_dir, '20180621-115258-284000.meta')

with tf.Graph().as_default():

    sess = tf.Session()


    image_size = 64
    embedding_size = 128
    network = importlib.import_module('network.squeezenet_v11')
    image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, image_size, image_size, 1], name='input_image')

    prelogits, _ = network.inference(image_placeholder, keep_probability = 1.0,
                                             phase_train=False, bottleneck_layer_size=embedding_size,
                                             weight_decay=0.0)
    logits = slim.fully_connected(prelogits, 36, activation_fn=None,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            weights_regularizer=slim.l2_regularizer(0.0),
                                          scope='Logits', reuse=False)
    logits = tf.identity(logits, 'squeezenet/output_logits_stu')
    output = tf.nn.softmax(logits=logits, name="squeezenet/output_softmax_stu")
    # embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='student_embeddings')
    saver = model_tools.load_pretrained_model(sess, model_dir)
    saver.save(sess, '/home/public/nfs72/hanfy/models/hand/fine_model/hand_36_no_fused_squeezenet-20180621-115258/renamed/hand_36_no_fused_squeezenet-20180621-115258')


