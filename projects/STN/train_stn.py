import tensorflow as tf
import tensorflow.contrib.slim as slim
from projects.STN.spatial_transformer import transformer
from utilities.data_preparation.save_read_tfrecord import load_tfrecord

def inference(input, num_classes, is_training=True, dropout_keep_prob=0.5, scope='STN', global_pool=False):
    with tf.variable_scope(scope, 'stn', [input]) as sc:



