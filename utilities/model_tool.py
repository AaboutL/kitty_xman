from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import re

def get_model_filenames(model_dir):
    model_dir = os.path.expanduser(model_dir)
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory %s' %model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        step = ckpt_file.split('-')[-1]
        return meta_file, ckpt_file, step

def load_model(sess, model_dir):
    model_exp = os.path.expanduser(model_dir)
    if os.path.isfile(model_exp): # pb
        print('model name %s' %model_exp)
        with gfile.FastGFile(model_exp, 'rb') as fp:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fp.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('model directory %s' %model_exp)
        meta_file, ckpt_file, step = get_model_filenames(model_exp)
        print('meta_file %s' %meta_file)
        print('ckpt_file %s' %ckpt_file)
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(sess, os.path.join(model_exp, ckpt_file))
        return step

def show_op_name():
    graph = tf.get_default_graph()
    operas = graph.get_operations()
    ops_names = [ops.name.split(':',1)[0] for ops in operas]
    for name in ops_names:
        # if name.startswith('alexnet_v2'):
        print(name)

def show_var_name():
    vars = tf.all_variables()
    var_names = [var.name.split(':', 1)[0] for var in vars]
    # for var in vars:
    #     print(var.get_shape())
    for name in var_names:
        print(name)

def show_tensor():
    graph_def = tf.get_default_graph().as_graph_def()
    graph = tf.get_default_graph()
    nodes = graph_def.node
    for node in nodes:
        tensor = graph.get_tensor_by_name(node.name+':0')
        print('name: %s\t shape: %s' %(tensor.name, tensor.get_shape()))

# get_model_filenames('/home/public/nfs132_1/hanfy/models/align_model')
