from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from utilities import tfrecords_util
from utilities import dataset

def convert_to_tfrecord(image_set, points_set, output_file):
    assert len(image_set)==len(points_set)
    print('writing')
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for i in range(len(image_set)):
            image = image_set[i]
            image_raw = image.tostring()
            pts = points_set[i]
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': tfrecords_util.bytes_feature(image_raw),
                    'label': tfrecords_util.float_list_feature(pts)
                }
            ))
            record_writer.write(example.SerializeToString())
