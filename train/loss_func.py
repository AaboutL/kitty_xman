from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import tensorflow as tf
import math


def wing_loss(predLandmarks, gtLandmarks, num_points, w=10.0, epsilon=2.0):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, num_landmarks, 2].
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """
    Gt = tf.reshape(gtLandmarks, [-1, num_points, 2])
    Pt = tf.reshape(predLandmarks, [-1, num_points, 2])
    with tf.name_scope('wing_loss'):
        x = Pt - Gt
        c = w * (1.0 - math.log(1.0 + w/epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(
            tf.greater(w, absolute_x),
            w * tf.log(1.0 + absolute_x/epsilon),
            absolute_x - c
        )
        loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)
        return loss

def NormRmse(gtLandmarks, predLandmarks, num_points):
    Gt = tf.reshape(gtLandmarks, [-1, num_points, 2])
    Pt = tf.reshape(predLandmarks, [-1, num_points, 2])
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(Gt, Pt), 2)), 1)
    norm = tf.norm(tf.reduce_mean(Gt[:, 36:42, :],1) - tf.reduce_mean(Gt[:, 42:48, :],1), axis=1)
    return loss/norm

def l1_loss(gtLandmarks, predLandmarks):
    loss = tf.abs(tf.subtract(gtLandmarks, predLandmarks), name='l1_loss')
    return loss

def smooth_l1_loss(gtLandmarks, predLandmarks, num_points):
    Gt = tf.reshape(gtLandmarks, [-1, num_points, 2])
    Pt = tf.reshape(predLandmarks, [-1, num_points, 2])

    loss = tf.losses.huber_loss(Gt, Pt, scope='smooth_l1_loss')
    return loss