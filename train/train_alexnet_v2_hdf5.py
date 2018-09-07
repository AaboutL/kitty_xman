from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
import numpy as np
import argparse
import time

from utilities import dataset
# from net_archs import AlexNet as net
# from net_archs import AlexNet_BN as net
from net_archs import squeezenet_v11 as net
from evaluate import landmark_eval
from train import loss_func
from utilities import model_tool


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(args):
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.mid_result_dir, exist_ok=True)
    dset = dataset.Dataset()
    image_set, points_set= dset.read_hdf5(args.input_file)
    points_set = points_set.reshape((len(points_set), 68*2))

    imgs_val, shapes_val = dset.read_hdf5(args.test_file)
    shapes_val = shapes_val.reshape(len(shapes_val), 68*2)

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # placeholders
        lr = tf.placeholder(tf.float32, name='learning_rate')
        images = tf.placeholder(tf.float32, [None, 224, 224, 3], name='image_ph')
        points_gt = tf.placeholder(tf.float32, [None, args.num_landmarks*2], name='points_gt_ph')
        is_training = tf.placeholder(tf.bool, name='is_training')

        # construct loss
        inference, _ = net.inference(images, args.num_landmarks*2, is_training, args.dropout_keep_prob)
        with tf.variable_scope('squeezenet'):
            # loss = tf.reduce_mean(loss_func.NormRmse(gtLandmarks=points_gt, predLandmarks=inference, num_points=args.num_landmarks))
            # loss = tf.reduce_mean(loss_func.l1_loss(gtLandmarks=points_gt, predLandmarks=inference))
            loss = tf.reduce_mean(loss_func.smooth_l1_loss(gtLandmarks=points_gt, predLandmarks=inference, num_points=args.num_landmarks))
            tf.summary.scalar('landmark_loss', loss)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'squeezenet')):
            optimizer = tf.train.AdamOptimizer(0.001).minimize(loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'alexnet_v2'))

        Saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
        merged = tf.summary.merge_all()
        Writer = tf.summary.FileWriter(args.log_dir, tf.get_default_graph())

        mid_result_path = os.path.join(args.mid_result_dir, 'result.txt')
        print(mid_result_path)
        min_error = float('inf')
        with open(mid_result_path, 'a+') as mid_result:
            results = np.loadtxt(mid_result_path)
            if len(results) != 0:
                min_error = np.min(results[:,1])

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                step = 0
                if args.pretrained_model_dir is not None:
                    step = int(model_tool.load_model(sess, model_dir=args.pretrained_model_dir))

                print('min_error', min_error)
                for epoch in range(args.epochs):
                    batch_id = 0
                    while batch_id < args.epoch_size:
                        RandomIdx = np.random.choice(image_set.shape[0], args.batch_size, False)
                        start_time = time.time()
                        summary, _, lm_loss = sess.run([merged, optimizer, loss],
                                                       feed_dict={images : image_set[RandomIdx],
                                                                  points_gt : points_set[RandomIdx],
                                                                  is_training : args.is_training})
                        Writer.add_summary(summary, step)
                        duration = time.time() - start_time
                        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                              (epoch, batch_id+1, args.epoch_size, duration, lm_loss))

                        batch_id += 1
                        step += 1
                        if batch_id % 100 == 0:
                            pred_shapes = sess.run([inference], feed_dict={images:imgs_val, is_training:False})
                            pred_shapes = np.reshape(pred_shapes, [len(shapes_val), 68, 2])
                            pts_val = np.reshape(shapes_val, [len(shapes_val), 68, 2])
                            norm_errors, errors = landmark_eval.landmark_error(pts_val, pred_shapes)
                            mean_error = np.mean(errors)
                            mid_result.write("{0} {1}".format(step, str(mean_error))+'\n')
                            if mean_error < min_error:
                                min_error = mean_error
                                Saver.save(sess, args.model_dir + '/model', global_step=step)
                Writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='path to the dataset',
                        default='/home/public/nfs132_1/hanfy/align/ibugs/trainset.hdf5')
    parser.add_argument('--test_file', type=str, help='path to the validation set',
                        default='/home/public/nfs132_1/hanfy/align/ibugs/testset.hdf5')
    parser.add_argument('--num_landmarks', type=int, help='number of landmarks on a face',
                        default=68)
    parser.add_argument('--is_training', type=bool, help='which mode, training or inference',
                        default=True)
    parser.add_argument('--batch_size', type=int, help='size of a batch',
                        default=64)
    parser.add_argument('--epochs', type=int, help='how many epoches should train',
                        default=90)
    parser.add_argument('--epoch_size', type=int, help='how many batches in one epoch',
                        default=1000)
    parser.add_argument('--log_dir', type=str, help='Directory to the log file',
                        default='/home/public/nfs132_1/hanfy/result_files/0905_h5_BN_smooth/log')
    parser.add_argument('--mid_result_dir', type=str, help='file to keep test error',
                        default='/home/public/nfs132_1/hanfy/result_files/0905_h5_BN_smooth')
    parser.add_argument('--model_dir', type=str, help='Director to the model file',
                        default='/home/public/nfs132_1/hanfy/result_files/0905_h5_BN_smooth/model')
    parser.add_argument('--pretrained_model_dir', type=str, help='Directory to the pretrain model')
                        # , default='/home/public/nfs132_1/hanfy/models/align_model/0905_h5_BN_smooth')
    parser.add_argument('--dropout_keep_prob', type=float, help='dropout rate',
                        default=0.5)

    args = parser.parse_args()
    main(args)
