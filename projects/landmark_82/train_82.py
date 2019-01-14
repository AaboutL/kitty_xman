import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import sys
import numpy as np
import argparse

from net_archs.tinynet import inference
from utilities.data_preparation.save_read_tfrecord import load_tfrecord
from train import loss_func
from utilities import model_tool

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def main(args):
    trainset = args.trainset
    testset = args.testset

    meanShape = np.genfromtxt('../../meanshape_untouch.txt')
    print(meanShape.dtype)
    meanShape = np.reshape(meanShape, [164]).astype(np.float32) * 112

    with tf.Graph().as_default():
        step = 0
        if args.pretrained_model_dir is not None:
            step = int(model_tool.load_model(tf.Session(), model_dir=args.pretrained_model_dir))
        global_steps = tf.Variable(step, trainable=False)
        train_queue = tf.train.string_input_producer([trainset])
        test_queue = tf.train.string_input_producer([testset], num_epochs=1)
        images_train, points_train = load_tfrecord(train_queue, pts_num=82, img_shape=[112, 112, 1], batch_size=args.batch_size, is_shuffle=True)
        images_test, points_test = load_tfrecord(test_queue, pts_num=82, img_shape=[112, 112, 1], batch_size=128, is_shuffle=False)

        lr_ph = tf.placeholder(tf.float32, name='learning_rate_ph')
        imgs_ph = tf.placeholder(tf.float32, [None, 112, 112, 1], 'images_ph')
        pts_ph = tf.placeholder(tf.float32, [None, 164], 'points_ph')
        is_train_ph = tf.placeholder(tf.bool, name='is_train')

        learning_rate = tf.train.exponential_decay(lr_ph, global_step=global_steps,
                                                   decay_steps=args.learning_rate_decay_steps,
                                                   decay_rate=args.learning_rate_decay_rate,
                                                   staircase=True)

        pts_pre = inference(imgs_ph, meanshape=meanShape, pts_num=82, is_training=is_train_ph)
        loss = tf.reduce_mean(loss_func.NormRmse(pts_ph, pts_pre, 82))
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = opt.minimize(loss, global_step=global_steps)
        opt.compute_gradients(loss)


        Saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        Writer = tf.summary.FileWriter(args.log_dir, tf.get_default_graph())

        mid_result_path = os.path.join(args.mid_result_dir, 'train_result.txt')
        min_error = float('inf')

        with open(mid_result_path, 'a+') as mid_result:
            results = np.loadtxt(mid_result_path)
            if len(results) != 0:
                min_error = np.min(results[:,1])
            print('min_error', min_error)

            with tf.Session() as sess:
                sess.run(init_op)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                # step = 0
                # if args.pretrained_model_dir is not None:
                #     step = int(model_tool.load_model(sess, model_dir=args.pretrained_model_dir))
                imgs_test, pts_test = sess.run([images_test, points_test])
                while True:
                    step = sess.run(global_steps, feed_dict=None)
                    if step % 100 == 0:
                        summary, error = sess.run([merged, loss],
                                                    feed_dict={imgs_ph: imgs_test,
                                                               pts_ph: pts_test,
                                                               is_train_ph: False
                                                    })
                        mid_result.write("{0} {1}".format(step, str(error))+'\n')
                        print('evaluate on testset -> step: %d, loss: %2.4f'%(step, error))
                        if error < min_error:
                            min_error = error
                            print('saving model...')
                            Saver.save(sess, args.model_dir + '/model', global_step=step)

                    imgs_train, pts_train = sess.run([images_train, points_train])
                    summary, step, _, error, lr = sess.run([merged, global_steps, optimizer, loss, learning_rate],
                                                feed_dict={imgs_ph: imgs_train,
                                                           pts_ph: pts_train,
                                                           is_train_ph: True,
                                                           lr_ph: args.learning_rate
                                                           })
                    Writer.add_summary(summary, step)
                    print('step: %d, loss: %2.4f, lr: %2.7f'%(step, error, lr))
                coord.request_stop()
                coord.join(threads)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainset', type=str, default='/home/public/nfs132_0/landmark/dataset/untouch/untouch_labeled/total/train_112.record')
    parser.add_argument('--testset', type=str, default='/home/public/nfs132_0/landmark/dataset/untouch/untouch_labeled/total/test_112.record')
    parser.add_argument('--model_dir', type=str, default='/home/public/nfs71/hanfy/models/landmark_82')
    parser.add_argument('--log_dir', type=str, default='/home/public/nfs71/hanfy/logs/landmark_82')
    parser.add_argument('--mid_result_dir', type=str, default='/home/public/nfs71/hanfy/models/landmark_82/results')
    parser.add_argument('--pretrained_model_dir', type=str, help='Directory to the pretrain model')
                        # ,default='/home/public/nfs71/hanfy/models/landmark_82')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_rate_decay_steps', type=int, default=100000)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.1)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

