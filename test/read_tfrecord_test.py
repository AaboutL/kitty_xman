import cv2
import tensorflow as tf

import numpy as np
from utilities.data_preparation.save_read_tfrecord import load_tfrecord

def main():
    output_tfrecords = '/home/slam/nfs132_0/landmark/dataset/untouch/train_116.record'
    # filename_queue = tf.train.string_input_producer([output_tfrecords], num_epochs=1)
    filename_queue = tf.train.string_input_producer([output_tfrecords])
    images, labels = load_tfrecord(filename_queue, pts_num=82, img_shape=[116, 116], is_shuffle=True)
    # images = read_tfrecord1(filename_queue, is_shuffle=True)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(3):
            imgs, labs = sess.run([images, labels])
            print('imgs shape: ', imgs.shape)
            print('labs shape: ', labs.shape)
            print('imgs len: ', len(imgs))
            for j in range(len(imgs)):
                print(labs[j])
                lab = labs[j]
                pts = lab
                # pts = np.array([[lab[0], lab[2], lab[4]], [lab[1], lab[3], lab[5]]])
                # dst = np.zeros((224, 224), dtype=np.uint8)
                # dst = cv2.warpAffine(imgs[j], pts, (224, 224))
                # print(pts)
                img = imgs[j]
                img = np.stack([img, img, img], axis=2)
                for k in range(len(pts)//2):
                    cv2.circle(img, (int(pts[k*2]), int(pts[k*2+1])), 2, (0, 255, 0))
                cv2.imshow("img", img)
                # cv2.imshow('dst', dst)
                cv2.waitKey(0)

        coord.request_stop()
        coord.join(threads)
if __name__ == '__main__':
    main()