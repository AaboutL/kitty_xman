import cv2
import tensorflow as tf
import os

import numpy as np
from utilities.data_preparation.save_read_tfrecord import load_tfrecord
from projects.landmark_82.layers import AffineTransformLayer, TransformParamsLayer, LandmarkImageLayer, LandmarkTransformLayer, GetBBox

os.environ['CUDA_VISIBLE_DEVICES'] = ''
def main():
    output_tfrecords = '/home/slam/nfs132_0/landmark/dataset/untouch/test_116.record'
    # filename_queue = tf.train.string_input_producer([output_tfrecords], num_epochs=1)
    filename_queue = tf.train.string_input_producer([output_tfrecords])
    images, labels = load_tfrecord(filename_queue, pts_num=82, img_shape=[116, 116, 1], is_shuffle=True)
    # labels = np.asarray(labels)
    labels = tf.reshape(labels, [128, 82, 2])

    min_xy_tf = tf.reduce_min(labels, axis=1)
    max_xy_tf = tf.reduce_max(labels, axis=1)
    bboxes_tf = tf.stack([min_xy_tf[:, 1], min_xy_tf[:, 0], max_xy_tf[:, 1], max_xy_tf[:, 0]], axis=1)/116

    mouth_tf = labels[:, 60:80]
    mouth_min_xy_tf = tf.reduce_min(mouth_tf, axis=1)
    mouth_max_xy_tf = tf.reduce_max(mouth_tf, axis=1)
    mouth_bboxes_tf = tf.stack([mouth_min_xy_tf[:, 1], mouth_min_xy_tf[:, 0], mouth_max_xy_tf[:, 1], mouth_max_xy_tf[:, 0]], axis=1)/116
    img_crop_tf = tf.image.crop_and_resize(images, mouth_bboxes_tf, tf.range(0, 128, 1), [40, 40])
    img_crop_tf = tf.squeeze(img_crop_tf)

    nose_tf = labels[:, 35:44]
    nose_min_xy_tf = tf.reduce_min(nose_tf, axis=1)
    nose_max_xy_tf = tf.reduce_max(nose_tf, axis=1)
    wh = tf.subtract(nose_max_xy_tf, nose_min_xy_tf)
    delta_s = tf.multiply(wh, 0.25)
    xy_min_nose = tf.subtract(nose_min_xy_tf, delta_s)
    xy_max_nose = tf.add(nose_max_xy_tf, delta_s)
    nose_bboxes_tf = tf.stack([xy_min_nose[:, 1], xy_min_nose[:, 0], xy_max_nose[:, 1], xy_max_nose[:, 0]], axis=1)/116
    nose_img_crop_tf = tf.image.crop_and_resize(images, nose_bboxes_tf, tf.range(0, 128, 1), [40, 40])
    nose_img_crop_tf = tf.squeeze(nose_img_crop_tf)

    eye_l = labels[:, 44:52]
    brow_l = labels[:, 17:26]
    broweye_l = tf.concat([brow_l, eye_l], 1)
    be_bbox_tf = GetBBox(broweye_l, 0.2) / 116
    be_img_crop_tf = tf.image.crop_and_resize(images, be_bbox_tf, tf.range(0, 128, 1), [40, 40])
    be_img_crop_tf = tf.squeeze(be_img_crop_tf)


    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(3):
            # imgs, labs, min_xy_tf_, max_xy_tf_ = sess.run([images, labels, min_xy_tf, max_xy_tf])
            imgs, labs, bboxes, img_crop, nose, be = sess.run([images, labels, bboxes_tf, img_crop_tf, nose_img_crop_tf, be_img_crop_tf])
            img_crop = np.asarray(img_crop, dtype=np.uint8)
            nose = np.asarray(nose, dtype=np.uint8)
            be = np.asarray(be, dtype=np.uint8)
            print(img_crop.shape, img_crop.dtype)
            print(bboxes.shape)
            print(labs.shape)
            print(nose.shape)
            print(be.shape)

            for j in range(len(imgs)):
                lab = labs[j]
                pts = lab
                bbox = bboxes[j]
                cv2.imshow('crop', img_crop[j])
                cv2.imshow('nose', nose[j])
                cv2.imshow('be', be[j])

                # print('min max tf: ', min_xy_tf_[j], max_xy_tf_[j])
                # pts = np.array([[lab[0], lab[2], lab[4]], [lab[1], lab[3], lab[5]]])
                # dst = np.zeros((224, 224), dtype=np.uint8)
                # dst = cv2.warpAffine(imgs[j], pts, (224, 224))
                img = imgs[j]
                img = np.squeeze(np.stack([img, img, img], axis=2))
                for k in range(len(pts)):
                    cv2.circle(img, (int(pts[k][0]), int(pts[k][1])), 2, (0, 255, 0))
                cv2.rectangle(img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 255, 0), 2)
                # cv2.rectangle(img, (min_xy_tf_[j][0], min_xy_tf_[j][1]), (max_xy_tf_[j][0], max_xy_tf_[j][1]), (0, 255, 0), 2)
                cv2.imshow("img", img)
                # cv2.imshow('dst', dst)
                cv2.waitKey(0)

        coord.request_stop()
        coord.join(threads)
if __name__ == '__main__':
    main()