import cv2
import tensorflow as tf

from utilities.data_preparation.save_read_tfrecord import load_tfrecord

output_tfrecords = '/home/hanfy/workspace/DL/alignment/align_untouch/temp/test.record'
filename_queue = tf.train.string_input_producer([output_tfrecords], num_epochs=1)
images, labels = load_tfrecord(filename_queue, is_shuffle=True)
# images = read_tfrecord1(filename_queue, is_shuffle=True)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        imgs, labs = sess.run([images, labels])
        # imgs = sess.run([images])
        for j in range(len(imgs)):
            print(imgs[j].shape)
            print(labs[j])
            pts = labs[j]
            for k in range(len(pts)//2):
                cv2.circle(imgs[j], (int(pts[k*2]), int(pts[k*2+1])), 2, 255)
            cv2.imshow("img", imgs[j])
            cv2.waitKey(0)

    coord.request_stop()
    coord.join(threads)