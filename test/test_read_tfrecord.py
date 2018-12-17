from utilities.save_read_tfrecord import read_tfrecord1
import tensorflow as tf
import cv2

output_tfrecords = '/home/hanfy/workspace/DL/alignment/align_untouch/temp/test.record'
filename_queue = tf.train.string_input_producer([output_tfrecords], num_epochs=1)
images, labels = read_tfrecord1(filename_queue, is_shuffle=True)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    while True:
        imgs, labs = sess.run([images, labels])
    # for i in range(len(imgs)):
    #     cv2.imshow("img", imgs[i])
    #     cv2.waitKey(0)

    coord.request_stop()
    coord.join(threads)