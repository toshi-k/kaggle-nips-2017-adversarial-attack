"""Implementation of sample defense.

This defense loads inception resnet v2 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
from scipy.misc import imread

import tensorflow as tf

import inception_resnet_v2

from tensorflow.contrib.keras.python.keras.utils import Progbar

slim = tf.contrib.slim

tf.flags.DEFINE_string(
        'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
        'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
        'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
        'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
        'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
        'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
        'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
        input_dir: input directory
        batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
        filenames: list file names without path of each image
            Lenght of this list could be less than batch_size, in this case only
            first few images of the result are elements of the minibatch.
        images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]

    filepaths = tf.gfile.Glob(os.path.join(input_dir, '*.png'))

    p_bar = Progbar(len(filepaths))

    for count, filepath in enumerate(filepaths):

        p_bar.update(count + 1)

        with tf.gfile.Open(filepath) as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            images[idx, :, :, :] = image * 2.0 - 1.0

        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

    p_bar.update(len(filepaths))


def main(_):

    start_time = time.time()

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    confidence = 0.0
    num_image = 0

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):

            # ------------------------------
            # approach (A) flip and sub-pixel shift

            s1 = [1, 0, 0.5, 0, 1, 0.5, 0, 0]
            s2 = [1, 0, 0.5, 0, 1, -0.5, 0, 0]
            s3 = [1, 0, -0.5, 0, 1, 0.5, 0, 0]
            s4 = [1, 0, -0.5, 0, 1, -0.5, 0, 0]

            # [1] original image
            input_1_1 = tf.map_fn(lambda img: tf.contrib.image.transform(img, s1, interpolation='BILINEAR'), x_input)
            input_1_2 = tf.map_fn(lambda img: tf.contrib.image.transform(img, s2, interpolation='BILINEAR'), x_input)
            input_1_3 = tf.map_fn(lambda img: tf.contrib.image.transform(img, s3, interpolation='BILINEAR'), x_input)
            input_1_4 = tf.map_fn(lambda img: tf.contrib.image.transform(img, s4, interpolation='BILINEAR'), x_input)

            _, end_points1_1 = inception_resnet_v2.inception_resnet_v2(input_1_1, num_classes=num_classes,
                                                                       is_training=False)
            _, end_points1_2 = inception_resnet_v2.inception_resnet_v2(input_1_2, num_classes=num_classes,
                                                                       is_training=False, reuse=True)
            _, end_points1_3 = inception_resnet_v2.inception_resnet_v2(input_1_3, num_classes=num_classes,
                                                                       is_training=False, reuse=True)
            _, end_points1_4 = inception_resnet_v2.inception_resnet_v2(input_1_4, num_classes=num_classes,
                                                                       is_training=False, reuse=True)

            # [2] flip image (ref)
            flip_input_1_1 = tf.map_fn(lambda img: tf.image.flip_left_right(img), input_1_1)
            flip_input_1_2 = tf.map_fn(lambda img: tf.image.flip_left_right(img), input_1_2)
            flip_input_1_3 = tf.map_fn(lambda img: tf.image.flip_left_right(img), input_1_3)
            flip_input_1_4 = tf.map_fn(lambda img: tf.image.flip_left_right(img), input_1_4)

            _, end_points2_1 = inception_resnet_v2.inception_resnet_v2(flip_input_1_1, num_classes=num_classes,
                                                                       is_training=False, reuse=True)
            _, end_points2_2 = inception_resnet_v2.inception_resnet_v2(flip_input_1_2, num_classes=num_classes,
                                                                       is_training=False, reuse=True)
            _, end_points2_3 = inception_resnet_v2.inception_resnet_v2(flip_input_1_3, num_classes=num_classes,
                                                                       is_training=False, reuse=True)
            _, end_points2_4 = inception_resnet_v2.inception_resnet_v2(flip_input_1_4, num_classes=num_classes,
                                                                       is_training=False, reuse=True)

            # sum all result
            predicted_values_a = (end_points1_1['Predictions'] + end_points1_2['Predictions'] +
                                  end_points1_3['Predictions'] + end_points1_4['Predictions'] +
                                  end_points2_1['Predictions'] + end_points2_2['Predictions'] +
                                  end_points2_3['Predictions'] + end_points2_4['Predictions']) / 8.0
            predicted_labels_a = tf.argmax(predicted_values_a, 1)

            # ------------------------------
            # approach (b) only flip

            # [1] original image
            _, end_points1 = inception_resnet_v2.inception_resnet_v2(x_input, num_classes=num_classes,
                                                                     is_training=False, reuse=True)

            # [2] flip image (ref)
            flip_input = tf.map_fn(lambda img: tf.image.flip_left_right(img), x_input)
            _, end_points2 = inception_resnet_v2.inception_resnet_v2(flip_input, num_classes=num_classes,
                                                                     is_training=False, reuse=True)

            # sum all result
            predicted_values_b = (end_points1['Predictions'] + end_points2['Predictions']) / 2.0
            predicted_labels_b = tf.argmax(predicted_values_b, 1)

            # ------------------------------
            # compare result

            # count number of same prediction
            # when ens_adv_inception_resnet_v2.ckpt is attacked, `count_match` is expected small
            # when ens_adv_inception_resnet_v2.ckpt is NOT attacked, `count_match` is expected large
            count_match = tf.reduce_sum(tf.cast(tf.equal(predicted_labels_a, predicted_labels_b), tf.int32))

            # decide which prediction to use
            # this is decided by weather more than half of batches are consistent or not
            # it is too conservative to use sub-pixel shift when ens_adv_inception_resnet_v2.ckpt is NOT attacked
            predicted_values = tf.cond(tf.less(count_match, 8), lambda: predicted_values_a, lambda: predicted_values_b)

        predicted_labels = tf.argmax(predicted_values, 1)
        predict_confidence = tf.reduce_max(predicted_values, reduction_indices=[1])

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
                scaffold=tf.train.Scaffold(saver=saver),
                checkpoint_filename_with_path=FLAGS.checkpoint_path,
                master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
                for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                    labels, confidences = sess.run([predicted_labels, predict_confidence], feed_dict={x_input: images})
                    for filename, label in zip(filenames, labels):
                        out_file.write('{0},{1}\n'.format(filename, label))

                    confidence += sum(confidences)
                    num_image += len(filenames)

    confidence /= float(num_image)
    print('  confidence: {0:.1f} [%]'.format(confidence * 100))

    elapsed_time = time.time() - start_time
    print('elapsed time: {0:.0f} [s]'.format(elapsed_time))

if __name__ == '__main__':
    tf.app.run()
