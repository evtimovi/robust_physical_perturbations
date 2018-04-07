'''
Applies noise from model_path to image from big_image
and saves it to output_path
'''
from utils.attack import setup_attack_graph
from utils.dataproc import read_img, write_img
import keras
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

import numpy as np
import cv2

FLAGS = flags.FLAGS
flags.DEFINE_string('output_path', '', 'Filepath to where to save the image')
flags.DEFINE_string('device', 'cpu', 'Which device to run this operation on')
flags.DEFINE_string('src_image', '', 'Filepath to image to apply the noise to')


def main(argv=None):
    with tf.device(FLAGS.device):
        with tf.Session() as sess:
            print "Noise loaded from", FLAGS.model_path
            print "Mask", FLAGS.attack_mask
            print "Source image", FLAGS.src_image
            bimg = cv2.resize(read_img(FLAGS.src_image), (FLAGS.img_rows, FLAGS.img_cols))/255.0 - 0.5

            noise= tf.Variable(tf.random_uniform( \
                [FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels], -0.5, 0.5), \
                name='noiseattack/noise', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'adv_var'])

            saver = tf.train.Saver(var_list=[noise])
            saver.restore(sess, FLAGS.model_path)

            noise_val = sess.run(noise)
            write_img('noise.png', (noise_val)*255.0)
            mask = read_img(FLAGS.attack_mask)/255.0
            noise_val = noise_val*mask
            write_img(FLAGS.output_path,(bimg+noise_val+0.5)*255)
            print "Wrote image to", FLAGS.output_path

if __name__ == "__main__":
    app.run()


