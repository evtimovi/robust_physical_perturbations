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
flags.DEFINE_string('big_image', '', 'Filepath to big image')
flags.DEFINE_string('device', 'cpu', 'Which device to run this operation on')
#flags.DEFINE_string('resize_method', '',  \
#                    'Resize method to use for the noise. Must be one of \
#                     area, bicubic, bilinear, nearestneighbor')
flags.DEFINE_integer('resize_rows', 300, 'How many rows in the result of the resize.')
flags.DEFINE_integer('resize_cols', 300, 'How many columns in the result of the resize.')
flags.DEFINE_boolean('resize_noise_only', True, 
                     'If true, resizes only the noises and uses a full-resolution mask. \
                        Else resizes the result of applying the mask to the noise.')
flags.DEFINE_boolean('downsize_first', False, \
                     'whether to resize down to 32 by 32 before upsizing')

def main(argv=None):
    with tf.device(FLAGS.device):
        with tf.Session() as sess:
            print "Noise loaded from", FLAGS.model_path
            print "Mask", FLAGS.attack_mask
            print "Source image", FLAGS.big_image

            if FLAGS.resize_method == "area":
                resize_met = tf.image.ResizeMethod.AREA
            elif FLAGS.resize_method == "bicubic":
                resize_met = tf.image.ResizeMethod.BICUBIC
            elif FLAGS.resize_method == "bilinear":
                resize_met = tf.image.ResizeMethod.BILINEAR
            elif FLAGS.resize_method == "nearestneighbor":
                resize_met = tf.image.ResizeMethod.NEAREST_NEIGHBOR
            else:
                raise Exception("resize method needs to be one of: area, bicubic, bilinear, nearestneighbor")

            bimg = cv2.resize(read_img(FLAGS.big_image), (FLAGS.resize_rows, FLAGS.resize_cols))/255.0 - 0.5
            print 'bimg shape', bimg.shape

            noise= tf.Variable(tf.random_uniform( \
                [FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels], -0.5, 0.5), \
                name='noiseattack/noise', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'adv_var'])

            mask = tf.placeholder(tf.float32, \
                                  shape= \
                                    (FLAGS.img_rows, \
                                    FLAGS.img_cols, \
                                    FLAGS.nb_channels), \
                                   name="noiseattack/noise_mask")

            saver = tf.train.Saver(var_list=[noise])
            saver.restore(sess, FLAGS.model_path)

            if not FLAGS.resize_noise_only:
                if FLAGS.downsize_first:
                    noise_val = sess.run(tf.image.resize_images( \
                                            tf.image.resize_images(noise * mask,(32,32)), \
                                            size=(bimg.shape[0], bimg.shape[1]), \
                                            method=resize_met), \
                                         feed_dict={mask: read_img(FLAGS.attack_mask)/255.0}
                                        )
                else:
                    noise_val = sess.run(tf.image.resize_images( \
                                            noise*mask,
                                            size=(bimg.shape[0], bimg.shape[1]), \
                                            method=resize_met), \
                                         feed_dict={mask: read_img(FLAGS.attack_mask)/255.0}
                                        )
            else:
                noise_val = sess.run(tf.image.resize_images(noise, \
                                                            size=(bimg.shape[0], bimg.shape[1]), \
                                                            method=resize_met))
                print 'noise shape', noise_val.shape
                mask = read_img(FLAGS.attack_mask)/255.0
                print 'mask shape', mask.shape
                noise_val = noise_val*mask

            write_img(FLAGS.output_path,(bimg+noise_val+0.5)*255)
            print "Wrote image to", FLAGS.output_path

if __name__ == "__main__":
    app.run()


