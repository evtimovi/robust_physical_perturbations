'''
Applies noise from model_path to image from big_image
and saves it to output_path
'''
from utils import setup_attack_graph, load_norm_mask
import keras
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import cv2
from scipy.misc import imsave, imread

FLAGS = flags.FLAGS

def main(argv=None):
    print "Noise loaded from", FLAGS.model_path
    print "Mask", FLAGS.attack_mask
    print "Source image", FLAGS.big_image

    bimg = cv2.resize(imread(FLAGS.big_image), (300,300))/255.0
    print 'bimg shape', bimg.shape

    op, model, sess, pholders, varops = setup_attack_graph()
    bigimage = tf.placeholder(shape=bimg.shape, dtype='float32')

    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.model_path)

    noise = sess.run(
                tf.image.resize_images(varops['noise_mul'],(bimg.shape[0],bimg.shape[1])), 
                feed_dict={bigimage: bimg, 
                           pholders['noise_mask']: load_norm_mask(), 
                           keras.backend.learning_phase(): 0})

    imsave(FLAGS.output_path,(bimg+noise)*255)
    print "Wrote image to", FLAGS.output_path

if __name__ == "__main__":
    app.run()

