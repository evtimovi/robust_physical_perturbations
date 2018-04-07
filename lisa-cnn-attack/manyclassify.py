'''
Classifies all images found in the folder specified in attack_srcdir
with the model specified in model_path
and prints out the filename and the top 3 most probable classes.
'''
from utils import top3
import keras
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from utils import setup_model_and_sess, load_norm_img_from_source

import numpy as np
import cv2
import os

FLAGS = flags.FLAGS

def main(argv=None):

    model, sess = setup_model_and_sess()
    x = tf.placeholder('float32', (1, 32, 32, 3))
    model_op = model(x)

    for image in os.listdir(FLAGS.attack_srcdir):
        model_out = sess.run(model_op, feed_dict={x: [load_norm_img_from_source(os.path.join(FLAGS.attack_srcdir,image))], keras.backend.learning_phase(): 0})
        print image, 
        for c, p in top3(model_out, 0):
            print c, p,
        print

if __name__ == "__main__":
    app.run()


