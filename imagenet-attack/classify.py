import os
import numpy as np
import tensorflow as tf
from attack_util import read_data_inception, top3_as_string
from attack import inception, restore_model_vars
import flags
tf.flags.DEFINE_string("input_dir", "", \
                        "Directory to classify")
tf.flags.DEFINE_string('output_file', '', \
                       'Output file to save labels.')
FLAGS = flags.FLAGS

def main(_):
    flags.load_config_file()
    filenames, images = read_data_inception(FLAGS.input_dir)

    batch_shape = [len(images), FLAGS.image_height, FLAGS.image_width, FLAGS.image_channels]
    num_classes = FLAGS.num_classes
    
    sess = tf.Session()
    with sess.as_default():
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        logits_activations = inception(x_input)
        restore_model_vars(sess, tf.global_variables(), FLAGS.model_path)

        with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
            classifications = sess.run(tf.nn.softmax(logits_activations), feed_dict={x_input: images})
            for j in xrange(len(filenames)):
                out_file.write("{0},{1}\n".format(filenames[j], top3_as_string(classifications, j)))

if __name__ == '__main__':
  tf.app.run()
