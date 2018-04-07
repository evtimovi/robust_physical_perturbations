import os
import numpy as np
import cv2 
import sys
import math

import utils
from utils.attack import setup_attack_graph, get_adv_target, get_print_triplets
from utils.model import YadavModel
from utils.dataproc import read_img, preprocess_yadav, write_img
from utils.eval import top3_as_string

import tensorflow as tf
from tensorflow.python.platform import app 
from tensorflow.python.platform import flags

from cleverhans.utils_tf import model_loss

FLAGS = flags.FLAGS
flags.DEFINE_string('attack_srcdir', '',
                    'Filepath to the directory containing the images to train on.')
flags.DEFINE_integer('attack_epochs', 100, 
                     'How many epochs to run the attack optimization for')
flags.DEFINE_string('checkpoint', '', 'Name to use when saving the checkpoint')
flags.DEFINE_string('device', 'cpu', 'Which device to run this operation on')
flags.DEFINE_float('min_rate_to_save', 0.9, 
                   'The minimum misclassification rate after which to begin saving')
flags.DEFINE_boolean('save_all_noisy_images', False, 
                     'whether to save the noisy images each epoch')

def main(argv=None):
    with tf.device(FLAGS.device):
        print "Parameters"
        for k in sorted(FLAGS.__dict__["__flags"].keys()):
            print k, FLAGS.__dict__["__flags"][k]

        op, model_obj, sess, pholders, varops = setup_attack_graph()

        model = varops['adv_pred']

        data = map(lambda z: preprocess_yadav(z),
                   map(lambda y: read_img(os.path.join(FLAGS.attack_srcdir, y)),
                       filter(lambda x: x.endswith(".png"), os.listdir(FLAGS.attack_srcdir))))
        num_images = len(data)

        feed_dict = {pholders['image_in']: data,
                     pholders['attack_target']: get_adv_target(nb_inputs=num_images),
                     pholders['noise_mask']: read_img(FLAGS.attack_mask)/255.0,
                     model_obj.keep_prob: 1.0}

        if FLAGS.printability_optimization:
            feed_dict[pholders['printable_colors']] = get_print_triplets()

        # used to save checkpoints after each epoch
        saver = tf.train.Saver(max_to_keep=5)

        clean_model_loss = model_loss(pholders['attack_target'], 
                                      varops['adv_pred'], mean=True)

        latest_misrate = FLAGS.min_rate_to_save
        latest_loss = 10000

        for i in xrange(FLAGS.attack_epochs):
            print 'Epoch %d'%i,
            sys.stdout.flush()
            _,  train_loss, mod_loss, noisy_in, noisy_classes = sess.run( \
                (op, \
                varops['adv_loss'], \
                varops['loss'], \
                varops['noisy_inputs'], \
                varops['adv_pred']) \
                , feed_dict=feed_dict)

            if FLAGS.regloss != "none":
                reg_loss = sess.run(varops['reg_loss'], feed_dict=feed_dict)
            else:
                reg_loss = 0

            clean_loss, clean_classes = sess.run((clean_model_loss, model), feed_dict={
                     pholders['image_in']: data,
                     pholders['attack_target']: get_adv_target(nb_inputs=num_images),
                     pholders['noise_mask']: np.zeros([FLAGS.input_rows, FLAGS.input_cols, FLAGS.nb_channels]),
                     model_obj.keep_prob: 1.0
                })

            print "adversarial loss %.5f reg loss %.5f model loss %.5f model loss on clean img: %.5f"%(train_loss, reg_loss, mod_loss, clean_loss),
            sys.stdout.flush()

            if FLAGS.printability_optimization:
                print "noise NPS %.5f"%sess.run(varops['printer_error'], feed_dict=feed_dict),

            num_misclassified = 0

            for j in xrange(num_images):
                clean_classification = np.argmax(clean_classes[j])
                noise_classification = np.argmax(noisy_classes[j])
                if clean_classification != noise_classification and noise_classification == FLAGS.target_class:
                    num_misclassified += 1

            proportion_misclassified = float(num_misclassified)/float(num_images)
            print 'percent misclassified images %.1f'%(proportion_misclassified*100.0)

            if proportion_misclassified > latest_misrate or \
                    (proportion_misclassified == latest_misrate and train_loss < latest_loss) \
                    or ("octagon" in FLAGS.attack_mask and train_loss < latest_loss):
                latest_misrate = proportion_misclassified
                latest_loss = train_loss
                saver.save(sess, os.path.join('optimization_output', FLAGS.checkpoint,
                                              'model', FLAGS.checkpoint), global_step=i)
            if FLAGS.save_all_noisy_images:
                write_img(os.path.join('optimization_output', FLAGS.checkpoint,
                               "noisy_images",
                               "noisyimg_%s_epoch_%d.png"%(FLAGS.checkpoint, i)),
                         ((noisy_in[0]+0.5)*255).astype(int))

if __name__ == '__main__':
    app.run()
