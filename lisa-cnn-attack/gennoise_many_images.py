'''
Generates noise by using multiple images by using the Adam optimizer method.
This script DOES use the inverse mask.
'''
from utils import get_adv_target, load_norm_mask, setup_attack_graph, report_extreme_values, load_many_images, get_print_triplets
import keras
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import os
import numpy as np
import cv2

from cleverhans.utils_tf import model_loss

from scipy.misc import imread, imsave

import sys
import math

#set parameters for attack
FLAGS = flags.FLAGS

def main(argv=None):
    print "going into setup"
    op, model, sess, pholders, varops = setup_attack_graph()

    data = load_many_images(FLAGS.attack_srcdir)
    num_images = len(data)

    feed_dict = {pholders['image_in']: data, 
                 pholders['attack_target']: get_adv_target(nb_inputs = num_images), 
                 pholders['noise_mask']: load_norm_mask(), 
                 keras.backend.learning_phase(): 0}

    if FLAGS.printability_optimization:
        feed_dict[pholders['printable_colors']] = get_print_triplets(FLAGS.printability_tuples)

    # used to save checkpoints after each epoch
    saver = tf.train.Saver(max_to_keep=50)

    # debug: sanity check to make sure the model isn't being adjusted
    # i.e. this should stay constant

    if FLAGS.fullres_input:
        clean_model_loss = model_loss(pholders['attack_target'], model(tf.image.resize_images(pholders['image_in'], (FLAGS.img_rows,FLAGS.img_cols))), mean=True) 
    else:
        clean_model_loss = model_loss(pholders['attack_target'], model(pholders['image_in']), mean=True) 
    
    for i in xrange(FLAGS.attack_epochs):
        print 'Epoch %d'%i,
        sys.stdout.flush()
        
        if not FLAGS.fullres_input:
            _, train_loss, noisy_in, clean_loss, clean_classes, noisy_classes = sess.run( \
                (op, \
                varops['adv_loss'], \
                varops['noise_inputs'], \
                clean_model_loss, \
                model(pholders['image_in']), \
                varops['adv_pred']) \
                , feed_dict=feed_dict)
        else:
            _, train_loss, noisy_in, clean_loss, clean_classes, noisy_classes, rnin = sess.run( \
                (op, \
                varops['adv_loss'], \
                varops['noise_inputs'], \
                clean_model_loss, \
                model(tf.image.resize_images(pholders['image_in'], (FLAGS.img_rows,FLAGS.img_cols))), \
                varops['adv_pred'], \
                varops['resized_noise_in']) \
                , feed_dict=feed_dict)

        print "adversarial loss %.5f model loss on clean img: %.5f"%(train_loss, clean_loss),
        sys.stdout.flush()
       
        if FLAGS.printability_optimization:
            print "noise NPS %.5f"%sess.run(varops['printer_error'], feed_dict=feed_dict),

        num_misclassified = 0

        for j in xrange(num_images):
            clean_classification = np.argmax(clean_classes[j])
            noise_classification = np.argmax(noisy_classes[j])
            if clean_classification != noise_classification:
                num_misclassified += 1

        proportion_misclassified = float(num_misclassified)/float(num_images)
        print 'percent misclassified images %.1f'%(proportion_misclassified*100.0)

        if i%FLAGS.save_frequency == 0 or proportion_misclassified > 0.9: 
            saver.save(sess, os.path.join('optimization_output', FLAGS.checkpoint, 'model', FLAGS.checkpoint), global_step=i)
            imsave(os.path.join('optimization_output', FLAGS.checkpoint, "noisy_images", "noisyimg_%s_epoch_%d.png"%(FLAGS.checkpoint,i)), (noisy_in[0]*255).astype(int))

            if FLAGS.fullres_input:
                imsave(os.path.join('optimization_output', FLAGS.checkpoint, "nimage_downsized_%d.png"%i), rnin[0])
                imsave(os.path.join('optimization_output', FLAGS.checkpoint, "noise_downsized_%d.png"%i),sess.run(varops['noise']))


        print 
       ### end of epoch
    sess.close()

if __name__ == '__main__':
    app.run()
