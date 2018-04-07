#imports
import keras
from keras import backend

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils import cnn_model
from cleverhans.utils_tf import tf_model_load, model_loss

import os
import numpy as np
import random
import cv2

from scipy.misc import imread
import sys

FLAGS = flags.FLAGS
flags.DEFINE_integer('attack_epochs', 500, 'Number of epochs to use when solving the attack optimization') 
flags.DEFINE_integer('tf_seed', 12345, 'The random seed for tensorflow to use') 
flags.DEFINE_integer("save_frequency", 50, "Save at every x-th epoch where x is the number specified at this parameter")
flags.DEFINE_float('attack_lambda', 0.02, 'The lambda parameter in the attack optimization problem')
flags.DEFINE_float('optimization_rate', 0.01, 'The optimization rate (learning rate) for the Adam optimizer of the attack objective')
flags.DEFINE_integer('nb_classes', 18, 'Number of classification classes') 

flags.DEFINE_integer('target_class', 5, 'The class being targeted for this attack as a number')

flags.DEFINE_integer('img_rows', 32, 'Input row dimension')
flags.DEFINE_integer('img_cols', 32, 'Input column dimension')
flags.DEFINE_integer('nb_channels', 3, 'Nb of color channels in the input.')

flags.DEFINE_string('attack_srcimg', './source_images/307_resized.png', 'Filepath to image to mount the attack on, if running a one-image script.')
flags.DEFINE_string('attack_mask', '', 'Filepath to the mask used in the attack.')
flags.DEFINE_string('attack_srcdir', '', 'Filepath to the directory containing the images to train on, if running a multi-image script.')
flags.DEFINE_string('model_path', './models/all_r_ivan.ckpt', 'Path to load model from.')

flags.DEFINE_boolean('clipping', True, 'Specifies whether to use clipping')
flags.DEFINE_float('noise_clip_max', 0.5, 'The maximum value for clipping the noise, if clipping is set to True')
flags.DEFINE_float('noise_clip_min', -0.5, 'The minimum value for clipping the noise, if clipping is set to True')
flags.DEFINE_float('noisy_input_clip_max', 1.0, 'The maximum value for clipping the image+noise, if clipping is set to True')
flags.DEFINE_float('noisy_input_clip_min', 0.0, 'The minimum value for clipping the image+noise, if clipping is set to True')

flags.DEFINE_boolean('inverse_mask', True, 'Specifies whether to use an inverse mask (set all pixels in the original image to a specified value)')
flags.DEFINE_float('inverse_mask_setpoint', 0.5, 'The value to set the pixels within the mask region in the original image to if using an inverse mask')

flags.DEFINE_string('checkpoint', 'attack_single', 'Prefix to use when saving the checkpoint')

flags.DEFINE_string('regloss', '', 'Specifies the regularization loss to use. Options: l1. Anything else defaults to l2')
flags.DEFINE_string('optimization_loss', '', 'Specifies the optimization loss to use (function J in the writeup). Options: mse. Anything else defaults to cross-entropy')

flags.DEFINE_boolean('printability_optimization', True, 'Specifies whether to add an extra term to the loss function to optimize for printability')
flags.DEFINE_string('printability_tuples','../printer_error/triplets/30values.txt', 'Specifies which file to load the printability tuples from. Used only if printability_optimization is set to True')

flags.DEFINE_float('adam_beta1', 0.9, 'The beta1 parameter for the Adam optimizer of the attack objective')
flags.DEFINE_float('adam_beta2', 0.999, 'The beta2 parameter for the Adam optimizer of the attack objective')
flags.DEFINE_float('adam_epsilon', 1e-08, 'The epsilon parameter for the Adam optimizer of the attack objective')

flags.DEFINE_string('initial_value_for_noise', '', 'An RGB color triplet represented by a comma-separted list of exactly 3 numbers representing the initial color of the noise in the 0-255 range or an empty string for random initialization')

flags.DEFINE_boolean('fullres_input', False, 'Specifies whether to use 300 by 300 input images (and resize them down to 32 by 32 for model input) or just use 32 by 32')

flags.DEFINE_string('output_path', '', 'Filepath to where to save the image')
flags.DEFINE_string('big_image', '', 'Filepath to big image')
#flags.DEFINE_float('adam_learning_rate', 0.5, 'The value to set the pixels within the mask region in the original image to')

def get_print_triplets(src):
    '''
    Reads the printability triplets from the specified file
    and returns a numpy array of shape (num_triplets, FLAGS.img_cols, FLAGS.img_rows, nb_channels)
    where each triplet has been copied to create an array the size of the image
    :param src: the source file for the printability triplets
    :return: as described 
    '''
    p = []
    
    # load the triplets and create an array of the speified size
    with open(FLAGS.printability_tuples) as f:
        for l in f:
            p.append(l.split(","))
    p = map(lambda x: [[x for _ in xrange(FLAGS.img_cols)] for __ in xrange(FLAGS.img_rows)], p)
    p = np.float32(p)

#    mask = load_norm_mask()
#
#   # mask out triplets outside the mask
#    for t in p:
#        for x in xrange(FLAGS.img_cols):
#            for y in xrange(FLAGS.img_rows):
#                if np.all(mask[x][y] == 0.0):
#                    t[x][y] = [0.0,0.0,0.0]

    return p


def top3(model_out, j = 0):
    '''
    Given a classification output, returns the top 3 
    classes in an array of tuples (class, probability).
    :param model_out: the output from the classification model
    :param j: the index in the output to use, in case there is more than one output vector in model_out. Defaults to 0
    :return: an array of 3 (class, probability) tuples in decreasing order of probability
    '''
    classes = zip(range(len(model_out[j])), model_out[j])
    return sorted(classes, key=lambda x: x[1], reverse=True)[:3]


def report_extreme_values(arr, min_val=0.0, max_val=1.0, name="array"):
    '''
    Prints out the average value and the proportion of values
    outside the given range in the arr.
    :param arr: the array to be analyzed
    :param min_val: the lower bound of the range
    :param max_val: the upper bound of the range
    '''
    arr_flat = arr.flatten()
    print 'avg %s value'%name, np.average(arr_flat)
    print '%.3f percent of all values in %s are outside (%d,%d)'%((float(len(filter(lambda x: x>max_val or x<min_val, arr_flat)))/float(len(arr_flat)))*100.0, name, min_val, max_val)
    print '%.3f percent of all values in %s are bigger than %d'%((float(len(filter(lambda x: x>max_val, arr_flat)))/float(len(arr_flat)))*100.0, name, max_val)
    print '%.3f percent of all values in %s are less than %d'%((float(len(filter(lambda x: x<min_val, arr_flat)))/float(len(arr_flat)))*100.0, name, min_val)

def load_angles_inverse_mask(folder, dist="1a", angles=['A', 'B', 'C', 'D']):
    '''
    Load all images for all angles for a given distance.
    Assumes folder structure folder/<angle-name>/<dist-name>.<file-extension>
    :param folder: the filepath to the folder where the images are located
    :param dist: the distance code/filename of the image
    :return: a numpy array of shape (<# of files in folder>, img_rows, img_cols, nb_channels)
    '''
    all_imgs = []

    for subfolder in angles:
        filtered_filenames = filter(lambda x: x.split(".")[0] == dist, os.listdir(os.path.join(folder, subfolder)))
        all_imgs.extend(map(lambda x: os.path.join(folder, subfolder, x), filtered_filenames))

    return map(lambda f: load_img_inverse_mask(f), all_imgs)

def load_many_images(src):
    '''
    Loads all images in the specified folder
    by using load_img_inverse_mask
    SIDE EFFECT: prints the image names and the indices they were loaded to
    :param src: the path to the source directory to load
    :return: a Python array of the images loaded
    '''
    imgs = []
    filenames = os.listdir(src)
    for fname in filenames:
        if FLAGS.inverse_mask:
            imgs.append(load_img_inverse_mask(os.path.join(src, fname)))
        else:
            imgs.append(load_norm_img_from_source(os.path.join(src, fname)))
    print 'Loaded images in directory %s'%src
    map(lambda x: sys.stdout.write('Index %d image %s\n'%(x[0],x[1])), zip(range(len(filenames)), filenames))
    print 
    sys.stdout.flush()
    return imgs

def load_many_images_twomask(src):
    '''
    Loads all images in the specified folder
    by using load_img_inverse_mask_twomask
    SIDE EFFECT: prints the image names and the indices they were loaded to
    :param src: the path to the source directory to load
    :return: a Python array of the images loaded
    '''
    imgs = []
    filenames = os.listdir(src)
    for fname in filenames:
        imgs.append(load_img_inverse_mask_twomask(os.path.join(src, fname)))
    print 'Loaded images in directory %s'%src
    map(lambda x: sys.stdout.write('Index %d image %s\n'%(x[0],x[1])), zip(range(len(filenames)), filenames))
    print 
    sys.stdout.flush()
    return imgs


def load_img_inverse_mask_twomask(src):
    '''
    Loads the specified image and sets all pixels within
    the non-zero region of the two masks specified in mask1 and mask2
    (only if the gennoise_two_masks script is run)
    to FLAGS.inverse_mask_setpoint  (i.e. the values are normalized)
    :param src: full filepath to the image being loaded
    :return: the image the attack is being performed on, modified as described
    '''
    mask1 = load_norm_mask(FLAGS.mask1)
    mask2 = load_norm_mask(FLAGS.mask2)
    img = load_norm_img_from_source(src)
    assert mask1.shape == img.shape and mask2.shape == img.shape, "The shape of an image and a mask do not match up. Image shape %s, mask1 shape: %s, mask2 shape: %s"%(img.shape, mask1.shape, mask2.shape)
    shape = img.shape
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            if np.sum(mask1[i][j]) != 0 or np.sum(mask2[i][j]) != 0:
                img[i][j] = [FLAGS.inverse_mask_setpoint,FLAGS.inverse_mask_setpoint,FLAGS.inverse_mask_setpoint]
    return img


def load_img_inverse_mask(src):
    '''
    Loads the specified image and sets all pixels within
    the non-zero region of the mask at attack_mask
    to 0.5  (i.e. the values are normalized)
    :param src: full filepath to the image being loaded
    :return: the image the attack is being performed on, modified as described
    '''
    mask = load_norm_mask()
    img = load_norm_img_from_source(src)
    assert mask.shape == img.shape, "Image and mask shape do not match up. Image shape %s, mask shape: %s"%(img.shape, mask.shape)
    shape = mask.shape
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            if np.sum(mask[i][j]) != 0:
                img[i][j] = [FLAGS.inverse_mask_setpoint,FLAGS.inverse_mask_setpoint,FLAGS.inverse_mask_setpoint]
    return img


def round_tensor(tensor, digits):
    '''
    Returns an op that rounds the tensor to the specified number of digits
    by multiplying by 10**digits, rounding, and then dividing by the same number
    :param tensor: the tensor we want rounded
    :param digits: the precision (in decimal digits) for rounding
    :return: an op that rounds the given tensor
    '''
    N = tf.pow(10.0, digits)
    return tf.divide(tf.round(tf.multiply(tensor,N)), N)


def load_norm_img_from_source(src):
    '''
    read an image from the filepath specified
    and normalizes it to the [0.0,1.0] range
    :param src: the filepath to the image
    :return: the normalized image as a numpy array of shape (img_rows, img_cols, nb_channels)
    '''
#    img = cv2.imread(src)
    img = imread(src, mode='RGB')
    assert img is not None, "No image found at filepath %s"%src
    return img/255.0

def load_single_image():
    '''
    Reads the image that FLAGS.attack_srcimg is pointing to,
    normalizes its pixel values to [0.0,1.0]
    and returns it in a numpy array of shape
    (1, img_cols, img_rows, nb_channels)
    :return: a Numpy array containing the image
    '''
    # get the names of all images in the attack source directory
    # and filter by extension to only include image files
#    img = np.float32(cv2.imread(FLAGS.attack_srcimg))
    img = np.float32(imread(FLAGS.attack_srcimg))
    assert img is not None, "Image at %s not loaded"%full_path

     #Note that the pixel values are being normalized to [0,1]
    return [img/255.0]

def load_norm_mask(src=FLAGS.attack_mask):
    '''
    Reads the mask at the path that FLAGS.attack_mask is pointing to
    :return: the mask as a numpy array
    '''
#    mask = cv2.imread(FLAGS.attack_mask)/255.0
    mask = imread(src)/255.0

    assert mask is not None, "No image found at %s for mask"%FLAGS.attack_mask
    return mask

def get_adv_target(nb_inputs=1):
    '''
    Generates a one-hot vector of shape (1, nb_classes)
    that represents a classification in the specified class
    The class needs to be specified in FLAGS.target_class

    :return: a one-hot vector representing that class
    '''
    target = np.zeros([nb_inputs,FLAGS.nb_classes])
    target[:,FLAGS.target_class] = 1.0
    return target

def setup_model_and_sess():
    '''
    Sets up and loads the model used for classifying 
    the signs with the help of keras
    and the corresponding TF session. (Code from cleverhans example)
    Needs FLAGS.model_path in order to locate the stored model
    :return: a tuple (model, sess) 
    '''
    # print all parameters for the current run
    # print "Parameters"
    # for k in sorted(FLAGS.__dict__["__flags"].keys()):
    #     print k, FLAGS.__dict__["__flags"][k]

    ###### setup code from cleverhans example ######
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(FLAGS.tf_seed)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)
    print("Created TensorFlow session and set Keras backend.")

    # Define TF model graph
    model = cnn_model(img_rows=FLAGS.img_rows, img_cols=FLAGS.img_cols, channels=FLAGS.nb_channels, nb_classes=FLAGS.nb_classes)

    # Restore the model from previously saved parameters
    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.model_path)
    print("Loaded the parameters for the model from %s"%FLAGS.model_path)
    
    return model, sess

def l2_loss(tensor1, tensor2):
    '''
    Provides a Tensorflow op that computess the L2 loss (the Euclidean distance)
    between the tensors provided.
    :param tensor1: the first tensor
    :param tensor2: the other tensor
    :return: a TF op that computes the L2 distance between the tensors
    '''
    return tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(tensor1,tensor2), 2)))

def l2_norm(tensor):
    '''
    Provides a Tensorflow op that computess the L2 norm of the given tensor
    :param tensor: the tensor whose L2 norm is to be computed
    :return: a TF op that computes the L2 norm of the tensor
    '''
    return tf.sqrt(tf.reduce_sum(tf.pow(tensor, 2)))


def l1_norm(tensor):
    '''
    Provides a Tensorflow op that computes the L1 norm of the given tensor
    :param tensor: the tensor whose L1 norm is to be computed
    :return: a TF op that computes the L1 norm of the tensor
    '''
    return tf.reduce_sum(tf.abs(tensor))


def setup_attack_graph():
    '''
    Sets up the attack graph
    :return: a tuple of (the optimization op, the model, the session, the placeholders, the variable and other ops)
    '''

    # the NPS calculation also depends on the correct input being provided in the correct size, 
    # so for now, the NPS calculation will stay undefined when the full resolution input is used
    assert not(FLAGS.printability_optimization and FLAGS.fullres_input), "Printability optimization and full resolution input are not set up to work together. Set at least one of them to false."
    
    # set up the input (to the TF graph) image size 
    # by setting up the variables here, there is no need to do this same if each time we define an operation dependent on the input size
    if FLAGS.fullres_input:
        # if we are providing a full-res image, the noise and mask placeholders and vars
        # will also have this size; they will then be resized back down to FLAGS.img_rows, FLAGS.img_cols for input to the classification model
        img_rows = 300
        img_cols = 300
    else:
        # this is just the standard, 32 by 32 input
        img_rows = FLAGS.img_rows
        img_cols = FLAGS.img_cols

    # this handles the setup of the Keras model, the initialization of the TF session,
    # and the loading of the model parameters from previously saved values
    model, sess = setup_model_and_sess()
    
    # at this point, only the model variables exist; we will need this set later
    # in order to tell TF to not initialize those variables again
    model_vars = set(tf.global_variables())

    # will hold the placeholders so that they can be returned 
    placeholders = {}

    # set up the placeholders -- these are "input" places to the computation graph that change from run to run
    # these are "filled in" for each session run by using a feed_dict
    # image_in was x: the input to the neural network
    placeholders['image_in'] = tf.placeholder(tf.float32, shape = (None, img_rows, img_cols, FLAGS.nb_channels))


    # attack_target used to be y: the one-hot vector for the class 
    # we are trying to mimic when feeding image_in into the network
    placeholders['attack_target'] = tf.placeholder(tf.float32, shape = (None, FLAGS.nb_classes)) 

    # this is the mask being applied to limit the region of the perturbations
    placeholders['noise_mask'] = tf.placeholder(tf.float32, shape=(img_rows, img_cols, FLAGS.nb_channels))

    if FLAGS.printability_optimization:
        ####!!! Assumption: the printable tuples were all expanded to match
        ### the size of the image, so one tuple (x, y, z) gets replicated 32x32 times
        placeholders['printable_colors'] = tf.placeholder(tf.float32, shape=(None,32,32,3))


    # will hold the variables and operations defined from now on
    varops = {}

    # the noise variable is what is actually being optimized
    # the values stored in variables are persisted across session runs (but not across program runs, unless saved)
    if FLAGS.initial_value_for_noise != "" and FLAGS.initial_value_for_noise != " ": 
        # if a specific color for the initialization has been specified,
        # set the initial value of the noise to that color
        noise_init_color = np.float32(FLAGS.initial_value_for_noise.split(","))/255.0
        assert noise_init_color.shape == (3,), "You must provide 3 comma-separated values or no value for the initial_value_for_noise argument"
        noise_init = np.ndarray([img_rows, img_cols, FLAGS.nb_channels], dtype='float32')
        noise_init[:,:] = noise_init_color
        varops['noise'] = tf.Variable(noise_init,
                name='noise', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'adv_var'])
    else:
        varops['noise'] = tf.Variable(tf.random_uniform([img_rows, img_cols, FLAGS.nb_channels], 0.0, 1.0), 
                name='noise', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'adv_var'])

    if FLAGS.clipping:
        varops['noise']= tf.clip_by_value(varops['noise'], FLAGS.noise_clip_min, FLAGS.noise_clip_max)
        varops['noise_mul']=tf.multiply(placeholders['noise_mask'], varops['noise'])
        varops['noise_inputs'] = tf.clip_by_value(tf.add(placeholders['image_in'], varops['noise_mul']), 
                                FLAGS.noisy_input_clip_min, FLAGS.noisy_input_clip_max)
    else:
        varops['noise_mul'] = tf.multiply(placeholders['noise_mask'], varops['noise'])
        varops['noise_inputs'] = tf.add(placeholders['image_in'], varops['noise_mul'])

    # add a resize before feeding into the model if the input to the TF graph
    # was provided in full resolution (needs to be down-scaled to fit into the classification model)
    if FLAGS.fullres_input:
        varops['resized_noise_in'] = tf.image.resize_images(varops['noise_inputs'],(FLAGS.img_rows, FLAGS.img_cols))
        varops['adv_pred'] = model(varops['resized_noise_in']) 
    else:
        # adv_pred is what comes out of the model (network) for a given input
        varops['adv_pred'] = model(varops['noise_inputs']) 

    # Regularization term to control size of perturbation
    if FLAGS.regloss == 'l1':
        varops['reg_loss'] = FLAGS.attack_lambda * l1_norm(tf.multiply(placeholders['noise_mask'], varops['noise']))
    else:
        varops['reg_loss'] = FLAGS.attack_lambda * l2_norm(tf.multiply(placeholders['noise_mask'], varops['noise'])) 

    # Compares adv predictions to given predictions
    # Default to cross-entropy (as defined in the model_loss cleverhans utility)
    if FLAGS.optimization_loss == 'mse':
        varops['loss'] = l2_loss(placeholders['attack_target'], varops['adv_pred']) 
    else: 
        varops['loss'] = model_loss(placeholders['attack_target'], varops['adv_pred'], mean=True) 

    if FLAGS.printability_optimization:
        ####!!! Assumption: the printable tuples were all expanded to match
        ### the size of the image, so one tuple (x, y, z) gets replicated 32x32 times
        varops['printab_pixel_element_diff']= tf.squared_difference(varops['noise_mul'],placeholders['printable_colors'])
        varops['printab_pixel_diff'] = tf.sqrt(tf.reduce_sum(varops['printab_pixel_element_diff'], 3))
        varops['printab_reduce_prod'] = tf.reduce_prod(varops['printab_pixel_diff'], 0)
        varops['printer_error'] = tf.reduce_sum(varops['printab_reduce_prod'])
        varops['adv_loss'] = varops['loss'] + varops['reg_loss'] + varops['printer_error']
    else:
        varops['adv_loss'] = varops['loss'] + varops['reg_loss']

    op = tf.train.AdamOptimizer(learning_rate=FLAGS.optimization_rate, beta1=FLAGS.adam_beta1, beta2=FLAGS.adam_beta2, epsilon=FLAGS.adam_epsilon).minimize(varops['adv_loss'], var_list = tf.get_collection('adv_var'))
    
    # initialize the noise variable 
    sess.run(tf.variables_initializer(set(tf.global_variables()) - model_vars))

    return op, model, sess, placeholders, varops

def get_nps_op(noise_op, printable_colors):
    '''
    Computes the nps op for the provided noise and printable colors.
    !!!!Assumes that printable_colors has been expanded to a 4D array
    of shape (number_of_triplets, noise_img_rows, noise_img_cols, number_of_channels)
    by replicating the same printability triplet to fill an image
    :param noise_op: the noise (could be masked)
    :param printable_colors: the printable colors
    :return: an operation computing the 
    '''
    printab_pixel_element_diff= tf.squared_difference(noise_op, printable_colors)
    printab_pixel_diff = tf.sqrt(tf.reduce_sum(printab_pixel_element_diff, 3))
    printab_reduce_prod = tf.reduce_prod(printab_pixel_diff, 0)
    return tf.reduce_sum(printab_reduce_prod)

def setup_attack_graph_two_masks():
    '''
    Sets up the attack graph assuming two different masks will be fed in
    :return: a tuple of (the optimization op, the model, the session, the placeholders, the variable and other ops)
    '''

    assert not(FLAGS.fullres_input), "High resolution input not supported with two masks"
    
    # this handles the setup of the Keras model, the initialization of the TF session,
    # and the loading of the model parameters from previously saved values
    model, sess = setup_model_and_sess()
    
    # at this point, only the model variables exist; we will need this set later
    # in order to tell TF to not initialize those variables again
    model_vars = set(tf.global_variables())

    # will hold the placeholders so that they can be returned 
    placeholders = {}

    # will hold the variables and operations defined from now on
    varops = {}


    # set up the placeholders -- these are "input" places to the computation graph that change from run to run
    # these are "filled in" for each session run by using a feed_dict
    placeholders['image_in'] = tf.placeholder(tf.float32, shape = (None, FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels))


    # attack_target is the one-hot vector for the class 
    # we are trying to mimic when feeding image_in into the network
    placeholders['attack_target'] = tf.placeholder(tf.float32, shape = (None, FLAGS.nb_classes)) 

    # this is the first mask being applied to limit the region of the perturbations
    placeholders['mask1'] = tf.placeholder(tf.float32, shape=(FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels))

    # this is the second mask being applied to limit the region of the perturbations
    placeholders['mask2'] = tf.placeholder(tf.float32, shape=(FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels))

    # this is the sum of mask1 and mask2
    varops['combined_mask'] = tf.add(placeholders['mask1'], placeholders['mask2']) 


    if FLAGS.printability_optimization:
        ####!!! Assumption: the printable tuples were all expanded to match
        ### the size of the image, so one tuple (x, y, z) gets replicated 32x32 times
        # we will have a different set of printable colors for each mask
        placeholders['printable_colors_region_1'] = tf.placeholder(tf.float32, shape=(None,32,32,3))
        placeholders['printable_colors_region_2'] = tf.placeholder(tf.float32, shape=(None,32,32,3))

       # the noise variable is what is actually being optimized
    # the values stored in variables are persisted across session runs (but not across program runs, unless saved)
    if FLAGS.initial_value_for_noise != "" and FLAGS.initial_value_for_noise != " ": 
        # if a specific color for the initialization has been specified,
        # set the initial value of the noise to that color
        noise_init_color = np.float32(FLAGS.initial_value_for_noise.split(","))/255.0
        assert noise_init_color.shape == (3,), "You must provide 3 comma-separated values or no value for the initial_value_for_noise argument"
        noise_init = np.ndarray([FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels], dtype='float32')
        noise_init[:,:] = noise_init_color
        varops['noise'] = tf.Variable(noise_init,
                name='noise', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'adv_var'])
    else:
        varops['noise'] = tf.Variable(tf.random_uniform([img_rows, img_cols, FLAGS.nb_channels], 0.0, 1.0), 
                name='noise', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'adv_var'])

    if FLAGS.clipping:
        varops['noise']= tf.clip_by_value(varops['noise'], FLAGS.noise_clip_min, FLAGS.noise_clip_max)
        varops['noise_mul']=tf.multiply(varops['combined_mask'], varops['noise'])
        varops['noise_inputs'] = tf.clip_by_value(tf.add(placeholders['image_in'], varops['noise_mul']), 
                                FLAGS.noisy_input_clip_min, FLAGS.noisy_input_clip_max)
    else:
        varops['noise_mul'] = tf.multiply(varops['combined_mask'], varops['noise'])
        varops['noise_inputs'] = tf.add(placeholders['image_in'], varops['noise_mul'])

    varops['adv_pred'] = model(varops['noise_inputs']) 

    # Regularization term to control size of perturbation
    if FLAGS.regloss == 'l1':
        varops['reg_loss'] = FLAGS.attack_lambda * l1_norm(tf.multiply(varops['combined_mask'], varops['noise']))
    else:
        varops['reg_loss'] = FLAGS.attack_lambda * l2_norm(tf.multiply(varops['combined_mask'], varops['noise'])) 

    # Compares adv predictions to given predictions
    # Default to cross-entropy (as defined in the model_loss cleverhans utility)
    if FLAGS.optimization_loss == 'mse':
        varops['loss'] = l2_loss(placeholders['attack_target'], varops['adv_pred']) 
    else: 
        varops['loss'] = model_loss(placeholders['attack_target'], varops['adv_pred'], mean=True) 

    if FLAGS.printability_optimization:
        varops['nps1'] = get_nps_op(tf.multiply(varops['noise'],placeholders['mask1']), placeholders['printable_colors_region_1'])  
        varops['nps2'] = get_nps_op(tf.multiply(varops['noise'],placeholders['mask2']), placeholders['printable_colors_region_2'])  
        varops['adv_loss'] = varops['loss'] + varops['reg_loss'] + varops['nps1'] + varops['nps2']
    else:
        varops['adv_loss'] = varops['loss'] + varops['reg_loss']

    op = tf.train.AdamOptimizer(learning_rate=FLAGS.optimization_rate, beta1=FLAGS.adam_beta1, beta2=FLAGS.adam_beta2, epsilon=FLAGS.adam_epsilon).minimize(varops['adv_loss'], var_list = tf.get_collection('adv_var'))
    
    # initialize the noise variable 
    sess.run(tf.variables_initializer(set(tf.global_variables()) - model_vars))

    return op, model, sess, placeholders, varops
