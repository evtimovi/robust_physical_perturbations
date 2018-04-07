'''
Module that contains all functions pertaining
to the generation of adversarial inputs.
'''
import numpy as np
import keras
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.contrib.graph_editor import connect
from .model import l2_loss, l2_norm, l1_norm, YadavModel
from cleverhans.utils_tf import model_loss

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', '',  \
                    'The path to load the weights for the model being attacked from')
flags.DEFINE_boolean('printability_optimization', True, \
                    'Specifies whether to add an extra term to the loss function \
                    to optimize for printability')
flags.DEFINE_string('printability_tuples', '../printer_error/triplets/30values.txt',
                    'Specifies which file to load the printability tuples from. \
                    Used only if printability_optimization is set to True')
flags.DEFINE_boolean('clipping', True, 'Specifies whether to use clipping')
flags.DEFINE_float('noise_clip_max', 0.5, \
    'The maximum value for clipping the noise, if clipping is set to True')
flags.DEFINE_float('noise_clip_min', -0.5, \
    'The minimum value for clipping the noise, if clipping is set to True')
flags.DEFINE_float('noisy_input_clip_max', 1.0, \
    'The maximum value for clipping the image+noise, if clipping is set to True')
flags.DEFINE_float('noisy_input_clip_min', 0.0, \
    'The minimum value for clipping the image+noise, if clipping is set to True')
flags.DEFINE_float('attack_lambda', 0.02, \
    'The lambda parameter in the attack optimization problem')
flags.DEFINE_string('regloss', '', \
    'Specifies the regularization loss to use. Options: l1, l2')
flags.DEFINE_string('optimization_loss', '', \
    'Specifies the optimization loss to use (function J in the writeup). \
    Options: mse, cross-entropy')
flags.DEFINE_float('optimization_rate', 0.1,
    'The rate of optimization for the attack')
flags.DEFINE_float('adam_beta1', 0.9, \
    'The beta1 parameter for the Adam optimizer of the attack objective')
flags.DEFINE_float('adam_beta2', 0.999, \
    'The beta2 parameter for the Adam optimizer of the attack objective')
flags.DEFINE_float('adam_epsilon', 1e-08, \
    'The epsilon parameter for the Adam optimizer of the attack objective')
flags.DEFINE_string('attack_mask', '', 'Filepath to the mask used in the attack.')
flags.DEFINE_integer('true_class', 44,  \
                    'The class that the victim images truly belong to.')
flags.DEFINE_integer('target_class', 5, \
                     'The class being targeted for this attack as a number')
flags.DEFINE_integer('input_rows', 32,
                     'how many rows the input will actually have \
                      (if different from img_rows, the input will be resized to img_rows)')
flags.DEFINE_integer('input_cols', 32, 
                     'how many columns the input will actually have \
                     (if different from img_cols, the input is resized to img_cols')
flags.DEFINE_string('resize_method', '',  \
                    'Resize method to use for the noise. Must be one of \
                     avgpool, area, bicubic, bilinear, nearestneighbor')

def _confmin_loss(tensor, \
                  class1_index, \
                  class2_index):
    '''
    Returns an operation to compute the difference
    between the confidence of class1 and the confidence of class2.
    These should be used as true_class = class1 and target_class = class2
    because the operation is class1 - class2
    :param tensor: a 2D tensor "containing images" in the 0 dimension 
                   and classes for those images in the 1 dimension
    :param class1_index: the class whose confidence is being subtracted from
    :param class2_index: the class whose confidence is being subtracted
    :return: a TF op as specified above
    '''
    return tf.reduce_mean(tensor[:, class1_index] - tensor[:, class2_index])

def setup_attack_graph():
    '''
    This function sets up an attack graph
    based on the Robust Physical Perturbations
    optimization algorithm and returns the Tensorflow operation
    to run that graph, along with the model, session, variables, operations,
    and placeholders defined along the way in dictionaries
    addressed by their names.
    :return: a tuple of (operation, model, session, placeholders, varops)
    where operation is the operation to run,
    provided initially, session is the TF session used to run the attack,
    and placeholder and varops are dictionaries holding the placeholders,
    variables, and intermediate TF operations defined
    '''
    assert FLAGS.img_rows, 'img_rows needs to be defined in the parameters'
    assert FLAGS.img_cols, 'img_cols needs to be defined in the parameters'
    assert FLAGS.nb_channels, 'nb_channels needs to be defined in the parameters'
    assert FLAGS.nb_classes, 'nb_classes needs to be defined in the parameters'
    assert FLAGS.model_path, 'model_path needs to be defined in the parameters'
    assert FLAGS.printability_optimization is not None, \
            'printability_optimization needs to be defined in the parameters'
    if FLAGS.printability_optimization:
        assert FLAGS.printability_tuples, \
        'printability_tuples needs to be defined if printability_optimization is True'
    assert FLAGS.regloss, 'regloss needs to be defined in the parameters'
    assert FLAGS.optimization_loss, 'optimization_loss needs to be defined in the params'

    # begin by setting up the session that will be used
    sess = tf.Session()

    # place all placeholders in this dict
    # so that they can be returned for use outside of this function
    placeholders = {}

    # note that these are set to the size of the input,
    # resizing happens later (before building model) if different
    placeholders['image_in'] = tf.placeholder(tf.float32, \
            shape=(None, FLAGS.input_rows, FLAGS.input_cols, FLAGS.nb_channels),
            name="noiseattack/image_in")

    placeholders['attack_target'] = tf.placeholder(tf.float32, \
        shape=(None, FLAGS.nb_classes),
        name="noiseattack/attack_targe")

    # resize later
    placeholders['noise_mask'] = tf.placeholder(tf.float32, \
                                                shape= \
                                                (FLAGS.input_rows, \
                                                FLAGS.input_cols, \
                                                FLAGS.nb_channels), \
                                                name="noiseattack/noise_mask")

    if FLAGS.printability_optimization:
        ####!!! Assumption: the printable tuples were all expanded to match
        ### the size of the image, so one tuple (x, y, z) gets replicated 32x32 times
        placeholders['printable_colors'] = tf.placeholder(tf.float32, \
                                                          shape=(None, \
                                                          FLAGS.input_rows, \
                                                          FLAGS.input_cols, \
                                                          FLAGS.nb_channels), \
                                                          name="noiseattack/printable_colors")

    # will hold the variables and operations defined from now on
    varops = {}

    varops['noise'] = tf.Variable(tf.random_normal( \
        [FLAGS.input_rows, FLAGS.input_cols, FLAGS.nb_channels]), \
        name='noiseattack/noise', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'adv_var'])

    # the following operations are these:
    # noise: a clipped value of the noise
    # noise_mul: the multiplication of the noise by the mask
    # noisy_inputs: the addition of the masked noise to the image
    if FLAGS.clipping:
        varops['noise'] = tf.clip_by_value(varops['noise'], \
            FLAGS.noise_clip_min, FLAGS.noise_clip_max, \
            name="noiseattack/noise_clipped")
        varops['noise_mul'] = tf.multiply(placeholders['noise_mask'], varops['noise'], \
            name="noiseattack/noise_mul")
        varops['noisy_inputs'] = tf.clip_by_value(tf.add(placeholders['image_in'], \
                                                  varops['noise_mul']), \
                                FLAGS.noisy_input_clip_min, FLAGS.noisy_input_clip_max, \
                                name="noiseattack/noisy_inputs")
    else:
        varops['noise_mul'] = tf.multiply(placeholders['noise_mask'], varops['noise'], \
                                          name="noiseattack/noise_mul")
        varops['noisy_inputs'] = tf.add(placeholders['image_in'], varops['noise_mul'], \
                                        name="noiseattack/noisy_inputs")

    if FLAGS.img_rows != FLAGS.input_rows or FLAGS.img_cols != FLAGS.input_cols:
        if FLAGS.resize_method != "avgpool" and FLAGS.resize_method != "convresize":
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

            varops['noisy_inputs'] = tf.image.resize_images(varops['noisy_inputs'], \
                                                            size=(FLAGS.img_rows, FLAGS.img_cols), \
                                                            method=resize_met)
        elif FLAGS.resize_method == "convresize":
            assert FLAGS.img_rows == 32 and FLAGS.input_rows == 256, \
                    "Convresize only guaranteed to work with input 256 and a 32 model"
            f = [[[[1,0,0], [0,1,0], [0,0,1]] for _ in xrange(8)] for __ in xrange(8)]
            f = np.array(f).astype('float32')/(64.0)
            varops['noisy_inputs'] = tf.nn.conv2d(varops['noisy_inputs'], \
                                                 tf.constant(f), \
                                                 strides=[1, 8, 8, 1], \
                                                 padding='SAME')
        else:
            s = FLAGS.input_rows/FLAGS.img_rows
            assert FLAGS.input_rows%FLAGS.img_rows == 0, \
                   "Input size should be a multiple of model input size, \
                   currently input: %d model input: %d"%(FLAGS.input_rows, FLAGS.img_rows)
            varops['noisy_inputs'] = tf.nn.avg_pool(varops['noisy_inputs'], \
                                                    ksize=[1, s, s, 1], \
                                                    strides=[1, s, s, 1], \
                                                    padding='SAME')
 
    # instantiate the model
    model = YadavModel(train=False, custom_input=varops['noisy_inputs'])

    model_vars = filter(lambda x: not str(x.name).startswith("noiseattack"), \
                        tf.global_variables())
    print map(lambda x: x.name, model_vars)
    # load the model
    saver = tf.train.Saver(var_list=model_vars)
    saver.restore(sess, FLAGS.model_path)
    print 'Loaded the parameters for the model from', FLAGS.model_path

    # adv_pred is the output of the model for an image (or images) with noise
    varops['adv_pred'] = model.labels_pred

     # Regularization term to control size of perturbation
    if FLAGS.regloss != "none":
        if FLAGS.regloss == 'l1':
            varops['reg_loss'] = FLAGS.attack_lambda * \
                l1_norm(tf.multiply(placeholders['noise_mask'], varops['noise']))
        elif FLAGS.regloss == 'l2':
            varops['reg_loss'] = FLAGS.attack_lambda * \
                l2_norm(tf.multiply(placeholders['noise_mask'], varops['noise']))
        else:
            raise Exception("Regloss may only be none or l1 or l2. Now%s"%FLAGS.regloss)

    # Compares adv predictions to given predictions
    # Default to cross-entropy (as defined in the model_loss cleverhans utility)
    if FLAGS.optimization_loss == 'justmse':
        varops['loss'] = l2_loss(placeholders['attack_target'], varops['adv_pred'])
    elif FLAGS.optimization_loss == "justcrossentropy":
        varops['loss'] = model_loss(placeholders['attack_target'], varops['adv_pred'], mean=True)
    elif FLAGS.optimization_loss == "justconfmin":
        varops['loss'] = _confmin_loss(varops['adv_pred'], FLAGS.true_class, FLAGS.target_class) 
    elif FLAGS.optimization_loss == "confminandcrossentropy":
        varops['loss'] = model_loss(placeholders['attack_target'], varops['adv_pred'], mean=True) + \
                         _confmin_loss(varops['adv_pred'], FLAGS.true_class, FLAGS.target_class)
    else:
        raise Exception("Optimization_loss needs to be justmse or justcrossentropy \
                        or justconfmin. Now %s"%FLAGS.optimization_loss)

    if FLAGS.printability_optimization:
        ####!!! Assumption: the printable tuples were all expanded to match
        ### the size of the image, so one tuple (x, y, z) gets replicated 32x32 times
        varops['printab_pixel_element_diff'] = tf.squared_difference(varops['noise_mul'], \
            placeholders['printable_colors'])
        varops['printab_pixel_diff'] = tf.sqrt(tf.reduce_sum( \
            varops['printab_pixel_element_diff'], 3))
        varops['printab_reduce_prod'] = tf.reduce_prod(varops['printab_pixel_diff'], 0)
        varops['printer_error'] = tf.reduce_sum(varops['printab_reduce_prod'])
        if FLAGS.regloss != "none":
            varops['adv_loss'] = varops['loss'] + varops['reg_loss'] + varops['printer_error']
        else:
            varops['adv_loss'] = varops['loss'] + varops['printer_error']
    else:
        if FLAGS.regloss != "none":
            varops['adv_loss'] = varops['loss'] + varops['reg_loss']
        else:
            varops['adv_loss'] = varops['loss']

    optimization_op = tf.train.AdamOptimizer(learning_rate=FLAGS.optimization_rate, \
        beta1=FLAGS.adam_beta1, \
        beta2=FLAGS.adam_beta2, \
        epsilon=FLAGS.adam_epsilon).minimize(varops['adv_loss'], \
        var_list=tf.get_collection('adv_var'))

    
    sess.run(tf.variables_initializer(set(tf.global_variables()) - set(model_vars)))
    print 'Initialized the model variables'

    return optimization_op, model, sess, placeholders, varops

def get_adv_target(nb_inputs=1):
    '''
    Generates a one-hot vector of shape (1, nb_classes)
    that represents a classification in the specified class
    The class needs to be specified in FLAGS.target_class

    :return: a one-hot vector representing that class
    '''
    target = np.zeros([nb_inputs, FLAGS.nb_classes])
    target[:, FLAGS.target_class] = 1.0
    return target


def get_print_triplets():
    '''
    Reads the printability triplets from the specified file
    and returns a numpy array of shape (num_triplets, FLAGS.img_cols, FLAGS.img_rows, nb_channels)
    where each triplet has been copied to create an array the size of the image
    :return: as described 
    '''     
    assert FLAGS.printability_tuples, \
           'If using NPS, need to define printability_tuples'
    p = []  
        
    # load the triplets and create an array of the speified size
    with open(FLAGS.printability_tuples) as f:
        for l in f:
            p.append(l.split(",")) 
    p = map(lambda x: [[x for _ in xrange(FLAGS.img_cols)] for __ in xrange(FLAGS.img_rows)], p)
    p = np.float32(p)
    p -= 0.5
    return p
