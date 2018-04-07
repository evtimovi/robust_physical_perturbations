#work-around for pylint bug 1869: https://github.com/PyCQA/pylint/issues/1869
from __future__ import print_function 

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import flags
from scipy.misc import imread, imsave, imresize
from sklearn.metrics import accuracy_score
import os

FLAGS = flags.FLAGS

class AttackGraph(object):
    '''
    Holds placeholders and variables 
    and the Tensorflow computation graph
    for the attack.
    '''
    def __init__(self, batch_size, \
                       image_height, image_width, image_channels, \
                       num_classes, noise_initializer, \
                       pixel_low, pixel_high):
        '''
        Builds the graph for the attack only at the bare minimum, i.e. only the noise
        and its application on the images. Does NOT create the model or the optimization.

        Creates these attributes: 
        clean_input: the clean, rank-4 tensor of inputs of shape 
                    (batch_size, image_height, image_width, image_channels)
        mask: the mask for the noise (a placeholder), 
              shape is (image_height, image_width, image_channels)
        noise: the noise that is applied to an image (a variable), 
               shape is (image_height, image_width, image_channels)
        noise_transforms: the (batch_size, 8) vectors describing the transforms
                          needed to apply the masked noise to the image
        noisy_inputs: the clean_inputs, after noise is applied to them as a patch
        noisy_inputs_transform: the noisy inputs, after they have been color-shifted, resized, etc.

        Any further graph building (model, optimization) is left for dedicated methods.
        This helps save computation when only extracting the noise. Use build_everything
        to finish building for an attack.

        Parameters
        ----------
        batch_size : int
            the size of batches to use
        image_height : int
            the number of rows in the images
        image_width : int
            the number of columns in the images
        image_channels : int
            the number of channels in the images, e.g. 3 for RGB
        noise_initializer : tf.initializer
            an intializer for the 
        '''
        self.image_shape = (image_height, image_width, image_channels)
        self.input_shape = (batch_size, image_height, image_width, image_channels)
        self.pixel_low = pixel_low
        self.pixel_high = pixel_high

        # Hopefully, when building another graph with this, the noise gets reused
        with tf.variable_scope("noiseattack", reuse=tf.AUTO_REUSE):
            self.clean_input = tf.placeholder(tf.float32, self.input_shape, name="clean_input")
            self.mask = tf.placeholder(tf.float32, self.image_shape, name="mask")
            self.noise =  tf.get_variable("noise", shape=self.image_shape, \
                                          dtype="float32", initializer=noise_initializer) 
            self.noisy_inputs = apply_noise_to_images(self.clean_input, self.noise, \
                                                      self.mask)
            self.noisy_inputs_transform, self.color_shifts, self.boxes, self.dest_points = \
                transform_noisy_inputs(self.noisy_inputs, image_height, image_width)
            self.final_noisy_inputs = tf.clip_by_value(self.noisy_inputs_transform, \
                                                       self.pixel_low, \
                                                       self.pixel_high)
 
    def build_model(self, model_func):
        '''
        Builds the model that is to be attacked
        Parameters
        ----------
        model_func : function
            a function that takes in an input vector and returns the logits of the model
            that is to be attacked
        
        Returns
        -------
            the logits of the model given adversarial inputs
        ''' 
        assert self.final_noisy_inputs is not None
        
        self.adv_pred = model_func(self.final_noisy_inputs)
            
        self.model_vars = filter(lambda x: "noiseattack" not in str(x.name), \
                                 tf.global_variables())
        self.model_vars = set(self.model_vars) - set([self.noise])
        self.output_shape = self.adv_pred.shape
        return self.adv_pred
    
    def build_optimization(self, regularization_losses, *args, **kwargs):
        #kwargs to allow overriding beta1, beta2, and epsilon of adam,
        # kwargs will be passed as is to Adam's initialization 
        assert self.adv_pred is not None, "build_model must be called before build_optimization"
        assert self.output_shape is not None

        self.attack_target =  tf.placeholder(tf.float32, shape=self.output_shape, name="attack_target")
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        self.reg_names = regularization_losses
        self.total_loss = self._build_loss(regularization_losses, self.adv_pred, self.attack_target, self.noise)
        
        with tf.name_scope("adamoptimizer"):
            self.optimization_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, **kwargs) \
                                                .minimize(self.total_loss, \
                                                var_list=[self.noise])

        self.init_adam = tf.variables_initializer(filter(lambda x: "adam" in x.name.lower(), tf.global_variables()))
        self.init_noise = tf.variables_initializer(set(tf.global_variables()) - set(self.model_vars))
        return self.optimization_op

    def _build_loss(self, reg_losses, model_out, target_vec, noise):
        '''
        Builds up the optimization loss given regularzation losses strings
        (one or more of l2norm, tv, l1norm, l2image) by defining
        necessary placeholders for each and returns the total loss.
        '''
        self.class_loss = crossentropy_loss(target_vec, model_out, mean=True)
        total_loss = self.class_loss
        self.reg_lambdas = {}
        self.reg = {}
        if len(reg_losses) > 0:
            for l in reg_losses:
                self.reg_lambdas[l] = tf.placeholder(tf.float32, shape=(), name="%s_lambda"%l)
                if l == "l2norm":
                    self.reg[l] = self.reg_lambdas[l] * l2_norm(noise)
                elif l == "tv":
                    self.reg[l] = self.reg_lambdas[l] * tf.reduce_mean(tf.image.total_variation(noise))
                elif l == "l1norm":
                    self.reg[l] = self.reg_lambdas[l] * l1_norm(noise)
                elif l == "l2image":
                    self.l2image = tf.placeholder(tf.float32, shape=self.image_shape, name="l2image")
                    self.reg[l] = self.reg_lambdas[l] * l2_loss(noise, self.l2image)
                elif l == "nps":
                    self.nps_triplets = tf.placeholder(tf.float32, \
                                                        shape=[None] + list(self.image_shape), \
                                                        name="nps_triplets")
                    nps = tf.squared_difference(noise, self.nps_triplets)
                    nps = tf.reduce_sum(nps, 3)
                    nps = tf.sqrt(nps)
                    nps = tf.reduce_prod(nps, 0)
                    nps = tf.reduce_sum(nps)
                    self.reg[l] = self.reg_lambdas[l] * nps
                
                total_loss += self.reg[l]
        return total_loss

    def build_everything(self, model_func, regularization_losses, *args, **kwargs):
        self.build_model(model_func)
        return self.build_optimization(regularization_losses, **kwargs)

    def get_noise_extract_op(self):
        return tf.clip_by_value(self.noise * self.mask, self.pixel_low, self.pixel_high)

def _verify_images_shape(images):
    '''
    determine if images is a 5d or a 4d tensor
    and reshape if necessary (make a 5d tensor a 4d tensor)
    
    Parameters:
    -----------
    images : Tensor
        a tensor of rank 4 or 5 holding all the image inputs
    
    Returns:
    --------
    (Tensor, int) tuple
        a tensor of rank 4 and the size of its first dimension
    '''
    images_shape = images.shape
    if images_shape.ndims == 5:
        n = images_shape[0]*images_shape[1]
        clean_images = tf.reshape(images, (n, images_shape[2], images_shape[3], images_shape[4]))
    elif images_shape.ndims == 4:
        n = images_shape[0]
        clean_images = images
    else:
        raise Exception("Only supporting images tensors of rank 4 or 5. Currently: %d"%images_shape.ndims)
    return clean_images, n

def apply_noise_to_images(images, noise, mask):
    clean_images, n = _verify_images_shape(images)
    noises = tf.stack([noise] * n)
    masks = tf.stack([mask] * n)

    inverse_masks = 1.0 - masks

    return clean_images * inverse_masks + noises * masks


def transform_noisy_inputs(noisy_inputs, image_height, image_width):
    '''
    applies several transformations to the noisy_images
    (a tensor of rank 4 or 5)
    1) multiplies every pixel by a chosen value 
    2) crops each image based on a boundig box specified 
    and resizes the result to the original size
    3) perspective transforms based on a series of destination points,
    while assuming that the source is always the corner of the images

    the parameters for these transformations are specified in placeholders
    that are returned as part of this function

    For 1, a placeholder of shape (n, image_height, image_width, channels)
    is provided (color_shift_placeholder). It is recommended
    that the values in here are uniform for all pixels and in the (1.0, 2.0)
    range. Such values shift the white balance of the image. 

    For 2, the coordinates of the boundig box to crop to for each image are given.
    These are normalized to the 0.0 to 1.0 range and are specified in the returned 
    boxes placeholder.

    For 3, a placeholder of shape (n, 4, 2) specifies where the corners of the image
    are mapped to for a perspective transform.

    Sizes of placeholders:
    n is either the batch size or frames*batch_size

    color_shift_placeholder: (n, image_height, image_width, channels) 
    boxes: (n, 4)
    dest_points: (n, 4, 2)
    '''
    imgs_shape = noisy_inputs.shape
    clean_images, n = _verify_images_shape(noisy_inputs)
    img_cols = float(image_width)
    img_rows = float(image_height)

    # color shifts
    color_shift_placeholder = tf.placeholder(tf.float32, shape=clean_images.shape, name="color_shift")

    color_shifted_noisy_inputs = color_shift_placeholder * clean_images

    # crops
    boxes = tf.placeholder(tf.float32, shape=(n, 4), name="boxes")
    cropped_images = tf.image.crop_and_resize(color_shifted_noisy_inputs, \
                                            boxes=boxes, box_ind=[x for x in range(n)], \
                                           crop_size=clean_images.shape[1:3])
    

    # perspective transforms
    src_points = \
        tf.stack([
                [[0., 0.],[0., img_cols],[img_rows, 0.],[img_rows, img_cols]] \
                for _ in range(n)])
    dest_points = tf.placeholder(tf.float32, shape=(n, 4, 2))

    final_images = []
    for i in range(n):
        transforms = homography(src_points[i], dest_points[i])
        final_images.append(tf.contrib.image.transform(cropped_images[i], transforms))
    
    final_images = tf.stack(final_images)
    final_images = tf.reshape(final_images, shape=imgs_shape)

    return final_images, color_shift_placeholder, boxes, dest_points


def ax(p, q):
    return [ p[0], p[1], -1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0] ]

def ay(p, q):
    return [ 0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1] ]

def homography(x1s, x2s):
    p = []

    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence
    p.append(ax(x1s[0], x2s[0]))
    p.append(ay(x1s[0], x2s[0]))

    p.append(ax(x1s[1], x2s[1]))
    p.append(ay(x1s[1], x2s[1]))

    p.append(ax(x1s[2], x2s[2]))
    p.append(ay(x1s[2], x2s[2]))

    p.append(ax(x1s[3], x2s[3]))
    p.append(ay(x1s[3], x2s[3]))

    # A is 8x8
    A = tf.stack(p, axis=0)

    m = [[x2s[0][0], x2s[0][1], x2s[1][0], x2s[1][1], x2s[2][0], x2s[2][1], x2s[3][0], x2s[3][1]]]

    # P is 8x1
    P = tf.transpose(tf.stack(m, axis=0))

    # here we solve the linear system
    # we transpose the result for convenience
    return tf.transpose(tf.matrix_solve_ls(A, P, fast=True))

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

def l1_norm(tensor):
    '''
    Provides a Tensorflow op that computes the L1 norm of the given tensor
    :param tensor: the tensor whose L1 norm is to be computed
    :return: a TF op that computes the L1 norm of the tensor
    '''
    return tf.reduce_sum(tf.abs(tensor))

def l2_norm(tensor):
    '''
    Provides a Tensorflow op that computess the L2 norm of the given tensor
    :param tensor: the tensor whose L2 norm is to be computed
    :return: a TF op that computes the L2 norm of the tensor
    ''' 
    return tf.sqrt(tf.reduce_sum(tf.pow(tensor, 2)))

def l2_loss(tensor1, tensor2):
    '''
    Provides a Tensorflow op that computess the L2 loss (the Euclidean distance)
    between the tensors provided.
    :param tensor1: the first tensor
    :param tensor2: the other tensor
    :return: a TF op that computes the L2 distance between the tensors
    '''
    return tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(tensor1,tensor2), 2)))

def crossentropy_loss(y, model, mean=True):
    """
    Define crossentropy loss of TF graph
    Adapted from cleverhans library implementation
    :param y: correct labels
    :param model: output of the model, function only works with logits
    :param mean: boolean indicating whether should return mean of loss
    or vector of losses for each input of the batch
    :return: return mean of loss if True, otherwise return vector with per
    sample loss
    """
    out = tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y)
    if mean:
        out = tf.reduce_mean(out)
    return out

def read_img(path):
    '''
    Reads the image at path, checking if it was really loaded
    '''
    img = imread(path, mode="RGB")
    assert img is not None, "No image found at %s"%path
    return img

def write_img(path, image):
    '''
    Wrapper to allow easy replacement of image write function
    '''
    imsave(path, image)

def write_reverse_preprocess_inception(path, img):
    img += 1.0
    img /= 2.0
    img *= 255.0
    write_img(path, img)

def read_preprocessed_inception(path):
    '''
    Gives the image preprocessed for use in the inception classifier.
    '''
    img = read_and_resize_image(path, (FLAGS.image_height, FLAGS.image_width))/255.0
    return img * 2.0 - 1.0

def read_and_crop_image(path, x1, x2, y1, y2):
    '''
    Reads the image specified at path 
    and crops it according to the specified parameters, 
    returning it as a numpy array.
    '''
    img = read_img(path)
    return img[x1:x2,y1:y2]

def read_and_resize_image(path, newsize):
    '''
    Wrapper to allow easy substitution of resize function.
    Might be extended to allow for different resize methods
    '''
    img = read_img(path)
    if img.shape[0] != newsize[0] or img.shape[1] != newsize[1]:
        return imresize(img, newsize)
    else:
        return img

def model_eval(labels, net_predictions):
    return accuracy_score(labels, net_predictions, normalize=True) 

def load_all_pngjpg_in_dir(path_to_dir, limit=None):
    '''
    helper to load all jpeg and png files in a directory
    and return a nice tuple of (filenames, images)
    where images is in numpy format, preprocessed for inception

    if limit is set, only reads the first limit files after sorting alphabetically
    '''
    filenames = filter(lambda x: x.endswith(".png") or x.endswith(".jpg"), \
                        os.listdir(path_to_dir))
    if limit is not None:
        filenames = sorted(filenames)[:limit]
    
    data = map(lambda y: read_img( \
                         os.path.join(path_to_dir, y)), filenames)
    return filenames, np.array(data)

def read_data_inception(folder_path):
    data = []
    filenames = os.listdir(folder_path)
    np.random.shuffle(filenames)
    fnames_new = []
    for f in filenames:
        if not f.startswith(".") and \
           (f.endswith(".jpg") or f.endswith(".png")):
            data.append( \
                read_preprocessed_inception( \
                    os.path.join(folder_path, f)))
            fnames_new.append(f)

    return fnames_new, np.array(data)

def top3(model_out, j = 0):
    '''
    Given a classification output, returns the top 3 
    classes in an array of tuples (class, probability).
    for the specified index of model_out
    :param model_out: the output from the classification model
    :param j: the index in the output to use, in case there is more than one output vector in model_out. Defaults to 0
    :return: an array of 3 (class, probability) tuples in decreasing order of probability
    '''
    classes = zip(range(len(model_out[j])), model_out[j])
    return sorted(classes, key=lambda x: x[1], reverse=True)[:3]


def top3_as_string(model_out, j=0):
    '''
    Given the output of a classification, returns the top 3 classes
    as a string for the specified index of model_out
    '''
    return "".join(map(lambda x: str(x[0]) + " " + str(x[1]) + " ", top3(model_out, j)))
