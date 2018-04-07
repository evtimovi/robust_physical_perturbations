#work-around for pylint bug 1869: https://github.com/PyCQA/pylint/issues/1869
from __future__ import print_function 

import os
import numpy as np
import sys
import math

import tensorflow as tf
from tensorflow.python.platform import app 
from tensorflow.python.platform import flags

import flags
from attack_util import AttackGraph, read_img, \
                        model_eval, write_img, \
                        load_all_pngjpg_in_dir, \
                        read_preprocessed_inception, \
                        write_reverse_preprocess_inception, \
                        read_data_inception

from sklearn.metrics import accuracy_score
import random
import gc

FLAGS = flags.FLAGS

def inception(x_input):
    '''
    Builds the inception network model,
    loads its weights from FLAGS.checkpoint_path,
    and returns the softmax activations tensor.
    '''
    from tensorflow.contrib.slim.nets import inception as inception_tf
    slim = tf.contrib.slim
    with slim.arg_scope(inception_tf.inception_v3_arg_scope()):
        _, end_points = inception_tf.inception_v3(x_input, \
                                                  num_classes=FLAGS.num_classes, \
                                                  is_training=False)

    return end_points['Logits']

def get_print_triplets(file_path):
    '''
    Reads the printability triplets from the specified file
    and returns a numpy array of shape (num_triplets, FLAGS.img_cols, FLAGS.img_rows, nb_channels)
    where each triplet has been copied to create an array the size of the image
    :return: as described 
    '''     
    p = []  
        
    # load the triplets and create an array of the speified size
    with open(file_path) as f:
        for l in f:
            p.append(l.split(",")) 
    p = map(lambda x: [[x for _ in xrange(FLAGS.image_width)] for __ in xrange(FLAGS.image_height)], p)
    p = np.float32(p)
    p *= 2.0
    p -= 1.0
    return p


def get_noise_init_from_flags():
    if FLAGS.noise_initial == "zeros":
        return tf.constant_initializer(0.0)
    elif FLAGS.noise_initial == "random_normal":
        return tf.random_normal_initializer(mean=FLAGS.noise_init_mean, \
                                            stddev=FLAGS.noise_init_stddev)
    else:
        raise Exception("FLAGS.noise_initial must be zeros or random_normal. Currently %s"%\
        FLAGS.noise_initial)

def get_reg_losses_from_flags():
    if FLAGS.reglosses != "":
        return [x.strip() for x in FLAGS.reglosses.split(",")]
    else:
        return []

def restore_or_init_noise(sess, attack_graph, noise_path):
    if noise_path != "":
        noise_restorer = tf.train.Saver(var_list=[attack_graph.noise])
        noise_restorer.restore(sess, noise_path)
        return True
    else:
        sess.run(attack_graph.init_noise)
        return False

def restore_model_vars(sess, model_vars, path):
    restorer = tf.train.Saver(var_list=model_vars)
    restorer.restore(sess, path)


def get_all_subfolders(path):
    folders = [os.path.join(path, x) for x in os.listdir(path)]
    return filter(lambda x: not x.endswith("DS_Store"), folders)

class Attack(object):
    def __init__(self, just_apply_noise, batch_size):
        self.just_apply_noise = just_apply_noise

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = FLAGS.tf_allow_growth
        self.sess = tf.Session(config=tfconfig)
        with self.sess.as_default():
            self.attack_graph = AttackGraph(batch_size=batch_size, \
                                            image_height=FLAGS.image_height, \
                                            image_width=FLAGS.image_width, \
                                            image_channels=FLAGS.image_channels, \
                                            noise_initializer=get_noise_init_from_flags(), \
                                            num_classes=FLAGS.num_classes, \
                                            pixel_low=FLAGS.pixel_low, \
                                            pixel_high=FLAGS.pixel_high)

            if not self.just_apply_noise:
                self.losses_dict = None
                self.reg_losses = get_reg_losses_from_flags()
                self.optimization_op = self.attack_graph.build_everything(inception, \
                                            self.reg_losses, \
                                            epsilon=FLAGS.adam_epsilon, \
                                            beta1=FLAGS.adam_beta1, \
                                            beta2=FLAGS.adam_beta2)
                self.sess.run(self.attack_graph.init_adam)
                restore_model_vars(self.sess, self.attack_graph.model_vars, FLAGS.model_path)
                
                _, self.train_data = read_data_inception(FLAGS.attack_srcdir)
                _, self.val_data = read_data_inception(FLAGS.validation_set)

                self.saver = tf.train.Saver(max_to_keep=2, \
                                            var_list=[self.attack_graph.noise])
            
            restore_or_init_noise(self.sess, self.attack_graph, FLAGS.noise_restore_checkpoint)

    def _get_color_shifts(self, shape_of_color_shifts):
        return np.ones(shape_of_color_shifts)*np.random.uniform(FLAGS.color_shifts_min, FLAGS.color_shifts_max) \
                                        if not self.just_apply_noise \
                                        else np.ones(shape_of_color_shifts)
    
    def _get_boxes(self, n):
        return [[np.random.uniform(0.0, 0.2), \
                np.random.uniform(0.0, 0.2), \
                np.random.uniform(0.8, 1.0), \
                np.random.uniform(0.8, 1.0)] for _ in range(n)] \
                if not self.just_apply_noise \
                else [[0, 0, 1, 1] for _ in range(n)]
    
    def _get_dest_points(self, shape):
        
        n = shape[0]
        img_rows = shape[-3]
        img_cols = shape[-2]

        # source points
        src = [[[0,0],[0,img_cols],[img_rows,0],[img_rows,img_cols]] for _ in range(n)]
        
        if self.just_apply_noise:
            return src

        import scipy.stats as stats

        lower, upper = -img_rows/3, img_rows/3
        mu, sigma = FLAGS.transform_mean, FLAGS.transform_stddev
        X = stats.truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        # we will add this to the source points, i.e. these are random offsets
        # random = np.random.normal(FLAGS.transform_mean, FLAGS.transform_stddev, (n, 4, 2))
        random = X.rvs((n, 4, 2))
        return src + random

    def create_feed_dict(self, data, attack_graph):    
        if len(data.shape) == 4:
            n = data.shape[0]
            shape_of_color_shifts = data.shape
        else:
            raise Exception("data needs to be of rank 4, currently %d"%len(data.shape))
            
        feed_dict = {
            attack_graph.clean_input: data, \
            attack_graph.mask: read_img(FLAGS.attack_mask)/255.0, \
            attack_graph.color_shifts: self._get_color_shifts(shape_of_color_shifts),  
            attack_graph.boxes: self._get_boxes(n), \
            attack_graph.dest_points: self._get_dest_points(shape_of_color_shifts)
        }

        if not self.just_apply_noise:
            feed_dict[attack_graph.learning_rate] = FLAGS.attack_learning_rate

            if self.losses_dict is None:
                self.losses_dict = {}
                targets = np.zeros(attack_graph.output_shape)
                targets[:, FLAGS.attack_target] = 1.0
                self.losses_dict[attack_graph.attack_target] = targets

                for l in attack_graph.reg_names:
                    self.losses_dict[attack_graph.reg_lambdas[l]] = FLAGS.__dict__["__flags"]["lambda_%s"%l]
                    if l == "l2image":
                        self.losses_dict[attack_graph.l2image] = read_preprocessed_inception(FLAGS.l2image)
                    elif l == "nps":
                        self.losses_dict[attack_graph.nps_triplets] = get_print_triplets(FLAGS.printability_tuples)
            
            feed_dict.update(self.losses_dict)
        return feed_dict
        
    def calculate_acc(self):
        assert FLAGS.validation_set is not None
        assert self.val_data is not None

        val_feed_dict = self.create_feed_dict(np.array(self.val_data), self.attack_graph)

        net_predictions = self.sess.run(tf.argmax(self.attack_graph.adv_pred, axis=1), \
                                        feed_dict=val_feed_dict)
        labels = [FLAGS.attack_target for _ in range(len(net_predictions))]
        
        val_feed_dict = None
        gc.collect()

        return accuracy_score(labels, net_predictions, normalize=True)

    def epoch(self, epoch_num):
        assert self.train_data is not None
        n_imgs = len(self.train_data)
        batch_size = FLAGS.attack_batch_size
        assert n_imgs%batch_size == 0, "n_imgs is %d, batch size is %d"%(n_imgs, batch_size)
        num_batches = n_imgs/batch_size

        report = (epoch_num % FLAGS.save_freq == 0)
        
        num_reg_losses = len(self.attack_graph.reg)

        if report:
            avg_losses = np.zeros((2 + num_reg_losses,))
        else:
            avg_losses = np.ones((2+num_reg_losses)) * 100000.0        

        for b in range(0, n_imgs, batch_size):
            curr_data = self.train_data[b : b+batch_size]
            self.feed_dict = self.create_feed_dict(np.array(curr_data), self.attack_graph)

            if epoch_num < FLAGS.first_epoch_with_reg:
                for l in self.attack_graph.reg_names:
                    self.feed_dict[self.attack_graph.reg_lambdas[l]] = 0.0
            else:
                self.feed_dict[self.attack_graph.learning_rate] = FLAGS.attack_learning_rate/FLAGS.learning_rate_reduce_factor

            if report:
                ops_to_run = [self.optimization_op, self.attack_graph.total_loss, self.attack_graph.class_loss]
                for r in self.attack_graph.reg_names:
                    ops_to_run.append(self.attack_graph.reg[r])
            else:
                ops_to_run = self.optimization_op
            
            results = self.sess.run(ops_to_run, feed_dict=self.feed_dict)
            
            if report:
                avg_losses += results[1:]

        if report:
            avg_losses /= float(num_batches)

            report_string = "Epoch %d"%epoch_num
            report_string += " total_loss %.4f"%avg_losses[0]
            report_string += " class_loss %.4f"%avg_losses[1]

            for rname, val in zip(self.attack_graph.reg_names, avg_losses[2:]):
                report_string += " %s_loss %.4f"%(rname, val)
            
            acc = self.calculate_acc()
            report_string += " val_accuracy %.2f"%acc

            print(report_string)
            sys.stdout.flush()

            noise_img, victim_img = self.sess.run(\
                [self.attack_graph.get_noise_extract_op(), \
                self.attack_graph.final_noisy_inputs[0]],  \
                feed_dict=self.feed_dict)
        else:
            noise_img = None
            victim_img = None
            acc = 0.0

        self.feed_dict = None
        gc.collect()

        return acc, avg_losses[0], noise_img, victim_img

    def optimize(self, num_epochs):
        latest_acc = 0.0
        latest_loss = 10000.0
        for e in range(num_epochs):
            acc, loss, noise_img, victim_img = self.epoch(e)

            if acc > latest_acc or (acc == latest_acc and loss < latest_loss):
                latest_acc = acc
                latest_loss = loss
                self.saver.save(self.sess, \
                                os.path.join(FLAGS.save_folder, FLAGS.save_prefix, "%s_epoch_%d"%(FLAGS.save_prefix, e)))
            if noise_img is not None:
                write_reverse_preprocess_inception( \
                    os.path.join(FLAGS.save_folder, FLAGS.save_prefix, "noise-epoch-%04d.png"%e), noise_img)

            if victim_img is not None:
                write_reverse_preprocess_inception( \
                    os.path.join(FLAGS.save_folder, FLAGS.save_prefix, "victim-epoch-%04d.png"%e), victim_img)


    def extract_noise(self, folder, fnames, data):
        feed_dict = self.create_feed_dict(data, self.attack_graph) 
        noisy_images = self.sess.run( \
                        tf.clip_by_value(self.attack_graph.noisy_inputs, FLAGS.pixel_low, FLAGS.pixel_high), \
                        feed_dict=feed_dict)

        save_folder = os.path.join(folder, "noisy_%s"%FLAGS.noise_restore_checkpoint)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        
        for name, img in zip(fnames, noisy_images):
            fpath = os.path.join(save_folder, name)
            print("Writing %s"%fpath)
            write_reverse_preprocess_inception(fpath, img)

def main(argv=None):
    flags.load_config_file()
    flags.print_flags()
    if not FLAGS.just_apply_noise:
        attack = Attack(False, FLAGS.attack_batch_size)
        attack.optimize(FLAGS.attack_epochs)
    else:
        assert FLAGS.apply_folder != ""
        fnames, data = read_data_inception(FLAGS.apply_folder)
        attack = Attack(True, len(fnames)) 
        print("apply folder", FLAGS.apply_folder)
        attack.extract_noise(FLAGS.apply_folder, fnames, data)

if __name__ == '__main__':
    app.run()
