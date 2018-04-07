'''
Holds parameter definitions for model.
'''
from tensorflow.python.platform import flags
import json
import math

FLAGS = flags.FLAGS
flags.DEFINE_integer("attack_epochs", 1, \
                    "how many iterations to run the attack for")
flags.DEFINE_integer("attack_batch_size", 1, 
                    "batch size to use in the attack")
flags.DEFINE_integer('image_height', 299, \
                        'Height of each input images.')
flags.DEFINE_integer('image_width', 299, \
                        'Width of each input images.')
flags.DEFINE_integer("image_channels", 3, \
                        "Number of channels in input image")
flags.DEFINE_string('noise_initial', '', \
    'Specifies the initialization of the noise \
    Options: zeros, random_normal')
flags.DEFINE_float("noise_init_mean", 0.0, \
                    "when random normally initializing the noise, use this as mean")
flags.DEFINE_float("noise_init_stddev", 1.0, \
                    "when random normally initializing the noise, use this as stddev")
flags.DEFINE_integer('num_classes', 1001, \
                        'The number of classes this network is trained on')
flags.DEFINE_float('pixel_high', 1.0, \
    'The maximum pixel value in this setting')
flags.DEFINE_float('pixel_low', 0.0, \
    'The minimum pixel value in this setting')
flags.DEFINE_boolean("tf_allow_growth", False, \
                    "Whether to let tensorflow take up all the memory. True doesn't let it grow, False lets it grow.")
flags.DEFINE_boolean('just_apply_noise', False, \
                    'simply applies the noise to the video in folder FLAGS.apply_folder and does not optimize')
flags.DEFINE_string('apply_folder', "", \
                    'if just_apply_noise is set, applies noise to videos in this folder')
flags.DEFINE_string("reglosses", "", \
                    "specify a comma separated list of regularization losses")
flags.DEFINE_float("lambda_l2norm", 0.1, \
                    "specifies the l2 norm lambda")
flags.DEFINE_float("lambda_tv", 0.1, \
                    "specifies the tv norm lambda")
flags.DEFINE_float("lambda_l1norm", 0.1, \
                    "specifies the l1norm norm lambda")
flags.DEFINE_float("lambda_l2image", 0.1, \
                    "specifies the l2image norm lambda")
flags.DEFINE_string("l2image", "", \
                    "path to the image to be used in l2 comparison")
flags.DEFINE_float('attack_learning_rate', 0.1,
                    'The rate of optimization for the attack')
flags.DEFINE_float('adam_beta1', 0.9, \
    'The beta1 parameter for the Adam optimizer of the attack objective')
flags.DEFINE_float('adam_beta2', 0.999, \
    'The beta2 parameter for the Adam optimizer of the attack objective')
flags.DEFINE_float('adam_epsilon', 1e-08, \
    'The epsilon parameter for the Adam optimizer of the attack objective')
flags.DEFINE_string('model_path', '', 'Filepath to original model')
flags.DEFINE_string('attack_srcdir', '',
                    'Filepath to the directory containing the images to train on.')
flags.DEFINE_string('attack_mask', '', 'Filepath to the mask used in the attack.')
flags.DEFINE_integer('attack_target', 5, \
                     'The class being targeted for this attack as a number')
flags.DEFINE_string('config_file', '', \
    'if not empty, overrides any defaults or command line args with arguments in this config file')
flags.DEFINE_string("noise_restore_checkpoint", "",
                    "If not empty, loads noise from here.")
flags.DEFINE_string("validation_set", "", \
                    "path to a folder containing a validation set")
flags.DEFINE_string("save_folder", "", \
                    "where to save checkpoints and noise images")
flags.DEFINE_string("save_prefix", "", \
                    "what prefix to use when saving checkpoints and noise images")
flags.DEFINE_integer("save_freq", 50, \
                    "how often to save, i.e. whenever an epoch number is a perfect multiple of this number")
flags.DEFINE_float("transform_mean", 0.0, \
                    "the mean of the destination points offset when generating the perspective transforms")
flags.DEFINE_float("transform_stddev", 30.0, \
                    "the std dev of the destination points offset when generating the perspective transforms")
flags.DEFINE_float("color_shifts_min", 0.1, 
                    "minimum value to multiply by")
flags.DEFINE_float("color_shifts_max", 1.0,
                    "maximum value to multiply by")
flags.DEFINE_string('printability_tuples', "",
                    'Specifies which file to load the printability tuples from. \
                    Used only if nps is included in the reg losses')
flags.DEFINE_integer("first_epoch_with_reg", 0, \
                    "the first epoch in which regularization will be applied")
flags.DEFINE_float("learning_rate_reduce_factor", 1.0, \
                    "how much to reduce the learning rate when starting to use regularization")


def load_config_file():
    if FLAGS.config_file != "":
        with open(FLAGS.config_file) as json_data:
            config = json.load(json_data)
        for key, value in config.iteritems():
            setattr(FLAGS, key, value)

def print_flags():
    print("Parameters")
    for k in sorted(FLAGS.__dict__["__flags"].keys()):
        print("%s: %s"%(k, FLAGS.__dict__["__flags"][k]))
