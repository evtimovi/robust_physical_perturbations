from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

# flags generic to training and attack
flags.DEFINE_integer('nb_classes', 43, 'Number of classification classes') #Used for All_r
flags.DEFINE_integer('img_rows', 32, 'Input row dimension')
flags.DEFINE_integer('img_cols', 32, 'Input column dimension')
flags.DEFINE_integer('nb_channels', 3, 'Nb of color channels in the input.')

flags.DEFINE_integer('tf_random_seed', 12345, 'Tensorflow random seed')
