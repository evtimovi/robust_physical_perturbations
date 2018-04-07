
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge
from keras.layers import Convolution2D, AveragePooling2D, MaxPooling2D

from tensorflow.python.platform import flags
import tensorflow as tf
FLAGS = flags.FLAGS

def get_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def get_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def conv_layer(input,
               num_inp_channels,
               filter_size,
               num_filters,
              use_pooling):
    shape = [filter_size, filter_size, num_inp_channels,num_filters]
    weights = get_weights(shape)
    biases = get_biases(num_filters)
    layer = tf.nn.conv2d(input = input,
                        filter = weights,
                        strides = [1,1,1,1],
                        padding = 'SAME')

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME')
    layer = tf.nn.relu(layer)

    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def fc_layer(input,          # The previous layer.
             num_inputs,     # Num. inputs from prev. layer.
             num_outputs,    # Num. outputs.
             use_relu=True): # Use Rectified Linear Unit (ReLU)?
    weights = get_weights(shape=[num_inputs, num_outputs])
    biases = get_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer,weights

def dropout_layer(layer, keep_prob):
    layer_drop = tf.nn.dropout(layer, keep_prob)
    return layer_drop

def fc_layer(input,          # The previous layer.
             num_inputs,     # Num. inputs from prev. layer.
             num_outputs,    # Num. outputs.
             use_relu=True): # Use Rectified Linear Unit (ReLU)?
    weights = get_weights(shape=[num_inputs, num_outputs])
    biases = get_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer,weights

def dropout_layer(layer, keep_prob):
    layer_drop = tf.nn.dropout(layer, keep_prob)
    return layer_drop

#############################Model Def########################################################

def vgg16(img_rows=FLAGS.img_rows, img_cols=FLAGS.img_cols, channels=FLAGS.nb_channels, nb_classes=FLAGS.nb_classes):
    '''
    Defines the VGG 16 model using the Keras Sequential model
    Follows architecture D defined in this paper:
    https://arxiv.org/abs/1409.1556
    :param img_rows: number of row in the image
    :param img_cols: number of columns in the image
    :param channels: number of color channels (e.g., 1 for MNIST)
    :param nb_classes: the number of output classes
    :return: a Keras model. Call with model(<input_tensor>)
    '''
    model = Sequential()

    input_shape = (img_rows, img_cols, channels)

# see Note_Convolution2D.md for details on the usage of Convolution2D
    layers = [Convolution2D(64,3,3, border_mode='same', subsample=(1,1), input_shape=input_shape),
        Activation('relu'),
        Convolution2D(64,3,3, border_mode='same', subsample=(1,1)),
        Activation('relu'),
        MaxPooling2D((2,2), strides=(2,2)),
        Convolution2D(128,3,3, border_mode='same', subsample=(1,1)),
        Activation('relu'),
        Convolution2D(128,3,3, border_mode='same', subsample=(1,1)),
        Activation('relu'),
        MaxPooling2D((2,2), strides=(2,2)),
        Convolution2D(256,3,3, border_mode='same', subsample=(1,1)),
        Activation('relu'),
        Convolution2D(256,3,3, border_mode='same', subsample=(1,1)),
        Activation('relu'),
        Convolution2D(256,3,3, border_mode='same', subsample=(1,1)),
        Activation('relu'),
        MaxPooling2D((2,2), strides=(2,2)),
        Convolution2D(512,3,3, border_mode='same', subsample=(1,1)),
        Activation('relu'),
        Convolution2D(512,3,3, border_mode='same', subsample=(1,1)),
        Activation('relu'),
        Convolution2D(512,3,3,  border_mode='same', subsample=(1,1)),
        Activation('relu'),
        MaxPooling2D((2,2), strides=(2,2)),
        Convolution2D(512,3,3, border_mode='same', subsample=(1,1)),
        Activation('relu'),
        Convolution2D(512,3,3, border_mode='same', subsample=(1,1)),
        Activation('relu'),
        Convolution2D(512,3,3,  border_mode='same', subsample=(1,1)),
        Activation('relu'),
        MaxPooling2D((2,2), strides=(2,2)),
        Flatten(),
        Dense(FLAGS.fc_number),
        Activation('relu'),
        Dropout(0.5),
        Dense(FLAGS.fc_number),
        Activation('relu'),
        Dropout(0.5),
        Dense(nb_classes),
        Activation('softmax')
    ]

    for layer in layers:
        model.add(layer)

    return model


def cunn_keras(img_rows=FLAGS.img_rows, img_cols=FLAGS.img_cols, channels=FLAGS.nb_channels, nb_classes=FLAGS.nb_classes):
    '''
    Defines the VGG 16 model using the Keras Sequential model
    :param img_rows: number of row in the image
    :param img_cols: number of columns in the image
    :param channels: number of color channels (e.g., 1 for MNIST)
    :param nb_classes: the number of output classes
    :return: a Keras model. Call with model(<input_tensor>)
    '''

    input = Input(shape=(img_rows, img_cols, channels))

    conv1 = Convolution2D(32,5,5, border_mode='same', subsample=(1,1), activation='relu')(input)
    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1)

    conv2 = Convolution2D(64,5,5, border_mode='same', subsample=(1,1), activation='relu')(pool1)
    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)

    conv3 = Convolution2D(128,5,5, border_mode='same', subsample=(1,1), activation='relu')(pool2)
    pool3 = MaxPooling2D((2,2), strides=(2,2))(conv3)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    flat_all = merge([flat1, flat2, flat3], mode='concat', concat_axis=1) #If this gives an error, update the keras tensorflow backend. It is likely that is making the call tf.concat(axis, [to_dense(x) for x in tensors]) in of tf.concat([to_dense(x) for x in tensors], axis)

    fc = Dense(1024)(flat_all)
    drop = Dropout(0.5)(fc)
    fc2 = Dense(nb_classes)(drop)
    output = Activation('softmax',name='prob')(fc2)

    model = Model(input=input, output=output)

    return model

def cunn_tf(img_rows=FLAGS.img_rows, img_cols=FLAGS.img_cols, channels=FLAGS.nb_channels, nb_classes=FLAGS.nb_classes):
    '''
    Defines the VGG 16 model using the Keras Sequential model
    :param img_rows: number of row in the image
    :param img_cols: number of columns in the image
    :param channels: number of color channels (e.g., 1 for MNIST)
    :param nb_classes: the number of output classes
    :return: a Keras model. Call with model(<input_tensor>)
    '''

    features = tf.placeholder(tf.float32, shape=[None, img_rows, img_cols, channels], name='features')
    labels_true = tf.placeholder(tf.float32,shape=[None,nb_classes], name='y_true')
    labels_true_cls = tf.argmax(labels_true, dimension=1)

    ## Dropout
    #drop_prob = 0.5

    keep_prob = tf.placeholder(tf.float32)

    layer_conv1, weights_conv1 = conv_layer(input=features,
                   num_inp_channels=3,
                   filter_size=5,
                   num_filters=32,
                   use_pooling=True)
    layer_conv2, weights_conv2 = conv_layer(input=layer_conv1,
                   num_inp_channels=32,
                   filter_size=5,
                   num_filters=64,
                   use_pooling=True)
    layer_conv3, weights_conv3 = conv_layer(input=layer_conv2_drop,
                   num_inp_channels=64,
                   filter_size=5,
                   num_filters=128,
                   use_pooling=True)

    layer_flat1, num_fc_layers1 = flatten_layer(layer_conv1)
    layer_flat2, num_fc_layers2 = flatten_layer(layer_conv2)
    layer_flat3, num_fc_layers3 = flatten_layer(layer_conv3)

    layer_flat = tf.concat([layer_flat1, layer_flat2, layer_f3at3],1)
    num_fc_layers = num_fc_layers1+num_fc_layers2+num_fc_layers6

    fc_layer1,weights_fc1 = fc_layer(layer_flat,          # The previous layer.
             num_fc_layers,     # Num. inputs from prev. layer.
             1024,    # Num. outputs.
             use_relu=True)
    fc_layer1_drop = dropout_layer(fc_layer1, keep_prob)

    fc_layer2,weights_fc2 = fc_layer(fc_layer1_drop,          # The previous layer.
             1024,     # Num. inputs from prev. layer.
             nb_classes,    # Num. outputs.
             use_relu=False)

    predictions = tf.nn.softmax(fc_layer2)

    return predictions

class YadavModel():
    '''
    Implements the road sign recognition algorithm given here:
    https://github.com/evtimovi/p2-TrafficSigns and described here:
    https://chatbotslife.com/german-sign-classification-using-deep-learning-neural-networks-98-8-solution-d05656bf51ad
    :return: a TF op that can be run and processes the model
    '''

    def __init__(self, img_rows=FLAGS.img_rows, 
                 img_cols=FLAGS.img_cols, num_channels=FLAGS.nb_channels, 
                 n_classes=FLAGS.nb_classes, train=True,
                 custom_input=None):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.num_channels = num_channels
        self.n_classes = n_classes
        if custom_input is not None:
            self.features = custom_input
        else:
            self.features = tf.placeholder(tf.float32, shape=[None, self.img_rows, self.img_cols, self.num_channels],
                       name='features')
        ## Convlayer 0
        self.filter_size0 = 1
        self.num_filters0 = 3

        ## Convlayer 1
        self.filter_size1 = 5
        self.num_filters1 = 32
        ## Convlayer 2
        self.filter_size2 = 5
        self.num_filters2 = 32

        ## Convlayer 3
        self.filter_size3 = 5
        self.num_filters3 = 64
        ## Convlayer 4
        self.filter_size4 = 5
        self.num_filters4 = 64

        ## Convlayer 5
        self.filter_size5 = 5
        self.num_filters5 = 128
        ## Convlayer 6
        self.filter_size6 = 5
        self.num_filters6 = 128

        ## FC_size
        self.fc_size1 = 1024
        ## FC_size
        self.fc_size2 = 1024

        ## Dropout
        #drop_prob = 0.5

        self.keep_prob = tf.placeholder(tf.float32)

        self.layer_conv0, self.weights_conv0 = \
            self.conv_layer(input=self.features,
                       num_inp_channels=self.num_channels,
                       filter_size=self.filter_size0,
                       num_filters=self.num_filters0,
                       use_pooling=False)

        self.layer_conv1, self.weights_conv1 = \
                self.conv_layer(input=self.layer_conv0,
                           num_inp_channels=self.num_filters0,
                           filter_size=self.filter_size1,
                           num_filters=self.num_filters1,
                           use_pooling=False)
        self.layer_conv2, self.weights_conv2 = \
                self.conv_layer(input=self.layer_conv1,
                           num_inp_channels=self.num_filters1,
                           filter_size=self.filter_size2,
                           num_filters=self.num_filters2,
                           use_pooling=True)
        self.layer_conv2_drop = self.dropout_layer(self.layer_conv2, self.keep_prob)

        self.layer_conv3, self.weights_conv3 = \
                self.conv_layer(input=self.layer_conv2_drop,
                           num_inp_channels=self.num_filters2,
                           filter_size=self.filter_size3,
                           num_filters=self.num_filters3,
                           use_pooling=False)
        self.layer_conv4, self.weights_conv4= \
                self.conv_layer(input=self.layer_conv3,
                           num_inp_channels=self.num_filters3,
                           filter_size=self.filter_size4,
                           num_filters=self.num_filters4,
                           use_pooling=True)
        self.layer_conv4_drop = dropout_layer(self.layer_conv4, self.keep_prob)

        self.layer_conv5, self.weights_conv5 = \
                self.conv_layer(input=self.layer_conv4_drop,
                           num_inp_channels=self.num_filters4,
                           filter_size=self.filter_size5,
                           num_filters=self.num_filters5,
                           use_pooling=False)
        self.layer_conv6, self.weights_conv6 = \
                conv_layer(input=self.layer_conv5,
                           num_inp_channels=self.num_filters5,
                           filter_size=self.filter_size6,
                           num_filters=self.num_filters6,
                           use_pooling=True)    
        self.layer_conv6_drop = dropout_layer(self.layer_conv6, self.keep_prob)


        self.layer_flat2, self.num_fc_layers2 = self.flatten_layer(self.layer_conv2_drop)
        self.layer_flat4, self.num_fc_layers4 = self.flatten_layer(self.layer_conv4_drop)
        self.layer_flat6, self.num_fc_layers6 = self.flatten_layer(self.layer_conv6_drop)

        self.layer_flat = tf.concat([self.layer_flat2, self.layer_flat4, self.layer_flat6], 1)
        self.num_fc_layers = self.num_fc_layers2+self.num_fc_layers4+self.num_fc_layers6

        self.fc_layer1,self.weights_fc1 = self.fc_layer(self.layer_flat,          # The previous layer.
                     self.num_fc_layers,     # Num. inputs from prev. layer.
                     self.fc_size1,    # Num. outputs.
                     use_relu=True)
        self.fc_layer1_drop = self.dropout_layer(self.fc_layer1, self.keep_prob)

        self.fc_layer2,self.weights_fc2 = self.fc_layer(self.fc_layer1_drop,          # The previous layer.
                     self.fc_size1,     # Num. inputs from prev. layer.
                     self.fc_size2,    # Num. outputs.
                     use_relu=True)
        self.fc_layer2_drop = self.dropout_layer(self.fc_layer2, self.keep_prob)

        self.fc_layer3,self.weights_fc3 = self.fc_layer(self.fc_layer2_drop,          # The previous layer.
                     self.fc_size2,     # Num. inputs from prev. layer.
                     n_classes,
                     use_relu=False)

        self.labels_pred = tf.nn.softmax(self.fc_layer3)

        if train:
            self.labels_true = tf.placeholder(tf.float32,shape=[None,self.n_classes], name='y_true')
            self.labels_true_cls = tf.argmax(self.labels_true, dimension=1)
            self.labels_pred_cls = tf.argmax(self.labels_pred, dimension=1)


            self.regularizers = (tf.nn.l2_loss(self.weights_conv0) 
                + tf.nn.l2_loss(self.weights_conv1) + tf.nn.l2_loss(self.weights_conv2) 
                + tf.nn.l2_loss(self.weights_conv3) + tf.nn.l2_loss(self.weights_conv4) 
                + tf.nn.l2_loss(self.weights_conv5) + tf.nn.l2_loss(self.weights_conv6) 
                + tf.nn.l2_loss(self.weights_fc1)  + tf.nn.l2_loss(self.weights_fc2) +
                tf.nn.l2_loss(self.weights_fc3))

            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_layer3,
                                                        labels=self.labels_true)
            self.cost = tf.reduce_mean(self.cross_entropy)+1e-5*self.regularizers
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.cost)

            self.correct_prediction = tf.equal(self.labels_pred_cls, self.labels_true_cls)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def get_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def get_biases(self, length):
        return tf.Variable(tf.constant(0.05, shape=[length]))
    
    def conv_layer(self, input,
                   num_inp_channels,
                   filter_size,
                   num_filters,
                  use_pooling):
        shape = [filter_size, filter_size, num_inp_channels,num_filters]
        weights = get_weights(shape)
        biases = get_biases(num_filters)
        layer = tf.nn.conv2d(input = input,
                            filter = weights,
                            strides = [1,1,1,1],
                            padding = 'SAME')
    
        layer += biases
    
        if use_pooling:
            layer = tf.nn.max_pool(value=layer,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
        layer = tf.nn.relu(layer)
    
        return layer, weights
    def flatten_layer(self, layer):
        # Get the shape of the input layer.
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat, num_features
    def fc_layer(self, input,          # The previous layer.
             num_inputs,     # Num. inputs from prev. layer.
             num_outputs,    # Num. outputs.
             use_relu=True): # Use Rectified Linear Unit (ReLU)?
        weights = get_weights(shape=[num_inputs, num_outputs])
        biases = get_biases(length=num_outputs)
        layer = tf.matmul(input, weights) + biases

        if use_relu:
            layer = tf.nn.relu(layer)

        return layer,weights

    def dropout_layer(self, layer, keep_prob):
        layer_drop = tf.nn.dropout(layer, keep_prob)
        return layer_drop

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
