import sys 

import cv2

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils.dataproc import gtsrb
from utils.model import YadavModel

FLAGS = flags.FLAGS
flags.DEFINE_string('weights', './models/old/model_best_test', 'path to the weights for the Yadav model')
flags.DEFINE_string('train_dataset', './clean_model/training_data/resized', 'Path to the training dataset')
flags.DEFINE_string('test_dataset', './clean_model/test_data/resized', 'Path to the testing dataset')

def pre_process_image(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    #print(image)
	
    image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    image[:,:,1] = cv2.equalizeHist(image[:,:,1])
    image[:,:,2] = cv2.equalizeHist(image[:,:,2])
    image = image/255. - 0.5
    return image

def main(argv=None):
    X_train, Y_train, X_test, Y_test = gtsrb(FLAGS.train_dataset, FLAGS.test_dataset)
    print 'Loaded GTSRB data'

    X_test = np.asarray(map(lambda x: pre_process_image(x), X_test.astype(np.uint8)),dtype=np.float32)

    with tf.Session() as sess:
        model = YadavModel()
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=FLAGS.weights)
        print 'Accuracy on test data', sess.run(model.accuracy, feed_dict={model.features: X_test, model.labels_true: Y_test,  model.keep_prob:1.0})


if __name__ == "__main__":
    app.run()
