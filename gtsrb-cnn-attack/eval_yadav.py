'''
Evaluates the accuracy of the Yadav model on the GTSRB dataset.
Uses independent accuracy function from the one used during training.
'''
import sys
import tensorflow as tf
from utils.dataproc import gtsrb, preprocess_yadav
from utils.eval import model_eval, model_eval_custom
from utils.model import YadavModel

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('test_dataset', './clean_model/test_data/resized', 'Path to the testing dataset')
flags.DEFINE_string('labels_filename', 'labels_usstop14.csv', 'Filename to use in the test folder to load the labels from')
flags.DEFINE_string('weights', './models/gtsrb_usstop/model_best_test', 'Location of the weights to load for this model')

def main(argv=None):
    with tf.Session() as sess:
        model = YadavModel(train=False)
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.weights)
        print 'Loaded model from %s'%FLAGS.weights
        sys.stdout.flush()
        _, __, X_test, Y_test = gtsrb(train_path=None, test_path=FLAGS.test_dataset, labels_filename=FLAGS.labels_filename)
        X_test = map(lambda x: preprocess_yadav(x), X_test)
        print "Loaded GTSRB test data"
        y = tf.placeholder(tf.float32, shape=Y_test.shape)
        initial_feed_dict = {model.keep_prob: 1.0}
        print 'Accuracy (Keras)', model_eval(sess, model.features, y, model.labels_pred, X_test, Y_test, initial_feed_dict)
        print 'Accuracy (manual)', model_eval_custom(sess, model.features, model.labels_pred, X_test, Y_test, initial_feed_dict)

if __name__ == "__main__":
    app.run()
