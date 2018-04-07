'''
Functions to evaluate the accuracy of different models.
'''
import keras
import numpy as np

def model_eval(sess, x, y, model, X_test, Y_test, additional_feed={}):
    """
    Generic function working with any classification model
    Borrowed from cleverhans utils tf, but does not process accuracy in batches.
    :param sess: TF session to use when running the graph
    :param x: input placeholder
    :param y: output placeholder holding CORRECT labels
    :param model: model output predictions
    :param X_test: numpy array with evaluation inputs
    :param Y_test: numpy array with CORRECT evaluation outputs
    :param additional_feed: extra arguments to include in the feed_dict, if necessary
    :return: a float with the accuracy value
    """

    # Define symbol for accuracy
    acc_value = keras.metrics.categorical_accuracy(y, model)

    # Init result var
    feed = additional_feed
    feed[x] = X_test
    feed[y] = Y_test
    feed[keras.backend.learning_phase()] = 0

    with sess.as_default():
        accuracy = acc_value.eval(feed_dict=feed)

    return accuracy

def model_eval_custom(sess, x, model, X_test, Y_test, additional_feed={}):
    """
    Evaluates the accuracy of the dataset
    by "manually" counting up the number of correct classifications
    :param sess: TF session to use when running the graph
    :param x: input placeholder
    :param model: model output predictions
    :param X_test: numpy array with evaluation inputs
    :param Y_test: numpy array with CORRECT evaluation outputs
    :param additional_feed: extra arguments to include in the feed_dict, if necessary
    :return: a float with the accuracy value
    """

    # Init result var
    feed = additional_feed
    feed[x] = X_test

    with sess.as_default():
        model_out = model.eval(feed_dict=feed)
    
    correct = 0
    total = len(model_out)
    for i in xrange(total):
        if np.argmax(model_out[i]) == np.argmax(Y_test[i]):
            correct += 1

    accuracy = float(correct)/float(total)

    return accuracy

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
