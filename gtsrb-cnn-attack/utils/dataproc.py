'''
Module to process any image data. Includes functions to process the original training set
data and others to load up data for training and evaluation.
'''
import os
import csv
import random
import numpy as np
import cv2

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

def preprocess_yadav(image):
    '''
    Pre-process the image given as a Numpy array
    for the Yadav model.
    :param image: the image as a numpy array
    :return: a preprocessed image
    '''
    image = image.astype(np.uint8)
#    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
#    image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
#    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
    image = image/255. - 0.5
    return image.astype(np.float32)

def read_img(path):
    '''
    Reads the image at path, checking if it was really loaded
    '''
    img = cv2.imread(path)
    assert img is not None, "No image found at %s"%path
    return img

def write_img(path, image):
    '''
    Wrapper to allow easy replacement of image write function
    '''
    cv2.imwrite(path, image)

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
    return cv2.resize(img, newsize)

def count_images_for_labels_file(path, labels_filename="labels.csv"):
    '''
    Returns the number of images that will be processed
    for the given labels file
    '''
    total = len(_read_labels(path, labels_filename))
    return total #+ total%FLAGS.batch_size

def _read_labels(path, labels_filename="labels.csv"):
    '''
    Reads the labels.csv produced in the format of process_orig_data
    :return: a list of (filename, label) tuples
    '''
    filelist = []
    with open(os.path.join(path, labels_filename)) as f:
        f.readline()
        for line in f:
            fname, label = line.split(",")
            filelist.append((fname, label))
    
    return filelist

def _process_labels_file(path, labels_filename="labels.csv"):
    '''
    Processes a labels.csv at the specified path
    :param path: the path to the DIRECTORY of the labels.csv file 
    :return: X, Y where X is the images and Y is the corresponding labels
    '''
    filelist = _read_labels(path, labels_filename)
    random.SystemRandom().shuffle(filelist)
    X = []
    Y = []
    for i in xrange(len(filelist)):
        X.append(read_img(os.path.join(path,filelist[i][0])))
        Y.append(filelist[i][1])

    return  np.array(X, dtype='float32'), np.array(Y, dtype='float32')

def yield_gtsrb_preprocessed(path, batch_size, labels_filename="labels.csv"):
    '''
    Processes a labels.csv at the specified path
    and yields the images in batches, generator-style
    Batches are shuffled using SystemRandom each time.
    :param path: the path to the DIRECTORY of the labels.csv file 
    :param batch_size: how many images to yield at once
    :return: X, Y where X is the images and Y is the corresponding labels
    '''
    from keras.utils import np_utils
    filelist = _read_labels(path, labels_filename)
    total = len(filelist)
    random.SystemRandom().shuffle(filelist)
    X = []
    Y = []
    i = -1
    while True:
        if i > total or i < 0 or i+batch_size > total:
            i = 0
        else:
            i += batch_size

        yield ( \
                np.asarray(map(lambda x: \
                        (preprocess_vgg16(read_img(os.path.join(path, x[0])))), \
                    filelist[i:(i+batch_size)]), dtype='float32'), \
                np_utils.to_categorical(map(lambda x: x[1], filelist[i:(i+batch_size)]), FLAGS.nb_classes) \
             )


def preprocess_vgg16(img):
    '''
    Subtracts the mean value of each channel 
    taken over the training set of GTSRB images
    and returns the image.
    :param img: the image as a numpy array
    :return: a "normalized" image  
    '''
    return img - [85.7040159959 , 81.1492690065 , 91.4233658836] 

def gtsrb(train_path, test_path, labels_filename="labels.csv"):
    '''
    Loads the *processed* data from the GTSRB dataset
    for training and evaluation.
    Assumes each folder specified in train_path, test_path
    contains a labels.csv in the formate of process_orig_data
    Also subtracts the average pixel value of the test dataset.
    If either of train_path or test_path is unspecified, returns
    an empty array in the corresponding positions
    :return: X_train, Y_train, X_test, Y_test
    '''
    from tensorflow.python.platform import flags
    from keras.utils import np_utils

    if train_path is not None:
        X_train, Y_train = _process_labels_file(train_path, labels_filename)
    #X_train -= [86 , 81 , 91] 
    #X_train -= [85.7040159959 , 81.1492690065 , 91.4233658836] 
        Y_train = np_utils.to_categorical(Y_train, flags.FLAGS.nb_classes)
    else:
        X_train = []
        Y_train = []

    if test_path is not None:
        X_test, Y_test = _process_labels_file(test_path, labels_filename)
    #X_test -= [86 , 81 , 91] 
    #X_test -= [85.7040159959 , 81.1492690065 , 91.4233658836] 
        Y_test = np_utils.to_categorical(Y_test, flags.FLAGS.nb_classes)
    else:
        X_test = []
        Y_test = []

    return X_train, Y_train, X_test, Y_test 


def process_orig_data(rootpath, savedir):
    '''
    Processes the original data (.ppm, original-sized files, as unzipped from download)
    and saves everything in the same directory, along with a csv file titled "labels.csv" 
    that gives the format of the resulting images in a two-column csv: imgname, label
    :param rootpath: path to the traffic sign data, for example './GTSRB/Training'
    :return: 
    '''
    with open(os.path.join(savedir,'labels.csv'), 'w') as lblfile:
        lblfile.write('imgname,label\n')

        # loop over each of the 42 classes
        for classnum in range(0,43):
            # get the subdirectory for the current class
            prefix = os.path.join(rootpath,format(classnum, '05d')) 

            print 'now in folder %s'%prefix

            # open the annotations file
            with open(os.path.join(prefix, 'GT-'+ format(classnum, '05d') + '.csv')) as annotfile:
                # csv parser for annotations file
                annotreader = csv.reader(annotfile, delimiter=';')

                # skip header
                annotreader.next()

                # loop over all images in current annotations file
                for row in annotreader:
                    orig_fname = row[0].split(".")[0]
                    # the 8th column is the label
                    new_fname = "%03d_%s.png"%(int(row[7]), orig_fname)
                    lblfile.write("%s,%d\n"%(new_fname, int(row[7])))

                    orig_fpath = os.path.join(prefix,row[0])
                    write_img(os.path.join(savedir, new_fname), read_and_crop_image(orig_fpath, int(row[3]), int(row[5]), int(row[4]), int(row[6])))

    print 'Done.'

def process_orig_test_data(rootpath, savedir, csvfile='GT-final_test.csv'):
    '''
    Same as process_orig_data but assumes folder structure is one level less deep
    (i.e. all images are in only one folder)
    and allows you to specify the name of the annotations file explicitly in
    csvfile.
    '''
    with open(os.path.join(savedir,'labels.csv'), 'w') as lblfile:
        lblfile.write('imgname,label\n')


        with open(os.path.join(rootpath, csvfile)) as annotfile:
            # csv parser for annotations file
            annotreader = csv.reader(annotfile, delimiter=';')

            # skip header
            annotreader.next()

            # loop over all images in current annotations file
            for row in annotreader:
                orig_fname = row[0].split(".")[0]
                # the 8th column is the label
                new_fname = "%03d_%s.png"%(int(row[7]), orig_fname)
                lblfile.write("%s,%d\n"%(new_fname, int(row[7])))

                orig_fpath = os.path.join(rootpath, row[0])
                write_img(os.path.join(savedir, new_fname), 
                          read_and_crop_image(orig_fpath, int(row[3]), int(row[5]), int(row[4]), int(row[6])))

    print 'Done.'
