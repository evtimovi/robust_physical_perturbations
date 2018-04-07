import sys 

import cv2
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils.dataproc import gtsrb
from utils.model import YadavModel

FLAGS = flags.FLAGS
flags.DEFINE_string('weights', './clean_model/checkpoints/model_best_test', 'path to the weights for the Yadav model')
flags.DEFINE_string('train_dataset', './clean_model/training_data/resized', 'Path to the training dataset')
flags.DEFINE_string('test_dataset', './clean_model/test_data/resized', 'Path to the testing dataset')
flags.DEFINE_string('labels', 'labels_usstop14.csv', 'The name of the labels file to use.')

def pre_process_image(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    #print(image)
	
    #image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    #image[:,:,1] = cv2.equalizeHist(image[:,:,1])
    #image[:,:,2] = cv2.equalizeHist(image[:,:,2])
    image = image/255.-0.5
    return image

def transform_image(image,ang_range,shear_range,trans_range):

    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = image.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    image = cv2.warpAffine(image,Rot_M,(cols,rows))
    image = cv2.warpAffine(image,Trans_M,(cols,rows))
    image = cv2.warpAffine(image,shear_M,(cols,rows))

    image = pre_process_image(image.astype(np.uint8))

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    #image = image[:,:,0]
    #image = cv2.resize(image, (img_resize,img_resize),interpolation = cv2.INTER_CUBIC)

    return image


def gen_transformed_data(X_train,Y_train,N_classes,n_each,ang_range,shear_range,trans_range,randomize_Var):

   X_arr = []
   Y_arr = []
   for i in range(len(X_train)):
       print(i)
       lab =  Y_train[i]

       X_arr.append(X_train[i,:,:,:])
       Y_arr.append(lab)
       for i_n in range(n_each):
           img_trf = transform_image(X_train[i,:,:,:], ang_range,shear_range,trans_range)
           X_arr.append(img_trf)
           Y_arr.append(lab)

   X_arr = np.array(X_arr,dtype = np.float32())
   Y_arr = np.array(Y_arr,dtype = np.float32())

   if (randomize_Var == 1):
       len_arr = np.arange(len(Y_arr))
       np.random.shuffle(len_arr)
       X_arr[len_arr] = X_arr
       Y_arr[len_arr] = Y_arr


   return X_arr,Y_arr

def acc_eval(session, model, X_input, Y_input, batch_size):

    num_batches =  X_input.shape[0] // batch_size
    cur_acc = 0
    for acc_batch in range(num_batches):
        start = acc_batch*batch_size
	end = min(len(X_input), start + batch_size)
	cur_batch_len = end-start
        feed_dict_eval = {model.features: X_input[start:end,:,:,:],
                           model.labels_true: Y_input[start:end,:],
                           model.keep_prob:1.0}
        cur_acc += session.run(model.accuracy, feed_dict=feed_dict_eval)*cur_batch_len
    return cur_acc/X_input.shape[0]
	
def random_batch(X_train, Y_train, batch_size):
    # Number of images in the training-set.
    num_images = len(X_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=batch_size,
                           replace=False)
    # Use the random index to select random images and labels.
    features_batch = X_train[idx, :,:,:]
    labels_batch = Y_train[idx, :]

    return features_batch, labels_batch

def optimize(session,model, x_train, y_train, X_test, Y_test, num_iterations, batch_size):
    total_iterations = 0
    best_validation_accuracy = 0.0
    last_improvement = 0
    best_test_accuracy = 0.0
    require_improvement = 10000
	
    X_train, X_valid, Y_train , Y_valid = train_test_split(x_train,
                                                 y_train,
                                                 test_size=0.1,
                                                 random_state=22)

    valida = X_valid[:np.int(np.floor(X_valid.shape[0]/2)),:,:,:].shape[0]
    validb = X_valid[np.int(np.floor(X_valid.shape[0]/2)+1):,:,:,:].shape[0]

    size1=X_test[:np.int(np.floor(X_test.shape[0]/4)),:,:,:].shape[0]
    size2=X_test[np.int(np.floor(X_test.shape[0]/4)+1):np.int(np.floor(X_test.shape[0]/2)),:,:,:].shape[0]
    size3=X_test[np.int(np.floor(X_test.shape[0]/2)+1):np.int(np.floor(X_test.shape[0]*3/4)),:,:,:].shape[0]
    size4=X_test[np.int(np.floor(X_test.shape[0]*3/4)+1):,:,:,:].shape[0]

    val_acc_list = []
    batch_acc_list = []
    test_acc_list = []

    #feed_dict_valid = {model.features: X_valid[:np.int(np.floor(X_valid.shape[0]/2)),:,:,:],
    #     model.labels_true: Y_valid[:np.int(np.floor(X_valid.shape[0]/2)),:],
    #     model.keep_prob:1.0}
    #feed_dict_valid2 = {model.features: X_valid[np.int(np.floor(X_valid.shape[0]/2)+1):,:,:,:],
    #     model.labels_true: Y_valid[np.int(np.floor(X_valid.shape[0]/2)+1):,:],
    #     model.keep_prob:1.0}

    #feed_dict_test = {model.features: X_test[:np.int(np.floor(X_test.shape[0]/4)),:,:,:],
    #     model.labels_true: Y_test[:np.int(np.floor(X_test.shape[0]/4)),:],
    #     model.keep_prob:1.0}
    #feed_dict_test2 = {model.features: X_test[np.int(np.floor(X_test.shape[0]/4)+1):np.int(np.floor(X_test.shape[0]/2)),:,:,:],
    #     model.labels_true: Y_test[np.int(np.floor(X_test.shape[0]/4)+1):np.int(np.floor(X_test.shape[0]/2)),:],
    #     model.keep_prob:1.0}
    #feed_dict_test3 = {model.features: X_test[np.int(np.floor(X_test.shape[0]/2)+1):np.int(np.floor(X_test.shape[0]*3/4)),:,:,:],
    #     model.labels_true: Y_test[np.int(np.floor(X_test.shape[0]/2)+1):np.int(np.floor(X_test.shape[0]*3/4)),:],
    #     model.keep_prob:1.0}
    #feed_dict_test4 = {model.features: X_test[np.int(np.floor(X_test.shape[0]*3/4)+1):,:,:,:],
    #     model.labels_true: Y_test[np.int(np.floor(X_test.shape[0]*3/4)+1):,:],
    #     model.keep_prob:1.0}

    #num_iter = np.math.floor(len(labels_train_rot)/batch_size)
    #num_iter = min(num_iter-2,num_iterations)
    num_iter = num_iterations
    for i in range(num_iter):
	print(i)
        total_iterations+=1
        # Get batch for training
        features_batch, labels_true_batch = random_batch(X_train, Y_train, batch_size)
        #idx = np.arange(batch_size*i,batch_size*(i+1))
        #features_batch = Image_GS_train_rot[idx, :,:,:]
        #labels_true_batch = labels_train_rot[idx, :]f
        feed_dict_batch = {model.features:features_batch,
                            model.labels_true: labels_true_batch,
                            model.keep_prob: 0.5}
        session.run(model.optimizer,feed_dict = feed_dict_batch)
        if ((total_iterations % 200 == 0) or (i == (num_iter - 1))):
            # Calculate the accuracy on the training-set.
	    acc_valid = acc_eval(session, model, X_valid, Y_valid, batch_size)
            #acc_batch = session.run(model.accuracy, feed_dict=feed_dict_batch)
            #acc_valida = session.run(model.accuracy,feed_dict=feed_dict_valid)
            #acc_validb = session.run(model.accuracy,feed_dict=feed_dict_valid2)
	    #acc_valid = (acc_valida *valida + acc_validb*validb)/(valida+validb)
            val_acc_list.append(acc_valid)
            #batch_acc_list.append(acc_batch)
            print(acc_valid)
            if acc_valid > best_validation_accuracy:
                best_validation_accuracy = acc_valid
                last_improvement = total_iterations
                improved_str = '*'
                saver = tf.train.Saver()
                saver.save(sess=session, save_path='models/model_best_batch')
            else:
                improved_str = ''
            
            if total_iterations - last_improvement > require_improvement:
                print("No improvement found in a while, stopping optimization.")
                break

            # Message for printing.
            if ((total_iterations % 5000 == 0) or (i == (num_iter - 1))):
                msg = "# {0:>6}, Train Acc.: {1:>6.1%}, Val Acc.: {2:>6.1%}, Test Acc.: {3:>6.1%}"
                acc_test = acc_eval(session, model, X_test, Y_test, batch_size)
                #acc_test = session.run(model.accuracy,feed_dict=feed_dict_test)
                #acc_testa = session.run(model.accuracy,feed_dict=feed_dict_test)
                #acc_testb = session.run(model.accuracy,feed_dict=feed_dict_test2)
                #acc_testc = session.run(model.accuracy,feed_dict=feed_dict_test3)
                #acc_testd = session.run(model.accuracy,feed_dict=feed_dict_test4)
                #acc_test = (acc_testa*size1 + acc_testb*size2+acc_testc*size3 + acc_testd*size4)/(size1+size2+size3+size4)
                
                if best_test_accuracy<acc_test:
                    saver = tf.train.Saver()
                    saver.save(sess=session, save_path='models/model_best_test')
                    best_test_accuracy = acc_test
                    #print_accuracy()
                # Print it.
                print(msg.format(i+1, acc_valid,acc_valid,
                                     acc_test))

def main(argv=None):
    X_train, Y_train, X_test, Y_test = gtsrb(FLAGS.train_dataset, FLAGS.test_dataset, labels_filename=FLAGS.labels)
    print 'Loaded GTSRB data'

    X_train = np.asarray(map(lambda x: pre_process_image(x), X_train.astype(np.uint8)),dtype=np.float32)
    X_test = np.asarray(map(lambda x: pre_process_image(x), X_test.astype(np.uint8)),dtype=np.float32)
    global total_iterations 
    global best_validation_accuracy
    global last_improvement
    global best_test_accuracy 
    
    global val_acc_list 
    global batch_acc_list 
    global test_acc_list


    with tf.Session() as sess:
        model = YadavModel()
	sess.run(tf.initialize_all_variables())
        #X_train, Y_train = gen_transformed_data(X_train,Y_train,43,10,30,5,5,1)
	print(X_train.shape)
	print(Y_train.shape)
	optimize(sess, model, X_train, Y_train, X_test, Y_test, 10000, 128)
       
if __name__ == "__main__":
    app.run()
