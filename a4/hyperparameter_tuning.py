from cs231n.classifiers.neural_net import TwoLayerNet
import numpy as np
input_size = 32 * 32 * 3
num_classes = 10

import logging

logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='ht.log',
                filemode='w')

from cs231n.data_utils import load_CIFAR10
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()


# hyperparameter: 
# hidden layer size -- increasing h_l_s will 
# decrease the training error but there is over-fitting possibility.
# num_iters 
# these two hyperparameters will increase computational cost
# learing_rate
# regularation parameter
# use grid search to tune the parameter
learning_rate_list=[1e-4,5*1e-4,25*1e-4,125*1e-4,625*1e-4]
reg_list=[0.05,0.1,0.5,0.9,1.5]
hidden_size_list=[50,70,90]
num_iteration_increment_list=[500,500,1000]
best_val_acc=0
logging.info("*****begin of hyperparameter tuning*****")
for i in learning_rate_list:
    logging.info("for learning_rate=%.4f tuning "%i)
    for j in reg_list:
        for h in hidden_size_list:
            net = TwoLayerNet(input_size, h, num_classes)
            iteration_total=0
            for k in num_iteration_increment_list:
                net.train(X_train, y_train, X_val, y_val,
                    num_iters=k, batch_size=200,
                    learning_rate=i, learning_rate_decay=0.95,
                    reg=j, verbose=False)
                val_acc = (net.predict(X_val) == y_val).mean()
                iteration_total += k
                if(val_acc>best_val_acc):
                    best_val_acc=val_acc
                    logging.info("best net: acc=%.3f,for learning_rate=%.4f,\
                    reg=%.3f,num_iters=%d,hidden_size=%d"%(best_val_acc,i,j,iteration_total,h))
logging.info("*****end of hyperparameter tuning*****")

