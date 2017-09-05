from keras.datasets import mnist
from keras.utils import np_utils

"""
this file reads the data and preprocesses it.

"""
#load and preprocess dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train=X_train.reshape(-1,784)
X_test=X_test.reshape(-1,784)
X_train =X_train*1.0/255
X_test=X_test*1.0/255
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, n_output)
Y_test = np_utils.to_categorical(y_test, n_output)