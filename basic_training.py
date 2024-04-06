import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models, utils, preprocessing

def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        images = batch['data'].reshape((len(batch['data']),3,32,32)).transpose(0,2,3,1)
        labels = batch['labels']
        return images, labels
    

def load_data(path):
    num_train_samples = 50000
    x_train = np.empty((num_train_samples,32,32,3), dtype='uint8')
    y_train = np.empty((num_train_samples,),dtype='uint8')

    for i in range(1,6):
        images, labels = load_cifar10_batch(os.path.join(path, f'data_batch_{i}'))
        x_train[(i-1) * 10000: i*10000, :, :,:] = images
        y_train[(i-1) * 10000: i*10000, :, :,:] = labels
    
    x_test, y_test = load_cifar10_batch(os.path.join(path, 'test_batch'))

    return (x_train, y_train), (x_test, y_test)



path_to_dataset = '/data/cifar-10-batches-py'

(train_images, train_labels), (test_images, test_labels) = load_data(path_to_dataset)
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)
