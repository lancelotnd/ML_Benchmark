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
        y_train[(i-1) * 10000: i*10000] = labels
    
    x_test, y_test = load_cifar10_batch(os.path.join(path, 'test_batch'))

    return (x_train, y_train), (x_test, y_test)



path_to_dataset = '/data/cifar-10-batches-py'

(train_images, train_labels), (test_images, test_labels) = load_data(path_to_dataset)
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)


## Training part

base_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3))

# Create the model
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(10, activation='softmax'))  # CIFAR-10 has 10 classes


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# Data Augmentation
data_augmentation = preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# Fit the model
batch_size = 64
epochs = 10  # For simplicity, but you may need more epochs for higher accuracy

model.fit(data_augmentation.flow(train_images, train_labels, batch_size=batch_size),
          steps_per_epoch=len(train_images) // batch_size,
          epochs=epochs,
          validation_data=(test_images, test_labels))