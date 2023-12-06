import keras
from keras.preprocessing import image, image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, Model
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD


import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt


train_data = np.load('bottleneck_features/bn_features_train.npy', 'r')
train_labels = np.array([0] * 1000 + [1] * 1000)

validation_data = np.load('bottleneck_features/bn_features_validation.npy', 'r')
validation_labels = np.array([0] * 1000 + [1] * 1000)


fc_model = Sequential()
fc_model.add(Flatten(input_shape=train_data.shape[1:]))
fc_model.add(Dense(64, activation='relu', name='dense_one'))
fc_model.add(Dropout(0.5, name='dropout_one'))
fc_model.add(Dense(64, activation='relu', name='dense_two'))
fc_model.add(Dropout(0.5, name='dropout_two'))
fc_model.add(Dense(1, activation='sigmoid', name='output'))

fc_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


fc_model.fit(train_data, train_labels,
            epochs=50, batch_size=256,
            validation_data=(validation_data, validation_labels))
'''
fc_model.save_weights('bottleneck_features/fc_inception_cats_dogs_250.hdf5') # сохраняем веса

fc_model.evaluate(validation_data, validation_labels)'''
