from keras.preprocessing import image, image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

from keras import backend as K
#K.set_image_dim_ordering('th')

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt

inc_model = InceptionV3(include_top=False, weights='imagenet',input_shape=(256, 256, 3), classes=3)

image_size = (256, 256) #определяем размер изображения
batch_size = 256#размер батча(количество загруежнных в память изображений за раз

train = image_dataset_from_directory('C:/PyProject/Inseption/datasets/cats and dogs/train',
                                     subset='training',
                                     seed=42,
                                     validation_split=0.01,
                                     batch_size=batch_size,
                                     image_size=image_size)

validation = image_dataset_from_directory('C:/PyProject/Inseption/datasets/cats and dogs/valid',
                                          subset='validation',
                                          batch_size=batch_size,
                                          seed=42,
                                          validation_split=0.99,
                                          image_size=image_size)

test = image_dataset_from_directory('C:/PyProject/Inseption/datasets/cats and dogs/test',
                                    batch_size=batch_size,
                                    image_size=image_size)

bottleneck_features_train = inc_model.predict(train, steps=2000)
np.save('bottleneck_features/bn_features_train.npy', bottleneck_features_train)

bottleneck_features_validation = inc_model.predict(validation, steps=2000)
np.save('bottleneck_features/bn_features_validation.npy', bottleneck_features_validation)















