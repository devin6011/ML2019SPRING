import models

import numpy as np
import pandas as pd
import io
import keras
import sys

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

print('Loading data...')
trainData = pd.read_csv('train.csv', skiprows=[0], header=None)
X_train = pd.read_csv(io.StringIO(trainData.iloc[:, 1].to_csv(header=False, index=False)), sep='\s+', header=None).to_numpy(dtype=float)
y_train = trainData.iloc[:, 0].to_numpy(dtype=float)
print('Loading data done')

print('Preprocessing')

y_train = keras.utils.to_categorical(y_train)
X_train = X_train.reshape((-1, 48, 48, 1))

print('Preprocessing done')

datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        brightness_range=[0.5, 1.5],
        rescale=1/255,
        horizontal_flip=True)

model = models.Model2()
print(model.summary())

sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.1, min_lr=0.00001, patience=3, verbose=1, cooldown=5, min_delta=0.01)
early_stopper = EarlyStopping(monitor='loss', min_delta=0.01, patience=30, verbose=1, mode='auto')

bs = 256

model.fit_generator(datagen.flow(X_train, y_train, batch_size=bs), steps_per_epoch=np.ceil(X_train.shape[0] / bs), epochs=300, callbacks=[lr_reducer, early_stopper])

model.save_weights('model2.h5')
