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

X_train /= 255

X_val = X_train[-2871:, :]
y_val = y_train[-2871:]
X_train = X_train[:-2871, :]
y_train = y_train[:-2871]

y_train = keras.utils.to_categorical(y_train)
y_val = keras.utils.to_categorical(y_val)
X_train = X_train.reshape((-1, 48, 48, 1))
X_val = X_val.reshape((-1, 48, 48, 1))

print('Preprocessing done')

datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

model = models.Model1()
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.8, patience=5, verbose=1)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
checkpointer = ModelCheckpoint('./model1.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)

bs = 256

model.fit_generator(datagen.flow(X_train, y_train, batch_size=bs), steps_per_epoch=np.ceil(X_train.shape[0] / bs), epochs=300, validation_data=(X_val, y_val), callbacks=[lr_reducer, early_stopper, checkpointer])
