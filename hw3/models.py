from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import regularizers

def Model1():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))

    return model

def Model2():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(Dropout(0.4))
    model.add(Dense(7, activation='softmax', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))

    return model

def Model3():
    model = Sequential()

    model.add(Conv2D(60, (3, 3), activation='relu', input_shape=(48, 48, 1), kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(60, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(MaxPooling2D())

    model.add(BatchNormalization())
    model.add(Conv2D(90, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(90, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(90, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(MaxPooling2D())

    model.add(BatchNormalization())
    model.add(Conv2D(120, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(120, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(7, (1, 1), kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    return model

def Model4():
    model = Sequential()

    model.add(Conv2D(30, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1), kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(30, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(30, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(MaxPooling2D())

    model.add(BatchNormalization())
    model.add(Conv2D(60, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(60, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(60, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(MaxPooling2D())

    model.add(BatchNormalization())
    model.add(Conv2D(90, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(90, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(90, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(MaxPooling2D())

    model.add(BatchNormalization())
    model.add(Conv2D(120, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(120, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(7, (1, 1), kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    return model

def Model5():
    model = Sequential()

    model.add(Conv2D(30, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1), kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(30, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(30, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(MaxPooling2D())

    model.add(BatchNormalization())
    model.add(Conv2D(60, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(60, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(60, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(MaxPooling2D())

    model.add(BatchNormalization())
    model.add(Conv2D(90, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(90, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(90, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(MaxPooling2D())

    model.add(BatchNormalization())
    model.add(Conv2D(120, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(120, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(7, (1, 1), kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))

    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax', kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='he_uniform'))

    return model
