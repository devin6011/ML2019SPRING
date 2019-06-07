import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, DepthwiseConv2D, LeakyReLU
from keras import regularizers

def Model1():
    model = Sequential()

    model.add(Conv2D(30, 3, strides=1, padding='same', input_shape=(48, 48, 1), kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())

    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, strides=1, padding='same', depthwise_regularizer=regularizers.l2(4e-5), depthwise_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(30, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())

    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, strides=1, padding='same', depthwise_regularizer=regularizers.l2(4e-5), depthwise_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(30, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())

    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, strides=1, padding='same', depthwise_regularizer=regularizers.l2(4e-5), depthwise_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(30, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())


    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, strides=2, padding='same', depthwise_regularizer=regularizers.l2(4e-5), depthwise_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(45, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())
    
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, strides=1, padding='same', depthwise_regularizer=regularizers.l2(4e-5), depthwise_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(45, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())

    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, strides=1, padding='same', depthwise_regularizer=regularizers.l2(4e-5), depthwise_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(45, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())

    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, strides=1, padding='same', depthwise_regularizer=regularizers.l2(4e-5), depthwise_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(45, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())


    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, strides=2, padding='same', depthwise_regularizer=regularizers.l2(4e-5), depthwise_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(45, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())

    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, strides=1, padding='same', depthwise_regularizer=regularizers.l2(4e-5), depthwise_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(45, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())

    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, strides=1, padding='same', depthwise_regularizer=regularizers.l2(4e-5), depthwise_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(45, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())

    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, strides=1, padding='same', depthwise_regularizer=regularizers.l2(4e-5), depthwise_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(45, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())


    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, strides=2, padding='same', depthwise_regularizer=regularizers.l2(4e-5), depthwise_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(45, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())

    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, strides=1, padding='same', depthwise_regularizer=regularizers.l2(4e-5), depthwise_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(45, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())

    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, strides=1, padding='same', depthwise_regularizer=regularizers.l2(4e-5), depthwise_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(45, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())

    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(3, strides=1, padding='same', depthwise_regularizer=regularizers.l2(4e-5), depthwise_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(45, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())


    model.add(GlobalAveragePooling2D())
    model.add(Dense(7, kernel_regularizer=regularizers.l2(4e-5), kernel_initializer='he_uniform'))
    model.add(Activation('softmax'))

    return model
