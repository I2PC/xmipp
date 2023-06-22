#!/usr/bin/env python3

import math
import numpy as np
from operator import itemgetter
import os
import sys
import xmippLib
from time import time
from scipy.ndimage import shift, rotate
import matplotlib.pyplot as plt

if __name__ == "__main__":

    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    fnModel = sys.argv[2]
    sigma = float(sys.argv[3])
    numEpochs = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    gpuId = sys.argv[6]
    numModels = int(sys.argv[7])
    learning_rate = float(sys.argv[8])
    patience = int(sys.argv[9])
    pretrained = sys.argv[10]
    if pretrained == 'yes':
        fnPreModel = sys.argv[11]

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    from keras.callbacks import TensorBoard, ModelCheckpoint
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, concatenate, \
        Activation, Subtract, SeparableConv2D, GlobalAveragePooling2D, AveragePooling2D, LeakyReLU, Add
    from keras.optimizers import *
    import keras
    from keras import callbacks
    from keras.callbacks import Callback
    from keras import regularizers
    from keras.models import load_model
    import tensorflow as tf


    class DataGenerator(keras.utils.Sequence):
        'Generates data for fnImgs'

        def __init__(self, fnImgs, labels, sigma, batch_size, dim, readInMemory):
            'Initialization'
            self.fnImgs = fnImgs
            self.labels = labels
            self.sigma = sigma
            self.batch_size = batch_size
            if self.batch_size > len(self.fnImgs):
                self.batch_size = len(self.fnImgs)
            self.dim = dim
            self.readInMemory = readInMemory
            self.on_epoch_end()


            # Read all data in memory
            if self.readInMemory:
                self.Xexp = np.zeros((len(self.labels), self.dim, self.dim, 1), dtype=np.float64)
                for i in range(len(self.labels)):
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[i]).getData(), (self.dim, self.dim, 1))
                    self.Xexp[i,] = (Iexp - np.mean(Iexp)) / np.std(Iexp)

        def __len__(self):
            'Denotes the number of batches per epoch'
            num_batches = int(np.floor((len(self.labels)) / self.batch_size))
            return num_batches

        def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            # Find list of IDs
            list_IDs_temp = []
            for i in range(int(self.batch_size)):
                list_IDs_temp.append(indexes[i])
            # Generate data
            Xexp, y = self.__data_generation(list_IDs_temp)

            return Xexp, y

        def on_epoch_end(self):
            'Updates indexes after each epoch'
            self.indexes = [i for i in range(len(self.labels))]
            np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):
            'Generates data containing batch_size samples'
            yvalues = np.array(itemgetter(*list_IDs_temp)(self.labels))

            # Functions to handle the data
            def get_image(fn_image):
                img = np.reshape(xmippLib.Image(fn_image).getData(), (self.dim, self.dim, 1))
                return (img - np.mean(img)) / np.std(img)

            def shift_image(img, shiftx, shifty):
                return shift(img, (shiftx, shifty, 0), order=1, mode='reflect')

            if self.readInMemory:
                Iexp = list(itemgetter(*list_IDs_temp)(self.Xexp))
            else:
                fnIexp = list(itemgetter(*list_IDs_temp)(self.fnImgs))
                Iexp = list(map(get_image, fnIexp))
            # Data augmentation
            rX = self.sigma * np.random.normal(0, 1, size=self.batch_size)
            rY = self.sigma * np.random.normal(0, 1, size=self.batch_size)
            rX = rX + self.sigma * np.random.uniform(-1, 1, size=self.batch_size)
            rY = rY + self.sigma * np.random.uniform(-1, 1, size=self.batch_size)
            # Shift image a random amount of px in each direction
            Xexp = np.array(list((map(shift_image, Iexp, rX, rY))))
            y = yvalues + np.vstack((rX, rY)).T
            return Xexp, y

    def conv_block1(x, filters, kernel_size, strides=(1, 1), activation='relu', batch_normalization=True):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x


    def identity_block1(tensor, filters, kernel_size, activation='relu', batch_normalization=True):
        x = conv_block(tensor, filters, kernel_size, activation=activation, batch_normalization=batch_normalization)
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        x = Add()([tensor, x])
        x = Activation(activation)(x)
        return x

    def identity_block(tensor, filters):

        x = Conv2D(filters, (3, 3), padding='same')(tensor)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=3)(x)

        x = Add()([x, tensor])
        x = Activation('relu')(x)
        return x

    def conv_block(tensor, filters):
        x = Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(tensor)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=3)(x)

        x_res = Conv2D(filters, (1, 1), strides=(2, 2))(tensor)

        x = Add()([x, x_res])
        x = Activation('relu')(x)
        return x

    def constructModel(Xdim):
        inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")

        x = conv_block(inputLayer, filters=32)

        x = conv_block(x, filters=64)

        for _ in range(0):
            x = identity_block(x, filters=64)

        x = conv_block(x, filters=128)

        for _ in range(0):
            x = identity_block(x, filters=128)

        x = conv_block(x, filters=256)

        for _ in range(0):
            x = identity_block(x, filters=256)

        x = conv_block(x, filters=512)

        for _ in range(0):
            x = identity_block(x, filters=512)

        x = GlobalAveragePooling2D()(x)

        x = Dense(2, name="output", activation="linear")(x)

        return Model(inputLayer, x)

    def get_labels(fnImages):
        Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnImages)
        mdExp = xmippLib.MetaData(fnImages)
        fnImgs = mdExp.getColumnValues(xmippLib.MDL_IMAGE)
        shiftX = mdExp.getColumnValues(xmippLib.MDL_SHIFT_X)
        shiftY = mdExp.getColumnValues(xmippLib.MDL_SHIFT_Y)
        labels = []
        for x, y in zip(shiftX, shiftY):
            labels.append(np.array((x, y)))
        return Xdim, fnImgs, labels

    import tensorflow as tf
    import keras.backend as K

    Xdim, fnImgs, labels = get_labels(fnXmdExp)
    start_time = time()

    if numModels == 1:
        lenTrain = int(len(fnImgs)*0.8)
        lenVal = len(fnImgs)-lenTrain
    else:
        lenTrain = int(len(fnImgs) / 5)
        print('lenTrain', lenTrain, flush=True)
        lenVal = int(len(fnImgs) / 20)

    for index in range(numModels):

        random_sample = np.random.choice(range(0, len(fnImgs)), size=lenTrain+lenVal, replace=False)
        if pretrained == 'yes':
            model = load_model(fnPreModel, compile=False)
        else:
            model = constructModel(Xdim)
        adam_opt = Adam(lr=learning_rate)
        model.summary()

        model.compile(loss='mean_absolute_error', optimizer='adam')

        save_best_model = ModelCheckpoint(fnModel + str(index) + ".h5", monitor='val_loss',
                                          save_best_only=True)
        patienceCallBack = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

        training_generator = DataGenerator([fnImgs[i] for i in random_sample[0:lenTrain]],
                                           [labels[i] for i in random_sample[0:lenTrain]],
                                           sigma, batch_size, Xdim, readInMemory=False)
        validation_generator = DataGenerator([fnImgs[i] for i in random_sample[lenTrain:lenTrain + lenVal]],
                                             [labels[i] for i in random_sample[lenTrain:lenTrain + lenVal]],
                                             sigma, batch_size, Xdim, readInMemory=False)

        history = model.fit_generator(generator=training_generator, epochs=numEpochs,
                                      validation_data=validation_generator, callbacks=[save_best_model, patienceCallBack])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)
