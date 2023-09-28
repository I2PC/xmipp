#!/usr/bin/env python3

import math
import numpy as np
from operator import itemgetter
import os
import sys
import xmippLib
from time import time
from scipy.ndimage import shift
import glob

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

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    from keras.callbacks import ModelCheckpoint
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, concatenate, Activation
    from keras.optimizers import *
    import keras
    from keras.models import load_model
    import tensorflow as tf


    class DataGenerator(keras.utils.all_utils.Sequence):
        """Generates data for fnImgs"""

        def __init__(self, fnImgs, labels, sigma, batch_size, dim, readInMemory):
            """Initialization"""
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
            """Denotes the number of batches per epoch"""
            num_batches = int(np.floor((len(self.labels)) / self.batch_size))
            return num_batches

        def __getitem__(self, index):
            """Generate one batch of data"""
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
            """Updates indexes after each epoch"""
            self.indexes = [i for i in range(len(self.labels))]
            np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):
            """Generates data containing batch_size samples"""
            yvalues = np.array(itemgetter(*list_IDs_temp)(self.labels))

            # Functions to handle the data
            def get_image(fn_image):
                """Normalize image"""
                img = np.reshape(xmippLib.Image(fn_image).getData(), (self.dim, self.dim, 1))
                return (img - np.mean(img)) / np.std(img)

            def shift_image(img, shiftx, shifty):
                """Shifts image in X and Y"""
                return shift(img, (shiftx, shifty, 0), order=1, mode='wrap')

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


    def constructModel(Xdim):
        """CNN architecture"""
        inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")

        L = Conv2D(8, (3, 3), padding='same')(inputLayer)
        L = BatchNormalization()(L)
        L = Activation(activation='relu')(L)
        L = MaxPooling2D()(L)

        L = Conv2D(16, (3, 3), padding='same')(L)
        L = BatchNormalization()(L)
        L = Activation(activation='relu')(L)
        L = MaxPooling2D()(L)

        L = Conv2D(32, (3, 3), padding='same')(L)
        L = BatchNormalization()(L)
        L = Activation(activation='relu')(L)
        L = MaxPooling2D()(L)

        L = Conv2D(64, (3, 3), padding='same')(L)
        L = BatchNormalization()(L)
        L = Activation(activation='relu')(L)
        L = MaxPooling2D()(L)

        L = Flatten()(L)

        L = Dense(2, name="output", activation="linear")(L)

        return Model(inputLayer, L)


    def get_labels(fnImages):
        """Returns dimensions, images and shifts values from images files"""
        Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnImages)
        mdExp = xmippLib.MetaData(fnImages)
        fnImgs = mdExp.getColumnValues(xmippLib.MDL_IMAGE)
        shiftX = mdExp.getColumnValues(xmippLib.MDL_SHIFT_X)
        shiftY = mdExp.getColumnValues(xmippLib.MDL_SHIFT_Y)
        labels = []
        for x, y in zip(shiftX, shiftY):
            labels.append(np.array((x, y)))
        return Xdim, fnImgs, labels


    Xdim, fnImgs, labels = get_labels(fnXmdExp)
    start_time = time()

    # Train-Validation sets
    if numModels == 1:
        lenTrain = int(len(fnImgs) * 0.8)
        lenVal = len(fnImgs) - lenTrain
    else:
        lenTrain = int(len(fnImgs) / 3)
        print('lenTrain', lenTrain, flush=True)
        lenVal = int(len(fnImgs) / 12)


    for index in range(numModels):
        random_sample = np.random.choice(range(0, len(fnImgs)), size=lenTrain + lenVal, replace=False)
        model = constructModel(Xdim)
        adam_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
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

        history = model.fit(training_generator, epochs=numEpochs,
                            validation_data=validation_generator, callbacks=[save_best_model, patienceCallBack])

    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)