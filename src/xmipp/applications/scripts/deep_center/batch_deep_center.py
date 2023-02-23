#!/usr/bin/env python3

import math
import numpy as np
from operator import itemgetter
import os
import sys
import xmippLib
from time import time
from scipy.ndimage import shift, rotate

if __name__ == "__main__":
    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed
    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    fnModel = sys.argv[2]
    print("------------------", flush=True)
    print(fnModel, flush=True)
    mode = sys.argv[3]
    sigma = float(sys.argv[4])
    numEpochs = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    gpuId = sys.argv[7]

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    from keras.callbacks import TensorBoard, ModelCheckpoint
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, concatenate, \
        Subtract, SeparableConv2D, GlobalAveragePooling2D
    from keras.optimizers import *
    import keras
    from keras import callbacks
    from keras.callbacks import Callback
    from keras import regularizers
    from keras.models import load_model
    import tensorflow as tf


    class DataGenerator(keras.utils.Sequence):
        'Generates data for fnImgs'

        def __init__(self, fnImgs, labels, mode, sigma, batch_size, dim, readInMemory):
            'Initialization'
            self.fnImgs = fnImgs
            self.labels = labels
            self.mode = mode
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
            def rotate_image(img, angle):
                return rotate(img, angle, order=1, mode='reflect', reshape=False)
            def shift_image(img, shiftx, shifty):
                return shift(img, (shiftx, shifty, 0), order=1, mode='reflect')
            def get_angles_radians(angles, angle):
                return np.array((math.sin(angles[0]), math.cos(angles[0]), math.sin(angles[1]),
                                 math.cos(angles[1]), math.sin(angles[2] + angle),
                                 math.cos(angles[2] + angle)))
            if self.readInMemory:
                Iexp = list(itemgetter(*list_IDs_temp)(self.Xexp))
            else:
                fnIexp = list(itemgetter(*list_IDs_temp)(self.fnImgs))
                Iexp = list(map(get_image, fnIexp))
            # Data augmentation
            if self.sigma > 0:
                if mode == 'Shift':
                    # Shift image a random amount of px in each direction
                    rX = (self.sigma/5) * np.random.normal(0, 1, size=self.batch_size)
                    rY = (self.sigma/5) * np.random.normal(0, 1, size=self.batch_size)
                    rX = rX + self.sigma * np.random.uniform(-1, 1, size=self.batch_size)
                    rY = rY + self.sigma * np.random.uniform(-1, 1, size=self.batch_size)
                    Xexp = np.array(list((map(shift_image, Iexp, rX, rY))))
                    y = yvalues + np.vstack((rX, rY)).T
                else:
                    # Rotates image a random angle. Thus, Psi must be updated
                    rAngle = self.sigma * np.random.normal(size=self.batch_size)
                    Xexp = np.array(list(map(rotate_image, Iexp, rAngle)))
                    rAngle = rAngle * math.pi / 180
                    yvalues = yvalues * np.pi / 180
                    y = np.array(list((map(get_angles_radians, yvalues, rAngle))))

            else:
                Xexp = np.array(list(Iexp))
                if mode == 'Shift':
                    y = yvalues
                else:
                    y = np.array(list((map(get_angles_radians, yvalues, np.zeros(self.batch_size)))))
            return Xexp, y

    def constructModel(Xdim, mode):
        inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")

        # Network model
        L = Conv2D(64, (int(Xdim / 20), int(Xdim / 20)), activation="relu")(inputLayer)  # 11 filter size before
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)

        L = Conv2D(64, (int(Xdim / 20), int(Xdim / 20)), activation="relu")(L)  # 11 filter size before
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Dropout(0.2)(L)

        L = Conv2D(32, (3, 3), activation="relu")(L)  # 11 filter size before
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Dropout(0.2)(L)

        L = Flatten()(L)
        L = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(L)
        L = BatchNormalization()(L)

        # L = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(L)
        # L = BatchNormalization()(L)

        if mode == 'Shift':
            L = Dense(2, name="output", activation="linear")(L)
        else:
            L = Dense(6, name="output", activation="linear")(L)
        return Model(inputLayer, L)


    Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnXmdExp)
    mdExp = xmippLib.MetaData(fnXmdExp)
    fnImgs = mdExp.getColumnValues(xmippLib.MDL_IMAGE)
    shiftX = mdExp.getColumnValues(xmippLib.MDL_SHIFT_X)
    shiftY = mdExp.getColumnValues(xmippLib.MDL_SHIFT_Y)
    rots = mdExp.getColumnValues(xmippLib.MDL_ANGLE_ROT)
    tilts = mdExp.getColumnValues(xmippLib.MDL_ANGLE_TILT)
    psis = mdExp.getColumnValues(xmippLib.MDL_ANGLE_PSI)

    # labels depends on mode (shift or Rot, Tilt, Psi)

    if mode == "Shift":
        labels = []
        for x, y in zip(shiftX, shiftY):
            labels.append(np.array((x, y)))
        ntraining = math.floor(len(fnImgs) * 0.8)
        training_generator = DataGenerator(fnImgs[0:ntraining], labels[0:ntraining], mode, sigma, batch_size, Xdim,
                                           readInMemory=False)
        validation_generator = DataGenerator(fnImgs[(ntraining + 1):], labels[(ntraining + 1):], mode, sigma,
                                             batch_size, Xdim, readInMemory=False)
    elif mode == "Angular":
        labels = []
        for r, t, p in zip(rots, tilts, psis):
            labels.append(np.array((r, t, p)))
        training_generator = DataGenerator(fnImgs, labels, mode, sigma, batch_size, Xdim,
                                           readInMemory=False)


    start_time = time()
    model = constructModel(Xdim, mode)

    model.summary()
    adam_opt = Adam(lr=0.001)


    # def custom_loss_function(y_true, y_pred):
    #     d = tf.abs(y_true - y_pred)
    #     return tf.reduce_mean(d, axis=-1)


    model.compile(loss="mean_absolute_error", optimizer='adam')

    steps = round(len(fnImgs) / batch_size)
    if mode == 'Shift':
        history = model.fit_generator(generator=training_generator, steps_per_epoch=steps, epochs=numEpochs, validation_data = validation_generator) #https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/
    else:
        history = model.fit_generator(generator=training_generator, steps_per_epoch=steps, epochs=numEpochs)

    model.save(fnModel)
    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)
