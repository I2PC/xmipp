#!/usr/bin/env python3

import math
import numpy as np
import os
import sys
import xmippLib
from time import time
from scipy.ndimage import shift, rotate

if __name__=="__main__":
    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed
    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    fnModel = sys.argv[2]
    mode = sys.argv[3]
    sigma = float(sys.argv[4])
    numEpochs = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    gpuId = sys.argv[7]

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    from keras.callbacks import TensorBoard, ModelCheckpoint
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, concatenate, Subtract, SeparableConv2D, GlobalAveragePooling2D
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
            self.dim = dim
            self.readInMemory=readInMemory
            self.on_epoch_end()

            # Read all data in memory
            if self.readInMemory:
                self.Xexp = np.zeros((len(self.labels),self.dim,self.dim,1),dtype=np.float64)
                for i in range(len(self.labels)):
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[i]).getData(),(self.dim,self.dim,1))
                    self.Xexp[i,] = (Iexp-np.mean(Iexp))/np.std(Iexp)

        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor((len(self.labels)) / self.batch_size))

        def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

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
            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
            # Initialization
            Xexp = np.zeros((self.batch_size,self.dim,self.dim,1),dtype=np.float64)
            y = np.empty((self.batch_size,2), dtype=np.int64)

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Read image
                if self.readInMemory:
                    Iexp = self.Xexp[ID]
                else:
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[ID]).getData(),(self.dim,self.dim,1))
                    Iexp = (Iexp-np.mean(Iexp))/np.std(Iexp)

                # Iexp = Iexp*0;
                # Iexp[48:80,48:80] = 1

                if mode=="Shift":
                    rX = self.sigma * np.random.normal()
                    rY = self.sigma * np.random.normal()
                    Xexp[i,] = shift(Iexp, (rX, rY, 0), order=1, mode='reflect')
                    y[i,] = np.array((rX,rY))+self.labels[ID]
                else:
                    rAngle = self.sigma * np.random.normal()
                    if rAngle!=0:
                        Xexp[i,] = rotate(Iexp, rAngle, order=1, mode='reflect', reshape=False)
                    angle = (self.labels[ID]+rAngle)*math.pi/180
                    y[i,] = np.array((math.cos(angle), math.sin(angle)))

            return Xexp, y


    def constructModel(Xdim):
        inputLayer = Input(shape=(Xdim,Xdim,1), name="input")

        #Network model
        L = Conv2D(8, (int(Xdim/10), int(Xdim/10)), activation="relu") (inputLayer) #33 filter size before
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        # L = Dropout(0.2)(L)
        L = Conv2D(4, (int(Xdim/20), int(Xdim/20)), activation="relu") (L) #11 filter size before
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        # L = Dropout(0.2)(L)
        L = Conv2D(4, (int(Xdim/20), int(Xdim/20)), activation="relu") (L) #11 filter size before
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        # L = Dropout(0.2)(L)
        L = Flatten() (L)
        L = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001))(L)
        L = BatchNormalization()(L)
        L = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001))(L)
        L = BatchNormalization()(L)
        L = Dense(2, name="output", activation="linear") (L)
        return Model(inputLayer, L)

    Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnXmdExp)

    mdExp = xmippLib.MetaData(fnXmdExp)
    fnImgs = mdExp.getColumnValues(xmippLib.MDL_IMAGE)
    shiftX = mdExp.getColumnValues(xmippLib.MDL_SHIFT_X)
    shiftY = mdExp.getColumnValues(xmippLib.MDL_SHIFT_Y)
    rots = mdExp.getColumnValues(xmippLib.MDL_ANGLE_ROT)
    tilts = mdExp.getColumnValues(xmippLib.MDL_ANGLE_TILT)
    psis = mdExp.getColumnValues(xmippLib.MDL_ANGLE_PSI)
    if mode == "Shift":
        labels = []
        for x, y in zip(shiftX, shiftY):
            labels.append(np.array((x,y)))
    elif mode == "Psi":
        labels = psis
    elif mode == "Rot":
        labels = rots
    elif mode == "Tilt":
        labels = tilts

    # Generator
    training_generator = DataGenerator(fnImgs, labels, mode, sigma, batch_size, Xdim, readInMemory=False)

    start_time = time()
    model = constructModel(Xdim)

    model.summary()
    adam_opt = Adam(lr=0.001)
    model.compile(loss='mean_absolute_error', optimizer=adam_opt, metrics=['accuracy'])

    steps = round(len(fnImgs)/batch_size)
    history = model.fit_generator(generator = training_generator, steps_per_epoch = steps, epochs=numEpochs)
    model.save(fnModel)
    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)
