#!/usr/bin/env python3

import math
import numpy as np
import os
import sys
import xmippLib
from time import time
from scipy.ndimage import shift, rotate

print("------------------------------------------------", flush=True)
if __name__=="__main__":
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
            print("----------Initialization---------")
            self.fnImgs = fnImgs
            print("number of fnImgs:", len(fnImgs))
            self.labels = labels
            print("number of labels", len(labels))
            self.mode = mode
            print("mode:", mode)
            self.sigma = sigma
            print("sigma:", sigma)
            self.batch_size = batch_size
            if self.batch_size>len(self.fnImgs):
                self.batch_size=len(self.fnImgs)
            print("batch_size:", batch_size)
            self.dim = dim
            print("dim:", dim)
            self.readInMemory = readInMemory
            print("readInMemory:", readInMemory)
            self.on_epoch_end()
            print("on_epoch_end:", self.on_epoch_end())

            print("-----------Reading data in memory-----------")
            # Read all data in memory
            if self.readInMemory:
                self.Xexp = np.zeros((len(self.labels),self.dim,self.dim,1),dtype=np.float64)
                for i in range(len(self.labels)):
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[i]).getData(),(self.dim,self.dim,1))
                    self.Xexp[i,] = (Iexp-np.mean(Iexp))/np.std(Iexp)

        def __len__(self):
            'Denotes the number of batches per epoch'
            num_batches = int(np.floor((len(self.labels)) / self.batch_size))
            return num_batches

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
            print("on_epoch_end", flush=True)
            self.indexes = [i for i in range(len(self.labels))]
            np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):
            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
            # Initialization
            Xexp = np.zeros((self.batch_size,self.dim,self.dim,1),dtype=np.float64)
            y = np.empty((self.batch_size, 6), dtype=np.float64)

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

                if mode == "Shift":
                    rX = self.sigma * np.random.normal()
                    rY = self.sigma * np.random.normal()
                    Xexp[i,] = shift(Iexp, (rX, rY, 0), order=1, mode='reflect')
                    y[i,] = np.array((rX, rY))+self.labels[ID]
                else:
                    rAngle = self.sigma * np.random.normal()
                    if rAngle != 0:
                        Xexp[i,] = rotate(Iexp, rAngle, order=1, mode='reflect', reshape=False)
                    else:
                        Xexp[i,] = Iexp
                    anglePsi = (self.labels[ID][2] + rAngle) * math.pi / 180
                    angleRot = (self.labels[ID][0]) * math.pi / 180
                    angleTilt = (self.labels[ID][1]) * math.pi / 180

                    y[i,] = np.array((math.sin(angleRot), math.cos(angleRot), math.sin(angleTilt), math.cos(angleTilt),
                                      math.sin(anglePsi), math.cos(anglePsi)))
            return Xexp, y


    def constructModel(Xdim):
        print("constructModel", flush=True)
        inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")

        # Network model
        L = Conv2D(16, (int(Xdim / 10), int(Xdim / 10)), activation="relu")(inputLayer)  # 11 filter size before
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)

        L = Conv2D(16, (int(Xdim / 20), int(Xdim / 20)), activation="relu")(L)  # 11 filter size before
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Dropout(0.2)(L)

        L = Flatten()(L)
        L = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(L)
        L = BatchNormalization()(L)

        L = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(L)
        L = BatchNormalization()(L)

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

    if mode == "Shift":
        labels = []
        for x, y in zip(shiftX, shiftY):
            labels.append(np.array((x, y)))
    elif mode == "Angular":
        labels = []
        for r, t, p in zip(rots, tilts, psis):
            labels.append(np.array((r, t, p)))


    # Generator
    training_generator = DataGenerator(fnImgs, labels, mode, sigma, batch_size, Xdim, readInMemory=False)


    start_time = time()
    model = constructModel(Xdim)

    model.summary()
    adam_opt = Adam(lr=0.001)


    def custom_loss_function(y_true, y_pred):
        d = tf.abs(y_true - y_pred)
        return tf.reduce_mean(d, axis=-1)

#    model.compile(loss=custom_loss_function, optimizer=adam_opt, metrics=['accuracy'])
    model.compile(loss=custom_loss_function, optimizer='adam')


    steps = round(len(fnImgs)/batch_size)
    history = model.fit_generator(generator=training_generator, steps_per_epoch=steps, epochs=numEpochs)
    model.save(fnModel)
    print(fnModel)
    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)
