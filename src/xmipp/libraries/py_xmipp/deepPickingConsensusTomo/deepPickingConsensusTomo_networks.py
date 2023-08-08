#!/usr/bin/env python3
# **************************************************************************
# *
# * Authors:    Mikel Iceta Tena (miceta@cnb.csic.es)
# *
# * Unidad de Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'coss@cnb.csic.es'
# *
# *
# * Neural Network utils for the deep picking consensus protocol found in
# * scipion-em-xmipptomo
# *
# * Initial release: september 2023
# **************************************************************************

import tensorflow as tf

from tensorflow.python.client import device_lib
import keras
from keras import layers as l
from keras import callbacks as cb
from keras import backend
from keras import Model
from keras.models import Sequential

import os

# Structural globals
CONV_LAYERS = 2
PREF_SIDE = 64

# Configuration globals
PROB_DROPOUT = 0.3
CHECK_POINT_AT= 50 #In batches

class NetMan():
    def __init__(self, nThreads:int, gpuIDs:list, rootPath:str):
        """
        rootPath: str. Root directory for the NN data to be saved.
                                Normally: "extra/nnetData/"
                                                        "/tfchkpoints"
                                                        "/tflogs"
                                                        "..."
        nThreads: int. Number of threads for execution
        gpuIDs: list<int>. GPUs to use
        """

        self.nnPath = rootPath
        self.nThreads = nThreads
        self.gpustrings : list

        checkpointsName = os.path.join(rootPath, "checkpoint")
        self.checkpointsNameTemplate = os.path.join(checkpointsName, "model.hdf5")

        self.optimizer = None
        self.model = None
        self.net = None
        self.wantedGPUs = gpuIDs

        if gpuIDs is not None:
            # Set TF so it only sees the desired GPUs
            self.gpusConfig()

        self.checkPointsName = os.path.join(rootPath,"tfchkpoint.hdf5")

    def gpusConfig(self):
        """
        This function allows TF only to see the GPUs wanted for the processing,
        thus avoiding the use of unwanted GPUs on multi-user systems.
        """

        # Check GPUs in system
        availGPUs = tf.config.list_physical_devices('GPU')
        print("Found this many GPUs in the system: ", availGPUs)
        # Compare with the asked amount
        assert len(self.wantedGPUs) <= len(availGPUs), "Not enough GPUs in the system for the asked amount"
        print("Trying to lock GPUs with id: ", self.wantedGPUs)
        self.gpustrings = ["GPU:%d" % id for id in self.wantedGPUs]
        print(self.gpustrings)
        gpustring = self.gpustrings[0]
        self.strategy = tf.distribute.OneDeviceStrategy(device=gpustring)
        # self.mirrored_strategy = tf.distribute.MirroredStrategy(devices=self.gpustrings)      
        #atexit.register(self.mirrored_strategy._extended._collective_ops._pool.close)  
    
    def createNetwork(self, size, nData=2**12):      

        print("Compiling the model into a network")
        # Get the model structure
        self.net = self.getNetwork(dataset_size=nData, input_shape=(size,size,size,1))

    def loadNetwork(self, modelFile, size):
        if not os.path.isfile(modelFile):
            raise ValueError("Model file %s not found",modelFile)
        with self.strategy.scope():
            aux : keras.Model = keras.models.load_model(modelFile, custom_objects={"PREF_SIDE":PREF_SIDE})
            self.net = aux.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def trainNetwork(self, nEpochs, dataman, learningRate, autoStop=True):
        """
        nEpochs: int. Number of epochs to be run in training stage
        dataman: DataMan. Data provider for training batch.
        """

        # Print the input information
        print("NN train stage started")
        print("Max epochs: ", nEpochs)
        print("Learning rate: %.1e"%(learningRate))
        print("Auto stop feature: ", autoStop)

        n_batches_per_epoch_train, n_batches_per_epoch_val= dataman.getNBatchesPerEpoch()
        epochN = max(1, nEpochs * float(n_batches_per_epoch_train/CHECK_POINT_AT))

        currentChkName = self.checkPointsName
        cBacks = [cb.ModelCheckpoint(currentChkName, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False)]

        if autoStop:
            cBacks += [cb.EarlyStopping()]

        self.net.fit(dataman.getDataIterator(stage="train"), steps_per_epoch=CHECK_POINT_AT,
                                validation_data=dataman.getDataIterator(stage="validate", nBatches=n_batches_per_epoch_val),
                                validation_steps=n_batches_per_epoch_val, callbacks= cBacks, epochs=epochN,
                                use_multiprocessing=True, verbose=2)

    def evalNetwork():
        pass

    def predictNetwork():
        pass

    def getNetwork(self, dataset_size, input_shape):
        """
        Generate the structure of the Neural Network.
        dataset_size: int Expected number of picked subtomos
        input_shape: tuple<int,int,int,int> height,width,depth and nChannels
        """
        UMBRAL_1 = 500
        UMBRAL_2 = 2000
        # Filters assignment
        if dataset_size < UMBRAL_1:
            filtermult = 2
        elif UMBRAL_1 <= dataset_size < UMBRAL_2:
            filtermult = 3
        else:
            filtermult = 4

        # PRINT PARAMETERS
        print("Intermediate layer count: %d" % (CONV_LAYERS))
        print("Input shape %d,%d,%d"% (input_shape[0], input_shape[1], input_shape[2]))

        with self.strategy.scope():
            # Input size different than NN desired side
            model = Sequential()
        
            model.add(l.Input(shape = input_shape))
            srcDim = input_shape[0]
            destDim = PREF_SIDE
            # Cube modifications
            if srcDim < destDim: # Need to increase cube sizes
                factor = round(destDim / srcDim)
                model.add(l.Lambda(lambda img: backend.resize_volumes(img, factor, factor, factor, 'channels_last'), name="resize_tf"))
            elif srcDim > destDim: # Need to decrease cube sizes
                factor = round(srcDim / destDim)
                model.add(l.AveragePooling3D(pool_size=(factor,)*3))

            print("INIT SECSO")
            print(model.output_shape)
            print(input_shape)

            # DNN PART
            for i in range(1, CONV_LAYERS + 1):
                # Convolve with an increasing number of filters
                # Several convolutions before pooling assure a better grasp of 
                # the features are taken before shrinking the image
                model.add(l.Conv3D(filters=2**(filtermult), kernel_size=(4,4,4),
                                activation='relu', padding='same',kernel_regularizer='l1_l2'))
                
                # filtermult += 1
                # model = l.Conv3D(filters = 2**(filtermult), kernel_size = (8,8,8),
                #                 activation='relu', padding='same', kernel_regularizer='l1_l2')(model)
                # filtermult += 1
                # Normalize
                model.add(l.BatchNormalization())
                # Activate
                model.add(l.Activation('relu'))

                if i != CONV_LAYERS:
                    model.add(l.MaxPooling3D(pool_size=(2,2,2), padding='same'))
                

            # Final touches
            # Desharpen edges
            model.add(l.AveragePooling3D(pool_size=(2,2,2), padding='same'))
            # Compact and drop
            model.add(l.Flatten())
            model.add(l.Dense(units=128, activation='relu', kernel_regularizer='l2'))
            model.add(l.Dropout(PROB_DROPOUT))
            model.add(l.Dense(units=64, activation='sigmoid'))
            model.add(l.Dropout(PROB_DROPOUT))
            model.add(l.Dense(units=32, activation='relu', kernel_regularizer='l2'))
            model.add(l.Dropout(PROB_DROPOUT))

            # Final predictions - 2 classes probabilities (p(GOOD),p(BAD))
            model.add(l.Dense(units=2, activation='softmax'))
            print(model.summary())
            #model.build((None,PREF_SIDE))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model