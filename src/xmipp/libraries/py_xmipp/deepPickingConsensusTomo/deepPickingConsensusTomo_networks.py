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
# * Initial release: june 2023
# **************************************************************************

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
from keras import layers as l
from keras import callbacks as cb
import os

CONV_LAYERS = 3
PREF_SIDE = 256
PROB_DROPOUT = 0.3
CHECK_POINT_AT= 50 #In batches

class DeepPickingConsensusTomoNetworkManager():
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
        self.compiledNetwork = None
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

        self.gpustrings = ["/gpu:%d" % id for id in self.wantedGPUs]
        self.mirrored_strategy = tf.distribute.MirroredStrategy(devices=self.gpustrings)        
    
    def createNetwork(self, xdim, ydim, zdim, num_chan, nData=2**12):      

        print("Compiling the model into a network")
        # Get the model structure
        self.compiledNetwork = self.getNetwork(dataset_size=nData, input_shape=(xdim,ydim,zdim,num_chan))

    def loadNetwork(self, modelFile, keepTraining=True):
        if not os.path.isfile(modelFile):
            raise ValueError("Model file %s not found",modelFile)
        with self.mirrored_strategy.scope():
            aux : keras.Model = keras.models.load_model(modelFile, custom_objects={"PREF_SIDE":PREF_SIDE})
            self.compiledNetwork = aux.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if keepTraining:
            pass

    def trainNetwork(self, nEpochs, dataman, learningRate, autoStop=True):
        """
        nEpochs: int. Number of epochs to be run in training stage
        dataman: DataMan. Data provider for training batch.
        """

        # Print the input information
        print("Max epochs: ", nEpochs)
        print("Learning rate: %.1e"%(learningRate))
        print("Auto stop feature: ", autoStop)

        n_batches_per_epoch_train, n_batches_per_epoch_val= dataman.getNBatchesPerEpoch()
        epochN = max(1, nEpochs * float(n_batches_per_epoch_train/CHECK_POINT_AT))

        currentChkName = self.checkPointsName
        cBacks = [cb.ModelCheckpoint(currentChkName, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False)]

        if autoStop:
            cBacks += [cb.EarlyStopping()]

        self.compiledNetwork.fit(dataman.getDataIterator(stage="train"), steps_per_epoch=CHECK_POINT_AT,
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
            filtermult = 4
        else:
            filtermult = 8

        # PRINT PARAMETERS
        print("Intermediate layer count: %d", CONV_LAYERS)
        print("Box size %d,%d,%d", input_shape[0], input_shape[1], input_shape[2])

        with self.mirrored_strategy.scope():
            # Input size different than NN desired side
            if input_shape != (PREF_SIDE, PREF_SIDE, PREF_SIDE, 1):
                input_layer = l.Input(shape = (None, None, None, input_shape[-1]))
                model = l.Lambda( lambda img: keras.backend.resize_volumes
                                (img, PREF_SIDE, PREF_SIDE, PREF_SIDE), name="Resizing layer")(input_layer)
            else:
                input_layer = l.Input(shape = input_shape)
                model = input_layer

            # DNN PART
            for i in range(1, CONV_LAYERS + 1):
                # Convolve with an increasing number of filters
                # Several convolutions before pooling assure a better grasp of 
                # the features are taken before shrinking the image
                model = l.Conv3D(filters=2**(filtermult), kernel_size=(3,3,3),
                                activation='relu', padding='same',kernel_regularizer='l1_l2')(model)
                filtermult += 1
                model = l.Conv3D(filters = 2**(filtermult), kernel_size = (2,2,2),
                                activation='relu', padding='same', kernel_regularizer='l1_l2')(model)
                filtermult += 1
                # Normalize
                model = l.BatchNormalization()(model)
                # Activate
                model = l.Activation('relu')(model)
                # Until last layer, sharpen for edges to be easily detectable
                if i != CONV_LAYERS:
                    model = l.MaxPooling3D(pool_size = 2, strides = 2, paddint='same')(model)

            # Final touches
            # Desharpen edges
            l.AveragePooling3D(pool_size=4, strides=(2,2,2), padding='same')
            # Compact and drop
            l.Flatten()
            l.Dense(units=512, activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l1_l2')
            l.Dense(units=256, activation='relu', kernel_regularizer='l2')
            l.Dropout(PROB_DROPOUT)

            # Final predictions - 2 classes (GOOD / BAD)
            pred_layer = l.Dense(units=2, activation='softmax')(model)
            network = keras.Model(inputs=input_layer, output=pred_layer)
            network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return network