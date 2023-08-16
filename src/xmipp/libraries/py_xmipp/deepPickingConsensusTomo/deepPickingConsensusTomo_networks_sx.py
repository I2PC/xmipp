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
import numpy as np

from tensorflow.python.client import device_lib
import keras
from keras import layers as l
from keras import callbacks as cb
from keras import backend
from keras.models import Sequential

import os
import xmippLib


# Structural globals
CONV_LAYERS = 3
PREF_SIDE = 64

# Configuration globals
PROB_DROPOUT = 0.3
CHECK_POINT_AT= 50 #In batches
PREFECTH_N = 128 # in subtomograms

class NetMan():
    def __init__(self, nThreads:int, gpuIDs:list, rootPath:str, batchSize:int,
                 posPath: str = None, negPath: str = None, doubtPath : str = None,   
                 netName:str = "mimodelo.h5"):
        """
        rootPath: str. Root directory for the NN data to be saved.
                                Normally: "extra/nnetData/"
                                                        "/tfchkpoints"
                                                        "/tflogs"
                                                        "..."
        nThreads: int. Number of threads for execution
        gpuIDs: list<int>. GPUs to use
        """

        self.train = (posPath is not None) and (negPath is not None)

        self.nnPath = rootPath
        self.nThreads = nThreads
        self.batchSize = batchSize
        self.gpustrings : list

        checkpointsName = os.path.join(rootPath, "checkpoint")
        self.checkpointsNameTemplate = os.path.join(checkpointsName, netName)

        self.optimizer = None
        self.model = None
        self.net = None
        self.wantedGPUs = gpuIDs

        if gpuIDs is not None:
            # Set TF so it only sees the desired GPUs
            self.gpusConfig()

        self.checkPointsName = os.path.join(rootPath,"tfchkpoint.hdf5")

        # Dataset preparation
        if self.train:
            # TRAINING
            print("NetMan will train, load pos+neg")
            self.posVolsFns = self.getFolderContent(posPath, ".mrc")
            self.nPos = len(self.posVolsFns)
            self.negVolsFns = self.getFolderContent(negPath, ".mrc")
            self.nNeg = len(self.negVolsFns)
            # All MD with label
            self.allLabeled = zip(self.posVolsFns, [1]*len(self.nPos)) + zip(self.negVolsFns, [0]*len(self.nNeg))
            self.nTotal = len(self.allVolsFns)

        else:
            # SCORING
            print("NetMan will score, load doubt")
            self.doubtVolsFns = self.getFolderContent(doubtPath, ".mrc")
            self.nDoubt = len(self.doubtVolsFns)
        
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
            raise ValueError("Model file %s not found", modelFile)
        else:
            with self.strategy.scope():
                self.net = keras.models.load_model(modelFile, custom_objects={"PREF_SIDE":PREF_SIDE})
                # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def compileNetwork(self, pLoss, pOptimizer, pMetrics):
        if self.net is not None and self.strategy is not None:
            with self.strategy.scope():
                self.net.compile(loss=pLoss, optimizer=pOptimizer, metrics=pMetrics)

    def trainNetwork(self, nEpochs, learningRate, autoStop=True):
        """
        nEpochs: int. Number of epochs to be run in training stage
        dataman: DataMan. Data provider for training batch.
        """

        # Print the input information
        print("NN train stage started")
        print("Max epochs: ", nEpochs)
        print("Learning rate: %.1e"%(learningRate))
        print("Auto stop feature: ", autoStop)

        currentChkName = self.checkPointsName
        cBacks = [cb.ModelCheckpoint(currentChkName, monitor='val_acc', verbose=2, save_best_only=True, save_weights_only=False)]

        if autoStop:
            cBacks += [cb.EarlyStopping()]

        # Index for each row of the whole dataset: pos + neg + augpos + augneg
        z = list(range(len(self.allLabeled)))
        # Do a TF dataset from the list of integer indices
        dataset : tf.data.Dataset
        dataset = tf.data.Dataset.from_generator(lambda : z, tf.uint8)
        dataset = dataset.shuffle(buffer_size = len(z), seed = 0, reshuffle_each_iteration = True)
        dataset = dataset.map(lambda i: tf.py_function(func = self.getRow,
                                                       inp = [i],
                                                       Tout = [tf.uint8, tf.float32]
                                        ), num_parallel_calls = self.nThreads)

        dataset = dataset.prefetch(PREFECTH_N)
        stepsInEpoch = self.getStepsInEpoch(nEpochs)
       
        with self.strategy.scope():
            self.net.fit(dataset,
                         steps_per_epoch = stepsInEpoch,
                         epochs = nEpochs)

    def getRow(self, i):
        ind = i.numpy()
        fname = self.allLabeled[ind][0]
        label = self.allLabeled[ind][1]
        data = loadXmippImage(fname)
        return data, label

    def getStepsInEpoch(nEpochs : int) -> int:
        res : int
        if nEpochs < 5:
            res = 50
        elif nEpochs < 10:
            res = 30
        else:
            res = 10
        return res
 
    def predictNetwork():
        pass

    def getNetwork(self, dataset_size: int, input_shape: tuple):
        """
        Generate the structure of the Neural Network.
        dataset_size: int Expected number of picked subtomos
        input_shape: tuple<int,int,int,int> height,width,depth and nChannels
        """

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

            

            # DNN PART
            filters = 32
            for i in range(1, CONV_LAYERS + 1):
                # Convolve with an increasing number of filters
                # Several convolutions before pooling assure a better grasp of 
                # the features are taken before shrinking the image
                model.add(l.Conv3D(filters=filters*i, kernel_size=(3,3,3),
                                activation='relu', padding='same', kernel_regularizer='l1_l2'))
                
                # Normalize
                model.add(l.BatchNormalization())
                
                # Activate
                # model.add(l.Activation('relu'))

                if i != CONV_LAYERS:
                    model.add(l.MaxPooling3D(pool_size=(3,3,3), padding='same'))
                

            # Final touches
            # Desharpen edges
            # model.add(l.AveragePooling3D(pool_size=(2,2,2), padding='same'))
            # Compact and drop
            model.add(l.Flatten())
            model.add(l.Dense(units=64, activation='relu', kernel_regularizer='l2'))
            model.add(l.Dropout(PROB_DROPOUT))
            # model.add(l.Dense(units=64, activation='sigmoid'))
            # model.add(l.Dropout(PROB_DROPOUT))
            model.add(l.Dense(units=16, activation='gelu', kernel_regularizer='l2'))
            model.add(l.Dropout(PROB_DROPOUT))

            # Final predictions - 2 classes probabilities (p(GOOD),p(BAD))
            model.add(l.Dense(units=2, activation='softmax'))
            print(model.summary())
            #model.build((None,PREF_SIDE))
            # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            print(model.output_shape)
            print(input_shape)
        return model
    

def loadXmippImage(fn) -> np.ndarray:
    image = xmippLib.Image()
    image.read(fn)
    img : np.ndarray = image.getData()
    return img