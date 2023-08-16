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
import random


# Structural globals
CONV_LAYERS = 3
PREF_SIDE = 64

# Configuration globals
PROB_DROPOUT = 0.4
CHECK_POINT_AT= 50 #In batches
PREFECTH_N = 128 # in subtomograms

class NetMan():
    def __init__(self, nThreads:int, gpuIDs:list, rootPath:str, batchSize:int, boxSize : int, doAugment : bool = True,
                 posPath: str = None, negPath: str = None, doubtPath : str = None, valFrac : int = 0.15, 
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
        self.valFrac = valFrac
        self.boxSize = boxSize
        self.doAugment = doAugment
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
            # POS
            print("NetMan will train, load pos+neg")
            self.posVolsFns = getFolderContent(posPath, ".mrc")
            self.nPos = len(self.posVolsFns)
            # NEG
            self.negVolsFns = getFolderContent(negPath, ".mrc")
            self.nNeg = len(self.negVolsFns)
            # Separate into train and validate

            if self.valFrac > 0:
                # Separate into train/val
                self.posVolsFnsTrain= random.choices(self.posVolsFns, k=int((1-valFrac)*self.nPos))
                self.posVolsFnsVal= list(set(self.posVolsFns).difference(self.posVolsFnsTrain))

                self.negVolsFnsTrain = random.choices(self.negVolsFns, k=int((1-valFrac)*self.nNeg))
                self.negVolsFnsVal = list(set(self.negVolsFns).difference(self.negVolsFnsTrain))
            else:
                # Don't separate, all goes to train si tu lo quiere asin
                self.posVolsFnsTrain = self.posVolsFns
                self.negVolsFnsTrain = self.negVolsFns
            
            self.allLabeled = list(zip(self.posVolsFns, [1]*self.nPos)) + list(zip(self.negVolsFns, [0]*self.nNeg))
            # Augmentation, if needed, should be launched here according to number of neg, pos...
            self.nTotal = len(self.allLabeled)

        else:
            # SCORING
            print("NetMan will score, load doubt")
            self.doubtVolsFns = getFolderContent(doubtPath, ".mrc")
            self.nDoubt = len(self.doubtVolsFns)
        
    def gpusConfig(self):
        """
        This function allows TF only to see the GPUs wanted for the processing,
        thus avoiding the use of unwanted GPUs on multi-user systems.
        """

        # Check GPUs in system
        # availGPUs = tf.config.list_physical_devices('GPU')
        # print("Found this many GPUs in the system: ", availGPUs)
        # Compare with the asked amount
        # assert len(self.wantedGPUs) <= len(availGPUs), "Not enough GPUs in the system for the asked amount"
        # print("Trying to lock GPUs with id: ", self.wantedGPUs)
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
        cBacks = [cb.ModelCheckpoint(currentChkName, monitor='val_accuracy', verbose=2, save_best_only=True, save_weights_only=False)]

        if autoStop:
            cBacks += [cb.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10, verbose=1)]

        with self.strategy.scope():
            # Create the dataset object from a generator: USEFUL FOR MULTIPROCESSING-MULTIGPU PURPOSES
            opt = tf.data.Options()
            opt.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            # TRAIN DATASET OBJECT
            dataset : tf.data.Dataset
            dataset = tf.data.Dataset.from_generator(self.data_generation_trainval, args=["train"], 
                                                     output_types=(tf.float32, tf.int32), 
                                                     output_shapes=((self.batchSize, self.boxSize,self.boxSize,self.boxSize,1),(self.batchSize, 1)))
            dataset = dataset.with_options(opt)
            dataset = dataset.repeat()
            # VALIDATION DATASET OBJECT
            datasetVal = None
            if self.valFrac > 0:
                datasetVal : tf.data.Dataset
                datasetVal = tf.data.Dataset.from_generator(self.data_generation_trainval, args=["validate"],
                                                             output_types=(tf.float32, tf.int32),
                                                             output_shapes=((self.batchSize, self.boxSize,self.boxSize,self.boxSize,1),(self.batchSize, 1)))
                datasetVal = datasetVal.with_options(opt)
                datasetVal = datasetVal.repeat()

            stepsInEpoch = getStepsInEpoch(nEpochs)
            history = self.net.fit(dataset,
                         steps_per_epoch = stepsInEpoch,
                         epochs = nEpochs,
                         verbose=1,
                         validation_data= datasetVal,
                         validation_steps = int(stepsInEpoch/2),
                         callbacks=cBacks,
                         use_multiprocessing=self.nThreads)
            
            last_val_acc = history.history['val_accuracy'][-1]
            print("Finished training with last validation accuracy %s" % str(last_val_acc))  

    def data_generation_trainval(self, mode: str):
        X = np.empty((self.batchSize, self.boxSize, self.boxSize, self.boxSize, 1))
        Y = np.empty((self.batchSize, 1), dtype=int)

        # Choose from the different data sets
        
        if mode == "train":
            negNames = self.negVolsFnsTrain
            posNames = self.posVolsFnsTrain
        else:
            negNames = self.negVolsFnsVal
            posNames = self.posVolsFnsVal

        image = xmippLib.Image()

        for i in range(self.batchSize):
            label = random.randrange(2)
            if label == 0:
                filename = random.choice(negNames)
            else:
                filename = random.choice(posNames)
            image.read(filename)
            xmippIm : np.ndarray = image.getData()
            # Data augment here
            if self.doAugment:
                pass
    
            X[i] = np.expand_dims(xmippIm, -1)
            Y[i] = label

        yield X, Y
 
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
        print("Input shape:")
        print(input_shape)

        with self.strategy.scope():
            # Input size different than NN desired side
            model = Sequential()
        
            # model.add(l.InputLayer(shape = input_shape))
            model.add(l.InputLayer(input_shape=input_shape))
            srcDim = input_shape[0]
            destDim = PREF_SIDE
            if srcDim < destDim: # Need to increase cube sizes
                factor = round(destDim / srcDim)
                model.add(l.Lambda(lambda img: backend.resize_volumes(img, factor, factor, factor, 'channels_last'), name="resize_tf"))
            elif srcDim > destDim: # Need to decrease cube sizes
                factor = round(srcDim / destDim)
                model.add(l.AveragePooling3D(pool_size=(factor,)*3))

            # DNN PART
            filters = 16
            for i in range(1, CONV_LAYERS + 1):
                # Convolve with an increasing number of filters
                # Several convolutions before pooling assure a better grasp of 
                # the features are taken before shrinking the image
                model.add(l.Conv3D(filters=filters*i, kernel_size=(3,3,3),
                                activation='relu', padding='same', kernel_regularizer='l1_l2'))
                
                # Normalize
                model.add(l.BatchNormalization())

                if i != CONV_LAYERS:
                    model.add(l.MaxPooling3D(pool_size=(3,3,3), padding='same'))
                

            # Final touches
            # Desharpen edges
            # model.add(l.AveragePooling3D(pool_size=(2,2,2), padding='same'))
            # Compact and drop
            model.add(l.Flatten())
            model.add(l.Dense(units=32, activation='gelu', kernel_regularizer='l2'))
            model.add(l.Dropout(PROB_DROPOUT))
            # model.add(l.Dense(units=64, activation='sigmoid'))
            # model.add(l.Dropout(PROB_DROPOUT))
            model.add(l.Dense(units=16, activation='relu', kernel_regularizer='l2'))
            model.add(l.Dropout(PROB_DROPOUT))

            # Final predictions - 2 classes probabilities (p(GOOD),p(BAD))
            # model.add(l.Dense(units=2, activation='softmax'))
            model.add(l.Dense(units=1, activation='sigmoid'))
            print(model.summary())
            #model.build((None,PREF_SIDE))
            # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
def getFolderContent(path: str, filter: str) -> list :
        fns = os.listdir(path)
        return [ path + "/" + fn for fn in fns if filter in fn] 

def getStepsInEpoch(nEpochs : int) -> int:
    res : int
    if nEpochs < 5:
        res = 75
    elif nEpochs < 10:
        res = 50
    else:
        res = 25
    return res

