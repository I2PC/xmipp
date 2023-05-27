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
import os

CONV_LAYERS = 4
PREF_SIDE = 256
PROB_DROPOUT = 0.3

class DeepPickingConsensusTomoNetworkManager():
    def __init__(self, nThreads, gpuIDs, rootPath):
        """
        rootPath: str. Root directory for the NN data to be saved.
                                Normally: "extra/nnetData/"
                                                        "/tfchkpoints"
                                                        "/tflogs"
                                                        "..."
        nThreads: int. Number of threads for execution
        gpuIDs: tuple<int>. GPUs to use


        """
        self.rootPath = rootPath
        self.nThreads = nThreads

        checkpointsName = os.path.join(rootPath, "nnchkpoints")
        self.checkpointsNameTemplate = os.path.join(checkpointsName)
        self.optimizer = None
        self.model = None
        self.compiledNetwork : keras.Model = None
        self.wantedGPUs = gpuIDs



    def startSessionAndInit(self):
        """
        """

        # Check GPUs and config TF
        availGPU = tf.config.list_physical_devices('GPU')
        print("Found the following GPUs in the system: ", len(availGPU))
        



    def createNetwork(self, xdim, ydim, zdim, num_chan, nData=2**12):      

        print("Compiling the model into a network")
        
        model : keras.models.Sequential = self.getModel(dataset_size=nData, input_shape=(xdim,ydim,zdim,num_chan))
        
        self.compiledNetwork = model.compile(loss='', optimizer='adam')

    def loadNetwork(self, modelFile, keepTraining=True):
        if not os.path.isfile(modelFile):
            raise ValueError("Model file %s not found",modelFile)
        self.compiledNetwork = keras.models.load_model(modelFile, custom_objects={"PREF_SIDE":PREF_SIDE})

        if keepTraining:
            pass

 

    def getModel(self, dataset_size, input_shape):
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

        model : keras.models.Sequential = keras.models.Sequential()
        
        if input_shape!=(PREF_SIDE, PREF_SIDE, PREF_SIDE, 1):
            model.add(l.Input(shape=(None,None,None, input_shape[-1])))
            assert keras.backend.backend() == 'tensorflow', 'You should be using Tensorflow for resizing'
            model.add(l.Lambda( lambda img: keras.backend.resize_volumes
                                            (img, PREF_SIDE, PREF_SIDE, PREF_SIDE),
                                              name="Resize layer"))
        else:
            model.add(keras.Input(input_shape))

        # DNN PART
        for i in range(1, CONV_LAYERS + 1):
            # Convolve with an increasing number of filters
            # Several convolutions before pooling assure a better grasp of 
            # the features are taken before shrinking the image
            model.add(l.Conv3D(filters=2**(filtermult), kernel_size=(3,3,3),
                             activation='relu', padding='same',kernel_regularizer='l1_l2'))
            filtermult += 1
            model.add()
            filtermult += 1
            model.add()
            filtermult +=1
            # Normalize
            model.add(l.BatchNormalization())
            # Activate
            model.add(l.Activation('relu'))
            # Until last layer, sharpen for edges to be easily detectable
            if i != CONV_LAYERS:
                model.add(l.MaxPooling3D())

        # Final touches
        # Desharpen edges
        model.add(l.AveragePooling3D(pool_size=4, strides=(2,2,2), padding='same'))
        # Compact and drop
        model.add(l.Flatten())
        model.add(l.Dense(units=512, activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l1_l2'))
        model.add(l.Dense(units=256, activation='relu', kernel_regularizer='l2'))
        model.add(l.Dropout(PROB_DROPOUT))
        # Final predictions - 2 classes (GOOD / BAD)
        model.add(l.Dense(units=2, activation='softmax'))
    
        return model