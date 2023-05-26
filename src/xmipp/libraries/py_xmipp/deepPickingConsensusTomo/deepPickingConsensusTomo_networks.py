# **************************************************************************
# *
# * Authors:    Mikel Iceta Tena (miceta@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
import os

INTERMEDIATE_LAYERS = 4

class DeepPickingConsensusTomoNetworkManager():
    def __init__(self, rootPath):
        """
        @param rootPath: str. Root directory for the NN data to be saved.
                                Normally: "extra/nnetData/"
                                                        "/tfchkpoints"
                                                        "/tflogs"
                                                        "..."

        """
        self.rootPath = rootPath

        checkpointsName = os.path.join(rootPath, "nnchkpoints")
        self.checkpointsNameTemplate = os.path.join(checkpointsName)
        self.optimizer = None
        self.model = None
        self.compiledNetwork = None

    def compileNetwork(self):      

        print("Compiling the model into a network")
        
        optimizer = self.getOptimizer()
        model : keras.models.Sequential = self.getModel()
        self.compiledNetwork = model.compile()
 
    def getOptimizer():
        return lambda learningRate: keras.optimizers.Adam(lr= learningRate, beta_1=0.9,beta_2=0.999,epsilon=1e-8)

    def getModel(self, input_shape):
        """
        Generate the structure of the Neural Network.
        """

        # PRINT PARAMETERS
        print("Intermediate layer depth: %d", INTERMEDIATE_LAYERS)
        print("Box size %d,%d", input_shape[0], input_shape[1])

        model : keras.models.Sequential = keras.models.Sequential()

        # INPUT LAYERS
        model.add(keras.layers.InputLayer(shape = input_shape))
        
        # CONVOLUTIONS
        for d in range (1, INTERMEDIATE_LAYERS+1):
            model.add(keras.layers.Conv3D())
            model.add(keras.layers.Conv3D())
            model.add(keras.layers.BatchNormalization())

        # FINAL FLATTENING AND OUTPUTS
        model.add(keras.layers.GlobalAveragePooling3D())
        model.add(keras.layers.Flatten())

        model.add(keras.layers.Dense())
        model.add(keras.layers.Activation())
        model.add(keras.layers.Dense())

        print("Generating the model")

        return model