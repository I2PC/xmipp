#!/usr/bin/env python3

import glob
import numpy as np
import os
import sys
import xmippLib
from time import time

maxSize = 32

if __name__ == "__main__":
    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

    checkIf_tf_keras_installed()
    fnExp = sys.argv[1]
    gpuId = sys.argv[2]
    fnExpResized = sys.argv[3]
    fnModel = sys.argv[4]
    numModels = len(glob.glob(fnModel+"*.h5"))

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    import keras
    import tensorflow as tf

    class DataGenerator(keras.utils.all_utils.Sequence):
        """Generates data for fnImgs"""

        def __init__(self, fnImgs, maxSize, dimResized, readInMemory):
            """Initialization"""
            self.fnImgs = fnImgs
            self.maxSize = maxSize
            self.dimResized = dimResized
            self.readInMemory = readInMemory
            self.on_epoch_end()

            # Read all data in memory
            if self.readInMemory:
                self.Xexp = np.zeros((len(self.fnImgs), self.dimResize, self.dimResized, 1), dtype=np.float64)
                for i in range(len(self.fnImgs)):
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[i]).getData(), (self.dimResized, self.dimResized, 1))
                    self.Xexp[i,] = (Iexp - np.mean(Iexp)) / np.std(Iexp)

        def __len__(self):
            """Denotes the number of batches per predictions"""
            num = len(self.fnImgs) // maxSize
            if len(self.fnImgs) % maxSize > 0:
                num = num + 1
            return num

        def __getitem__(self, index):
            """Generate one batch of data"""
            # Generate indexes of the batch
            indexes = self.indexes[index * maxSize:(index + 1) * maxSize]
            # Find list of IDs
            list_IDs_temp = []
            for i in range(len(indexes)):
                list_IDs_temp.append(indexes[i])

            # Generate data
            Xexp = self.__data_generation(list_IDs_temp)

            return Xexp

        def on_epoch_end(self):
            self.indexes = [i for i in range(len(self.fnImgs))]

        def getNumberOfBlocks(self):
            self.st = len(self.fnImgs) // maxSize
            if len(self.fnImgs) % maxSize > 0:
                self.st = self.st + 1

        def __data_generation(self, list_IDs_temp):
            """Generates data containing batch_size samples"""
            # Initialization
            Xexp = np.zeros((len(list_IDs_temp), self.dimResized, self.dimResized, 1), dtype=np.float64)
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Read image
                if self.readInMemory:
                    Xexp[i, ] = self.Xexp[ID]
                else:
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[ID]).getData(), (self.dimResized, self.dimResized, 1))
                    Xexp[i, ] = (Iexp - np.mean(Iexp)) / np.std(Iexp)
            return Xexp

    def produce_output(fnExp, Y, XdimResized):
        XdimOrig, _, _, _, _ = xmippLib.MetaDataInfo(fnExp)
        Y*=XdimOrig/XdimResized

        mdExp = xmippLib.MetaData(fnExp)
        ID = 0
        for objId in mdExp:
            # Set predictions in mdExp
            shiftX, shiftY = Y[ID]
            mdExp.setValue(xmippLib.MDL_SHIFT_X, float(shiftX), objId)
            mdExp.setValue(xmippLib.MDL_SHIFT_Y, float(shiftY), objId)
            ID += 1
        mdExp.write(fnExp)

    mdExpResized = xmippLib.MetaData(fnExpResized)
    fnImgsResized = mdExpResized.getColumnValues(xmippLib.MDL_IMAGE)
    XdimResized, _, _, _, _ = xmippLib.MetaDataInfo(fnExpResized)

    predictions = np.zeros((len(fnImgsResized), numModels, 2))
    ShiftManager = DataGenerator(fnImgsResized, 32, XdimResized, readInMemory=False)
    for index in range(numModels):
        ShiftModel = keras.models.load_model(fnModel + str(index) + ".h5", compile=False)
        predictions[:, index, :] = ShiftModel.predict_generator(ShiftManager, ShiftManager.getNumberOfBlocks())
    Y = tf.reduce_mean(predictions, axis=1)
    produce_output(fnExp, Y, XdimResized)
