#!/usr/bin/env python3

import math
import numpy as np
from numpy.linalg import norm
import os
import sys
import xmippLib
from time import time
from scipy.ndimage import shift, rotate

maxSize = 237

if __name__ == "__main__":
    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    fnShiftModel = sys.argv[2]
    fnAngModel = sys.argv[3]
    gpuId = sys.argv[4]
    outputDir = sys.argv[5]
    fnXmdImages = sys.argv[6]

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    from keras.models import Model
    import keras
    from keras.models import load_model
    import tensorflow as tf


    class DataGenerator(keras.utils.Sequence):
        'Generates data for fnImgs'

        def __init__(self, fnImgs, maxSize, dim, readInMemory):
            'Initialization'
            self.fnImgs = fnImgs
            self.maxSize = maxSize
            self.dim = dim
            self.readInMemory = readInMemory
            self.on_epoch_end()

            # Read all data in memory
            if self.readInMemory:
                self.Xexp = np.zeros((len(self.fnImgs), self.dim, self.dim, 1), dtype=np.float64)
                for i in range(len(self.fnImgs)):
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[i]).getData(), (self.dim, self.dim, 1))
                    self.Xexp[i,] = (Iexp - np.mean(Iexp)) / np.std(Iexp)

        def __len__(self):
            'Denotes the number of batches per predictions'
            return maxSize

        def __getitem__(self, index):
            'Generate one batch of data'
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
            # self.st = ImagesNumber/maxSize        
            self.st = len(self.fnImgs) // maxSize
            if len(self.fnImgs) % maxSize > 0:
                self.st = self.st + 1

        def __data_generation(self, list_IDs_temp):
            'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
            # Initialization
            Xexp = np.zeros((len(list_IDs_temp), self.dim, self.dim, 1), dtype=np.float64)
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Read image
                if self.readInMemory:
                    Xexp[i,] = self.Xexp[ID]
                else:
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[ID]).getData(), (self.dim, self.dim, 1))
                    Xexp[i,] = (Iexp - np.mean(Iexp)) / np.std(Iexp)
                    # Xexp[i,] = shift(Xexp[i, ], (-20, 0, 0), order=1, mode='reflect')
            return Xexp


    def produce_output(mdExp, mode, Y, fnImages):
        ID = 0
        for objId in mdExp:
            if mode == "Shift":
                shiftX, shiftY = Y[ID]
                mdExp.setValue(xmippLib.MDL_SHIFT_X, float(shiftX), objId)
                mdExp.setValue(xmippLib.MDL_SHIFT_Y, float(shiftY), objId)
                mdExp.setValue(xmippLib.MDL_IMAGE, fnImages[ID], objId)
            elif mode == "Angular":
                rots = Y[ID][0:2]
                tilts = Y[ID][2:4]
                psis = Y[ID][4:]
                psis /= norm(psis)
                psis_degree = (math.atan2(psis[0], psis[1])) * 180 / math.pi
                rots /= norm(rots)
                rots_degree = (math.atan2(rots[0], rots[1])) * 180 / math.pi
                tilts /= norm(tilts)
                tilts_degree = (math.atan2(tilts[0], tilts[1])) * 180 / math.pi
                mdExp.setValue(xmippLib.MDL_ANGLE_PSI, float(psis_degree), objId)
                mdExp.setValue(xmippLib.MDL_ANGLE_ROT, float(rots_degree), objId)
                mdExp.setValue(xmippLib.MDL_ANGLE_TILT, float(tilts_degree), objId)
            ID += 1


    Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnXmdExp)

    mdExp = xmippLib.MetaData(fnXmdExp)
    fnImgs = mdExp.getColumnValues(xmippLib.MDL_IMAGE)

    Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnXmdExp)
    mdExp = xmippLib.MetaData(fnXmdExp)
    fnImgs = mdExp.getColumnValues(xmippLib.MDL_IMAGE)

    mdExpImages = xmippLib.MetaData(fnXmdImages)
    fnImages = mdExpImages.getColumnValues(xmippLib.MDL_IMAGE)


    start_time = time()

    ShiftModel = load_model(fnShiftModel, compile=True)
    AngModel = load_model(fnAngModel, compile=True)
    # model.compile(loss=custom_loss_function, optimizer='adam')
    # model.compile(optimizer='adam')

    ShiftManager = DataGenerator(fnImgs, maxSize, Xdim, readInMemory=False)

    print("Number OF Blocks", ShiftManager.getNumberOfBlocks())

    Y = ShiftModel.predict_generator(ShiftManager, ShiftManager.getNumberOfBlocks())

    print(len(Y), flush=True)

    produce_output(mdExp, 'Shift', Y, fnImages)

    AngManager = DataGenerator(fnImgs, maxSize, Xdim, readInMemory=False)
    Y = AngModel.predict_generator(AngManager, AngManager.getNumberOfBlocks())

    produce_output(mdExp, 'Angular', Y, fnImages)

    mdExp.write(os.path.join(outputDir, "predict_results.xmd"))


    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)


