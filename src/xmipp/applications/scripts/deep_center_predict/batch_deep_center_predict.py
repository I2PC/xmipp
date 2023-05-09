#!/usr/bin/env python3

import math
import numpy as np
from numpy.linalg import norm
import os
import sys
import xmippLib
from time import time
from scipy.spatial.transform import Rotation
from scipy.ndimage import shift, rotate

maxSize = 128

if __name__ == "__main__":
    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    fnShiftModel = sys.argv[2]
    predictAngles = sys.argv[3]
    gpuId = sys.argv[4]
    outputDir = sys.argv[5]
    fnXmdImages = sys.argv[6]
    if predictAngles == 'yes':
        fnAngModel = sys.argv[7]
        representation = sys.argv[8]
        loss_function = sys.argv[9]

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
            num = len(self.fnImgs) // maxSize
            if len(self.fnImgs) % maxSize > 0:
                num = num + 1
            return num

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


    def rotation6d_to_matrix(rot):

        a1 = np.array((rot[0], rot[2], rot[4]))
        a2 = np.array((rot[1], rot[3], rot[5]))
        a1 = np.reshape(a1, (3, 1))
        a2 = np.reshape(a2, (3, 1))

        b1 = a1 / np.linalg.norm(a1)

        c1 = np.multiply(b1, a2)
        c2 = np.sum(c1)

        b2 = a2 - c2 * b1
        b2 = b2 / np.linalg.norm(b2)
        b3 = np.cross(b1, b2, axis=0)
        return np.concatenate((b1, b2, b3), axis=1)

    def matrix_to_euler(mat):
        r = Rotation.from_matrix(mat)
        angles = r.as_euler("xyz", degrees=True)
        return angles

    def produce_output(mdExp, mode, Y, fnImages):
        ID = 0
        for objId in mdExp:
            if mode == "Shift":
                shiftX, shiftY = Y[ID]
                mdExp.setValue(xmippLib.MDL_SHIFT_X, float(shiftX), objId)
                mdExp.setValue(xmippLib.MDL_SHIFT_Y, float(shiftY), objId)
                mdExp.setValue(xmippLib.MDL_IMAGE, fnImages[ID], objId)
            elif mode == "Angular":
                if representation == 'euler':
                    if loss_function == 'geodesic':
                        rots = Y[ID][0:2]
                        costilts = Y[ID][2]
                        psis = Y[ID][3:]
                        psis /= norm(psis)
                        psis_degree = (math.atan2(psis[0], psis[1])) * 180 / math.pi
                        rots /= norm(rots)
                        rots_degree = (math.atan2(rots[0], rots[1])) * 180 / math.pi
                        if costilts > 1:
                            costilts = 1
                        if costilts < -1:
                            costilts = -1
                            
                        tilts_degree = math.acos(costilts) * 180 / math.pi
                        mdExp.setValue(xmippLib.MDL_ANGLE_PSI, psis_degree, objId)
                        mdExp.setValue(xmippLib.MDL_ANGLE_ROT, rots_degree, objId)
                        mdExp.setValue(xmippLib.MDL_ANGLE_TILT, tilts_degree, objId)
                    else:
                        rots = Y[ID][0:2]
                        tilts = Y[ID][2:4]
                        psis = Y[ID][4:]
                        psis /= norm(psis)
                        psis_degree = (math.atan2(psis[0], psis[1])) * 180 / math.pi
                        rots /= norm(rots)
                        rots_degree = (math.atan2(rots[0], rots[1])) * 180 / math.pi
                        tilts /= norm(tilts)
                        tilts_degree = (math.atan2(tilts[0], tilts[1])) * 180 / math.pi
                    mdExp.setValue(xmippLib.MDL_ANGLE_PSI, psis_degree, objId)
                    mdExp.setValue(xmippLib.MDL_ANGLE_ROT, rots_degree, objId)
                    mdExp.setValue(xmippLib.MDL_ANGLE_TILT, tilts_degree, objId)
                elif representation == 'cartesian':
                    psis = Y[ID][3:]
                    psis /= norm(psis)
                    psis_degree = (math.atan2(psis[0], psis[1])) * 180 / math.pi
                    rots = math.atan2(Y[ID][1], Y[ID][0])
                    rots_degree = rots * 180 / math.pi
                    tilts = math.atan2(math.sqrt(math.pow(Y[ID][1], 2) + math.pow(Y[ID][0], 2)), Y[ID][2])
                    tilts_degree = tilts * 180 / math.pi
                    mdExp.setValue(xmippLib.MDL_ANGLE_PSI, psis_degree, objId)
                    mdExp.setValue(xmippLib.MDL_ANGLE_ROT, rots_degree, objId)
                    mdExp.setValue(xmippLib.MDL_ANGLE_TILT, tilts_degree, objId)

                else:
                    rotmatrix = rotation6d_to_matrix(Y[ID])
                    angles = matrix_to_euler(rotmatrix)
                    mdExp.setValue(xmippLib.MDL_ANGLE_PSI, angles[2], objId)
                    mdExp.setValue(xmippLib.MDL_ANGLE_ROT, angles[0], objId)
                    mdExp.setValue(xmippLib.MDL_ANGLE_TILT, angles[1] + 90, objId)
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

    ShiftModel = load_model(fnShiftModel, compile=False)
    ShiftModel.compile(loss="mean_squared_error", optimizer='adam')

    ShiftManager = DataGenerator(fnImgs, maxSize, Xdim, readInMemory=False)

    Y = ShiftModel.predict_generator(ShiftManager, ShiftManager.getNumberOfBlocks())

    produce_output(mdExp, 'Shift', Y, fnImages)

    if predictAngles == 'yes':
        AngModel = load_model(fnAngModel, compile=False)
        AngModel.compile(loss="mean_squared_error", optimizer='adam')
        AngManager = DataGenerator(fnImgs, maxSize, Xdim, readInMemory=False)
        Y = AngModel.predict_generator(AngManager, AngManager.getNumberOfBlocks())
        produce_output(mdExp, 'Angular', Y, fnImages)

    mdExp.write(os.path.join(outputDir, "predict_results.xmd"))


    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)


