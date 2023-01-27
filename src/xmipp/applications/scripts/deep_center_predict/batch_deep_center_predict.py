#!/usr/bin/env python3

import math
import numpy as np
from numpy.linalg import norm
import os
import sys
import xmippLib
from time import time

maxSize = 400

if __name__ == "__main__":
    print("----in deep_center_predict----", flush=True)
    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    fnModel = sys.argv[2]
    mode = sys.argv[3]
    gpuId = sys.argv[4]


    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    from keras.models import Model
    import keras
    from keras.models import load_model
    import tensorflow as tf


    class DataGenerator(keras.utils.Sequence):
        'Generates data for fnImgs'

        def __init__(self, fnImgs, mode, maxSize, dim, readInMemory):
            'Initialization'
            print("----------Initialization---------", flush=True)
            self.fnImgs = fnImgs
            self.mode = mode
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
            print("maxSize", flush=True)
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
                # print(ID)
                # Read image
                if self.readInMemory:
                    Xexp[i,] = self.Xexp[ID]
                else:
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[ID]).getData(), (self.dim, self.dim, 1))
                    Xexp[i,] = (Iexp - np.mean(Iexp)) / np.std(Iexp)
                # Iexp = Iexp*0;
                # Iexp[48:80,48:80] = 1
            return Xexp


    def produce_output(mdExp, mode, numIm, Y):

        print(mode)
        ID = 0
        for objId in mdExp:
            if mode == "Shift":
                shiftX, shiftY = Y[ID]
                # print(shiftX, shiftY)
                mdExp.setValue(xmippLib.MDL_SHIFT_X, float(shiftX), objId)
                mdExp.setValue(xmippLib.MDL_SHIFT_Y, float(shiftY), objId)
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

    print("----------------------------------------ahkbrvfeiwbfc", flush=True)
    Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnXmdExp)

    mdExp = xmippLib.MetaData(fnXmdExp)
    fnImgs = mdExp.getColumnValues(xmippLib.MDL_IMAGE)

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

    def custom_loss_function(y_true, y_pred):
        d = tf.abs(y_true - y_pred)
        d = tf.Print(d, [d], "Inside loss function")
        return tf.reduce_mean(d, axis=-1)


    start_time = time()
    print("----Loading model----", flush=True)
    print(fnModel, flush=True)

    model = load_model(fnModel, compile=False)
    model.compile(loss=custom_loss_function, optimizer='adam')
    print("----model loaded!----", flush=True)
    manager = DataGenerator(fnImgs, mode, maxSize, Xdim, readInMemory=False)
    Y = model.predict_generator(manager, manager.getNumberOfBlocks())


    produce_output(mdExp, mode, len(fnImgs), Y)


    print(mdExp)
    print("-----------Test values------------")
    if mode == "Shift":
        mdExp.write("shift_results.xmd")
    elif mode == "Angular":
        mdExp.write("ang_results.xmd")

    if mode == "Shift":
        print("Shift")
    else:
        print("Rot:", flush=True)
        pred = Y[: , 0:2]
        test = [i[0] for i in labels]
        pred_norm = norm(pred, axis = -1)
        pred_degree = np.zeros(len(pred_norm))
        error = np.zeros(len(pred_norm))
        for i in range(len(pred_norm)):
            pred[i,] = pred[i,]/pred_norm[i]
            pred_degree[i] = (math.atan2(pred[i, 0], pred[i, 1])) * 180 / math.pi
            error[i] = np.min(np.array([np.abs(pred_degree[i]-test[i]), np.abs(np.abs(pred_degree[i]-test[i])-360)]))
        maxAbsError = np.max(error)
        meanAbsError = np.mean(error)
        medianAbsError = np.median(error)
        print("Max Absolute Test Error", maxAbsError)
        print("Mean Absolute Test Error", meanAbsError)
        print("Median Absolute Test Error", medianAbsError)

    if mode == "Shift":
        print("Shift")
    else:
        print("Tilt:", flush=True)
        pred = Y[: , 2:4]
        test = [i[1] for i in labels]
        pred_norm = norm(pred, axis=-1)
        pred_degree = np.zeros(len(pred_norm))
        error = np.zeros(len(pred_norm))
        for i in range(len(pred_norm)):
            pred[i,] = pred[i,] / pred_norm[i]
            pred_degree[i] = (math.atan2(pred[i, 0], pred[i, 1])) * 180 / math.pi
            error[i] = np.min(
                np.array([np.abs(pred_degree[i] - test[i]), np.abs(np.abs(pred_degree[i] - test[i]) - 360)]))
        maxAbsError = np.max(error)
        meanAbsError = np.mean(error)
        medianAbsError = np.median(error)
        print("Max Absolute Test Error", maxAbsError)
        print("Mean Absolute Test Error", meanAbsError)
        print("Median Absolute Test Error", medianAbsError)

    if mode == "Shift":
        print("Shift")
    else:
        print("Psi:", flush=True)
        pred = Y[: , 4:6]
        test = [i[2] for i in labels]
        pred_norm = norm(pred, axis=-1)
        pred_degree = np.zeros(len(pred_norm))
        error = np.zeros(len(pred_norm))
        for i in range(len(pred_norm)):
            pred[i,] = pred[i,] / pred_norm[i]
            pred_degree[i] = (math.atan2(pred[i, 0], pred[i, 1])) * 180 / math.pi
            error[i] = np.min(
                np.array([np.abs(pred_degree[i] - test[i]), np.abs(np.abs(pred_degree[i] - test[i]) - 360)]))
        maxAbsError = np.max(error)
        meanAbsError = np.mean(error)
        medianAbsError = np.median(error)
        print("Max Absolute Test Error", maxAbsError)
        print("Mean Absolute Test Error", meanAbsError)
        print("Median Absolute Test Error", medianAbsError)

    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)
