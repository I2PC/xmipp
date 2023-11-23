#!/usr/bin/env python3

import math
import numpy as np
from numpy.linalg import norm
import os
import sys
import xmippLib
from scipy.ndimage import shift

maxSize = 32

if __name__ == "__main__":
    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed
    from xmippPyModules.xmipp_utils import fillSymList, symmetryOperatorSymList

    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    gpuId = sys.argv[2]
    outputDir = sys.argv[3]
    fnXmdImages = sys.argv[4]
    fnAngModel = sys.argv[5]
    numAngModels = int(sys.argv[6])
    tolerance = int(sys.argv[7])
    maxModels = int(sys.argv[8])
    symmetry = sys.argv[9]

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    import keras
    from keras.models import load_model

    def produce_output(mdExp, Y, fnImages):
        ID = 0
        for objId in mdExp:
            rot, tilt, psi = Y[ID]
            mdExp.setValue(xmippLib.MDL_ANGLE_ROT, rot, objId)
            mdExp.setValue(xmippLib.MDL_ANGLE_TILT, tilt, objId)
            mdExp.setValue(xmippLib.MDL_ANGLE_PSI, psi, objId)
            mdExp.setValue(xmippLib.MDL_IMAGE, fnImages[ID], objId)
            ID += 1

    def calculate_r6d(redundant):
        """Solves linear system to get 6D representation from 42 parameters output"""
        A = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
                      [2, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0],
                      [0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 2],
                      [1, -1, 0, 0, 0, 0], [0, 1, -1, 0, 0, 0], [0, 0, 1, -1, 0, 0],
                      [0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 1, -1], [-1, 0, 0, 0, 0, 1],
                      [2, 0, 0, -1, 0, 0], [0, 2, 0, 0, -1, 0], [0, 0, 2, 0, 0, -1],
                      [-1, 0, 0, 2, 0, 0], [0, -1, 0, 0, 2, 0], [0, 0, -1, 0, 0, 2],
                      [1, 0, 1, 0, -1, 0], [0, 1, 0, 1, 0, -1], [-1, 0, 1, 0, 1, 0],
                      [0, -1, 0, 1, 0, 1], [1, 0, -1, 0, 1, 0], [0, 1, 0, -1, 0, 1],
                      [1, 0, 0, 0, 0, -1], [-1, 1, 0, 0, 0, 0], [0, -1, 1, 0, 0, 0],
                      [0, 0, -1, 1, 0, 0], [0, 0, 0, -1, 1, 0], [0, 0, 0, 0, -1, 1],
                      [1, 0, 0, 0, -1, 0], [0, 1, 0, 0, 0, -1], [-1, 0, 1, 0, 0, 0],
                      [0, -1, 0, 1, 0, 0], [0, 0, -1, 0, 1, 0], [0, 0, 0, -1, 0, 1]])
        X = np.zeros((redundant.shape[0], 6))

        for i in range(X.shape[0]):
            X[i] = np.linalg.lstsq(A, redundant[i],rcond=-1)[0]

        return np.reshape(X,(6))

    def rotation6d_to_matrixZYZ(rot):
        """Return rotation matrix from 6D representation."""
        a1 = np.array([rot[0], rot[1], rot[2]]).reshape(1, 3)
        a2 = np.array([rot[3], rot[4], rot[5]]).reshape(1, 3)

        b1 = a1 / np.linalg.norm(a1)

        c1 = np.multiply(b1, a2)
        c2 = np.sum(c1)

        b2 = a2 - c2 * b1
        b2 = b2 / np.linalg.norm(b2)
        b3 = np.cross(b1, b2, axis=1)
        return np.concatenate((b1, b2, b3), axis=0)

    def decodePredictions(fnImages, p6d_redundant, symList):
        pred6d = list(map(calculate_r6d,p6d_redundant))
        matrices = list(map(rotation6d_to_matrixZYZ, pred6d))
        angles = list(map(xmippLib.Euler_matrix2angles, matrices))
        angles = list(map(lambda item: symmetryOperatorSymList(symList, item[0], item[1], item[2], -1), angles))
        return angles

    Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnXmdExp)
    symList = fillSymList(symmetry)

    mdExp = xmippLib.MetaData(fnXmdExp)
    fnImgs = mdExp.getColumnValues(xmippLib.MDL_IMAGE)

    mdExpImages = xmippLib.MetaData(fnXmdImages)
    fnImages = mdExpImages.getColumnValues(xmippLib.MDL_IMAGE)

    shiftX = mdExp.getColumnValues(xmippLib.MDL_SHIFT_X)
    shiftY = mdExp.getColumnValues(xmippLib.MDL_SHIFT_Y)
    shifts = [np.array((sX, sY)) for sX, sY in zip(shiftX, shiftY)]

    def shift_image(img, img_shifts):
        """Shift image to center particle"""
        return shift(img, (-img_shifts[0], -img_shifts[1], 0), order=1, mode='wrap')

    models = []
    for index in range(numAngModels):
        AngModel = load_model(fnAngModel + str(index) + ".h5", compile=False)
        AngModel.compile(loss="mean_squared_error", optimizer='adam')
        models.append(AngModel)

    numImgs = len(fnImgs)
    predictions = np.zeros((numImgs, numAngModels, 42))
    numBatches = numImgs // maxSize
    if numImgs % maxSize > 0:
        numBatches = numBatches + 1
    k = 0
    # perform batch predictions for each model
    for i in range(numBatches):
        numPredictions = min(maxSize, numImgs-i*maxSize)
        Xexp = np.zeros((numPredictions, Xdim, Xdim, 1), dtype=np.float64)
        for j in range(numPredictions):
            Iexp = np.reshape(xmippLib.Image(fnImgs[k]).getData(), (Xdim, Xdim, 1))
            Xexp[j, ] = (Iexp - np.mean(Iexp)) / np.std(Iexp)
            Xexp[j, ] = shift_image(Xexp[j, ], shifts[k])
            k += 1
        for index in range(numAngModels):
            predictions[i*maxSize:(i*maxSize + numPredictions), index, :] = models[index].predict(Xexp)

    Y = decodePredictions(fnImages, predictions, symList)
    produce_output(mdExp, Y, fnImages)
    mdExp.write(os.path.join(outputDir, "predict_results.xmd"))
