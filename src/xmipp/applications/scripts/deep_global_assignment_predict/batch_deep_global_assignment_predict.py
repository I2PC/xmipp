#!/usr/bin/env python3

import glob
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
    from xmippPyModules.deepGlobalAssignment import Redundancy
    from xmippPyModules.xmipp_utils import RotationAverager

    checkIf_tf_keras_installed()
    fnExp = sys.argv[1]
    fnExpResized = sys.argv[2]
    gpuId = sys.argv[3]
    fnModelDir = sys.argv[4]
    symmetry = sys.argv[5]
    fnOut = sys.argv[6]

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    import keras

    def produce_output(fnExp, Y, fnOut):
        mdExp = xmippLib.MetaData(fnExp)
        ID = 0
        for objId in mdExp:
            rot, tilt, psi = Y[ID]
            mdExp.setValue(xmippLib.MDL_ANGLE_ROT, rot, objId)
            mdExp.setValue(xmippLib.MDL_ANGLE_TILT, tilt, objId)
            mdExp.setValue(xmippLib.MDL_ANGLE_PSI, psi, objId)
            ID += 1
        mdExp.write(fnOut)

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

    def decodePredictions(p6d_redundant):
        pred6d = list(map(Redundancy().make_nonredundant,p6d_redundant))
        matrices = list(map(rotation6d_to_matrixZYZ, pred6d))
        angles = list(map(xmippLib.Euler_matrix2angles, matrices))
        return angles

    mdResized = xmippLib.MetaData(fnExpResized)
    XdimResized, _, _, _, _ = xmippLib.MetaDataInfo(fnExpResized)
    fnImgs = mdResized.getColumnValues(xmippLib.MDL_IMAGE)

    shiftX = mdResized.getColumnValues(xmippLib.MDL_SHIFT_X)
    shiftY = mdResized.getColumnValues(xmippLib.MDL_SHIFT_Y)
    shifts = [np.array((sX, sY)) for sX, sY in zip(shiftX, shiftY)]

    def shift_image(img, img_shifts):
        """Shift image to center particle"""
        return shift(img, (-img_shifts[0], -img_shifts[1], 0), order=1, mode='wrap')


    numImgs = len(fnImgs)
    numBatches = numImgs // maxSize
    if numImgs % maxSize > 0:
        numBatches = numBatches + 1

    Ylist = []
    numAngModels = len(glob.glob(os.path.join(fnModelDir,"model*.h5")))
    for index in range(numAngModels):
        AngModel = keras.models.load_model(os.path.join(fnModelDir,"model%d.h5"%index), compile=False)

        k = 0
        predictions = np.zeros((numImgs, 64))
        for i in range(numBatches):
            numPredictions = min(maxSize, numImgs-i*maxSize)
            Xexp = np.zeros((numPredictions, XdimResized, XdimResized, 1), dtype=np.float64)
            for j in range(numPredictions):
                Iexp = np.reshape(xmippLib.Image(fnImgs[k]).getData(), (XdimResized, XdimResized, 1))
                Xexp[j, ] = (Iexp - np.mean(Iexp)) / np.std(Iexp)
                Xexp[j, ] = shift_image(Xexp[j, ], shifts[k])
                k += 1
            predictions[i*maxSize:(i*maxSize + numPredictions), :] = AngModel.predict(Xexp)

        Ylist.append(decodePredictions(predictions))

    averager=RotationAverager(Ylist)
    averager.bringToAsymmetricUnit(symmetry)
    Y=averager.computeAverageAssignment()
    produce_output(fnExp, Y, fnOut)
