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

    def produce_output(fnExp, angles, shifts, itemIds, fnOut):
        Ydict = {itemId: index for index, itemId in enumerate(itemIds)}
        mdExp = xmippLib.MetaData(fnExp)
        for objId in mdExp:
            itemId = mdExp.getValue(xmippLib.MDL_ITEM_ID, objId)
            rot, tilt, psi = angles[Ydict[itemId]]
            x, y = shifts[Ydict[itemId]]
            mdExp.setValue(xmippLib.MDL_ANGLE_ROT, rot, objId)
            mdExp.setValue(xmippLib.MDL_ANGLE_TILT, tilt, objId)
            mdExp.setValue(xmippLib.MDL_ANGLE_PSI, psi, objId)
            mdExp.setValue(xmippLib.MDL_SHIFT_X, x, objId)
            mdExp.setValue(xmippLib.MDL_SHIFT_Y, y, objId)
        mdExp.write(fnOut)

    def rotation6d_to_matrixZYZ(rot):
        """Return rotation matrix from 6D representation."""
        a2 = np.array([rot[0], rot[1], rot[2]]).reshape(1, 3)
        a3 = np.array([rot[3], rot[4], rot[5]]).reshape(1, 3)

        b2 = a2 / np.linalg.norm(a2)

        c1 = np.dot(b2, a3)

        b3 = a3 - c1 * b2
        b3 = b3 / np.linalg.norm(b3)
        b1 = np.cross(b2, b3, axis=1)
        return np.concatenate((b1, b2, b3), axis=0)

    def decodePredictions(p6d):
        matrices = list(map(rotation6d_to_matrixZYZ, p6d))
        angles = list(map(xmippLib.Euler_matrix2angles, matrices))
        return angles

    mdResized = xmippLib.MetaData(fnExpResized)
    XdimResized, _, _, _, _ = xmippLib.MetaDataInfo(fnExpResized)
    fnImgs = mdResized.getColumnValues(xmippLib.MDL_IMAGE)
    itemIds = mdResized.getColumnValues(xmippLib.MDL_ITEM_ID)

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

    angleList=[]
    shiftList=[]
    numAngModels = len(glob.glob(os.path.join(fnModelDir,"model_angles*.h5")))
    for index in range(numAngModels):
        AngModel = keras.models.load_model(os.path.join(fnModelDir,"model_angles%d.h5"%index), compile=False)

        k = 0
        predictions = np.zeros((numImgs, 8))
        for i in range(numBatches):
            numPredictions = min(maxSize, numImgs-i*maxSize)
            Xexp = np.zeros((numPredictions, XdimResized, XdimResized, 1), dtype=np.float64)
            for j in range(numPredictions):
                Iexp = np.reshape(xmippLib.Image(fnImgs[k]).getData(), (XdimResized, XdimResized, 1))
                Xexp[j, ] = (Iexp - np.mean(Iexp)) / np.std(Iexp)
                Xexp[j, ] = shift_image(Xexp[j, ], shifts[k])
                k += 1
            predictions[i*maxSize:(i*maxSize + numPredictions), :] = AngModel.predict(Xexp)

        angleList.append(decodePredictions(predictions))
        shiftList.append(predictions[:,-2:])

    averager=RotationAverager(angleList)
    averager.bringToAsymmetricUnit(symmetry)
    angles=averager.computeAverageAssignment()
    shift = np.mean(np.stack(shiftList),axis=0)
    produce_output(fnExp, angles, shift, itemIds, fnOut)
