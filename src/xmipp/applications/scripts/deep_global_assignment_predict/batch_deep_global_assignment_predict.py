#!/usr/bin/env python3

import glob
import math
import numpy as np
from numpy.linalg import norm
import os
import sys
import xmippLib
from xmipp_base import *


class ScriptDeepGlobalAssignmentPredict(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Predict using a deep global assignment model')

        ## params
        self.addParamsLine(' --iexp <metadata>            : xmd file with the list of experimental images')
        self.addParamsLine(' --iexpResized <metadata>     : xmd file with the list of resized experimental images')
        self.addParamsLine(' --modelDir <dir>             : Directory with the neural network models')
        self.addParamsLine(' -o <metadata>                : Output filename')
        self.addParamsLine('[--gpu <id=0>]                : GPU Id')
        self.addParamsLine('[--sym <s=c1>]                : Symmetry')

    def run(self):
        fnExp = self.getParam("--iexp")
        fnExpResized = self.getParam("--iexpResized")
        fnModelDir = self.getParam("--modelDir")
        fnOut = self.getParam("-o")
        gpuId = self.getParam("--gpu")
        symmetry = self.getParam("--sym")
        maxSize = 32

        from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed
        checkIf_tf_keras_installed()

        from xmippPyModules.xmipp_utils import RotationAverager
        import xmippPyModules.deepGlobalAssignment as deepGlobal

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

            b3 = a3 / np.linalg.norm(a3)

            c2 = np.inner(a2, b3)

            b2 = a2 - c2 * b3
            b2 = b2 / np.linalg.norm(b2)
            b1 = np.cross(b2, b3, axis=1)
            return np.concatenate((b1, b2, b3), axis=0)

        def decodePredictions(p6d):
            matrices = list(map(rotation6d_to_matrixZYZ, p6d))
            angles = list(map(xmippLib.Euler_matrix2angles, matrices))
            return angles

        mdResized = xmippLib.MetaData(fnExpResized)
        Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnExp)
        XdimResized, _, _, _, _ = xmippLib.MetaDataInfo(fnExpResized)
        K = Xdim/XdimResized
        fnImgs = mdResized.getColumnValues(xmippLib.MDL_IMAGE)
        itemIds = mdResized.getColumnValues(xmippLib.MDL_ITEM_ID)

        numImgs = len(fnImgs)
        numBatches = numImgs // maxSize
        if numImgs % maxSize > 0:
            numBatches = numBatches + 1

        angleList=[]
        shiftList=[]
        numAngModels = len(glob.glob(os.path.join(fnModelDir,"model_angles*.tf")))
        for index in range(numAngModels):
            AngModel = keras.models.load_model(os.path.join(fnModelDir,"model_angles%d.tf"%index),
                                               custom_objects={'ConcatenateZerosLayer': deepGlobal.ConcatenateZerosLayer,
                                                               'ShiftImageLayer': deepGlobal.ShiftImageLayer,
                                                               'VAESampling': deepGlobal.VAESampling},
                                               compile=False)

            k = 0
            predictions = np.zeros((numImgs, 8))
            for i in range(numBatches):
                numPredictions = min(maxSize, numImgs-i*maxSize)
                Xexp = np.zeros((numPredictions, XdimResized, XdimResized, 1), dtype=np.float64)
                for j in range(numPredictions):
                    Iexp = np.reshape(xmippLib.Image(fnImgs[k]).getData(), (XdimResized, XdimResized, 1))
                    Xexp[j, ] = (Iexp - np.mean(Iexp)) / np.std(Iexp)
                    k += 1
                predictions[i*maxSize:(i*maxSize + numPredictions), :] = AngModel.predict(Xexp)

            for i, image in enumerate(Xexp):
                mean = np.mean(image)
                std = np.std(image)
                fn = fnImgs[i]
                print(f"Image {fn} {i}: Mean = {mean}, Std = {std}")
                np.set_printoptions(threshold=sys.maxsize)
                print(predictions[i])

            angleList.append(decodePredictions(predictions))
            shiftList.append(predictions[:,-2:]*K)

        averager=RotationAverager(angleList)
        averager.bringToAsymmetricUnit(symmetry)
        angles=averager.computeAverageAssignment()
        shift = np.mean(np.stack(shiftList),axis=0)
        produce_output(fnExp, angles, shift, itemIds, fnOut)

if __name__ == '__main__':
    ScriptDeepGlobalAssignmentPredict().tryRun()