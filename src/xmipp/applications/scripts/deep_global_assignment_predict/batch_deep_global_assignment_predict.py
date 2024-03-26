#!/usr/bin/env python3

import glob
import math
import numpy as np
from scipy.ndimage import shift
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
        self.addParamsLine(' -i <metadata>                : xmd file with the list of experimental images')
        self.addParamsLine(' --modelDir <dir>             : Directory with the neural network models')
        self.addParamsLine('[-o <metadata="">]            : Output filename')
        self.addParamsLine(' --mode <mode>                : Valid modes: shift, angles')
        self.addParamsLine('[--gpu <id=0>]                : GPU Id')
        self.addParamsLine('[--sym <sym=c1>]              : Symmetry')

    def run(self):
        fnIn = self.getParam("-i")
        fnModelDir = self.getParam("--modelDir")
        fnOut = self.getParam("-o")
        if fnOut=="":
            fnOut=fnIn
        gpuId = self.getParam("--gpu")
        mode = self.getParam("--mode")
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

        def produce_output(fnIn, angles, shifts, itemIds, fnOut):
            Ydict = {itemId: index for index, itemId in enumerate(itemIds)}
            md = xmippLib.MetaData(fnIn)
            for objId in md:
                itemId = md.getValue(xmippLib.MDL_ITEM_ID, objId)
                if angles is not None:
                    rot, tilt, psi = angles[Ydict[itemId]]
                    md.setValue(xmippLib.MDL_ANGLE_ROT, rot, objId)
                    md.setValue(xmippLib.MDL_ANGLE_TILT, tilt, objId)
                    md.setValue(xmippLib.MDL_ANGLE_PSI, psi, objId)
                if shifts is not None:
                    x, y = shifts[Ydict[itemId]]
                    md.setValue(xmippLib.MDL_SHIFT_X, x, objId)
                    md.setValue(xmippLib.MDL_SHIFT_Y, y, objId)
            md.write(fnOut)

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

        mdIn = xmippLib.MetaData(fnIn)
        Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnIn)
        fnImgs = mdIn.getColumnValues(xmippLib.MDL_IMAGE)
        itemIds = mdIn.getColumnValues(xmippLib.MDL_ITEM_ID)
        x = mdIn.getColumnValues(xmippLib.MDL_SHIFT_X)
        y = mdIn.getColumnValues(xmippLib.MDL_SHIFT_Y)
        mask = deepGlobal.create_circular_mask(Xdim, Xdim)

        numImgs = len(fnImgs)
        numBatches = numImgs // maxSize
        if numImgs % maxSize > 0:
            numBatches = numBatches + 1

        predictionsList=[]
        models = glob.glob(fnModelDir+"*.tf")
        numAngModels = len(models)
        for index in range(numAngModels):
            if mode=="shift":
                model = keras.models.load_model(models[index], compile=False)
                predictions = np.zeros((numImgs, 2))
            else:
                model = keras.models.load_model(models[index],
                                               custom_objects={'Angles2VectorLayer': deepGlobal.Angles2VectorLayer},
                                               compile=False)
                predictions = np.zeros((numImgs, 6))

            k = 0
            for i in range(numBatches):
                numPredictions = min(maxSize, numImgs-i*maxSize)
                X = np.zeros((numPredictions, Xdim, Xdim, 1), dtype=np.float64)
                for j in range(numPredictions):
                    I = xmippLib.Image(fnImgs[k]).getData()
                    I = (I - np.mean(I)) / np.std(I)
                    if mode=="angles":
                        # print(fnImgs[k], np.max(I), "shift", np.array([-x[k],-y[k]]))
                        I = shift(I, np.array([-x[k],-y[k]]), mode='wrap')
                    I *= mask
                    X[j, ] =  np.reshape(I, (Xdim, Xdim, 1))
                    k += 1
                ypred = model.predict(X)
                # print(ypred)
                predictions[i*maxSize:(i*maxSize + numPredictions), :] = ypred

            if mode=="angles":
                predictionsList.append(decodePredictions(predictions))
            else:
                predictionsList.append(predictions)

        angles = None
        shifts = None
        if mode=="angles":
            averager=RotationAverager(predictionsList)
            averager.bringToAsymmetricUnit(symmetry)
            angles=averager.computeAverageAssignment()
        else:
            shifts = np.mean(np.stack(predictionsList),axis=0)
        produce_output(fnIn, angles, shifts, itemIds, fnOut)

if __name__ == '__main__':
    ScriptDeepGlobalAssignmentPredict().tryRun()