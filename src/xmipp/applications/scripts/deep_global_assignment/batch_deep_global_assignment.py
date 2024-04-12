#!/usr/bin/env python3

import math
import numpy as np
from scipy.ndimage import shift
import os
import random
import sys
from time import time

import xmippLib
from xmipp_base import *

def euler_to_rotation6d(angles):
    mat = xmippLib.Euler_angles2matrix(angles[0], angles[1], angles[2])
    return np.concatenate((mat[1], mat[2]))

def rotation6d_to_euler(rot):
    a2 = np.array([rot[0], rot[1], rot[2]]).reshape(1, 3)
    a3 = np.array([rot[3], rot[4], rot[5]]).reshape(1, 3)

    b3 = a3 / np.linalg.norm(a3)

    c2 = np.inner(a2, b3)

    b2 = a2 - c2 * b3
    b2 = b2 / np.linalg.norm(b2)
    b1 = np.cross(b2, b3, axis=1)
    mat=np.concatenate((b1, b2, b3), axis=0)
    print("recons mat",mat)
    angles=xmippLib.Euler_matrix2angles(mat)
    print(angles)
    return angles

class ScriptDeepGlobalAssignment(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Train a deep global assignment model')
        ## params
        self.addParamsLine(' -i <metadata>                : xmd file with the list of aligned images')
        self.addParamsLine(' --oroot <root>               : Rootname for the models')
        self.addParamsLine('[--maxEpochs <N=500>]         : Max. number of epochs')
        self.addParamsLine('[--batchSize <N=8>]           : Batch size')
        self.addParamsLine('[--gpu <id=0>]                : GPU Id')
        self.addParamsLine('[--Nmodels <N=5>]             : Number of models')
        self.addParamsLine('[--learningRate <lr=0.0001>]  : Learning rate')
        self.addParamsLine('[--precision <s=0.5>]         : Alignment precision measured in pixels')
        self.addParamsLine('[--Nimgs <n=1000>]            : Subset size for training')
        self.addParamsLine('[--mode <mode>]               : Mode: shift, angles')

    def run(self):
        fnXmd = self.getParam("-i")
        fnModel = self.getParam("--oroot")
        maxEpochs = int(self.getParam("--maxEpochs"))
        batch_size = int(self.getParam("--batchSize"))
        gpuId = self.getParam("--gpu")
        numModels = int(self.getParam("--Nmodels"))
        learning_rate = float(self.getParam("--learningRate"))
        precision = float(self.getParam("--precision"))
        mode = self.getParam("--mode")
        Nimgs = int(self.getParam('--Nimgs'))

        from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

        if not gpuId.startswith('-1'):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpuId
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
        checkIf_tf_keras_installed()

        import tensorflow as tf
        import keras
        import xmippPyModules.deepGlobalAssignment as deepGlobal
        from xmippPyModules.xmipp_utils import XmippTrainingSequence

        class DataLoader():
            """Generates data for fnImgs"""

            def __init__(self, fnImgs, angles, shifts, batch_size, dim, mode):
                """Initialization"""
                self.fnImgs = fnImgs
                self.angles = angles
                self.shifts = shifts
                self.batch_size = batch_size
                self.dim = dim
                self.mode = mode
                self.mask = deepGlobal.create_circular_mask(dim, dim)
                self.loadData()

            def loadData(self):
                # Read all data in memory
                Nimgs=len(self.angles)

                self.X = np.zeros((Nimgs, self.dim, self.dim, 1), dtype=np.float64)
                if self.mode=="shift":
                    self.y = np.zeros((Nimgs, 2), dtype=np.float64)
                else:
                    self.y = np.zeros((Nimgs, 6), dtype=np.float64)
                for i in range(Nimgs):
                    I = xmippLib.Image(self.fnImgs[i]).getData()
                    I = (I - np.mean(I)) / np.std(I)
                    if self.mode == 'shift':
                        self.y[i] = self.shifts[i]
                    else:
                        # print(self.fnImgs[i], np.max(I), "shift", -self.shifts[i])
                        I = shift(I, -self.shifts[i], mode='wrap')
                        self.y[i] = euler_to_rotation6d(self.angles[i])
                    I *= self.mask
                    self.X[i] = np.reshape(I, (self.dim, self.dim, 1))

        def get_labels(fnXmd, Nimgs):
            """Returns dimensions, images, angles and shifts values from images files"""
            md = xmippLib.MetaData(fnXmd)
            Xdim, _, _, _ = md.getImageSize()
            fnImg = md.getColumnValues(xmippLib.MDL_IMAGE)
            shiftX = md.getColumnValues(xmippLib.MDL_SHIFT_X)
            shiftY = md.getColumnValues(xmippLib.MDL_SHIFT_Y)
            rots = md.getColumnValues(xmippLib.MDL_ANGLE_ROT)
            tilts = md.getColumnValues(xmippLib.MDL_ANGLE_TILT)
            psis = md.getColumnValues(xmippLib.MDL_ANGLE_PSI)

            angles = [np.array((r,t,p)) for r, t, p in zip(rots, tilts, psis)]
            img_shift = [np.array((sX,sY)) for sX, sY in zip(shiftX, shiftY)]

            Nimgs = min(Nimgs, len(angles))
            idx = random.sample([i for i in range(len(angles))], Nimgs)
            return Xdim, [fnImg[i] for i in idx], [angles[i] for i in idx], [img_shift[i] for i in idx]
            # return Xdim, fnImg, angles, img_shift

        def testModel(model, generator, dataLoader, mode):
            ypred = model.predict(generator.x)
            for i in range(len(dataLoader.fnImgs)):
                print(dataLoader.fnImgs[i])
                np.set_printoptions(threshold=sys.maxsize)
                print('i=%d X[i].max'%i,np.max(generator.x[i]))
                np.set_printoptions(threshold=sys.maxsize)
                print(generator.y[i])
                np.set_printoptions(threshold=sys.maxsize)
                print(ypred[i])
                if mode=="angles":
                    print("true angles",dataLoader.angles[i])
                    trueAngles=dataLoader.angles[i]
                    print("true matrix", xmippLib.Euler_angles2matrix(trueAngles[0],trueAngles[1],trueAngles[2]))
                    rotation6d_to_euler(ypred[i])

        def trainModel(model, dataLoader, mode, modeprec, fnThisModel, lossFunction):
            adam_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            sys.stdout.flush()
            if lossFunction is not None:
                model.compile(loss=lossFunction, optimizer=adam_opt)

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',  # Metric to be monitored
                factor=0.1,  # Factor by which the learning rate will be reduced. new_lr = lr * factor
                patience=10,  # Number of epochs with no improvement after which learning rate will be reduced
                min_lr=1e-8,  # Lower bound on the learning rate
            )

            generator = XmippTrainingSequence(dataLoader.X, dataLoader.y, batch_size, maxSize=len(dataLoader.fnImgs),
                                              randomize=False)

            # generator.shuffle_data()

            epoch = 0
            for i in range(maxEpochs):
                start_time = time()
                history = model.fit(generator, epochs=1, verbose=0, callbacks=[reduce_lr])
                end_time = time()
                epoch += 1
                loss = history.history['loss'][-1]
                print("Epoch %d loss=%f trainingTime=%d" % (epoch, loss, int(end_time-start_time)), flush=True)
                if loss < modeprec:
                    break
            # testModel(model, generator, dataLoader, mode)
            model.save_weights(fnThisModel, save_format="h5")

        SL = xmippLib.SymList()
        listSymmetryMatrices = SL.getSymmetryMatrices('c1')
        Xdim, fnImgs, angles, shifts = get_labels(fnXmd, Nimgs)
        dataLoader = DataLoader(fnImgs, angles, shifts, batch_size, Xdim, mode)

        angularLoss = deepGlobal.AngularLoss(listSymmetryMatrices, Xdim)

        try:
            for index in range(numModels):
                fnModelIndex = fnModel + str(index) + ".h5"
                # Learn shift
                if mode=="shift":
                    modelShift = deepGlobal.constructShiftModel(Xdim)
                    modelShift.summary()
                    if os.path.exists(fnModelIndex):
                        print("Loading weights from",fnModelIndex)
                        modelShift.build(input_shape=(None, Xdim, Xdim, 1))
                        modelShift.load_weights(fnModelIndex)
                    trainModel(modelShift, dataLoader, mode, precision, fnModelIndex, 'mae')
                else:
                    # Learn angles
                    model = deepGlobal.constructAnglesModel(Xdim)
                    model.summary()
                    if os.path.exists(fnModelIndex):
                        model.build(input_shape=(None, Xdim, Xdim, 1))
                        print("Loading weights from",fnModelIndex)
                        model.load_weights(fnModelIndex)
                    trainModel(model, dataLoader, mode, precision, fnModelIndex, angularLoss)
        except Exception as e:
            print(e)
            sys.exit(1)

if __name__ == '__main__':
    exitCode = ScriptDeepGlobalAssignment().tryRun()
    sys.exit(exitCode)
