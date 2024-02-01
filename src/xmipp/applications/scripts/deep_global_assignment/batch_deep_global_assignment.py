#!/usr/bin/env python3

import math
import numpy as np
from scipy.ndimage import shift
import os
import sys
import xmippLib
from time import time
from xmipp_base import *

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

        from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

        if not gpuId.startswith('-1'):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpuId
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
        checkIf_tf_keras_installed()

        import tensorflow as tf
        import xmippPyModules.deepGlobalAssignment as deepGlobal
        from xmippPyModules.xmipp_utils import XmippTrainingSequence

        class DataGenerator():
            """Generates data for fnImgs"""

            def __init__(self, fnImgs, angles, shifts, batch_size, dim, mode):
                """Initialization"""
                self.fnImgs = fnImgs
                self.angles = angles
                self.shifts = shifts
                self.batch_size = batch_size
                self.dim = dim
                self.mode = mode
                self.loadData()

            def loadData(self):
                def euler_to_rotation6d(angles):
                    mat =  xmippLib.Euler_angles2matrix(angles[0],angles[1],angles[2])
                    return np.concatenate((mat[1], mat[2]))

                # Read all data in memory
                self.X = np.zeros((len(self.angles), self.dim, self.dim, 1), dtype=np.float64)
                if self.mode=="shift":
                    self.y = np.zeros((len(self.angles), 2), dtype=np.float64)
                else:
                    self.y = np.zeros((len(self.angles), 6), dtype=np.float64)
                for i in range(len(self.angles)):
                    I = xmippLib.Image(self.fnImgs[i]).getData()
                    I = (I - np.mean(I)) / np.std(I)
                    if self.mode == 'shift':
                        self.y[i] = self.shifts[i]
                    else:
                        # print(self.fnImgs[i], np.max(I), "shift", -self.shifts[i])
                        I = shift(I, -self.shifts[i], mode='wrap')
                        self.y[i] = euler_to_rotation6d(self.angles[i])
                    self.X[i] = np.reshape(I, (self.dim, self.dim, 1))

        def get_labels(fnXmd):
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

            return Xdim, fnImg, angles, img_shift

        def trainModel(model, X, y, modeprec, fnThisModel, lossFunction):
            adam_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            sys.stdout.flush()
            if lossFunction is not None:
                model.compile(loss=lossFunction, optimizer=adam_opt)

            generator = XmippTrainingSequence(X, y, batch_size, maxSize=None)
            # generator.shuffle_data()

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',  # Metric to be monitored
                factor=0.1,  # Factor by which the learning rate will be reduced. new_lr = lr * factor
                patience=10,  # Number of epochs with no improvement after which learning rate will be reduced
                min_lr=1e-8,  # Lower bound on the learning rate
            )

            goOn = True
            while goOn:
                print("Current size =%d"%generator.maxSize)
                epoch = 0
                for i in range(maxEpochs):
                    start_time = time()
                    history = model.fit(generator, epochs=1, verbose=0, callbacks=[reduce_lr])
                    end_time = time()
                    goOn = not generator.isMaxSize()
                    epoch += 1
                    loss = history.history['loss'][-1]
                    print("Epoch %d loss=%f trainingTime=%d" % (epoch, loss, int(end_time-start_time)), flush=True)
                    if loss < modeprec:
                        break
                generator.increaseMaxSize(1.25)
            model.save(fnThisModel, save_format="tf")

        def testModel(model, X, y):
            ypred = model.predict(X)
            for i, _ in enumerate(ypred):
                np.set_printoptions(threshold=sys.maxsize)
                print(y[i])
                np.set_printoptions(threshold=sys.maxsize)
                print(ypred[i])

        SL = xmippLib.SymList()
        listSymmetryMatrices = SL.getSymmetryMatrices('c1')
        Xdim, fnImgs, angles, shifts = get_labels(fnXmd)
        training_generator = DataGenerator(fnImgs, angles, shifts, batch_size, Xdim, mode)

        angularLoss = deepGlobal.AngularLoss(listSymmetryMatrices, Xdim)

        try:
            for index in range(numModels):
                fnModelIndex = fnModel + str(index) + ".tf"
                # Learn shift
                if mode=="shift":
                    print("Learning shift")
                    modelShift = deepGlobal.constructShiftModel(Xdim)
                    modelShift.summary()
                    trainModel(modelShift, training_generator.X, training_generator.y,
                               precision, fnModelIndex, 'mae')
                else:
                    # Learn angles
                    print("Learning angular assignment")
                    model = deepGlobal.constructAnglesModel(Xdim)
                    model.summary()
                    trainModel(model, training_generator.X, training_generator.y, precision, fnModelIndex, angularLoss)
                    # testModel(model, training_generator.X, training_generator.y)
        except Exception as e:
            print(e)
            sys.exit(1)

if __name__ == '__main__':
    exitCode = ScriptDeepGlobalAssignment().tryRun()
    sys.exit(exitCode)
