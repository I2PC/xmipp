#!/usr/bin/env python3

import math
import numpy as np
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
        self.addParamsLine(' --isim <metadata>            : xmd file with the list of simulated images,'
                           'they must have a 3D alignment')
        self.addParamsLine('[--iexp <metadata="">]        : xmd file with the list of experimental images,'
                           'they must have a 3D alignment, the same angles and shifts as the simulated')
        self.addParamsLine(' --oroot <root>               : Rootname for the models')
        self.addParamsLine('[--maxEpochs <N=500>]         : Max. number of epochs')
        self.addParamsLine('[--batchSize <N=8>]           : Batch size')
        self.addParamsLine('[--firstBatch <N=32>]         : First batch size')
        self.addParamsLine('[--gpu <id=0>]                : GPU Id')
        self.addParamsLine('[--Nmodels <N=5>]             : Number of models')
        self.addParamsLine('[--learningRate <lr=0.0001>]  : Learning rate')
        self.addParamsLine('[--sym <s=c1>]                : Symmetry')
        self.addParamsLine('[--modelSize <s=0>]           : Model size (0=277k, 1=1M, 2=5M, 3=19M parameters)')
        self.addParamsLine('[--precision <s=0.07>]        : Alignment precision measured in percentage of '
                           'the image size')
        self.addParamsLine('[--vae <vaePrec=0.1>]         : Include VAE, the VAE precision is a fraction of'
                           '                              : the image standard deviation')
        self.addParamsLine('[--centerParticles]           : Learn a model to center particles')
        self.addParamsLine('[--approximateFirst]          : Learn a first approximation as a helper')

    def run(self):
        fnXmdSim = self.getParam("--isim")
        fnXmdExp = self.getParam("--iexp")
        fnModel = self.getParam("--oroot")
        maxEpochs = int(self.getParam("--maxEpochs"))
        batch_size = int(self.getParam("--batchSize"))
        gpuId = self.getParam("--gpu")
        numModels = int(self.getParam("--Nmodels"))
        learning_rate = float(self.getParam("--learningRate"))
        symmetry = self.getParam("--sym")
        modelSize = int(self.getParam("--modelSize"))
        firstBatch = int(self.getParam("--firstBatch"))
        precision = float(self.getParam("--precision"))
        includeVae = self.checkParam("--vae")
        if includeVae:
            vaePrecision = float(self.getParam("--vae"))
        centerParticles = self.checkParam("--centerParticles")
        approximateFirst = self.checkParam("--approximateFirst")

        from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

        if not gpuId.startswith('-1'):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpuId
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
        checkIf_tf_keras_installed()

        import tensorflow as tf
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
        import xmippPyModules.deepGlobalAssignment as deepGlobal
        from xmippPyModules.xmipp_utils import XmippTrainingSequence

        class DataGenerator():
            """Generates data for fnImgs"""

            def __init__(self, fnImgsSim, fnImgsExp, angles, shifts, batch_size, dim):
                """Initialization"""
                self.fnImgsSim = fnImgsSim
                self.fnImgsExp = fnImgsExp
                self.angles = angles
                self.shifts = shifts
                self.batch_size = batch_size
                self.dim = dim
                self.loadData()

            def loadData(self):
                def euler_to_rotation6d(angles, shifts):
                    mat =  xmippLib.Euler_angles2matrix(angles[0],angles[1],angles[2])
                    return np.concatenate((mat[1], mat[2], shifts))

                # Read all data in memory
                self.Xsim = np.zeros((len(self.angles), self.dim, self.dim, 1), dtype=np.float64)
                self.Xexp = np.zeros((len(self.angles), self.dim, self.dim, 1), dtype=np.float64)
                self.ysim = np.zeros((len(self.angles), 8), dtype=np.float64)
                readExp = len(fnImgsExp)>0
                for i in range(len(self.angles)):
                    Isim = np.reshape(xmippLib.Image(self.fnImgsSim[i]).getData(), (self.dim, self.dim, 1))
                    self.Xsim[i] = (Isim - np.mean(Isim)) / np.std(Isim)
                    self.ysim[i] = euler_to_rotation6d(self.angles[i], self.shifts[i])
                    if readExp:
                        Iexp = np.reshape(xmippLib.Image(self.fnImgsExp[i]).getData(), (self.dim, self.dim, 1))
                        self.Xexp[i] = (Iexp - np.mean(Iexp)) / np.std(Iexp)

                self.indexes = [i for i in range(len(self.fnImgsSim))]

                # K means of the projection directions
                kmeans = KMeans(n_clusters=6, random_state=0).fit(self.ysim[:,3:6])
                self.cluster_centers = normalize(kmeans.cluster_centers_)
                self.ylocation = np.dot(self.ysim[:,3:6], self.cluster_centers.T)

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

        def trainModel(model, X, y, modeprec, firstBatch, lossFunction=None, saveModel=False, fnModelIndex=None):
            adam_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            sys.stdout.flush()
            if lossFunction is not None:
                model.compile(loss=lossFunction, optimizer=adam_opt)

            generator = XmippTrainingSequence(X, y, batch_size, maxSize=firstBatch)
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
            if saveModel:
                model.save(fnModelIndex, save_format="tf")

        def testModel(model, X, y):
            ypred = model.predict(X)
            for i, image in enumerate(ypred):
                np.set_printoptions(threshold=sys.maxsize)
                print(y[i])
                np.set_printoptions(threshold=sys.maxsize)
                print(ypred[i])

        SL = xmippLib.SymList()
        listSymmetryMatrices = SL.getSymmetryMatrices(symmetry)
        Xdim, fnImgsSim, angles, shifts = get_labels(fnXmdSim)
        if fnXmdExp!="":
            _, fnImgsExp, _, _ = get_labels(fnXmdExp)
        else:
            fnImgsExp=[]
        training_generator = DataGenerator(fnImgsSim, fnImgsExp, angles, shifts, batch_size, Xdim)

        angularLoss = deepGlobal.AngularLoss(listSymmetryMatrices, Xdim)

        for index in range(numModels):
            fnModelIndex = fnModel + "_angles"+str(index) + ".tf"
            if os.path.exists(fnModelIndex):
                continue

            # Learn shift
            if centerParticles:
                print("Learning shift")
                angularLoss.setMode(deepGlobal.SHIFT_MODE)
                modelShift = deepGlobal.constructShiftModel(Xdim)
                modelShift.summary()
                trainModel(modelShift, training_generator.Xsim, training_generator.ysim,
                           0.1, firstBatch, angularLoss, saveModel=False) # Prec=0.1 pixels
                modelShift.trainable = False
            else:
                modelShift = None

            # Location model
            if approximateFirst:
                print("Learning approximate location")
                modelLocation = deepGlobal.constructLocationModel(Xdim, training_generator.cluster_centers.shape[0],
                                                                  modelShift)
                modelLocation.summary()
                trainModel(modelLocation, training_generator.Xsim, training_generator.ylocation,
                           precision, firstBatch, 'mae', saveModel=False)
                modelLocation.trainable = False
            else:
                modelLocation = None

            # VAE
            if includeVae:
                print("Learning VAE")
                vae, vaeEncoder = deepGlobal.constructVAEModel(Xdim, modelShift)
                trainModel(vae, training_generator.Xsim, training_generator.Xsim, vaePrecision, firstBatch,
                           saveModel=False)
                vaeEncoder.trainable = False
            else:
                vaeEncoder = None

            # Learn angles
            print("Learning angular assignment")
            angularLoss.setMode(deepGlobal.FULL_MODE)
            modeprec = precision
            model = deepGlobal.constructAnglesModel(Xdim, modelSize, modelShift, modelLocation, vaeEncoder)
            model.summary()
            trainModel(model, training_generator.Xsim, training_generator.ysim, modeprec, firstBatch, angularLoss,
                       saveModel=True, fnModelIndex=fnModelIndex)

            testModel(model, training_generator.Xsim, training_generator.ysim)

if __name__ == '__main__':
    exitCode = ScriptDeepGlobalAssignment().tryRun()
    sys.exit(exitCode)
