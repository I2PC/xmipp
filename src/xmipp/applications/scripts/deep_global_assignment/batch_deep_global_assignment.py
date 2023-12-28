#!/usr/bin/env python3

import math
import numpy as np
from operator import itemgetter
import os
import sys
import random
import xmippLib
from time import time
from scipy.ndimage import shift, rotate, affine_transform
from xmipp_base import *

SHIFT_MODE = 0
FULL_MODE = 1

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
        self.addParamsLine('[--maxEpochs <N=3000>]        : Max. number of epochs')
        self.addParamsLine('[--batchSize <N=8>]           : Batch size')
        self.addParamsLine('[--gpu <id=0>]                : GPU Id')
        self.addParamsLine('[--Nmodels <N=5>]             : Number of models')
        self.addParamsLine('[--learningRate <lr=0.0001>]  : Learning rate')
        self.addParamsLine('[--sym <s=c1>]                : Symmetry')
        self.addParamsLine('[--modelSize <s=0>]           : Model size (0=277k, 1=1M, 2=5M, 3=19M parameters)')
        self.addParamsLine('[--precision <s=0.07>]        : Alignment precision measured in percentage of '
                           'the image size')

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
        precision = float(self.getParam("--precision"))

        from keras.callbacks import TensorBoard, ModelCheckpoint
        from keras.models import Model
        from keras.layers import Input, Conv2D, BatchNormalization, Dense, concatenate, \
            Activation, GlobalAveragePooling2D, Add
        import keras
        from keras.models import load_model
        import tensorflow as tf
        from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

        if not gpuId.startswith('-1'):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpuId
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        checkIf_tf_keras_installed()

        class DataGenerator(keras.utils.all_utils.Sequence):
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
                    angles = np.reshape(mat[0:2,:],(6))
                    return np.concatenate((angles, shifts))

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

            def __len__(self):
                """Denotes the number of batches per epoch"""
                num_batches = int(np.floor((len(self.indexes)) / self.batch_size))
                return num_batches

            def __getitem__(self, index):
                """Generate one batch of data"""
                # Generate indexes of the batch
                indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

                # Generate data
                X_batch = self.Xsim[indexes]
                y_batch = self.ysim[indexes]

                return X_batch, y_batch

        def conv_block(tensor, filters):
            # Convolutional block of RESNET
            x = Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(tensor)
            x = BatchNormalization(axis=3)(x)
            x = Activation('relu')(x)
            x = Conv2D(filters, (3, 3), padding='same')(x)
            x = BatchNormalization(axis=3)(x)
            x_res = Conv2D(filters, (1, 1), strides=(2, 2))(tensor)
            x = Add()([x, x_res])
            x = Activation('relu')(x)
            return x

        def constructModel(Xdim, modelSize):
            """RESNET architecture"""
            inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")
            x = conv_block(inputLayer, filters=64)
            x = conv_block(x, filters=128)
            if modelSize>=1:
                x = conv_block(x, filters=256)
            if modelSize>=2:
                x = conv_block(x, filters=512)
            if modelSize>=3:
                x = conv_block(x, filters=1024)
            x = GlobalAveragePooling2D()(x)
            x = Dense(64)(x)
            x = Dense(8, name="output", activation="linear")(x)
            return Model(inputLayer, x)

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

        def custom_lossFull(y_true, y_pred):
            # y_6d = tf.matmul(y_pred, tfAt)
            y_6d = y_pred[:, :6]
            y_6dtrue = y_true[:, :6]
            # Take care of symmetry
            num_rows = tf.shape(y_6d)[0]
            min_errors = tf.fill([num_rows], float('inf'))
            rotated_versions = tf.TensorArray(dtype=tf.float32, size=num_rows)
            for i, symmetry_matrix in enumerate(listSymmetryMatrices):
                transformed = tf.matmul(y_6d, symmetry_matrix)
                errors = tf.reduce_mean(tf.abs(y_6dtrue - transformed), axis=1)

                # Update minimum errors and rotated versions
                for j in tf.range(num_rows):
                    if errors[j] < min_errors[j]:
                        min_errors = tf.tensor_scatter_nd_update(min_errors, [[j]], [errors[j]])
                        rotated_versions = rotated_versions.write(j, transformed[j])
            y_6d = rotated_versions.stack()

            e1_true = y_6dtrue[:, :3]  # First 3 components
            e2_true = y_6dtrue[:, 3:]  # Last 3 components
            e3_true = tf.linalg.cross(e1_true, e2_true)
            shift_true = y_true[:, 6:]  # Last 2 components

            e1_pred = y_6d[:, :3]  # First 3 components
            e2_pred = y_6d[:, 3:]  # Last 3 components
            shift_pred = y_pred[:, 6:]  # Last 2 components

            # Gram-Schmidt orthogonalization
            #        e1_pred = tf.clip_by_value(e1_pred, -1.0, 1.0)
            e1_pred = tf.nn.l2_normalize(e1_pred, axis=-1)  # Normalize e1
            projection = tf.reduce_sum(e2_pred * e1_pred, axis=-1, keepdims=True)
            e2_pred = e2_pred - projection * e1_pred
            #        e2_pred = tf.clip_by_value(e2_pred, -1.0, 1.0)
            e2_pred = tf.nn.l2_normalize(e2_pred, axis=-1)
            e3_pred = tf.linalg.cross(e1_pred, e2_pred)
            #        e3_pred = tf.clip_by_value(e3_pred, -1.0, 1.0)
            e3_pred = tf.nn.l2_normalize(e3_pred, axis=-1)

            epsilon = 1e-7
            angle1 = tf.acos(tf.clip_by_value(tf.reduce_sum(e1_true * e1_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))
            angle2 = tf.acos(tf.clip_by_value(tf.reduce_sum(e2_true * e2_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))
            angle3 = tf.acos(tf.clip_by_value(tf.reduce_sum(e3_true * e3_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))

            angular_error = tf.reduce_mean(tf.abs(angle1) + tf.abs(angle2) + tf.abs(angle3)) / 3.0
            shift_error = tf.reduce_mean(tf.abs(shift_true - shift_pred))
            return angular_error + shift_error/Xdim

        def custom_lossShift(y_true, y_pred):
            shift_true = y_true[:, 6:]  # Last 2 components
            shift_pred = y_pred[:, 6:]  # Last 2 components
            shift_error = tf.reduce_mean(tf.abs(shift_true - shift_pred))
            return shift_error/Xdim

        SL = xmippLib.SymList()
        listSymmetryMatrices = [tf.convert_to_tensor(np.kron(np.eye(2),np.transpose(np.array(R))), dtype=tf.float32)
                                for R in SL.getSymmetryMatrices(symmetry)]
        Xdim, fnImgsSim, angles, shifts = get_labels(fnXmdSim)
        if fnXmdExp!="":
            _, fnImgsExp, _, _ = get_labels(fnXmdExp)
        else:
            fnImgsExp=[]
        training_generator = DataGenerator(fnImgsSim, fnImgsExp, angles, shifts, batch_size, Xdim)
        for mode in range(1):
            if mode==SHIFT_MODE:
                modeprec=0.25*precision # 0.25 because the shift are 2 out of 8 numbers in the output vector
            else:
                modeprec=precision
            for index in range(numModels):
                fnModelIndex =  fnModel + str(index) + ".h5"
                save_best_model = ModelCheckpoint(fnModelIndex, monitor='loss', save_best_only=True)
                if os.path.exists(fnModel):
                    model = load_model(fnModelIndex)
                else:
                    model = constructModel(Xdim, modelSize)
                adam_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                model.summary()
                if mode==SHIFT_MODE:
                    model.compile(loss=custom_lossShift, optimizer=adam_opt)
                else:
                    model.compile(loss=custom_lossFull, optimizer=adam_opt)

                epoch=0
                for i in range(maxEpochs//2):
                    print("Iteration %d, SubIteration %d (%d images)"%(epoch,i,len(training_generator.indexes)))
                    history = model.fit(training_generator, epochs=2, callbacks=[save_best_model])
                    epoch+=1
                    loss=history.history['loss'][-1]
                    if loss<modeprec:
                        break

if __name__ == '__main__':
    ScriptDeepGlobalAssignment().tryRun()