#!/usr/bin/env python3

import math
import numpy as np
from operator import itemgetter
import os
import sys
import xmippLib
from time import time
from scipy.ndimage import shift, rotate, affine_transform

if __name__ == "__main__":

    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed
    from xmippPyModules.deepGlobalAssignment import Redundancy

    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    fnModel = sys.argv[2]
    sigma = float(sys.argv[3])
    numEpochs = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    gpuId = sys.argv[6]
    numModels = int(sys.argv[7])
    learning_rate = float(sys.argv[8])
    patience = int(sys.argv[9])
    pretrained = sys.argv[10]
    symmetry = sys.argv[11]
    if pretrained == 'yes':
        fnPreModel = sys.argv[12]

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    from keras.callbacks import TensorBoard, ModelCheckpoint
    from keras.models import Model
    from keras.layers import Input, Conv2D, BatchNormalization, Dense, concatenate, \
        Activation, GlobalAveragePooling2D, Add
    import keras
    from keras.models import load_model
    import tensorflow as tf

    class DataGenerator(keras.utils.all_utils.Sequence):
        """Generates data for fnImgs"""

        def __init__(self, fnImgs, labels, sigma, batch_size, dim, shifts, readInMemory):
            """Initialization"""
            self.fnImgs = fnImgs
            self.labels = labels
            self.sigma = sigma
            self.batch_size = batch_size
            if self.batch_size > len(self.fnImgs):
                self.batch_size = len(self.fnImgs)
            self.dim = dim
            self.readInMemory = readInMemory
            self.on_epoch_end()
            self.shifts = shifts

            # Read all data in memory
            if self.readInMemory:
                self.Xexp = np.zeros((len(self.labels), self.dim, self.dim, 1), dtype=np.float64)
                for i in range(len(self.labels)):
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[i]).getData(), (self.dim, self.dim, 1))
                    self.Xexp[i,] = (Iexp - np.mean(Iexp)) / np.std(Iexp)

        def __len__(self):
            """Denotes the number of batches per epoch"""
            num_batches = int(np.floor((len(self.labels)) / self.batch_size))
            return num_batches

        def __getitem__(self, index):
            """Generate one batch of data"""
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            # Find list of IDs
            list_IDs_temp = []
            for i in range(int(self.batch_size)):
                list_IDs_temp.append(indexes[i])
            # Generate data
            Xexp, y = self.__data_generation(list_IDs_temp)

            return Xexp, y

        def on_epoch_end(self):
            """Updates indexes after each epoch"""
            self.indexes = [i for i in range(len(self.labels))]
            np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):
            """Generates data containing batch_size samples"""
            yvalues = np.array(itemgetter(*list_IDs_temp)(self.labels))
            yshifts = np.array(itemgetter(*list_IDs_temp)(self.shifts))

            # Functions to handle the data
            def get_image(fn_image):
                img = np.reshape(xmippLib.Image(fn_image).getData(), (self.dim, self.dim, 1))
                return (img - np.mean(img)) / np.std(img)

            def euler_to_rotation6d(angles, psi_rotation):
                mat =  xmippLib.Euler_angles2matrix(angles[0],angles[1],angles[2] + psi_rotation)
                return np.reshape(mat[0:2,:],(6))

            if self.readInMemory:
                Iexp = list(itemgetter(*list_IDs_temp)(self.Xexp))
            else:
                fnIexp = list(itemgetter(*list_IDs_temp)(self.fnImgs))
                Iexp = list(map(get_image, fnIexp))

            def shift_then_rotate_image(img, shiftx, shifty, angle):
                imgShifted=shift(img, (shiftx, shifty, 0), order=1, mode='wrap')
                imgRotated=rotate(imgShifted, angle, order=1, mode='reflect', reshape=False)
                return imgRotated

            rX = self.sigma * np.random.uniform(-1, 1, size=self.batch_size)
            rY = self.sigma * np.random.uniform(-1, 1, size=self.batch_size)
            rAngle = 180 * np.random.uniform(-1, 1, size=self.batch_size)
            Xexp = np.array(list((map(shift_then_rotate_image, Iexp, rX-yshifts[:,0], rY-yshifts[:,1], rAngle))))
            # y_6d = np.array(list((map(euler_to_rotation6d, yvalues, rAngle))))
            # y = np.array(list((map(Redundancy().make_redundant, y_6d))))
            y = np.array(list((map(euler_to_rotation6d, yvalues, rAngle))))

            return Xexp, y

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

    def constructModel(Xdim):
        """RESNET architecture"""
        inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")
        x = conv_block(inputLayer, filters=64)
        x = conv_block(x, filters=128)
        x = conv_block(x, filters=256)
        x = conv_block(x, filters=512)
        x = conv_block(x, filters=1024)
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, name="output", activation="linear")(x)
        return Model(inputLayer, x)

    def get_labels(fnImages):
        """Returns dimensions, images, angles and shifts values from images files"""
        Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnImages)
        mdExp = xmippLib.MetaData(fnImages)
        fnImg = mdExp.getColumnValues(xmippLib.MDL_IMAGE)
        shiftX = mdExp.getColumnValues(xmippLib.MDL_SHIFT_X)
        shiftY = mdExp.getColumnValues(xmippLib.MDL_SHIFT_Y)
        rots = mdExp.getColumnValues(xmippLib.MDL_ANGLE_ROT)
        tilts = mdExp.getColumnValues(xmippLib.MDL_ANGLE_TILT)
        psis = mdExp.getColumnValues(xmippLib.MDL_ANGLE_PSI)

        label = []
        for r, t, p in zip(rots, tilts, psis):
            label.append(np.array((r,t,p)))
        img_shift = [np.array((sX,sY)) for sX, sY in zip(shiftX, shiftY)]

        return Xdim, fnImg, label, img_shift

    tfAt = tf.cast(tf.transpose(Redundancy().Apinv),tf.float32)
    SL = xmippLib.SymList()
    listSymmetryMatrices = [tf.convert_to_tensor(np.kron(np.eye(2),np.transpose(np.array(R))), dtype=tf.float32)
                            for R in SL.getSymmetryMatrices(symmetry)]
    def custom_lossAngles(y_true, y_pred):
        y_6d = tf.matmul(y_pred, tfAt)

        e1_true = y_true[:, :3] # First 3 components
        e2_true = y_true[:, 3:] # Last 3 components

        e1_pred = y_6d[:, :3] # First 3 components
        e2_pred = y_6d[:, 3:] # Last 3 components

        # Gram-Schmidt orthogonalization
        e1_pred = tf.clip_by_value(e1_pred, -1.0, 1.0)
        e1_pred = tf.nn.l2_normalize(e1_pred, axis=-1)  # Normalize e1
        projection = tf.reduce_sum(e2_pred * e1_pred, axis=-1, keepdims=True)
        e2_pred = e2_pred - projection * e1_pred
        e2_pred = tf.clip_by_value(e2_pred, -1.0, 1.0)
        e2_pred = tf.nn.l2_normalize(e2_pred, axis=-1)

        epsilon = 1e-7
        angle1 = tf.acos(tf.clip_by_value(tf.reduce_sum(e1_true * e1_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))
        angle2 = tf.acos(tf.clip_by_value(tf.reduce_sum(e2_true * e2_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))

        return tf.reduce_mean(tf.abs(angle1)+tf.abs(angle2))*0.5*(180.0 / np.pi)

    def custom_lossVectors(y_true, y_pred):
        y_6d = tf.matmul(y_pred, tfAt)
        # e=tf.reduce_mean(tf.abs(y_true-y_6d),axis=1)
        # return tf.reduce_mean(e)
        esym = tf.stack([tf.reduce_mean(tf.abs(y_true-tf.matmul(y_6d, tensor)),axis=1)
                          for tensor in listSymmetryMatrices],axis=1)
        return tf.reduce_mean(tf.reduce_min(esym, axis=1))

    Xdims, fnImgs, labels, shifts = get_labels(fnXmdExp)

    # Train-Validation sets
    lenTrain = int(len(fnImgs)*0.8)
    lenVal = len(fnImgs)-lenTrain

    for index in range(numModels):
        # chooses equal number of particles for each division
        random_sample = np.random.choice(range(0, len(fnImgs)), size=lenTrain+lenVal, replace=False)

        training_generator = DataGenerator([fnImgs[i] for i in random_sample[0:lenTrain]],
                                           [labels[i] for i in random_sample[0:lenTrain]],
                                           sigma, batch_size, Xdims, shifts, readInMemory=True)
        validation_generator = DataGenerator([fnImgs[i] for i in random_sample[lenTrain:lenTrain+lenVal]],
                                             [labels[i] for i in random_sample[lenTrain:lenTrain+lenVal]],
                                             sigma, batch_size, Xdims, shifts, readInMemory=True)

        if pretrained == 'yes':
            model = load_model(fnPreModel, compile=False)
        else:
            model = constructModel(Xdims)

        adam_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        save_best_model = ModelCheckpoint(fnModel + str(index) + ".h5", monitor='val_loss', save_best_only=True)
        patienceCallBack = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

        model.summary()
        model.compile(loss=custom_lossVectors, optimizer=adam_opt)
        history = model.fit_generator(generator=training_generator, epochs=numEpochs,
                                      validation_data=validation_generator, callbacks=[save_best_model,
                                                                                       patienceCallBack])
        # model.compile(loss=custom_lossAngles, optimizer=adam_opt)
        # history = model.fit_generator(generator=training_generator, epochs=numEpochs,
        #                               validation_data=validation_generator, callbacks=[save_best_model,
        #                                                                                patienceCallBack])
