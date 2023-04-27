#!/usr/bin/env python3

import math
import numpy as np
from operator import itemgetter
import os
import sys
import xmippLib
from time import time
from scipy.ndimage import shift, rotate

if __name__ == "__main__":

    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed
    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    fnModel = sys.argv[2]
    print("------------------", flush=True)
    mode = sys.argv[3]
    sigma = float(sys.argv[4])
    numEpochs = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    gpuId = sys.argv[7]
    if mode == 'Angular':
        representation = sys.argv[8]
        loss_function = sys.argv[9]

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    from keras.callbacks import TensorBoard, ModelCheckpoint
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, concatenate, \
        Subtract, SeparableConv2D, GlobalAveragePooling2D, AveragePooling2D
    from keras.optimizers import *
    import keras
    from keras import callbacks
    from keras.callbacks import Callback
    from keras import regularizers
    from keras.models import load_model
    import tensorflow as tf


    class DataGenerator(keras.utils.Sequence):
        'Generates data for fnImgs'

        def __init__(self, fnImgs, labels, mode, sigma, batch_size, dim, readInMemory):
            'Initialization'
            self.fnImgs = fnImgs
            self.labels = labels
            self.mode = mode
            self.sigma = sigma
            self.batch_size = batch_size
            if self.batch_size > len(self.fnImgs):
                self.batch_size = len(self.fnImgs)
            self.dim = dim
            self.readInMemory = readInMemory
            self.on_epoch_end()

            # Read all data in memory
            if self.readInMemory:
                self.Xexp = np.zeros((len(self.labels), self.dim, self.dim, 1), dtype=np.float64)
                for i in range(len(self.labels)):
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[i]).getData(), (self.dim, self.dim, 1))
                    self.Xexp[i,] = (Iexp - np.mean(Iexp)) / np.std(Iexp)

        def __len__(self):
            'Denotes the number of batches per epoch'
            num_batches = int(np.floor((len(self.labels)) / self.batch_size))
            return num_batches

        def __getitem__(self, index):
            'Generate one batch of data'
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
            'Updates indexes after each epoch'
            self.indexes = [i for i in range(len(self.labels))]
            np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):
            'Generates data containing batch_size samples'
            yvalues = np.array(itemgetter(*list_IDs_temp)(self.labels))

            # Functions to handle the data
            def get_image(fn_image):
                img = np.reshape(xmippLib.Image(fn_image).getData(), (self.dim, self.dim, 1))
                return (img - np.mean(img)) / np.std(img)

            def rotate_image(img, angle):
                # angle in degrees
                return rotate(img, angle, order=1, mode='reflect', reshape=False)

            def shift_image(img, shiftx, shifty):
                return shift(img, (shiftx, shifty, 0), order=1, mode='reflect')

            def get_angles_radians(angles, angle):
                return np.array((math.sin(angles[0]), math.cos(angles[0]), math.sin(angles[1]),
                                 math.cos(angles[1]), math.sin(angles[2] + angle),
                                 math.cos(angles[2] + angle)))

            def R_rot(theta):
                return np.matrix([[1, 0, 0],
                                  [0, math.cos(theta), -math.sin(theta)],
                                  [0, math.sin(theta), math.cos(theta)]])

            def R_tilt(theta):
                return np.matrix([[math.cos(theta), 0, math.sin(theta)],
                                  [0, 1, 0],
                                  [-math.sin(theta), 0, math.cos(theta)]])

            def R_psi(theta):
                return np.matrix([[math.cos(theta), -math.sin(theta), 0],
                                  [math.sin(theta), math.cos(theta), 0],
                                  [0, 0, 1]])

            def euler_angles_to_matrix(angles, psi_rotation):
                Rx = R_rot(angles[0])
                Ry = R_tilt(angles[1] - math.pi / 2)
                Rz = R_psi(angles[2] + psi_rotation)
                return Rz * Ry * Rx

            def matrix_to_rotation6d(mat):
                r6d = np.delete(mat, -1, axis=1)
                return np.array((r6d[0, 0], r6d[0, 1], r6d[1, 0], r6d[1, 1], r6d[2, 0], r6d[2, 1]))

            def euler_to_rotation6d(angles, psi_rotation):
                mat = euler_angles_to_matrix(angles, psi_rotation)
                return matrix_to_rotation6d(mat)

            def euler_to_cartesian(angles, psi_rotation):
                x = math.sin(angles[1])*math.cos(angles[0])
                y = math.sin(angles[1])*math.sin(angles[0])
                z = math.cos(angles[1])
                sinpsi = math.sin(angles[2] + psi_rotation)
                cospsi = math.cos(angles[2] + psi_rotation)
                return np.array((x, y, z, sinpsi, cospsi))

            if self.readInMemory:
                Iexp = list(itemgetter(*list_IDs_temp)(self.Xexp))
            else:
                fnIexp = list(itemgetter(*list_IDs_temp)(self.fnImgs))
                Iexp = list(map(get_image, fnIexp))
            # Data augmentation
            if self.sigma > 0:
                rX = (self.sigma / 5) * np.random.normal(0, 1, size=self.batch_size)
                rY = (self.sigma / 5) * np.random.normal(0, 1, size=self.batch_size)
                rX = rX + self.sigma * np.random.uniform(-1, 1, size=self.batch_size)
                rY = rY + self.sigma * np.random.uniform(-1, 1, size=self.batch_size)
                if mode == 'Shift':
                    # Shift image a random amount of px in each direction
                    Xexp = np.array(list((map(shift_image, Iexp, rX, rY))))
                    y = yvalues + np.vstack((rX, rY)).T
                else:
                    # Shift image a random amount of px in each direction
                    Xexp = np.array(list((map(shift_image, Iexp, rX, rY))))

                    # Rotates image a random angle. Thus, Psi must be updated
                    rAngle = 180 * np.random.uniform(-1, 1, size=self.batch_size)

                    Xexp = np.array(list(map(rotate_image, Xexp, rAngle)))
                    rAngle = rAngle * math.pi / 180
                    yvalues = yvalues * math.pi / 180

                    if representation == 'euler':
                        y = np.array(list((map(get_angles_radians, yvalues, rAngle))))
                    elif representation == 'cartesian':
                        y = np.array(list((map(euler_to_cartesian, yvalues, rAngle))))
                    else:
                        y = np.array(list((map(euler_to_rotation6d, yvalues, rAngle))))
            else:
                Xexp = np.array(list(Iexp))
                if mode == 'Shift':
                    y = yvalues
                else:
                    y = np.array(list((map(get_angles_radians, yvalues, np.zeros(self.batch_size)))))
            return Xexp, y


    def constructModel(Xdim, mode):
        inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")

        # Network model

        L = Conv2D(32, (3, 3), activation="relu")(inputLayer)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Dropout(0.2)(L)

        L = Conv2D(64, (3, 3), activation="relu")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Dropout(0.2)(L)

        L = Conv2D(128, (3, 3), activation="relu")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Dropout(0.2)(L)

        L = Conv2D(256, (3, 3), activation="relu")(L)
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Dropout(0.2)(L)

        L = Flatten()(L)
        L = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(L)
        L = BatchNormalization()(L)

        L = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(L)
        L = BatchNormalization()(L)
        L = Dropout(0.2)(L)

        L = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(L)
        L = BatchNormalization()(L)
        L = Dropout(0.2)(L)

        if mode == 'Shift':
            L = Dense(2, name="output", activation="linear")(L)
        else:
            if representation == 'cartesian':
                L = Dense(5, name="output", activation="linear")(L)
            else:
                L = Dense(6, name="output", activation="linear")(L)

        return Model(inputLayer, L)

    def get_labels(fnImages, mode):
        Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnImages)
        mdExp = xmippLib.MetaData(fnImages)
        fnImgs = mdExp.getColumnValues(xmippLib.MDL_IMAGE)
        shiftX = mdExp.getColumnValues(xmippLib.MDL_SHIFT_X)
        shiftY = mdExp.getColumnValues(xmippLib.MDL_SHIFT_Y)
        rots = mdExp.getColumnValues(xmippLib.MDL_ANGLE_ROT)
        tilts = mdExp.getColumnValues(xmippLib.MDL_ANGLE_TILT)
        psis = mdExp.getColumnValues(xmippLib.MDL_ANGLE_PSI)

        labels_val = []

        if mode == "Shift":
            for x, y in zip(shiftX, shiftY):
                labels_val.append(np.array((x, y)))
        elif mode == "Angular":
            for r, t, p in zip(rots, tilts, psis):
                labels_val.append(np.array((r, t, p)))

        ntraining = math.floor(len(fnImgs) * 0.8)
        ind = [i for i in range(len(labels_val))]
        np.random.shuffle(ind)
        train = ind[0:ntraining]
        valid = ind[(ntraining + 1):]

        return Xdim, fnImgs, labels_val, train, valid

    Xdim, fnImgs, labels, indTrain, indVal = get_labels(fnXmdExp, mode)

    training_generator = DataGenerator([fnImgs[i] for i in indTrain], [labels[i] for i in indTrain], mode, sigma,
                                       batch_size, Xdim, readInMemory=False)
    validation_generator = DataGenerator([fnImgs[i] for i in indVal], [labels[i] for i in indVal], mode, sigma,
                                         batch_size, Xdim, readInMemory=False)


    import tensorflow as tf
    import keras.backend as K
    def rotation6d_to_matrix(rot):
        a1 = rot[:, slice(0, 6, 2)]
        a2 = rot[:, slice(1, 6, 2)]
        a1 = K.reshape(a1, (-1, 3))
        a2 = K.reshape(a2, (-1, 3))
        b1 = tf.linalg.normalize(a1, axis=1)[0]
        b3 = K.tf.cross(b1, a2)
        b3 = tf.linalg.normalize(b3, axis=1)[0]
        b2 = K.tf.cross(b3, b1)
        b1 = K.expand_dims(b1, axis=2)
        b2 = K.expand_dims(b2, axis=2)
        b3 = K.expand_dims(b3, axis=2)
        return K.concatenate((b1, b2, b3), axis=2)

    def geodesic_distance(y_true, y_pred):
        mat_true = rotation6d_to_matrix(y_true)
        mat_pred = rotation6d_to_matrix(y_pred)
        R = tf.matmul(mat_true, mat_pred, transpose_b=True)
        val = (K.tf.linalg.trace(R)-1)/2
        val = tf.math.minimum(val, tf.constant([0.999]))
        val = tf.math.maximum(val, tf.constant([-0.999]))
        return K.mean(tf.math.acos(val), axis=-1)

    def geodesic_loss(y_true, y_pred):
        print('y_true', y_true)
        d = geodesic_distance(y_true, y_pred)
        return d


    def R_rot(x, y):
        # (x,y) must be a normalized
        # Construct the rotation matrix for each sample in the batch
        ones = tf.ones_like(x)
        zeros = tf.zeros_like(x)
        batch_size = tf.shape(x)[0]
        rot_matrix = tf.concat([
            tf.concat([ones, zeros, zeros], axis=1),
            tf.concat([zeros, y, -x], axis=1),
            tf.concat([zeros, x, y], axis=1)
        ], axis=1)
        return tf.reshape(rot_matrix, (batch_size, 3, 3))

    def R_tilt(x, y):
        # Construct the rotation matrix for each sample in the batch
        ones = tf.ones_like(x)
        zeros = tf.zeros_like(x)
        batch_size = tf.shape(x)[0]
        tilt_matrix = tf.concat([
            tf.concat([y, zeros, x], axis=1),
            tf.concat([zeros, ones, zeros], axis=1),
            tf.concat([-x, zeros, y], axis=1)
        ], axis=1)
        return tf.reshape(tilt_matrix, (batch_size, 3, 3))

    def R_psi(x, y):
        # Construct the rotation matrix for each sample in the batch
        ones = tf.ones_like(x)
        zeros = tf.zeros_like(x)
        batch_size = tf.shape(x)[0]
        psi_matrix = tf.concat([
            tf.concat([y, -x, zeros], axis=1),
            tf.concat([x, y, zeros], axis=1),
            tf.concat([zeros, zeros, ones], axis=1)
        ], axis=1)
        return tf.reshape(psi_matrix, (batch_size, 3, 3))

    def rotation_matrix(angles):
        rot = angles[:, slice(0, 2)]
        tilt = angles[:, slice(2, 4)]
        psi = angles[:, slice(4, 6)]
        nrot = tf.linalg.normalize(rot, axis=1)[0]
        ntilt = tf.linalg.normalize(tilt, axis=1)[0]
        npsi = tf.linalg.normalize(psi, axis=1)[0]
        return tf.matmul(R_psi(nrot[:, slice(0,1)], nrot[:, slice(1,2)]), tf.matmul(R_tilt(ntilt[:, slice(0,1)], ntilt[:, slice(1,2)]), R_psi(npsi[:, slice(0,1)], npsi[:, slice(1,2)])))


    def distanceBetweenMatrix(mat1, mat2):
        return True


    def custom_loss(y_true, y_pred):
        rotation_true = rotation_matrix(y_true)
        rotation_pred = rotation_matrix(y_pred)
        rotmul = tf.linalg.matmul(rotation_true, rotation_pred, transpose_b=True)
        d = -tf.linalg.trace(rotmul)
        return K.mean(d)

    start_time = time()
    model = constructModel(Xdim, mode)

    model.summary()
    adam_opt = Adam(lr=0.001)

    steps = round(len(fnImgs) / batch_size)
    if mode == 'Shift':
        model.compile(loss='mean_absolute_error', optimizer='adam')
    else:
        if loss_function == 'l1':
            model.compile(loss='mean_absolute_error', optimizer='adam')
        elif loss_function == 'l2':
            model.compile(loss='mean_squared_error', optimizer='adam')
        else:
            if representation == 'euler':
                model.compile(loss=custom_loss, optimizer='adam')
            else:
                model.compile(loss=geodesic_loss, optimizer='adam')

    history = model.fit_generator(generator=training_generator, steps_per_epoch=steps, epochs=numEpochs,
                                  validation_data=validation_generator)

    model.save(fnModel)
    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)




