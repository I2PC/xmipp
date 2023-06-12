#!/usr/bin/env python3

import math
import numpy as np
from operator import itemgetter
import os
import sys
import xmippLib
from time import time
from scipy.ndimage import shift, rotate
import matplotlib.pyplot as plt

if __name__ == "__main__":

    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    fnModel = sys.argv[2]
    mode = sys.argv[3]
    sigma = float(sys.argv[4])
    numEpochs = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    gpuId = sys.argv[7]
    numModels = int(sys.argv[8])
    if mode == 'Angular':
        learning_rate = float(sys.argv[9])
        pretrained = sys.argv[10]
        if pretrained == 'yes':
            fnPreModel = sys.argv[11]

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    from keras.callbacks import TensorBoard, ModelCheckpoint
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, concatenate, \
        Activation, Subtract, SeparableConv2D, GlobalAveragePooling2D, AveragePooling2D, LeakyReLU, Add
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
                    rX = rX * 3
                    rY = rY * 3
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

                    y = np.array(list((map(euler_to_rotation6d, yvalues, rAngle))))
            else:
                Xexp = np.array(list(Iexp))
                if mode == 'Shift':
                    y = yvalues
                else:
                    y = np.array(list((map(euler_to_rotation6d, yvalues, np.zeros(self.batch_size)))))
            return Xexp, y





    def conv_block(x, filters, kernel_size, strides=(1, 1), activation='relu', batch_normalization=True):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x


    def identity_block(tensor, filters, kernel_size, activation='relu', batch_normalization=True):
        x = conv_block(tensor, filters, kernel_size, activation=activation, batch_normalization=batch_normalization)
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        x = Add()([tensor, x])
        x = Activation(activation)(x)
        return x


    def resnet(Xdim, mode):
        inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")

        x = conv_block(inputLayer, filters=64, kernel_size=(7, 7), strides=(2, 2))
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        for _ in range(3):
            x = identity_block(x, filters=64, kernel_size=(3, 3))

        x = conv_block(x, filters=128, kernel_size=(3, 3), strides=(2, 2))

        for _ in range(3):
            x = identity_block(x, filters=128, kernel_size=(3, 3))

        x = conv_block(x, filters=256, kernel_size=(3, 3), strides=(2, 2))

        for _ in range(5):
            x = identity_block(x, filters=256, kernel_size=(3, 3))

        x = conv_block(x, filters=512, kernel_size=(3, 3), strides=(2, 2))

        for _ in range(2):
            x = identity_block(x, filters=512, kernel_size=(3, 3))

        x = GlobalAveragePooling2D()(x)
        if mode == 'Shift':
            x = Dense(2, name="output", activation="linear")(x)
        else:
            x = Dense(6, name="output", activation="linear")(x)

        model = Model(inputs=inputLayer, outputs=x)
        return model





    def constructModel(Xdim, mode):
        inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")

        # Network model
        L = Conv2D(32, (3, 3), padding='same')(inputLayer)
        L = BatchNormalization()(L)
        L = Activation(activation='relu')(L)
        L = MaxPooling2D()(L)
        L = Dropout(0.1)(L)

        L = Conv2D(64, (3, 3), padding='same')(L)
        L = BatchNormalization()(L)
        L = Activation(activation='relu')(L)
        L = MaxPooling2D()(L)
        L = Dropout(0.1)(L)

        L = Conv2D(128, (3, 3), padding='same')(L)
        L = BatchNormalization()(L)
        L = Activation(activation='relu')(L)
        L = MaxPooling2D()(L)
        L = Dropout(0.1)(L)

        L = Conv2D(256, (3, 3), padding='same')(L)
        L = BatchNormalization()(L)
        L = Activation(activation='relu')(L)
        L = MaxPooling2D()(L)
        L = Dropout(0.1)(L)

        L = Conv2D(512, (3, 3), padding='same')(L)
        L = BatchNormalization()(L)
        L = Activation(activation='relu')(L)
        L = MaxPooling2D()(L)
        L = Dropout(0.1)(L)

        L = Flatten()(L)
        # L = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(L)
        L = Dense(1024, activation='relu')(L)
        L = BatchNormalization()(L)
        L = Dropout(0.2)(L)

        L = Dense(1024, activation='relu')(L)
        L = BatchNormalization()(L)
        L = Dropout(0.2)(L)

        L = Dense(1024, activation='relu')(L)
        L = BatchNormalization()(L)

        if mode == 'Shift':
            L = Dense(2, name="output", activation="linear")(L)
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

        labels = []

        numTiltDivs = 5
        numRotDivs = 10
        limits_rot = np.linspace(-180.01, 180, num=(numRotDivs+1))
        limits_tilt = np.zeros(numTiltDivs+1)
        limits_tilt[0] = -0.01
        for i in range(1, numTiltDivs+1):
            limits_tilt[i] = math.acos(1-2*(i/numTiltDivs))
        limits_tilt = limits_tilt*180/math.pi
        zone = [[] for _ in range((len(limits_tilt)-1)*(len(limits_rot)-1))]

        i = 0
        if mode == "Shift":
            for x, y in zip(shiftX, shiftY):
                labels.append(np.array((x, y)))
        elif mode == "Angular":
            for r, t, p in zip(rots, tilts, psis):
                labels.append(np.array((r, t, p)))
                region_rot = np.digitize(r, limits_rot, right=True) - 1  # Índice de la región para el componente x
                region_tilt = np.digitize(t, limits_tilt, right=True) - 1  # Índice de la región para el componente y
                region_idx = region_rot * (len(limits_tilt)-1) + region_tilt  # Índice de la región combinada
                zone[region_idx].append(i)
                i += 1

        for i in range((len(limits_tilt)-1)*(len(limits_rot)-1)):
            print(len(zone[i]))

        return Xdim, fnImgs, labels, zone


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

    def geodesic_distance2(y_true, y_pred):
        a1 = y_pred[:, slice(0, 6, 2)]
        b1 = y_pred[:, slice(1, 6, 2)]
        a2 = tf.linalg.normalize(a1, axis=1)[0]
        c1 = tf.multiply(a1, b1)
        c2 = tf.reduce_sum(c1, axis=1, keepdims=True)
        b2 = b1 - tf.multiply(c2, a2)
        b2 = tf.linalg.normalize(b2, axis=1)[0]
        d = y_true[:, 0] * a2[:, 0]
        d += y_true[:, 2] * a2[:, 1]
        d += y_true[:, 4] * a2[:, 2]
        d += y_true[:, 1] * b2[:, 0]
        d += y_true[:, 3] * b2[:, 1]
        d += y_true[:, 5] * b2[:, 2]
        return -d
    def geodesic_distance(y_true, y_pred):
        a1 = y_pred[:, slice(0, 6, 2)]
        a2 = y_pred[:, slice(1, 6, 2)]
        a1 = tf.linalg.normalize(a1, axis=1)[0]
        a2 = tf.linalg.normalize(a2, axis=1)[0]
        d = y_true[:, 0] * a1[:, 0]
        d += y_true[:, 2] * a1[:, 1]
        d += y_true[:, 4] * a1[:, 2]
        d += y_true[:, 1] * a2[:, 0]
        d += y_true[:, 3] * a2[:, 1]
        d += y_true[:, 5] * a2[:, 2]
        return -d


    def geodesic_loss(y_true, y_pred):
        d = K.mean(geodesic_distance(y_true, y_pred))
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
        cos_tilt = angles[:, slice(2, 3)]
        tilt = tf.math.acos(cos_tilt)
        sin = tf.math.sin(tilt)
        ntilt = tf.concat([sin, cos_tilt], axis=1)
        psi = angles[:, slice(3, 5)]
        nrot = tf.linalg.normalize(rot, axis=1)[0]
        npsi = tf.linalg.normalize(psi, axis=1)[0]
        return tf.matmul(R_psi(nrot[:, slice(0, 1)], nrot[:, slice(1, 2)]),
                         tf.matmul(R_tilt(ntilt[:, slice(0, 1)], ntilt[:, slice(1, 2)]),
                                   R_psi(npsi[:, slice(0, 1)], npsi[:, slice(1, 2)])))



    Xdim, fnImgs, labels, zone = get_labels(fnXmdExp, mode)
    start_time = time()

    if numModels == 1:
        lenTrain = int(len(fnImgs)*0.8)
        lenVal = len(fnImgs)-lenTrain
    else:
        lenTrain = int(len(fnImgs) / 3)
        lenVal = int(len(fnImgs) / 12)


    elements_zone = int((lenVal+lenTrain)/len(zone))
    print('elements_zone', elements_zone, flush=True)

    for index in range(numModels):

        random_sample = np.random.choice(range(0, len(fnImgs)), size=lenTrain+lenVal, replace=False)
        if mode == 'Angular':
            for i in range(len(zone)):
                random_sample[i*elements_zone:(i+1)*elements_zone] = np.random.choice(zone[i], size=elements_zone, replace=True)
            np.random.shuffle(random_sample)
        training_generator = DataGenerator([fnImgs[i] for i in random_sample[0:lenTrain]],
                                           [labels[i] for i in random_sample[0:lenTrain]],
                                           mode, sigma, batch_size, Xdim, readInMemory=False)
        validation_generator = DataGenerator([fnImgs[i] for i in random_sample[lenTrain:lenTrain+lenVal]],
                                             [labels[i] for i in random_sample[lenTrain:lenTrain+lenVal]],
                                             mode, sigma, batch_size, Xdim, readInMemory=False)

        if mode == 'Shift':
            model = resnet(Xdim, mode)
        else:
            if pretrained == 'yes':
                model = load_model(fnPreModel, compile=False)
            else:
                model = resnet(Xdim, mode)
            adam_opt = Adam(lr=learning_rate)
        model.summary()


        if mode == 'Shift':
            model.compile(loss='mean_absolute_error', optimizer='adam')
        else:
            model.compile(loss=geodesic_loss, optimizer=adam_opt)
        save_best_model = ModelCheckpoint(fnModel + str(index) + ".h5", monitor='val_loss',
                                          save_best_only=True)

        history = model.fit_generator(generator=training_generator, epochs=numEpochs,
                                      validation_data=validation_generator, callbacks=[save_best_model])
        #if mode != 'Shift':
        #    plt.plot(history.history['loss'])
        #    plt.plot(history.history['val_loss'])
        #    plt.title('model loss')
        #    plt.ylabel('loss')
        #    plt.xlabel('epoch')
        #    plt.legend(['train', 'test'], loc='upper left')
        #    plt.show()

    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)
