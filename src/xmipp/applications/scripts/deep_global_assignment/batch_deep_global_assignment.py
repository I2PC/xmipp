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
    sigma = float(sys.argv[3])
    numEpochs = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    gpuId = sys.argv[6]
    numModels = int(sys.argv[7])
    learning_rate = float(sys.argv[8])
    patience = int(sys.argv[9])
    pretrained = sys.argv[10]
    if pretrained == 'yes':
        fnPreModel = sys.argv[11]
    #rotational_order = int(sys.argv[12])
    rotational_order = 4

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    from keras.callbacks import TensorBoard, ModelCheckpoint
    from keras.models import Model
    from keras.layers import Input, Conv2D, BatchNormalization, Dense, concatenate, \
        Activation, GlobalAveragePooling2D, Add
    from keras.optimizers import *
    import keras
    from keras.models import load_model
    import tensorflow as tf

    _AXES2TUPLE = {
        'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
        'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
        'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
        'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
        'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
        'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
        'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
        'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

    _TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())
    # axis sequences for Euler angles
    _NEXT_AXIS = [1, 2, 0, 1]

    _EPS = np.finfo(float).eps * 4.0


    def euler_matrix(ai, aj, ak, axes='sxyz'):
        try:
            firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
        except (AttributeError, KeyError):
            _TUPLE2AXES[axes]  # validation
            firstaxis, parity, repetition, frame = axes

        i = firstaxis
        j = _NEXT_AXIS[i + parity]
        k = _NEXT_AXIS[i - parity + 1]

        if frame:
            ai, ak = ak, ai
        if parity:
            ai, aj, ak = -ai, -aj, -ak

        si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
        ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
        cc, cs = ci * ck, ci * sk
        sc, ss = si * ck, si * sk

        M = np.identity(4)
        if repetition:
            M[i, i] = cj
            M[i, j] = sj * si
            M[i, k] = sj * ci
            M[j, i] = sj * sk
            M[j, j] = -cj * ss + cc
            M[j, k] = -cj * cs - sc
            M[k, i] = -sj * ck
            M[k, j] = cj * sc + cs
            M[k, k] = cj * cc - ss
        else:
            M[i, i] = cj * ck
            M[i, j] = sj * sc - cs
            M[i, k] = sj * cc + ss
            M[j, i] = cj * sk
            M[j, j] = sj * ss + cc
            M[j, k] = sj * cs - sc
            M[k, i] = -sj
            M[k, j] = cj * si
            M[k, k] = cj * ci
        return M


    def euler_to_matrix(angles):
        return euler_matrix(angles[0] * math.pi / 180, angles[1] * math.pi / 180, angles[2] * math.pi / 180,
                            axes='szyz')[:3, :3]


    def euler_from_matrix(matrix, axes='sxyz'):
        try:
            firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
        except (AttributeError, KeyError):
            _TUPLE2AXES[axes]  # validation
            firstaxis, parity, repetition, frame = axes

        i = firstaxis
        j = _NEXT_AXIS[i + parity]
        k = _NEXT_AXIS[i - parity + 1]

        M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
        if repetition:
            sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
            if sy > _EPS:
                ax = math.atan2(M[i, j], M[i, k])
                ay = math.atan2(sy, M[i, i])
                az = math.atan2(M[j, i], -M[k, i])
            else:
                ax = math.atan2(-M[j, k], M[j, j])
                ay = math.atan2(sy, M[i, i])
                az = 0.0
        else:
            cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
            if cy > _EPS:
                ax = math.atan2(M[k, j], M[k, k])
                ay = math.atan2(-M[k, i], cy)
                az = math.atan2(M[j, i], M[i, i])
            else:
                ax = math.atan2(-M[j, k], M[j, j])
                ay = math.atan2(-M[k, i], cy)
                az = 0.0

        if parity:
            ax, ay, az = -ax, -ay, -az
        if frame:
            ax, az = az, ax
        return ax, ay, az


    def matrix_to_euler(mat):
        return np.array(euler_from_matrix(mat, axes='szyz')) * 180 / math.pi


    SL = xmippLib.SymList()
    Matrices = np.array(SL.getSymmetryMatrices('o'))
    inverse_matrices = np.zeros_like(Matrices)
    for i in range(24):
        inverse_matrices[i] = np.transpose(Matrices[i])

    def move_to_fundamental_domain(input_matrix):
        candidates = [[378]]
        max_tilt = -999
        for sym_matrix in inverse_matrices:
            aux_mat = np.matmul(input_matrix, sym_matrix)
            aux_euler_angles = matrix_to_euler(aux_mat)
            aux_rot = aux_euler_angles[0]
            aux_tilt = aux_euler_angles[1]
            if (aux_rot >= 0) and (90 >= aux_rot):
                if aux_tilt >= max_tilt:
                    max_tilt = aux_tilt
                    candidates = aux_mat

        return candidates

    def transform_rot(input_matrix, order):
        euler = matrix_to_euler(input_matrix)
        euler[0] = order * euler[0]
        output_matrix = euler_to_matrix(euler)
        return output_matrix[:3, :3]

    def rodrigues_formula(axis, angle):
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * np.matmul(K, K)


    def factor_rot(rot_angle, tilt_angle):
        dist = 2.2300e-07 * tilt_angle**4 + 5.8474e-05 * tilt_angle**3 + 5.6207e-03 * tilt_angle**2 - 8.1743e-02 * tilt_angle
        return rot_angle-np.sign(rot_angle)*dist*((180-np.abs(rot_angle))/(90-dist))

    def perform_symmetry_transformation(euler_angles, psi_rotation):
        euler_angles[2] += psi_rotation
        y_matrix = euler_to_matrix(euler_angles)
        y_fundamental = move_to_fundamental_domain(y_matrix)

        rot_order = 4
        y_transformedRot1 = transform_rot(y_fundamental, rot_order)

        angle = math.pi / 4
        matrix_axis = rodrigues_formula([0, 1, 0], angle)

        y_new_axis = np.matmul(y_transformedRot1, matrix_axis)
        y_euler_new_axis = matrix_to_euler(y_new_axis)

        min_tilt = -99
        factor_tilt = -180 / min_tilt

        y_euler_new_axis[1] = factor_tilt * y_euler_new_axis[1]
        y_euler_new_axis[0] = factor_rot(y_euler_new_axis[0],  y_euler_new_axis[1])
        y_euler_new_axis[0] = 2 * (180 + y_euler_new_axis[0])

        return y_euler_new_axis


    class DataGenerator(keras.utils.Sequence):
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

            def shift_image(img, shiftx, shifty, yshift):
                return shift(img, (shiftx-yshift[0], shifty-yshift[1], 0), order=1, mode='wrap')

            def rotate_image(img, angle):
                # angle in degrees
                return rotate(img, angle, order=1, mode='reflect', reshape=False)

            def matrix_to_rotation6d(mat):
                r6d = np.delete(mat, -1, axis=1)
                return np.array((r6d[0, 0], r6d[0, 1], r6d[1, 0], r6d[1, 1], r6d[2, 0], r6d[2, 1]))

            def euler_to_rotation6d(angles):
                mat = euler_to_matrix(angles)
                return matrix_to_rotation6d(mat)

            if self.readInMemory:
                Iexp = list(itemgetter(*list_IDs_temp)(self.Xexp))
            else:
                fnIexp = list(itemgetter(*list_IDs_temp)(self.fnImgs))
                Iexp = list(map(get_image, fnIexp))

            rX = self.sigma * np.random.uniform(-1, 1, size=self.batch_size)
            rY = self.sigma * np.random.uniform(-1, 1, size=self.batch_size)
            # Shift image a random amount of px in each direction
            Xexp = np.array(list((map(shift_image, Iexp, rX, rY, yshifts))))
            # Rotates image a random angle. Psi must be updated
            rAngle = 180 * np.random.uniform(-1, 1, size=self.batch_size)
            Xexp = np.array(list(map(rotate_image, Xexp, rAngle)))

            y_euler = np.array(list(map(perform_symmetry_transformation, yvalues, rAngle)))

            y_6d = np.array(list((map(euler_to_rotation6d, y_euler))))

            return Xexp, y_6d

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

        x = Dense(6, name="output", activation="linear")(x)

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
        img_shift = []

        for r, t, p, sX, sY in zip(rots, tilts, psis, shiftX, shiftY):
            rotational_order = 1
            r = r*rotational_order
            r = r % 360
            label.append(np.array((r, t, p)))
            img_shift.append(np.array((sX, sY)))
            # Region index

        return Xdim, fnImg, label, img_shift

    Xdims, fnImgs, labels, shifts = get_labels(fnXmdExp)
    start_time = time()

    import keras.backend as K

    def geodesic_distance(y_true, y_pred):
        a1 = tf.linalg.normalize(y_pred[:, slice(0, 6, 2)], axis=-1)[0]
        a2 = tf.linalg.normalize(y_pred[:, slice(1, 6, 2)], axis=-1)[0]
        d = y_true[:, 0] * a1[:, 0]
        d += y_true[:, 2] * a1[:, 1]
        d += y_true[:, 4] * a1[:, 2]
        d += y_true[:, 1] * a2[:, 0]
        d += y_true[:, 3] * a2[:, 1]
        d += y_true[:, 5] * a2[:, 2]

        return -d


    def geodesic_loss(y_true, y_pred):
        d = geodesic_distance(y_true, y_pred)
        return K.mean(d)


    def rotation6d_to_matrix(rot):
        a1 = rot[:, slice(0, 6, 2)]
        a2 = rot[:, slice(1, 6, 2)]
        a1 = K.reshape(a1, (-1, 3))
        a2 = K.reshape(a2, (-1, 3))
        b1 = tf.linalg.normalize(a1, axis=1)[0]
        b3 = tf.linalg.cross(b1, a2)
        b3 = tf.linalg.normalize(b3, axis=1)[0]
        b2 = tf.linalg.cross(b3, b1)
        b1 = K.expand_dims(b1, axis=2)
        b2 = K.expand_dims(b2, axis=2)
        b3 = K.expand_dims(b3, axis=2)
        return K.concatenate((b1, b2, b3), axis=2)


    def eq_matrices(y_true, R):
        tfR = tf.constant(R)
        tfR = tf.transpose(tfR, perm=[1, 0, 2])
        # I could have a number and pass it...
        # Reshape the tensor to (3, ?)
        matrices = tf.reshape(tfR, (3, -1))

        eq_rotations = tf.matmul(y_true, matrices)

        shape = tf.shape(eq_rotations)
        eq_rotations = tf.reshape(eq_rotations, [shape[0], -1, shape[1]])
        eq_rotations6d = eq_rotations[:, :, :-1]
        return tf.reshape(eq_rotations6d, [shape[0], shape[1], -1])


    def compute_distances(y_pred, eq_y_true):
        a1 = y_pred[:, slice(0, 6, 2)]
        a2 = y_pred[:, slice(1, 6, 2)]
        a1 = tf.linalg.normalize(a1, axis=1)[0]
        a2 = tf.linalg.normalize(a2, axis=1)[0]
        a1 = tf.expand_dims(a1, axis=-1)
        a2 = tf.expand_dims(a2, axis=-1)

        mat_pred = tf.concat([a1, a2], axis=-1)

        odd_columns = eq_y_true[:, :, ::2]
        even_columns = eq_y_true[:, :, 1::2]
        odd_columns_expanded = tf.expand_dims(odd_columns, axis=0)
        even_columns_expanded = tf.expand_dims(even_columns, axis=0)

        # Perform dot multiplication between columns
        dot_product = tf.multiply(mat_pred[:, :, 0:1], odd_columns_expanded) + tf.multiply(mat_pred[:, :, 1:2],
                                                                                           even_columns_expanded)
        # return dot_product
        # Reduce the result along the second dimension
        distances = -tf.reduce_sum(dot_product, axis=2)

        return distances


    def geodesic_distance_symmetries(y_true, y_pred):
        mat_true = rotation6d_to_matrix(y_true)
        eq_true = eq_matrices(mat_true, Matrices)
        ds = compute_distances(y_pred, eq_true)
        d = tf.reduce_min(ds, axis=2, keepdims=True)

        return K.mean(d)

    # Train-Validation sets
    if numModels == 1:
        lenTrain = int(len(fnImgs)*0.8)
        lenVal = len(fnImgs)-lenTrain
    else:
        lenTrain = int(len(fnImgs) / 5)
        lenVal = int(len(fnImgs) / 20)


    for index in range(numModels):
        # chooses equal number of particles for each division
        random_sample = np.random.choice(range(0, len(fnImgs)), size=lenTrain+lenVal, replace=False)

        training_generator = DataGenerator([fnImgs[i] for i in random_sample[0:lenTrain]],
                                           [labels[i] for i in random_sample[0:lenTrain]],
                                           sigma, batch_size, Xdims, shifts, readInMemory=False)
        validation_generator = DataGenerator([fnImgs[i] for i in random_sample[lenTrain:lenTrain+lenVal]],
                                             [labels[i] for i in random_sample[lenTrain:lenTrain+lenVal]],
                                             sigma, batch_size, Xdims, shifts, readInMemory=False)

        if pretrained == 'yes':
            model = load_model(fnPreModel, compile=False)
        else:
            model = constructModel(Xdims)

        adam_opt = Adam(lr=learning_rate)
        model.summary()

        model.compile(loss='mean_squared_error', optimizer=adam_opt)
        save_best_model = ModelCheckpoint(fnModel + str(index) + ".h5", monitor='val_loss',
                                          save_best_only=True)
        patienceCallBack = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)


        history = model.fit_generator(generator=training_generator, epochs=numEpochs,
                                      validation_data=validation_generator, callbacks=[save_best_model, patienceCallBack])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
    plt.show()

    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)
