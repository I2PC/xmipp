#!/usr/bin/env python3

import math
import numpy as np
from operator import itemgetter
import os
import sys
import xmippLib
from time import time
from scipy.ndimage import shift, rotate
import random

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
    print('numModels', numModels)
    learning_rate = float(sys.argv[8])
    patience = int(sys.argv[9])
    pretrained = sys.argv[10]
    symmetry = sys.argv[12]
    if pretrained == 'yes':
        fnPreModel = sys.argv[11]

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    import tensorflow as tf
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, concatenate, \
        Activation, GlobalAveragePooling2D, Add, MaxPooling2D, Flatten
    import keras
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import to_categorical

    print('fnModel', fnModel)


    class DataGenerator(keras.utils.all_utils.Sequence):
        """Generates data for fnImgs"""

        def __init__(self, fnImgs, labels, sigma, batch_size, dim, shifts, readInMemory, bool_classifier,
                     map_domains, target_domain_matrices, inv_matrices, inv_sqrt_matrices):
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
            self.bool_classifier = bool_classifier
            self.map_domains = map_domains
            self.target_domain_matrices = target_domain_matrices
            self.inv_matrices = inv_matrices
            self.inv_sqrt_matrices = inv_sqrt_matrices

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
            domains = np.array(itemgetter(*list_IDs_temp)(self.map_domains))

            # Functions to handle the data
            def get_image(fn_image):
                img = np.reshape(xmippLib.Image(fn_image).getData(), (self.dim, self.dim, 1))
                return (img - np.mean(img)) / np.std(img)

            def shift_image(img, shiftx, shifty, yshift):
                return shift(img, (shiftx - yshift[0], shifty - yshift[1], 0), order=1, mode='wrap')

            def rotate_image(img, angle):
                # angle in degrees
                angle = (180 / math.pi) * angle
                return rotate(img, angle, order=1, mode='reflect', reshape=False)

            def matrix_to_rotation6d(mat):
                r6d = np.delete(mat, -1, axis=1)
                return np.array((r6d[0, 0], r6d[0, 1], r6d[1, 0], r6d[1, 1], r6d[2, 0], r6d[2, 1]))

            def euler_to_rotation6d(angles, psi_rotation):
                mat = euler_angles_to_matrix(angles, psi_rotation)
                return matrix_to_rotation6d(mat)

            def make_redundant(rep_6d):
                rep_6d = np.append(rep_6d, 2 * rep_6d)
                for i in range(6):
                    j = (i + 1) % 6
                    rep_6d = np.append(rep_6d, rep_6d[i] - rep_6d[j])
                for i in range(6):
                    j = (i + 3) % 6
                    rep_6d = np.append(rep_6d, rep_6d[i + 6] - rep_6d[j])
                for i in range(6):
                    j = (i + 2) % 6
                    k = (i + 4) % 6
                    rep_6d = np.append(rep_6d, rep_6d[i] + rep_6d[j] - rep_6d[k])
                for i in range(6):
                    j = (i + 5) % 6
                    rep_6d = np.append(rep_6d, rep_6d[i] - rep_6d[j])
                for i in range(6):
                    j = (i + 4) % 6
                    rep_6d = np.append(rep_6d, rep_6d[i] - rep_6d[j])
                return rep_6d

            def check_angle(euler_angles, angle, previous_map):
                j = 0
                index_map = 124

                while index_map != previous_map:
                    angle = (angle / (2 ** j))
                    j = j + 1
                    matrix = euler_angles_to_matrix(euler_angles, angle)
                    S_inv = map_symmetries(matrix, self.inv_matrices, np.eye(3))
                    index_map = map_symmetries_index(S_inv, inverse_sqrt_matrices, np.eye(3))

                    index_map = round(index_map / 1.99)

                    if index_map > 3:
                        index_map = 3
                return angle

            def compute_yvalues(euler_angles, angle):
                y_matrices = euler_angles_to_matrix(euler_angles, angle)
                map_matrices = map_symmetries(y_matrices, self.inv_matrices, self.target_domain_matrices)
                y_6d = matrix_to_rotation6d(map_matrices)
                y_6d_redundant = make_redundant(y_6d)
                return y_6d_redundant

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
            yvalues = yvalues * math.pi / 180
            rAngle = math.pi * np.random.uniform(-1, 1, size=self.batch_size)
            rAngle = np.array(list(map(check_angle, yvalues, rAngle, domains)))
            Xexp = np.array(list(map(rotate_image, Xexp, rAngle)))
            if self.bool_classifier:
                y = to_categorical(domains, num_classes=4)
            else:
                y = np.array(list(map(compute_yvalues, yvalues, rAngle)))
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

        x = Dense(42, name="output", activation="linear")(x)

        return Model(inputLayer, x)


    # def constructClassifier(Xdim, classes):
    #    """RESNET architecture"""
    #    inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")
    #
    #    L = Conv2D(8, (3, 3), padding='same')(inputLayer)
    #    L = BatchNormalization()(L)
    #    L = Activation(activation='relu')(L)
    #    L = MaxPooling2D()(L)
    #
    #    L = Conv2D(16, (3, 3), padding='same')(L)
    #    L = BatchNormalization()(L)
    #    L = Activation(activation='relu')(L)
    #    L = MaxPooling2D()(L)
    #
    #    L = Conv2D(32, (3, 3), padding='same')(L)
    #    L = BatchNormalization()(L)
    #    L = Activation(activation='relu')(L)
    #    L = MaxPooling2D()(L)
    #
    #    L = Conv2D(64, (3, 3), padding='same')(L)
    #    L = BatchNormalization()(L)
    #    L = Activation(activation='relu')(L)
    #    L = MaxPooling2D()(L)
    #
    #    L = Flatten()(L)
    #
    #    L = Dense(classes, name="output", activation="softmax")(L)
    #
    #    return Model(inputLayer, L)

    def constructClassifier(Xdim, classes):
        """RESNET architecture"""
        inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")

        x = conv_block(inputLayer, filters=8)

        x = conv_block(x, filters=16)

        x = conv_block(x, filters=32)

        x = conv_block(x, filters=64)

        x = conv_block(x, filters=128)

        x = GlobalAveragePooling2D()(x)

        x = Dense(classes, name="output", activation="softmax")(x)

        return Model(inputLayer, x)


    SL = xmippLib.SymList()
    Matrices = np.array(SL.getSymmetryMatrices(symmetry))

    print('Matrices', Matrices)


    def rodrigues_formula(axis, angle):
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * np.matmul(K, K)


    def sqrt_matrix(sym_matrix):
        eigenvalues, eigenvectors = np.linalg.eig(sym_matrix)
        real_eigenvalue_index = np.where(np.isreal(eigenvalues))[0]
        if len(real_eigenvalue_index) > 1:
            real_eigenvalue_index = np.where(eigenvalues[real_eigenvalue_index] > 0)[0]
        real_eigenvalue_index = real_eigenvalue_index[0]
        eigenvector = np.squeeze(np.real(eigenvectors[:, real_eigenvalue_index]))
        angle = np.arccos((np.trace(sym_matrix) - 1) / 2)
        sm1 = rodrigues_formula(eigenvector, angle / 2)
        sm2 = rodrigues_formula(eigenvector, math.pi + (angle / 2))
        if not np.allclose(np.matmul(sm1, sm1), sym_matrix, atol=1e-5):
            sm1 = rodrigues_formula(eigenvector, -angle / 2)
            sm2 = rodrigues_formula(eigenvector, math.pi + (-angle / 2))
        return sm1, sm2


    matrix_1 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    gen_matrix1 = sqrt_matrix(matrix_1)[0]
    matrix_1a = np.transpose(matrix_1)
    gen_matrix1a = sqrt_matrix(matrix_1a)[0]
    matrix_2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    gen_matrix2 = sqrt_matrix(matrix_2)[0]
    matrix_2a = np.transpose(matrix_2)
    gen_matrix2a = sqrt_matrix(matrix_2a)[0]
    sqrt_matrices = []

    sqrt_matrices.append(np.eye(3))
    sqrt_matrices.append(gen_matrix1)
    sqrt_matrices.append(gen_matrix1a)
    sqrt_matrices.append(gen_matrix2)
    sqrt_matrices.append(gen_matrix2a)

    for matrix in Matrices:
        sqrt1, sqrt2 = sqrt_matrix(matrix)
        print(len(sqrt_matrices))
        if not any(np.allclose(sqrt1, m, atol=1e-5) for m in sqrt_matrices):
            sqrt_matrices.append(sqrt1)
        else:
            print('sqrt1', sqrt1)
            print(matrix)
        if not any(np.allclose(sqrt2, m, atol=1e-5) for m in sqrt_matrices):
            sqrt_matrices.append(sqrt2)
        else:
            print('sqrt2', sqrt2)
            print(matrix)

    print('sqrt_matrices', sqrt_matrices)

    num_sqrt_matrices = np.shape(sqrt_matrices)[0]
    num_matrices = np.shape(Matrices)[0]
    inverse_sqrt_matrices = np.zeros_like(sqrt_matrices)
    for i in range(num_sqrt_matrices):
        inverse_sqrt_matrices[i] = np.transpose(sqrt_matrices[i])
    print('len sqrt matrices', len(sqrt_matrices))
    print('len matrices', len(Matrices))

    inverse_matrices = np.zeros_like(Matrices)
    for i in range(num_matrices):
        inverse_matrices[i] = np.transpose(Matrices[i])

    target_matrices = sqrt_matrices[0:2]
    target_matrices.append(sqrt_matrices[3])
    target_matrices.append(np.matmul(gen_matrix1, gen_matrix2))


    def frobenius_norm(matrix):
        return np.linalg.norm(matrix, ord='fro')


    def map_symmetries_index(input_matrix, inv_group_matrices, target):
        n = np.shape(inv_group_matrices)[0]
        reshaped_matrix = np.tile(input_matrix, (n, 1, 1))
        candidates = np.matmul(reshaped_matrix, inv_group_matrices)
        norms = np.array(list((map(frobenius_norm, candidates - target))))
        index_min = np.argmin(norms)
        return index_min


    def map_symmetries(input_matrix, inv_group_matrices, target):
        return np.matmul(input_matrix,
                         inv_group_matrices[map_symmetries_index(input_matrix, inv_group_matrices, target)])


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


    def euler_matrix(ai, aj, ak, axes='szyz'):
        """Return homogeneous rotation matrix from Euler angles and axis sequence.
        ai, aj, ak : Euler's roll, pitch and yaw angles
        axes : One of 24 axis sequences as string or encoded tuple
        """
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
        return M[:3, :3]


    def euler_angles_to_matrix(angles, psi_rotation):
        return euler_matrix(angles[0], angles[1], angles[2] + psi_rotation, axes='szyz')


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

        # For better performance, images are selected to be 'homogeneously' distributed in the sphere
        # 50 divisions with equal area
        numTiltDivs = 5
        numRotDivs = 10
        limits_rot = np.linspace(-180.01, 180, num=(numRotDivs + 1))
        limits_tilt = np.zeros(numTiltDivs + 1)
        limits_tilt[0] = -0.01
        for i in range(1, numTiltDivs + 1):
            limits_tilt[i] = math.acos(1 - 2 * (i / numTiltDivs))
        limits_tilt = limits_tilt * 180 / math.pi

        # Each particle is assigned to a division
        zone = [[] for _ in range((len(limits_tilt) - 1) * (len(limits_rot) - 1))]
        i = 0
        map_region = []
        regions_map = [[] for _ in range(4)]

        for r, t, p, sX, sY in zip(rots, tilts, psis, shiftX, shiftY):
            img_shift.append(np.array((sX, sY)))
            zone_rot = np.digitize(r, limits_rot, right=True) - 1
            zone_tilt = np.digitize(t, limits_tilt, right=True) - 1
            # Region index
            zone_idx = zone_rot * (len(limits_tilt) - 1) + zone_tilt
            zone[zone_idx].append(i)

            matrix = euler_angles_to_matrix([r * math.pi / 180, t * math.pi / 180, p * math.pi / 180], 0.)
            S_inv = map_symmetries(matrix, inverse_matrices, np.eye(3))
            index_map = map_symmetries_index(S_inv, inverse_sqrt_matrices, np.eye(3))
            index_map = round(index_map / 1.99)
            if index_map > 3:
                index_map = 3
            map_region.append(index_map)
            regions_map[index_map].append(i)
            label.append(np.array((r, t, p)))
            i += 1
        return Xdim, fnImg, label, zone, img_shift, map_region, regions_map


    import tensorflow as tf
    import keras.backend as K

    Xdims, fnImgs, labels, zones, shifts, map_regions, regions_map = get_labels(fnXmdExp)
    start_time = time()

    non_void_regions = []

    for index, sublist in enumerate(regions_map):
        if len(sublist) > 0:
            non_void_regions.append(index)

    num_non_void_regions = len(non_void_regions)


    def fill_regions_map(regions_list):
        max_num_elements = max(len(sublist) for sublist in regions_list if sublist)
        print('max_num_elements', max_num_elements)

        for sublist in regions_list:
            print(len(sublist))
            if sublist:
                while len(sublist) < max_num_elements:
                    sublist.append(random.choice(sublist))
        return regions_list


    for index in range(numModels):

        if numModels == 1:
            lenTrain = int(len(fnImgs) * 0.8)
            lenVal = len(fnImgs) - lenTrain
        else:
            lenTrain = int(len(fnImgs) / 3)
            lenVal = int(len(fnImgs) / 12)

        regions_map_train = []
        regions_map_val = []
        for sublist in regions_map:
            lenValList = int((len(sublist) / len(fnImgs)) * lenVal)
            random.shuffle(sublist)
            regions_map_val.append(sublist[0: lenValList])
            regions_map_train.append(sublist[lenValList: len(sublist)])

        regions_map_train_filled = fill_regions_map(regions_map_train)

        flat_regions_map_train = [item for sublist in regions_map_train_filled for item in sublist]
        flat_regions_map_val = [item for sublist in regions_map_val for item in sublist]

        elements_zone = int((lenVal + lenTrain) / len(zones))

        folder_path = fnModel + '/model' + str(index)
        # chooses equal number of particles for each division
        classifier = False
        if symmetry != 'C1':
            classifier = True

            random_sample_train = np.random.choice(flat_regions_map_train, size=lenTrain, replace=False)
            random_sample_val = flat_regions_map_val

            training_generator = DataGenerator([fnImgs[i] for i in random_sample_train],
                                               [labels[i] for i in random_sample_train],
                                               sigma, batch_size, Xdims, [shifts[i] for i in random_sample_train],
                                               readInMemory=False,
                                               bool_classifier=classifier,
                                               map_domains=[map_regions[i] for i in random_sample_train],
                                               target_domain_matrices=target_matrices, inv_matrices=inverse_matrices,
                                               inv_sqrt_matrices=inverse_sqrt_matrices)
            validation_generator = DataGenerator([fnImgs[i] for i in random_sample_val],
                                                 [labels[i] for i in random_sample_val],
                                                 sigma, batch_size, Xdims,
                                                 [shifts[i] for i in random_sample_val],
                                                 readInMemory=False,
                                                 bool_classifier=classifier,
                                                 map_domains=[map_regions[i] for i in
                                                              random_sample_val],
                                                 target_domain_matrices=target_matrices, inv_matrices=inverse_matrices,
                                                 inv_sqrt_matrices=inverse_sqrt_matrices)

            model = constructClassifier(Xdims, classes=4)
            adam_opt = tf.keras.optimizers.Adam(lr=learning_rate)
            model.summary()

            model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])
            save_best_model = ModelCheckpoint(folder_path + '/classifier' + '.h5', monitor='val_loss',
                                              save_best_only=True)
            patienceCallBack = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2 * patience)

            history = model.fit_generator(generator=training_generator, epochs=numEpochs,
                                          validation_data=validation_generator,
                                          callbacks=[save_best_model, patienceCallBack])

        classifier = False
        for j in range(4):
            print('len regions map', len(regions_map[j]))
            if numModels == 1:
                lenTrain = int(len(regions_map[j]) * 0.8)
                lenVal = len(regions_map[j]) - lenTrain
            else:
                lenTrain = int(len(regions_map[j]) / 3)
                lenVal = int(len(regions_map[j]) / 12)
            print('lenTrain', lenTrain)
            print('lenVal', lenVal)

            if lenVal > 0:
                random_sample = np.random.choice(regions_map[j], size=lenTrain + lenVal, replace=False)

                training_generator = DataGenerator([fnImgs[i] for i in random_sample[0:lenTrain]],
                                                   [labels[i] for i in random_sample[0:lenTrain]],
                                                   sigma, batch_size, Xdims,
                                                   [shifts[i] for i in random_sample[0:lenTrain]],
                                                   readInMemory=False, bool_classifier=classifier,
                                                   map_domains=[map_regions[i] for i in random_sample[0:lenTrain]],
                                                   target_domain_matrices=target_matrices[j],
                                                   inv_matrices=inverse_matrices,
                                                   inv_sqrt_matrices=inverse_sqrt_matrices)
                validation_generator = DataGenerator([fnImgs[i] for i in random_sample[lenTrain:lenTrain + lenVal]],
                                                     [labels[i] for i in random_sample[lenTrain:lenTrain + lenVal]],
                                                     sigma, batch_size, Xdims,
                                                     [shifts[i] for i in random_sample[lenTrain:lenTrain + lenVal]],
                                                     readInMemory=False, bool_classifier=classifier,
                                                     map_domains=[map_regions[i] for i in
                                                                  random_sample[lenTrain:lenTrain + lenVal]],
                                                     target_domain_matrices=target_matrices[j],
                                                     inv_matrices=inverse_matrices,
                                                     inv_sqrt_matrices=inverse_sqrt_matrices)

                # if pretrained == 'yes':
                #    model = load_model(fnPreModel, compile=False)
                # else:
                #    model = constructModel(Xdims)

                model = constructModel(Xdims)
                adam_opt = tf.keras.optimizers.Adam(lr=learning_rate)
                model.summary()

                model.compile(loss='mean_squared_error', optimizer=adam_opt)
                save_best_model = ModelCheckpoint(folder_path + '/modelAng' + str(j) + '.h5', monitor='val_loss',
                                                  save_best_only=True)
                patienceCallBack = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

                history = model.fit_generator(generator=training_generator, epochs=numEpochs,
                                              validation_data=validation_generator,
                                              callbacks=[save_best_model, patienceCallBack])

    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)
