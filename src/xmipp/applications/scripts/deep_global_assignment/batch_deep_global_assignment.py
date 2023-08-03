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

    """Preprocesing"""

    SL = xmippLib.SymList()
    Matrices = np.array(SL.getSymmetryMatrices(symmetry))

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
        return rodrigues_formula(eigenvector, angle / 2)

    num_rot_axis = 2
    matrix_1 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    gen_matrix1 = sqrt_matrix(matrix_1)
    matrix_2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    gen_matrix2 = sqrt_matrix(matrix_2)
    sqrt_matrices = [[] for _ in range(num_rot_axis)]


    def is_in_list(matrices_list, new_matrix):
        if any(np.allclose(new_matrix, m, atol=1e-05) for m in matrices_list):
            return True
        return False

    def complete_sqrt_matrices(sqrt_gen_matrices, void_list_sqrt_matrices):
        for i in range(len(void_list_sqrt_matrices)):
            j = 0
            candidate = np.eye(3)
            while not is_in_list(void_list_sqrt_matrices[i], candidate):
                void_list_sqrt_matrices[i].append(candidate)
                j = j + 1
                candidate = np.linalg.matrix_power(sqrt_gen_matrices[i], n=j)
        return void_list_sqrt_matrices

    def get_target_matrices(list_sqrt_matrices):
        list_matrices = []
        list_matrices.append(np.eye(3))
        for i in range(len(list_sqrt_matrices)):
            list_matrices.append(list_sqrt_matrices[i][1])
        list_matrices.append(np.matmul(list_matrices[1], list_matrices[2]))
        return list_matrices

    sqrt_matrices = complete_sqrt_matrices([gen_matrix1, gen_matrix2], sqrt_matrices)

    print('sqrt_matrices', sqrt_matrices)

    target_matrices = get_target_matrices(sqrt_matrices)

    print('target matrices', target_matrices)

    #def zeros_exists(matrix):
    #    for row in matrix:
    #        for element in row:
    #            if element == 0:
    #                return True
    #    return False
#
    #def next_zero_indices(matrix, n):
    #    for row_index in range(n):
    #        for column_index in range(n):
    #            if matrix[row_index][column_index] == 0:
    #                return row_index, column_index
    #    return -1, -1
#

#
    #def add_row_and_column_of_zeros(matrix, n):
    #    row_of_zeros = [0] * n
    #    matrix.append(row_of_zeros)
    #    for row in matrix:
    #        row.append(0)
    #    return matrix
#
    #def generate_group(gen_matrices):
    #    list_matrices = [m for m in gen_matrices]
    #    n = len(list_matrices)
    #    products_matrix = [[0 for _ in range(n)] for _ in range(n)]
    #    while zeros_exists(products_matrix):
    #        l, r = next_zero_indices(products_matrix, n)
    #        left = list_matrices[l]
    #        right = list_matrices[r]
    #        products_matrix[l][r] = 1
    #        candidate = np.matmul(left, right)
    #        print('l', l)
    #        print('r', r)
    #        if not is_in_list(list_matrices, candidate):
    #            list_matrices.append(candidate)
    #            products_matrix = add_row_and_column_of_zeros(products_matrix, n)
    #            n = n + 1
    #        print('n', n)
    #    return list_matrices


    #sqrt_matrices = generate_group(sqrt_matrices)

    def inverse_list_matrices(list_of_matrices):
        inverse_list = [
            [np.transpose(mat) for mat in sublist] for sublist in list_of_matrices
        ]
        return inverse_list

    inverse_sqrt_matrices = inverse_list_matrices(sqrt_matrices)
    print('inverse_sqrt_matrices', inverse_sqrt_matrices)
    inverse_matrices = inverse_list_matrices([Matrices])
    print('inverse_matrices', inverse_matrices)
    inverse_matrices = inverse_matrices[0]
    print('inverse_matrices', inverse_matrices)

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

        i = 0
        region = []
        elem_regions = [[[], []] for _ in range(num_rot_axis)]
        print('elem_regions', elem_regions)

        for r, t, p, sX, sY in zip(rots, tilts, psis, shiftX, shiftY):
            img_shift.append(np.array((sX, sY)))
            matrix = euler_angles_to_matrix([r * math.pi / 180, t * math.pi / 180, p * math.pi / 180], 0.)
            delta = []
            S_inv = map_symmetries(matrix, inverse_matrices, np.eye(3))
            for axis_ind in range(num_rot_axis):
                index_map = map_symmetries_index(S_inv, inverse_sqrt_matrices[axis_ind], np.eye(3))
                if index_map > 1:
                    index_map = 1
                elem_regions[axis_ind][index_map].append(i)
                delta.append(index_map)
            region.append(delta)
            label.append(np.array((r, t, p)))
            i += 1
        return Xdim, fnImg, label, img_shift, region, elem_regions

    import tensorflow as tf
    import keras.backend as K

    Xdims, fnImgs, labels, shifts, map_regions, regions_map = get_labels(fnXmdExp)
    start_time = time()

    print(len(regions_map))
    print(len(regions_map[0][0]))
    print(len(regions_map[0][1]))
    print(len(regions_map[1][0]))
    print(len(regions_map[1][1]))

    def fill_regions_map(regions_list):
        max_num_elements = max(len(sublist) for sublist in regions_list if sublist)
        print('max_num_elements', max_num_elements)

        for sublist in regions_list:
            print(len(sublist))
            if sublist:
                while len(sublist) < max_num_elements:
                    sublist.append(random.choice(sublist))
        return regions_list


    """Training"""

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
                index_map = 124 * np.ones_like(previous_map)
                # Distinguir entre clasificador y regresor utilizando el self.bool_classifier
                if self.bool_classifier:
                    while index_map != previous_map:
                        angle = (angle / (2 ** j))
                        j = j + 1
                        matrix = euler_angles_to_matrix(euler_angles, angle)
                        S_inv = map_symmetries(matrix, self.inv_matrices, np.eye(3))
                        index_map = map_symmetries_index(S_inv, self.inv_sqrt_matrices, np.eye(3))
                        if index_map > 1:
                            index_map = 1
                else:
                    while any(index_map != previous_map):
                        angle = (angle / (2 ** j))
                        j = j + 1
                        matrix = euler_angles_to_matrix(euler_angles, angle)
                        S_inv = map_symmetries(matrix, self.inv_matrices, np.eye(3))
                        for axis_index in range(len(previous_map)):
                            index_map[axis_index] = map_symmetries_index(S_inv, self.inv_sqrt_matrices[axis_index],
                                                                         np.eye(3))
                            if index_map[axis_index] > 1:
                                index_map[axis_index] = 1
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
                y = to_categorical(domains, num_classes=2)
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


    def constructClassifier(Xdim, classes):
        """RESNET architecture"""
        inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")

        L = Conv2D(8, (3, 3), padding='same')(inputLayer)
        L = BatchNormalization()(L)
        L = Activation(activation='relu')(L)
        L = MaxPooling2D()(L)

        L = Conv2D(16, (3, 3), padding='same')(L)
        L = BatchNormalization()(L)
        L = Activation(activation='relu')(L)
        L = MaxPooling2D()(L)

        L = Conv2D(32, (3, 3), padding='same')(L)
        L = BatchNormalization()(L)
        L = Activation(activation='relu')(L)
        L = MaxPooling2D()(L)

        L = Conv2D(64, (3, 3), padding='same')(L)
        L = BatchNormalization()(L)
        L = Activation(activation='relu')(L)
        L = MaxPooling2D()(L)

        L = Flatten()(L)

        L = Dense(classes, name="output", activation="softmax")(L)

        return Model(inputLayer, L)

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

        folder_path = fnModel + '/model' + str(index)
        # chooses equal number of particles for each division
        classifier = False
        if symmetry != 'C1':
            classifier = True

            for sym_axis in range(num_rot_axis):
                if numModels == 1:
                    lenTrain = int(len(fnImgs) * 0.8)
                    lenVal = len(fnImgs) - lenTrain
                else:
                    lenTrain = int(len(fnImgs) / 3)
                    lenVal = int(len(fnImgs) / 12)
                regions_map_train = []
                regions_map_val = []

                for sublist in regions_map[sym_axis]:
                    lenValList = int((len(sublist) / len(fnImgs)) * lenVal)
                    random.shuffle(sublist)
                    regions_map_val.append(sublist[0: lenValList])
                    regions_map_train.append(sublist[lenValList: len(sublist)])

                regions_map_train_filled = fill_regions_map(regions_map_train)

                flat_regions_map_train = [item for sublist in regions_map_train_filled for item in sublist]
                flat_regions_map_val = [item for sublist in regions_map_val for item in sublist]

                random_sample_train = np.random.choice(flat_regions_map_train, size=lenTrain, replace=False)
                random_sample_val = flat_regions_map_val

                training_generator = DataGenerator([fnImgs[i] for i in random_sample_train],
                                                   [labels[i] for i in random_sample_train],
                                                   sigma, batch_size, Xdims, [shifts[i] for i in random_sample_train],
                                                   readInMemory=False,
                                                   bool_classifier=classifier,
                                                   map_domains=[map_regions[i][sym_axis] for i in random_sample_train],
                                                   target_domain_matrices=target_matrices, inv_matrices=inverse_matrices,
                                                   inv_sqrt_matrices=inverse_sqrt_matrices[sym_axis])
                validation_generator = DataGenerator([fnImgs[i] for i in random_sample_val],
                                                     [labels[i] for i in random_sample_val],
                                                     sigma, batch_size, Xdims,
                                                     [shifts[i] for i in random_sample_val],
                                                     readInMemory=False,
                                                     bool_classifier=classifier,
                                                     map_domains=[map_regions[i][sym_axis] for i in
                                                                  random_sample_val],
                                                     target_domain_matrices=target_matrices, inv_matrices=inverse_matrices,
                                                     inv_sqrt_matrices=inverse_sqrt_matrices[sym_axis])

                model = constructClassifier(Xdims, classes=2)
                adam_opt = tf.keras.optimizers.Adam(lr=learning_rate)
                model.summary()

                model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])
                save_best_model = ModelCheckpoint(folder_path + '/classifier' + str(sym_axis) + '.h5', monitor='val_loss',
                                                  save_best_only=True)
                patienceCallBack = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2 * patience)

                history = model.fit_generator(generator=training_generator, epochs=numEpochs,
                                          validation_data=validation_generator,
                                          callbacks=[save_best_model, patienceCallBack])

        def get_subsets_regression(lists):
            subsets = []
            if num_rot_axis == 2:
                subsets.append(list(set(lists[0][0]) & set(lists[1][0])))
                subsets.append(list(set(set(lists[0][1]) & set(lists[1][0]))))
                subsets.append(list(set(set(lists[0][0]) & set(lists[1][1]))))
                subsets.append(list(set(set(lists[0][1]) & set(lists[1][1]))))
            return subsets

        classifier = False
        subsets_regression = get_subsets_regression(regions_map)
        for j in range(2**num_rot_axis):
            if numModels == 1:
                lenTrain = int(len(subsets_regression[j]) * 0.8)
                lenVal = len(subsets_regression[j]) - lenTrain
            else:
                lenTrain = int(len(subsets_regression[j]) / 3)
                lenVal = int(len(subsets_regression[j]) / 12)
            print('lenTrain', lenTrain)
            print('lenVal', lenVal)

            if lenVal > 0:
                random_sample = np.random.choice(subsets_regression[j], size=lenTrain + lenVal, replace=False)

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
