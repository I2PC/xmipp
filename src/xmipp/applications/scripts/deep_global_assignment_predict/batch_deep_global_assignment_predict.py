#!/usr/bin/env python3

import math
import numpy as np
from numpy.linalg import norm
import os
import sys
import xmippLib
from time import time
from scipy.spatial.transform import Rotation
from scipy.ndimage import shift, rotate

# from pwem.convert.transformations import quaternion_from_matrix, euler_from_quaternion

maxSize = 32

if __name__ == "__main__":
    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    gpuId = sys.argv[2]
    outputDir = sys.argv[3]
    fnXmdImages = sys.argv[4]
    fnAngModel = sys.argv[5]
    numAngModels = int(sys.argv[6])
    tolerance = int(sys.argv[7])
    maxModels = int(sys.argv[8])

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    from keras.models import Model
    import keras
    from keras.models import load_model
    import tensorflow as tf


    class DataGenerator(keras.utils.Sequence):
        'Generates data for fnImgs'

        def __init__(self, fnImgs, maxSize, dim, shifts, readInMemory):
            'Initialization'
            self.fnImgs = fnImgs
            self.maxSize = maxSize
            self.dim = dim
            self.shifts = shifts
            self.readInMemory = readInMemory
            self.on_epoch_end()

            # Read all data in memory
            if self.readInMemory:
                self.Xexp = np.zeros((len(self.fnImgs), self.dim, self.dim, 1), dtype=np.float64)
                for i in range(len(self.fnImgs)):
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[i]).getData(), (self.dim, self.dim, 1))
                    self.Xexp[i,] = (Iexp - np.mean(Iexp)) / np.std(Iexp)

        def __len__(self):
            'Denotes the number of batches per predictions'
            num = len(self.fnImgs) // maxSize
            if len(self.fnImgs) % maxSize > 0:
                num = num + 1
            return num

        def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch
            indexes = self.indexes[index * maxSize:(index + 1) * maxSize]
            # Find list of IDs
            list_IDs_temp = []
            for i in range(len(indexes)):
                list_IDs_temp.append(indexes[i])

            # Generate data
            Xexp = self.__data_generation(list_IDs_temp)

            return Xexp

        def on_epoch_end(self):
            self.indexes = [i for i in range(len(self.fnImgs))]

        def getNumberOfBlocks(self):
            self.st = len(self.fnImgs) // maxSize
            if len(self.fnImgs) % maxSize > 0:
                self.st = self.st + 1

        def __data_generation(self, list_IDs_temp):
            'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
            def shift_image(img, img_shifts):
                return shift(img, (-img_shifts[0], -img_shifts[1], 0), order=1, mode='wrap')
            # Initialization
            Xexp = np.zeros((len(list_IDs_temp), self.dim, self.dim, 1), dtype=np.float64)

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Read image
                if self.readInMemory:
                    Xexp[i,] = self.Xexp[ID]
                else:
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[ID]).getData(), (self.dim, self.dim, 1))
                    Xexp[i,] = (Iexp - np.mean(Iexp)) / np.std(Iexp)
                    Xexp[i,] = shift_image(Xexp[i,], self.shifts[i])
            return Xexp


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


    def quaternion_from_matrix(matrix, isprecise=False):
        """Return quaternion from rotation matrix.

        If isprecise is True, the input matrix is assumed to be a precise rotation
        matrix and a faster algorithm is used."""
        M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        if isprecise:
            q = np.empty((4,))
            t = np.trace(M)
            if t > M[3, 3]:
                q[0] = t
                q[3] = M[1, 0] - M[0, 1]
                q[2] = M[0, 2] - M[2, 0]
                q[1] = M[2, 1] - M[1, 2]
            else:
                i, j, k = 1, 2, 3
                if M[1, 1] > M[0, 0]:
                    i, j, k = 2, 3, 1
                if M[2, 2] > M[i, i]:
                    i, j, k = 3, 1, 2
                t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
                q[i] = t
                q[j] = M[i, j] + M[j, i]
                q[k] = M[k, i] + M[i, k]
                q[3] = M[k, j] - M[j, k]
            q *= 0.5 / math.sqrt(t * M[3, 3])
        else:
            m00 = M[0, 0]
            m01 = M[0, 1]
            m02 = M[0, 2]
            m10 = M[1, 0]
            m11 = M[1, 1]
            m12 = M[1, 2]
            m20 = M[2, 0]
            m21 = M[2, 1]
            m22 = M[2, 2]
            # symmetric matrix K
            K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                          [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                          [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                          [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
            K /= 3.0
            # quaternion is eigenvector of K that corresponds to largest eigenvalue
            w, V = np.linalg.eigh(K)
            q = V[[3, 0, 1, 2], np.argmax(w)]
        if q[0] < 0.0:
            np.negative(q, q)
        return q


    def quaternion_matrix(quaternion):
        """Return homogeneous rotation matrix from quaternion."""
        q = np.array(quaternion, dtype=np.float64, copy=True)
        n = np.dot(q, q)
        if n < _EPS:
            return np.identity(4)
        q *= math.sqrt(2.0 / n)
        q = np.outer(q, q)
        return np.array([
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0]])


    def euler_from_matrix(matrix, axes='sxyz'):
        """Return Euler angles from rotation matrix for specified axis sequence.

        axes : One of 24 axis sequences as string or encoded tuple

        Note that many Euler angle triplets can describe one matrix."""
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


    def euler_from_quaternion(quaternion, axes='sxyz'):
        """Return Euler angles from quaternion for specified axis sequence."""
        return euler_from_matrix(quaternion_matrix(quaternion), axes)


    def rotation6d_to_matrix(rot):

        a1 = np.array((rot[0], rot[2], rot[4]))
        a2 = np.array((rot[1], rot[3], rot[5]))
        a1 = np.reshape(a1, (3, 1))
        a2 = np.reshape(a2, (3, 1))

        b1 = a1 / np.linalg.norm(a1)

        c1 = np.multiply(b1, a2)
        c2 = np.sum(c1)

        b2 = a2 - c2 * b1
        b2 = b2 / np.linalg.norm(b2)
        b3 = np.cross(b1, b2, axis=0)
        return np.concatenate((b1, b2, b3), axis=1)


    def matrix_to_euler(mat):
        r = Rotation.from_matrix(mat)
        angles = r.as_euler("xyz", degrees=True)
        return angles


    def produce_output(mdExp, Y, distance, fnImages):
        ID = 0
        for objId in mdExp:
            angles = Y[ID] * 180 / math.pi
            mdExp.setValue(xmippLib.MDL_ANGLE_PSI, angles[2], objId)
            mdExp.setValue(xmippLib.MDL_ANGLE_ROT, angles[0], objId)
            mdExp.setValue(xmippLib.MDL_ANGLE_TILT, angles[1] + 90, objId)
            mdExp.setValue(xmippLib.MDL_IMAGE, fnImages[ID], objId)
            if distance[ID] > tolerance:
                mdExp.setValue(xmippLib.MDL_ENABLED, -1, objId)
            ID += 1


    def convert_to_matrix(rep6d):
        return np.array(list(map(rotation6d_to_matrix, rep6d)))


    def convert_to_quaternions(mat):
        return np.array(list(map(quaternion_from_matrix, mat)))


    def average_quaternions(quaternions):
        s = np.matmul(quaternions.T, quaternions)
        s /= len(quaternions)
        eigenValues, eigenVectors = np.linalg.eig(s)
        return np.real(eigenVectors[:, np.argmax(eigenValues)])


    def maximum_distance(av_mat, mat):
        c = mat * av_mat[np.newaxis, 0:3, 0:3]
        d = np.sum(c, axis=(1, 2))
        ang_Distances = np.arccos((d - 1) / 2)
        return np.max(ang_Distances), np.argmax(ang_Distances)


    def calculate_r6d(redundant):
        A = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
                      [2, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0],
                      [0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 2],
                      [1, -1, 0, 0, 0, 0], [0, 1, -1, 0, 0, 0], [0, 0, 1, -1, 0, 0],
                      [0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 1, -1], [-1, 0, 0, 0, 0, 1],
                      [2, 0, 0, -1, 0, 0], [0, 2, 0, 0, -1, 0], [0, 0, 2, 0, 0, -1],
                      [-1, 0, 0, 2, 0, 0], [0, -1, 0, 0, 2, 0], [0, 0, -1, 0, 0, 2],
                      [1, 0, 1, 0, -1, 0], [0, 1, 0, 1, 0, -1], [-1, 0, 1, 0, 1, 0],
                      [0, -1, 0, 1, 0, 1], [1, 0, -1, 0, 1, 0], [0, 1, 0, -1, 0, 1],
                      [1, 0, 0, 0, 0, -1], [-1, 1, 0, 0, 0, 0], [0, -1, 1, 0, 0, 0],
                      [0, 0, -1, 1, 0, 0], [0, 0, 0, -1, 1, 0], [0, 0, 0, 0, -1, 1],
                      [1, 0, 0, 0, -1, 0], [0, 1, 0, 0, 0, -1], [-1, 0, 1, 0, 0, 0],
                      [0, -1, 0, 1, 0, 0], [0, 0, -1, 0, 1, 0], [0, 0, 0, -1, 0, 1]])
        X = np.zeros((redundant.shape[0], 6))

        for i in range(X.shape[0]):
            X[i] = np.linalg.lstsq(A, redundant[i])[0]

        return np.array(X)


    def average_of_rotations(p6d_redundant):
        pred6d = calculate_r6d(p6d_redundant)
        matrix = convert_to_matrix(pred6d)
        minParticles = np.shape(matrix)[0] - maxModels
        quats = convert_to_quaternions(matrix)
        av_quats = average_quaternions(quats)
        av_matrix = quaternion_matrix(av_quats)
        max_distance, max_dif_particle = maximum_distance(av_matrix, matrix)
        max_distance = max_distance * 180 / math.pi
        while (np.shape(matrix)[0] > minParticles) and (max_distance > tolerance):
            matrix = np.delete(matrix, max_dif_particle, axis=0)
            quats = np.delete(quats, max_dif_particle, axis=0)
            av_quats = average_quaternions(quats)
            av_matrix = quaternion_matrix(av_quats)
            max_distance, max_dif_particle = maximum_distance(av_matrix, matrix)
            max_distance = max_distance * 180 / math.pi
        av_euler = euler_from_matrix(av_matrix)
        return np.append(av_euler, max_distance)


    def compute_ang_averages(pred6d):
        averages_mdistance = np.array(list(map(average_of_rotations, pred6d)))
        average = averages_mdistance[:, 0:3]
        mdistance = averages_mdistance[:, 3]
        return average, mdistance

    Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnXmdExp)

    mdExp = xmippLib.MetaData(fnXmdExp)
    fnImgs = mdExp.getColumnValues(xmippLib.MDL_IMAGE)

    mdExpImages = xmippLib.MetaData(fnXmdImages)
    fnImages = mdExpImages.getColumnValues(xmippLib.MDL_IMAGE)

    shiftX = mdExp.getColumnValues(xmippLib.MDL_SHIFT_X)
    shiftY = mdExp.getColumnValues(xmippLib.MDL_SHIFT_Y)
    shifts = []
    for sX, sY in zip(shiftX, shiftY):
        shifts.append(np.array((sX, sY)))

    start_time = time()


    def shift_image(img, img_shifts):
        return shift(img, (-img_shifts[0], -img_shifts[1], 0), order=1, mode='wrap')

    models = []
    for index in range(numAngModels):
        AngModel = load_model(fnAngModel + str(index) + ".h5", compile=False)
        AngModel.compile(loss="mean_squared_error", optimizer='adam')
        models.append(AngModel)

    numImgs = len(fnImgs)
    predictions = np.zeros((numImgs, numAngModels, 42))
    numBatches = numImgs // maxSize
    print('numBatches', numBatches)
    if numImgs % maxSize > 0:
        numBatches = numBatches + 1
    k = 0
    for i in range(numBatches):
        print(i, flush=True)
        numPredictions = min(maxSize, numImgs-i*maxSize)
        print('numPredictions', numPredictions, flush=True)
        Xexp = np.zeros((numPredictions, Xdim, Xdim, 1), dtype=np.float64)
        for j in range(numPredictions):
            Iexp = np.reshape(xmippLib.Image(fnImgs[k]).getData(), (Xdim, Xdim, 1))
            Xexp[j, ] = (Iexp - np.mean(Iexp)) / np.std(Iexp)
            Xexp[j, ] = shift_image(Xexp[j, ], shifts[k])
            k += 1
        for index in range(numAngModels):
            predictions[i*maxSize:(i*maxSize + numPredictions), index, :] = models[index].predict(Xexp)

    Y, distance = compute_ang_averages(predictions)
    produce_output(mdExp, Y, distance, fnImages)
    mdExp.write(os.path.join(outputDir, "predict_results.xmd"))

    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)
