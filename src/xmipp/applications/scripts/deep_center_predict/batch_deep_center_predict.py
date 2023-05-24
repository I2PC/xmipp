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
#from pwem.convert.transformations import quaternion_from_matrix, euler_from_quaternion

maxSize = 128

if __name__ == "__main__":
    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    fnShiftModel = sys.argv[2]
    predictAngles = sys.argv[3]
    gpuId = sys.argv[4]
    outputDir = sys.argv[5]
    fnXmdImages = sys.argv[6]
    if predictAngles == 'yes':
        fnAngModel = sys.argv[7]
        numModels = int(sys.argv[8])

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    from keras.models import Model
    import keras
    from keras.models import load_model
    import tensorflow as tf


    class DataGenerator(keras.utils.Sequence):
        'Generates data for fnImgs'

        def __init__(self, fnImgs, maxSize, dim, readInMemory):
            'Initialization'
            self.fnImgs = fnImgs
            self.maxSize = maxSize
            self.dim = dim
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
            # self.st = ImagesNumber/maxSize        
            self.st = len(self.fnImgs) // maxSize
            if len(self.fnImgs) % maxSize > 0:
                self.st = self.st + 1

        def __data_generation(self, list_IDs_temp):
            'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
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
                    #Xexp[i,] = shift(Xexp[i, ], (-10, 10, 0), order=1, mode='reflect')
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
        matrix and a faster algorithm is used.

        >>> q = quaternion_from_matrix(np.identity(4), True)
        >>> np.allclose(q, [1, 0, 0, 0])
        True
        >>> q = quaternion_from_matrix(np.diag([1, -1, -1, 1]))
        >>> np.allclose(q, [0, 1, 0, 0]) or np.allclose(q, [0, -1, 0, 0])
        True
        >>> R = rotation_matrix(0.123, (1, 2, 3))
        >>> q = quaternion_from_matrix(R, True)
        >>> np.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
        True
        >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
        ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
        >>> q = quaternion_from_matrix(R)
        >>> np.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
        True
        >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
        ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
        >>> q = quaternion_from_matrix(R)
        >>> np.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
        True
        >>> R = random_rotation_matrix()
        >>> q = quaternion_from_matrix(R)
        >>> is_same_transform(R, quaternion_matrix(q))
        True

        """
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
        """Return homogeneous rotation matrix from quaternion.

        >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
        >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
        True
        >>> M = quaternion_matrix([1, 0, 0, 0])
        >>> np.allclose(M, np.identity(4))
        True
        >>> M = quaternion_matrix([0, 1, 0, 0])
        >>> np.allclose(M, np.diag([1, -1, -1, 1]))
        True

        """
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

        Note that many Euler angle triplets can describe one matrix.

        >>> R0 = euler_matrix(1, 2, 3, 'syxz')
        >>> al, be, ga = euler_from_matrix(R0, 'syxz')
        >>> R1 = euler_matrix(al, be, ga, 'syxz')
        >>> np.allclose(R0, R1)
        True
        >>> angles = (4*math.pi) * (np.random.random(3) - 0.5)
        >>> for axes in _AXES2TUPLE.keys():
        ...    R0 = euler_matrix(axes=axes, *angles)
        ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
        ...    if not np.allclose(R0, R1): print(axes, "failed")

        """
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
        """Return Euler angles from quaternion for specified axis sequence.

        >>> angles = euler_from_quaternion([0.99810947, 0.06146124, 0, 0])
        >>> np.allclose(angles, [0.123, 0, 0])
        True

        """
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

    def produce_output(mdExp, mode, Y, fnImages):
        ID = 0
        for objId in mdExp:
            if mode == "Shift":
                shiftX, shiftY = Y[ID]
                mdExp.setValue(xmippLib.MDL_SHIFT_X, float(shiftX), objId)
                mdExp.setValue(xmippLib.MDL_SHIFT_Y, float(shiftY), objId)
                mdExp.setValue(xmippLib.MDL_IMAGE, fnImages[ID], objId)
            elif mode == "Angular":
                angles = Y[ID]*180/math.pi
                mdExp.setValue(xmippLib.MDL_ANGLE_PSI, angles[2], objId)
                mdExp.setValue(xmippLib.MDL_ANGLE_ROT, angles[0], objId)
                mdExp.setValue(xmippLib.MDL_ANGLE_TILT, angles[1] + 90, objId)
            ID += 1

    def convert_to_cuaternions(rep6d):
        matrix = np.array(list(map(rotation6d_to_matrix, rep6d)))
        return np.array(list(map(quaternion_from_matrix, matrix)))

    def average_quaternions(quaternions):
        s = np.matmul(quaternions.T, quaternions)
        s /= len(quaternions)
        eigenValues, eigenVectors = np.linalg.eig(s)
        return np.real(eigenVectors[:, np.argmax(eigenValues)])

    def compute_averages(pred6d):
        quats = np.array(list(map(convert_to_cuaternions, pred6d)))
        av_quats = np.array(list(map(average_quaternions, quats)))
        av_euler = np.array(list(map(euler_from_quaternion, av_quats)))
        return av_euler


    Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnXmdExp)

    mdExp = xmippLib.MetaData(fnXmdExp)
    fnImgs = mdExp.getColumnValues(xmippLib.MDL_IMAGE)

    mdExpImages = xmippLib.MetaData(fnXmdImages)
    fnImages = mdExpImages.getColumnValues(xmippLib.MDL_IMAGE)

    start_time = time()

    ShiftModel = load_model(fnShiftModel, compile=False)
    ShiftModel.compile(loss="mean_squared_error", optimizer='adam')

    ShiftManager = DataGenerator(fnImgs, maxSize, Xdim, readInMemory=False)

    Y = ShiftModel.predict_generator(ShiftManager, ShiftManager.getNumberOfBlocks())

    produce_output(mdExp, 'Shift', Y, fnImages)

    if predictAngles == 'yes':
        predictions = np.zeros((len(fnImgs), numModels, 6))
        for index in range(numModels):
           AngModel = load_model(fnAngModel + str(index) + ".h5", compile=False)
           AngModel.compile(loss="mean_squared_error", optimizer='adam')
           AngManager = DataGenerator(fnImgs, maxSize, Xdim, readInMemory=False)
           predictions[:, index, :] = AngModel.predict_generator(AngManager, AngManager.getNumberOfBlocks())
        Y = compute_averages(predictions)
        produce_output(mdExp, 'Angular', Y, fnImages)

        # AngModel = load_model(fnAngModel, compile=False)
        # AngModel.compile(loss="mean_squared_error", optimizer='adam')
        # AngManager = DataGenerator(fnImgs, maxSize, Xdim, readInMemory=False)
        # Y = AngModel.predict_generator(AngManager, AngManager.getNumberOfBlocks())
        # produce_output(mdExp, 'Angular', Y, fnImages)

    mdExp.write(os.path.join(outputDir, "predict_results.xmd"))


    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)


