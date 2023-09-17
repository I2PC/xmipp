#!/usr/bin/env python3

import math
import numpy as np
from numpy.linalg import norm
import os
import sys
import xmippLib
from time import time
from scipy.spatial.transform import Rotation
from scipy.ndimage import shift

maxSize = 512

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

    import keras
    from keras.models import load_model

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
            _TUPLE2AXES[axes]
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
        """Return rotation matrix from 6D representation."""
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
        """Return Euler angles from rotation matrix"""
        r = Rotation.from_matrix(mat)
        angles = r.as_euler("xyz", degrees=True)
        return angles


    def produce_output(mdExp, Y, distance, fnImages):
        ID = 0
        for objId in mdExp:
            angles = Y[ID]
            mdExp.setValue(xmippLib.MDL_ANGLE_PSI, angles[2], objId)
            mdExp.setValue(xmippLib.MDL_ANGLE_ROT, angles[0], objId)
            mdExp.setValue(xmippLib.MDL_ANGLE_TILT, angles[1], objId)
            mdExp.setValue(xmippLib.MDL_IMAGE, fnImages[ID], objId)
            if distance[ID] > tolerance:
                mdExp.setValue(xmippLib.MDL_ENABLED, -1, objId)
            ID += 1


    def convert_to_matrix(rep6d):
        return np.array(list(map(rotation6d_to_matrix, rep6d)))


    def convert_to_quaternions(mat):
        return np.array(list(map(quaternion_from_matrix, mat)))


    def average_quaternions(quaternions):
        """Calculates the average quaternion from a set"""
        s = np.matmul(quaternions.T, quaternions)
        s /= len(quaternions)
        eigenValues, eigenVectors = np.linalg.eig(s)
        return np.real(eigenVectors[:, np.argmax(eigenValues)])


    def maximum_distance(av_mat, mat):
        """Max and argMax distance in angles from a set of rotation matrix"""
        c = mat * av_mat[np.newaxis, 0:3, 0:3]
        d = np.sum(c, axis=(1, 2))
        ang_Distances = np.arccos((d - 1) / 2)
        return np.max(ang_Distances), np.argmax(ang_Distances)


    def calculate_r6d(redundant):
        """Solves linear system to get 6D representation from 42 parameters output"""
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


    def R_rot(theta):
        return np.array([[1, 0, 0],
                         [0, math.cos(theta), -math.sin(theta)],
                         [0, math.sin(theta), math.cos(theta)]])


    def R_tilt(theta):
        return np.array([[math.cos(theta), 0, math.sin(theta)],
                         [0, 1, 0],
                         [-math.sin(theta), 0, math.cos(theta)]])


    def R_psi(theta):
        return np.array([[math.cos(theta), -math.sin(theta), 0],
                         [math.sin(theta), math.cos(theta), 0],
                         [0, 0, 1]])


    def euler_angles_to_matrix(angles):
        Rx = R_rot(angles[0])
        Ry = R_tilt(angles[1])
        Rz = R_psi(angles[2])
        return np.matmul(np.matmul(Rz, Ry), Rx)


    def average_of_rotations(p6d_redundant):
        """Consensus tool"""
        # Calculates average angle for each particle
        matrix = convert_to_matrix(p6d_redundant)
        # min number of models
        euler_angles = np.array(list(map(euler_from_matrix, matrix)))
        euler_angles[:, 0] = euler_angles[:, 0] / 4

        matrix = np.array(list(map(euler_angles_to_matrix, euler_angles)))

        euler_angles = np.array(list(map(euler_from_matrix, matrix)))
        euler_angles[:, 0] = euler_angles[:, 0]

        minModels = np.shape(matrix)[0] - maxModels
        quats = convert_to_quaternions(matrix)
        av_quats = average_quaternions(quats)
        av_matrix = quaternion_matrix(av_quats)
        # max and argmax distance of a prediction to the average
        max_distance, max_dif_model = maximum_distance(av_matrix, matrix)
        max_distance = max_distance * 180 / math.pi
        while (np.shape(matrix)[0] > minModels) and (max_distance > tolerance):
            # deletes predictions from the max_dif_model and recalculates averages
            matrix = np.delete(matrix, max_dif_model, axis=0)
            quats = np.delete(quats, max_dif_model, axis=0)
            av_quats = average_quaternions(quats)
            av_matrix = quaternion_matrix(av_quats)
            max_distance, max_dif_model = maximum_distance(av_matrix, matrix)
            max_distance = max_distance * 180 / math.pi
        av_euler = euler_from_matrix(av_matrix)
        return np.append(av_euler, max_distance)


    def compute_ang_averages(pred6d):
        """Calls consensus tool"""
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
    rots = mdExp.getColumnValues(xmippLib.MDL_ANGLE_ROT)
    tilts = mdExp.getColumnValues(xmippLib.MDL_ANGLE_TILT)
    psis = mdExp.getColumnValues(xmippLib.MDL_ANGLE_PSI)
    shifts = []
    euler_angles = []
    for r, t, p in zip(rots, tilts, psis):
        euler_angles.append(np.array([r, t, p]))

    for sX, sY in zip(shiftX, shiftY):
        shifts.append(np.array((sX, sY)))

    start_time = time()


    def shift_image(img, img_shifts):
        """Shift image to center particle"""
        return shift(img, (-img_shifts[0], -img_shifts[1], 0), order=1, mode='wrap')


    models = []
    for index in range(numAngModels):
        print(fnAngModel + "/modelAngular" + str(index) + ".h5")
        AngModel = load_model(fnAngModel + "/modelAngular" + str(index) + ".h5", compile=False)
        AngModel.compile(loss="mean_squared_error", optimizer='adam')
        models.append(AngModel)

    numImgs = len(fnImgs)
    predictions = np.zeros((numImgs, numAngModels, 6))
    numBatches = numImgs // maxSize
    if numImgs % maxSize > 0:
        numBatches = numBatches + 1
    k = 0
    # perform batch predictions for each model
    for i in range(numBatches):
        numPredictions = min(maxSize, numImgs - i * maxSize)
        Xexp = np.zeros((numPredictions, Xdim, Xdim, 1), dtype=np.float64)
        for j in range(numPredictions):
            Iexp = np.reshape(xmippLib.Image(fnImgs[k]).getData(), (Xdim, Xdim, 1))
            Xexp[j,] = (Iexp - np.mean(Iexp)) / np.std(Iexp)
            Xexp[j,] = shift_image(Xexp[j,], shifts[k])
            k += 1
        for index in range(numAngModels):
            predictions[i * maxSize:(i * maxSize + numPredictions), index, :] = models[index].predict(Xexp)

    Y, distance = compute_ang_averages(predictions)


    def euler_matrix(ai, aj, ak, axes='sxyz'):
        """Return homogeneous rotation matrix from Euler angles and axis sequence.

        ai, aj, ak : Euler's roll, pitch and yaw angles
        axes : One of 24 axis sequences as string or encoded tuple

        >>> R = euler_matrix(1, 2, 3, 'syxz')
        >>> np.allclose(np.sum(R[0]), -1.34786452)
        True
        >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
        >>> np.allclose(np.sum(R[0]), -0.383436184)
        True
        >>> ai, aj, ak = (4*math.pi) * (np.random.random(3) - 0.5)
        >>> for axes in _AXES2TUPLE.keys():
        ...    R = euler_matrix(ai, aj, ak, axes)
        >>> for axes in _TUPLE2AXES.keys():
        ...    R = euler_matrix(ai, aj, ak, axes)

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
        return M


    def euler_to_matrix(angles):
        return euler_matrix(angles[0] * math.pi / 180, angles[1] * math.pi / 180, angles[2] * math.pi / 180,
                            axes='szyz')[:3, :3]


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


    def matrix_to_euler(mat):
        return np.array(euler_from_matrix(mat, axes='szyz')) * 180 / math.pi


    SL = xmippLib.SymList()
    Matrices = np.array(SL.getSymmetryMatrices('o'))
    inverse_matrices = np.zeros_like(Matrices)
    for i in range(24):
        inverse_matrices[i] = np.transpose(Matrices[i])


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


    def move_to_fundamental_domain(input_matrix, inv_group_matrices):
        candidates = [[378]]
        max_tilt = -999
        for sym_matrix in inv_group_matrices:
            aux_mat = np.matmul(input_matrix, sym_matrix)
            aux_euler_angles = matrix_to_euler(aux_mat)
            aux_rot = aux_euler_angles[0]
            aux_tilt = aux_euler_angles[1]
            if (aux_rot >= 0) and (90 >= aux_rot):
                if aux_tilt >= max_tilt:
                    max_tilt = aux_tilt
                    candidates = aux_mat
        if candidates[0][0] == 378:
            print('LIADA', flush=True)

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

    def inv_factor_rot(rot_f, tilt_angle):
        dist = 2.2300e-07 * tilt_angle**4 + 5.8474e-05 * tilt_angle**3 + 5.6207e-03 * tilt_angle**2 - 8.1743e-02 * tilt_angle
        return rot_f*((90-dist)/90) + np.sign(rot_f)*2*dist

    print('factor', factor_rot(160, -60))
    print('invfactor', inv_factor_rot(factor_rot(160, -60), -60))

    y_matrices = np.array(list(map(euler_to_matrix, euler_angles)))
    y_m = y_matrices[:, :3, :3]
    S_inv = []

    for matrix in y_m:
        S_inv.append(move_to_fundamental_domain(matrix, inverse_matrices))

    # Now rotations are in a fundamental domain
    rot_order = 4
    y_transformedRot1 = np.array(list(map(lambda input_matrix: transform_rot(input_matrix, rot_order), S_inv)))

    angle = math.pi / 4
    matrix_axis = rodrigues_formula([0, 1, 0], angle)
    y_new_axis = []
    # Ahora cambio de eje Z:
    for matrix in y_transformedRot1:
        y_new_axis.append(np.matmul(matrix, matrix_axis))

    y_euler_new_axis = np.array(list(map(matrix_to_euler, y_new_axis)))
    min_tilt = -99
    print('min tilt', min_tilt)
    factor_tilt = -180 / min_tilt
    Y = y_euler_new_axis
    Y[:, 1] = factor_tilt * y_euler_new_axis[:, 1]

    Y[:, 0] = np.array(list(map(factor_rot, Y[:, 0], Y[:, 1])))

    Y[:, 0] = 2*(180+Y[:, 0])

    Y[:, 0] = Y[:, 0]/2 - 180

    Y[:, 0] = np.array(list(map(inv_factor_rot, Y[:, 0], Y[:, 1])))

    Y[:, 1] = Y[:, 1]/factor_tilt

    Matrix_euler_new_axis = np.array(list(map(euler_to_matrix, Y)))

    matrix_axis = rodrigues_formula([0, 1, 0], -angle)
    Matrix_original_axis = []
    # Ahora cambio de eje Z:
    for matrix in Matrix_euler_new_axis:
        Matrix_original_axis.append(np.matmul(matrix, matrix_axis))

    Y = np.array(list(map(matrix_to_euler, Matrix_original_axis)))

    Y[:, 0] = Y[:, 0]/4

    #arr = Y
#
    ## Define the division ranges
    #division_ranges = [(-5, 0), (-10, -5), (-15, -10), (-20, -15), (-25, -20), (-30, -25), (-35, -30), (-40, -35),
    #                   (-45, -40), (-50, -45), (-55, -50), (-60, -55), (-65, -60), (-70, -65), (-75, -70), (-80, -75),
    #                   (-85, -80), (-90, -85), (-95, -90), (-100, -95), (-105, -100), (-110, -105), (-115, -110),
    #                   (-120, -115), (-125, -120), (-130, -125), (-135, -130), (-140, -135), (-145, -140), (-150, -145),
    #                   (-155, -150), (-160, -155), (-165, -160), (-170, -165), (-175, -170), (-180, -175)]
#
    ## Calculate the minimum distances for each first element
    #min_distances = []
    #min_distances_args = []
    #conteo_elementos = 0
    #for division_range in division_ranges:
    #    lower, upper = division_range
    #    first_elements = arr[(arr[:, 1] >= lower) & (arr[:, 1] < upper)][:, 0]
    #    second_elements = arr[(arr[:, 1] >= lower) & (arr[:, 1] < upper)][:, 1]
    #    distances = np.min(np.array([abs(90 - first_elements), abs(-90 - first_elements)]), axis=0)
    #    print(len(distances))
    #    print('numero de elementos en la division', len(first_elements))
    #    conteo_elementos += len(first_elements)
    #    if len(first_elements) > 0:
    #        min_distances.append(np.min(distances))
    #        min_distances_args.append(second_elements[np.argmin(distances)])
    #    else:
    #        min_distances.append(np.inf)
    #print('conteo final', conteo_elementos)
#
    #print('distances', min_distances)
#
    #print('tilts', min_distances_args)

    produce_output(mdExp, Y, distance, fnImages)

    mdExp.write(os.path.join(outputDir, "predict_results.xmd"))

    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)
