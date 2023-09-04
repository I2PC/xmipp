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
        return M

    def euler_from_matrix(matrix, axes='szyz'):
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

    def euler_from_quaternion(quaternion: object, axes: object = 'szyz') -> object:
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

    def produce_output(mdExp, Y, distance, fnImages, regions):
        ID = 0
        for objId in mdExp:
            angles = Y[ID] * 180 / math.pi
            mdExp.setValue(xmippLib.MDL_ANGLE_PSI, angles[2], objId)
            mdExp.setValue(xmippLib.MDL_ANGLE_ROT, angles[0], objId)
            mdExp.setValue(xmippLib.MDL_ANGLE_TILT, angles[1], objId)
            mdExp.setValue(xmippLib.MDL_IMAGE, fnImages[ID], objId)
            if regions[ID] != 1:
                mdExp.setValue(xmippLib.MDL_ENABLED, -1, objId)
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


    def average_of_rotations(p6d_redundant):
        """Consensus tool"""
        # Calculates average angle for each particle
        #pred6d = calculate_r6d(p6d_redundant)
        #matrix = convert_to_matrix(pred6d)
        matrix = convert_to_matrix(p6d_redundant)
        # min number of models
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


    #############################################################33

    SL = xmippLib.SymList()
    Matrices = np.array(SL.getSymmetryMatrices('o'))



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
        if not any(np.allclose(sqrt1, m, atol=1e-5) for m in sqrt_matrices):
            sqrt_matrices.append(sqrt1)
        if not any(np.allclose(sqrt2, m, atol=1e-5) for m in sqrt_matrices):
            sqrt_matrices.append(sqrt2)


    num_sqrt_matrices = np.shape(sqrt_matrices)[0]
    num_matrices = np.shape(Matrices)[0]
    inverse_sqrt_matrices = np.zeros_like(sqrt_matrices)
    for i in range(num_sqrt_matrices):
        inverse_sqrt_matrices[i] = np.linalg.inv(sqrt_matrices[i])

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

    Xdims, fnImgs, labels, zones, shifts, map_regions, regions_map = get_labels(fnXmdExp)

    for sublist in regions_map:
        print(len(sublist))

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

    dominio = 1
    def shift_image(img, img_shifts):
        """Shift image to center particle"""
        return shift(img, (-img_shifts[0], -img_shifts[1], 0), order=1, mode='wrap')

    models = [[] for _ in range(numAngModels)]
    for index in range(numAngModels):
        path_to_model = fnAngModel + '/modelAngular' + '/model' + str(index)
        print(path_to_model + '/classifier' + ".h5")
        AngModel = load_model(path_to_model + '/classifier' + ".h5", compile=False)
        AngModel.compile(loss="mean_squared_error", optimizer='adam')
        models[index].append(AngModel)
        for i in [0, 1, 2, 3]:
            AngModel = load_model(path_to_model + '/modelAng' + str(i) + ".h5", compile=False)
            AngModel.compile(loss="mean_squared_error", optimizer='adam')
            models[index].append(AngModel)
    print('models loaded')
    numImgs = len(fnImgs)
    predictions = np.random.uniform(size=(numImgs, numAngModels, 6))
    numBatches = numImgs // maxSize
    if numImgs % maxSize > 0:
        numBatches = numBatches + 1
    k = 0
    mdExp = xmippLib.MetaData(fnXmdExp)
    rots = mdExp.getColumnValues(xmippLib.MDL_ANGLE_ROT)
    # perform batch predictions for each model
    numClasses = len(models[index]) - 1
    fallosClasificadorDominio = 0
    for i in range(numBatches):
        numPredictions = min(maxSize, numImgs-i*maxSize)
        print('predicting %', 100 * (maxSize * i + numPredictions) / numImgs, flush=True)
        Xexp = np.zeros((numPredictions, Xdim, Xdim, 1), dtype=np.float64)
        realClasses = np.zeros(numPredictions)
        for j in range(numPredictions):
            Iexp = np.reshape(xmippLib.Image(fnImgs[k]).getData(), (Xdim, Xdim, 1))
            Xexp[j, ] = (Iexp - np.mean(Iexp)) / np.std(Iexp)
            Xexp[j, ] = shift_image(Xexp[j, ], shifts[k])
            realClasses[j] = map_regions[k]
            k += 1
        classes_predictions = np.zeros(shape=(numPredictions, numClasses))

        for index in range(numAngModels):
            classes_predictions += models[index][0].predict(Xexp)
            classes = np.argmax(classes_predictions, axis=1)
        for t in range(numPredictions):
            if realClasses[t] == dominio:
                if classes[t] != dominio:
                    fallosClasificadorDominio += 1
        for index in range(numAngModels):
            for class_index in range(numClasses):
                condicion = [a and b for a, b in zip(classes == class_index, realClasses == dominio)]
                indices = np.where(condicion)[0]
                #indices = np.where(classes == class_index)[0]
                Xexp_class = Xexp[indices]
                indices += i * maxSize
                if len(indices) >= 1:
                    predictions[indices, index, :] = models[index][class_index + 1].predict(Xexp_class)

    print('fallos Dominio', fallosClasificadorDominio)

    Y, distance = compute_ang_averages(predictions)
    produce_output(mdExp, Y, distance, fnImages, map_regions)
    mdExp.write(os.path.join(outputDir, "predict_results.xmd"))

    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)
