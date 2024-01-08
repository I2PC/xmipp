# ***************************************************************************
# *
# * Authors:     David Maluenda (dmaluenda@cnb.csic.es)
# *
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# ****************************************************************************
""" This is an example of a module which can be imported by any Xmipp script.
    It will be installed at build/pylib/xmippPyModules.
"""

import math
import numpy as np
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation
import xmippLib

def applyTransformNP(img, rot, x, y):
    "Transform a numpy array with rot, x and y"
    theta = rot*math.pi/180.0
    c, s = math.cos(theta), math.sin(theta)
    M = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])
    return affine_transform(img, M, order=1, mode='reflect', output_shape=img.shape)

class RotationHandler:
    """ This class can convert from quaternions to any Euler convention and viceversa.
        It can also compute the average of several quaternions and calculate the distance of the
        different quaternions to the average."""
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

    @staticmethod
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

    @staticmethod
    def quaternion_matrix(quaternion):
        """Return homogeneous rotation matrix from quaternion."""
        q = np.array(quaternion, dtype=np.float64, copy=True)
        n = np.dot(q, q)
        if n < RotationHandler._EPS:
            return np.identity(4)
        q *= math.sqrt(2.0 / n)
        q = np.outer(q, q)
        return np.array([
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]])

    @staticmethod
    def euler_from_matrix(matrix, axes='szyz'):
        """Return Euler angles from rotation matrix for specified axis sequence.

        axes : One of 24 axis sequences as string or encoded tuple

        Note that many Euler angle triplets can describe one matrix."""
        try:
            firstaxis, parity, repetition, frame = RotationHandler._AXES2TUPLE[axes.lower()]
        except (AttributeError, KeyError):
            RotationHandler._TUPLE2AXES[axes]
            firstaxis, parity, repetition, frame = axes

        i = firstaxis
        j = RotationHandler._NEXT_AXIS[i + parity]
        k = RotationHandler._NEXT_AXIS[i - parity + 1]

        M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
        if repetition:
            sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
            if sy > RotationHandler._EPS:
                ax = math.atan2(M[i, j], M[i, k])
                ay = math.atan2(sy, M[i, i])
                az = math.atan2(M[j, i], -M[k, i])
            else:
                ax = math.atan2(-M[j, k], M[j, j])
                ay = math.atan2(sy, M[i, i])
                az = 0.0
        else:
            cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
            if cy > RotationHandler._EPS:
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
        return -ax, -ay, -az

    @staticmethod
    def euler_from_quaternion(quaternion, axes='szyz'):
        """Return Euler angles from quaternion for specified axis sequence."""
        return euler_from_matrix(quaternion_matrix(quaternion), axes)

    @staticmethod
    def average_quaternions(quaternions):
        """Calculates the average quaternion from a list of quaternions.
        The input must be a np.array 2D matrix where each row is one quaternion"""
        s = np.matmul(quaternions.T, quaternions)
        s /= len(quaternions)
        eigenValues, eigenVectors = np.linalg.eig(s)
        return np.real(eigenVectors[:, np.argmax(eigenValues)])

    @staticmethod
    def quaternion_distance(quaternions: np.ndarray,
                            quaternion: np.ndarray) -> np.ndarray:
        dots = np.dot(quaternions, quaternion)
        arg = np.clip(2 * (dots ** 2) - 1,-1,1)
        return np.arccos(arg)

    @staticmethod
    def maximum_distance(av_mat, mat):
        """Max and argMax distance in angles from a set of rotation matrix"""
        c = mat * av_mat[np.newaxis, 0:3, 0:3]
        d = np.sum(c, axis=(1, 2))
        arg = np.clip((d - 1) / 2,-1,1)
        ang_Distances = np.arccos(arg)
        return np.max(ang_Distances), np.argmax(ang_Distances)

def getRotationMatrix(n):
    theta = np.radians(360.0/n)  # Convert angle to radians
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                [sin_theta, cos_theta, 0],
                                [0, 0, 1]])
    return rotation_matrix

class RotationAverager():
    def __init__(self, angleList):
        self.angleList = angleList # List of list of list of angles. axis=0: image number, axis=1 (rot, tilt, psi)

    def bringToAsymmetricUnit(self, symmetry):
        SL=xmippLib.SymList()
        SL.readSymmetryFile(symmetry)
        Nangles = len(self.angleList)
        for i in range(len(self.angleList[0])): # Loop over images
            rot1  = self.angleList[0][i][0]
            tilt1 = self.angleList[0][i][1]
            psi1  = self.angleList[0][i][2]
            for j in range(1,Nangles):
                rot2  = self.angleList[j][i][0]
                tilt2 = self.angleList[j][i][1]
                psi2  = self.angleList[j][i][2]
                self.angleList[j][i] = SL.computeClosestSymmetricAngles(rot1, tilt1, psi1, rot2, tilt2, psi2)

    def computeAverageAssignment(self):
        avgAngles = []
        RH=RotationHandler()
        Nangles = len(self.angleList)
        mask_size = Nangles // 2 + 1

        for i in range(len(self.angleList[0])): # Loop over images
            quaternions = np.empty((Nangles, 4))
            for j in range(Nangles):
                rot = self.angleList[j][i][0]
                tilt = self.angleList[j][i][1]
                psi = self.angleList[j][i][2]
                quaternions[j, :] = RH.quaternion_from_matrix(xmippLib.Euler_angles2matrix(rot, tilt, psi))
            avgQuaternion = RH.average_quaternions(quaternions)
            angleDiff = RH.quaternion_distance(quaternions, avgQuaternion)

            sorted_indices = np.argsort(angleDiff)
            selected_indices = sorted_indices[:mask_size]
            avgQuaternion = RH.average_quaternions(quaternions[selected_indices,:])
            rot, tilt, psi = xmippLib.Euler_matrix2angles(RH.quaternion_matrix(avgQuaternion))
            avgAngles.append((rot,tilt,psi))
        return avgAngles

try:
    from keras.utils.all_utils import Sequence

    class XmippTrainingSequence(Sequence):
        def __init__(self, x_set, y_set, batch_size, maxSize=64):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size
            self.maxSize = maxSize if maxSize is not None else len(self.x)
            self.maxSize = min(self.maxSize, len(self.x))

        def __len__(self):
            return int(np.ceil(min(self.maxSize, len(self.x)) / float(self.batch_size)))

        def __getitem__(self, idx):
            start_idx = idx * self.batch_size
            end_idx = min((idx + 1) * self.batch_size, self.maxSize)
            batch_x = self.x[start_idx:end_idx]
            batch_y = self.y[start_idx:end_idx]

            return batch_x, batch_y

        def increaseMaxSize(self, K):
            new_max_size = int(self.maxSize * K)
            self.maxSize = min(new_max_size, len(self.x))

        def isMaxSize(self):
            return self.maxSize == len(self.x)

        def shuffle_data(self):
            # Generate shuffled indices
            indices = np.arange(len(self.x))
            np.random.shuffle(indices)

            # Reorder X and Y according to the shuffled indices
            self.x = self.x[indices]
            self.y = self.y[indices]

    KERAS_INSTALLED = True
except:
    KERAS_INSTALLED = False