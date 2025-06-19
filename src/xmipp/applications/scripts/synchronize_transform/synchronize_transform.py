#!/usr/bin/env python

# ***************************************************************************
# * Authors:    Jose Luis Vilas Prieto (jlvilas@cnb.csic.es) 
#               Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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
# *  e-mail address 'xmipp@cnb.csic.es'
# ***************************************************************************/

from typing import List, Optional, Tuple
    
from xmipp_base import XmippScript
import xmippLib
import itertools
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
    


class ScriptSynchronizeTransform(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addParamsLine(' -i <pairwise>          : Pairwise alignment metadata')
        self.addParamsLine(' -o <alignment>         : Output alignment metadata')

    def run(self):
        inputFn = self.getParam('-i')
        outputFn = self.getParam('-o')

        inputMd = xmippLib.MetaData(inputFn)
        pairs, rotations, shifts, correlations = self._readPairwiseAlignments(inputMd)
        ids, indices = np.unique(pairs, return_inverse=True)
        indices = indices.reshape(pairs.shape)
        
        n = inputMd.size()
        synchronizedRotations = self._synchronizeRotations(indices, n, rotations, correlations)
        synchronizedShifts, _ = self._synchronizeShifts(indices, n, synchronizedRotations, shifts)
            
        outputMd = self._writeAlignments(ids, synchronizedRotations, synchronizedShifts)
        outputMd.write(outputFn)
            
    def _readPairwiseAlignments(self, md: xmippLib.MetaData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = md.size()
        ids = np.empty((n, 2), dtype=np.int64)
        rotations = np.empty((n, 3, 3))
        shifts = np.empty((n, 3))
        correlations = np.empty(n)
        
        for i, objId in enumerate(md):
            id0 = md.getValue(xmippLib.MDL_REF, objId)
            id1 = md.getValue(xmippLib.MDL_IDX, objId)
            rot = md.getValue(xmippLib.MDL_ANGLE_ROT, objId)
            tilt = md.getValue(xmippLib.MDL_ANGLE_TILT, objId)
            psi = md.getValue(xmippLib.MDL_ANGLE_PSI, objId)
            x = md.getValue(xmippLib.MDL_SHIFT_X, objId)
            y = md.getValue(xmippLib.MDL_SHIFT_Y, objId)
            z = md.getValue(xmippLib.MDL_SHIFT_Z, objId)
            corr = md.getValue(xmippLib.MDL_CORRELATION_IDX, objId)
            
            rotation = xmippLib.Euler_angles2matrix(rot, tilt, psi)
            shift =  -(rotation.T @ np.array([x, y, z])) # Determine why
            
            ids[i] = (id0, id1)
            rotations[i] = rotation
            shifts[i] = shift
            correlations[i] = corr
    
        return ids, rotations, shifts, correlations
    
    def _writeAlignments(self, ids: np.ndarray, rotations: np.ndarray, shifts: np.ndarray) -> xmippLib.MetaData:
        md = xmippLib.MetaData()
        
        for itemId, rotation, shift in zip(ids, rotations, shifts):
            rot, tilt, psi = xmippLib.Euler_matrix2angles(rotation)
            x, y, z = shift
            
            objId = md.addObject()
            md.setValue(xmippLib.MDL_ITEM_ID, int(itemId), objId)
            md.setValue(xmippLib.MDL_ANGLE_ROT, float(rot), objId)
            md.setValue(xmippLib.MDL_ANGLE_TILT, float(tilt), objId)
            md.setValue(xmippLib.MDL_ANGLE_PSI, float(psi), objId)
            md.setValue(xmippLib.MDL_SHIFT_X, float(x), objId)
            md.setValue(xmippLib.MDL_SHIFT_Y, float(y), objId)
            md.setValue(xmippLib.MDL_SHIFT_Z, float(z), objId)
            
        return md
    
    def _synchronizeRotations(self, indices: np.ndarray, n: int, rotations: np.ndarray, correlations: np.ndarray) -> np.ndarray:
        pairwise = scipy.sparse.lil_array((3*n, )*2)
        for (index0, index1), rotation, correlation in zip(indices, rotations, correlations):
            start0 = 3*index0
            end0 = start0 + 3
            start1 = 3*index1
            end1 = start1 + 3
            
            rotation = correlation*rotation
            pairwise[start0:end0, start1:end1] = rotation
            pairwise[start1:end1, start0:end0] = rotation.T
        pairwise = pairwise.tocsr()

        result = np.random.randn(n*3, 3)
        result, _ = np.linalg.qr(result, mode='reduced')
        result = result.reshape(n, 3, 3)

        MAX_ITER = 1024
        EPS = 1e-6
        lastObjective = -np.inf
        for _ in range(MAX_ITER):
            x = result.reshape(-1, 3)
            y = pairwise @ x
            objective = np.dot(x.ravel(), y.ravel()) # tr(x.T @ y)
            
            improvement = objective - lastObjective
            if improvement < EPS:
                break

            u, _, vt = np.linalg.svd(y.reshape(result.shape), full_matrices=False)
            result = u @ vt
            
            sign = np.linalg.det(result)
            result[:,:,2] *= sign[:,None]
            
            lastObjective = objective
        
        errors = []
        for (index0, index1), rotation in zip(indices, rotations):
            delta = result[index0].T @ rotation @ result[index1]
            error = np.degrees(np.arccos((np.trace(delta) - 1) / 2))
            errors.append(error)

        plt.scatter(correlations, errors)
        plt.show()
        
        plt.hist(correlations)
        plt.show()
        
        return result

    def _synchronizeShifts(self, indices: np.ndarray, n: int, synchronizedRotations: np.ndarray, shifts: np.ndarray) -> np.ndarray:
        nCols = 3*n
        nRows = 3*len(indices)

        desing = scipy.sparse.lil_array((nRows, nCols))
        y = np.empty(nRows)
        for i, (index0, index1), shift in zip(itertools.count(), indices, shifts):
            startRow = 3*i
            endRow = startRow + 3
            start0 = 3*index0
            end0 = start0 + 3
            start1 = 3*index1
            end1 = start1 + 3
            
            desing[startRow:endRow,start0:end0] = -synchronizedRotations[index1] @ synchronizedRotations[index0].T
            desing[startRow:endRow,start1:end1] = np.eye(3)
            y[startRow:endRow] = shift
        desing = desing.tocoo()
        
        solution = scipy.sparse.linalg.lsqr(desing, y)
        x = solution[0].reshape((n, 3))
        err = solution[3] / n
        
        return x, (err / n)
    
if __name__ == '__main__':
    ScriptSynchronizeTransform().tryRun()
