#!/usr/bin/env python3

"""
**************************************************************************
*
* Authors:  Oier Lauzirika Zarrabeitia (olauzirika@cnb.csic.es)
*
* Unidad de  Bioinformatica of Centro Nacional de Biotecnologia, CSIC
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
* 02111-1307  USA
*
*  All comments concerning this program package may be sent to the
*  e-mail address 'scipion@cnb.csic.es'
*
**************************************************************************
"""

from typing import Tuple, List, NamedTuple, Iterable, Optional
import numpy as np

from xmipp_base import XmippScript
import xmippLib

class TiltData(NamedTuple):
    projectionMatrix: np.ndarray
    shift: np.ndarray
    coordinates2d: np.ndarray

def _computeProjectionDeltas(positions: np.ndarray,
                             data: Iterable[TiltData],
                             outProjected: Optional[np.ndarray] = None,
                             outBackprojected: Optional[np.ndarray] = None) -> np.ndarray:
    n = 0
    for tilt in data:
        n += len(tilt.coordinates2d)
    k = len(positions)
    
    if outProjected is None:
        outProjected = np.empty((n, k, 2))
    elif outProjected.shape != (n, k, 2):
        raise ValueError(f"outProjected must have shape ({n}, {k}, 2), got {outProjected.shape}")
    
    if outBackprojected is None:
        outBackprojected = np.empty((n, k, 3))
    elif outBackprojected.shape != (n, k, 3):
        raise ValueError(f"outBackprojected must have shape ({n}, {k}, 3), got {outBackprojected.shape}")
        
    start = 0
    for tilt in data:
        end = start + len(tilt.coordinates2d)

        projectionMatrix = tilt.projectionMatrix
        detections = tilt.coordinates2d
        projections = (projectionMatrix @ positions.T).T
        np.subtract(
            detections[:,None], 
            projections[None,:], 
            out=outProjected[start:end]
        )

        projectionMatrix2 = projectionMatrix.T @ projectionMatrix
        projections2 = (projectionMatrix2 @ positions.T).T
        detections2 = (projectionMatrix.T @ detections.T).T
        np.subtract(
            detections2[:,None],
            projections2[None,:],
            out=outBackprojected[start:end]
        )
            
        start = end

    assert start == n, "Not all deltas have been computed"
    return outProjected, outBackprojected
    
def _computeGmmResponsibilities(deltas: np.ndarray,
                                sigma2: np.ndarray,
                                weights: np.ndarray,
                                returnLogLikelihood: bool = False,
                                out: Optional[np.ndarray] = None) -> np.ndarray:

    _, _, D = deltas.shape
    LOG2PI = np.log(2*np.pi)

    distance2 = np.sum(np.square(deltas), axis=2, out=out)
    distance2 *= -0.5 / sigma2
    exponent = distance2 # Alias

    logSigma2 = np.log(sigma2)
    logWeights = np.log(weights)
    logCoefficient = -0.5*D*(LOG2PI + logSigma2)
    logMantissa = logWeights + logCoefficient

    logProbabilities = exponent # Alias
    logProbabilities += logMantissa
    
    logNorm = np.logaddexp.reduce(logProbabilities, axis=1, keepdims=True)
    logProbabilities -= logNorm

    gamma = logProbabilities # Alias
    np.exp(gamma, out=gamma)
    if returnLogLikelihood:
        logLikelihood = np.sum(logNorm)
        return gamma, logLikelihood
    else:
        return gamma



class ScriptCoordinateBackProjection(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Coordinate backprojection')
        self.addParamsLine('-i <inputMetadata>          : Path to a metadata file with all the 2D coordinates')
        self.addParamsLine('-a <tsMetadata>             : Path to a metadata file with the tilt-series alignment information')
        self.addParamsLine('-o <outputMetadata>         : Path to a metadata with the reconstructed 3D coordinates')
        self.addParamsLine('-n <numberOfCoords>         : Number of reconstructed 3D coordinates')
        self.addParamsLine('--box <x> <y> <z>           : Box size')
        self.addParamsLine('[--sigma <noise=8.0>]       : Picking noise std deviation estimation')
        
    def run(self):
        inputMetadataFn = self.getParam('-i')
        inputTsAliMetadataFn = self.getParam('-a')
        outputMetadataFn = self.getParam('-o')
        nCoords = self.getIntParam('-n')
        boxSize = (
            self.getIntParam('--box', 0),
            self.getIntParam('--box', 1),
            self.getIntParam('--box', 2)
        )
        sigma = self.getDoubleParam('--sigma')
        
        
        tiltSeriesCoordinates = self.readTiltSeriesCoordinates(inputMetadataFn)
        transforms = self.readTiltSeriesProjectionTransforms(inputTsAliMetadataFn)
        tiltIds = set(tiltSeriesCoordinates.keys()) & set(transforms.keys())
        
        data = []
        for tiltId in tiltIds:
            matrix, shift = transforms[tiltId]
            coordinates2d = np.array(tiltSeriesCoordinates[tiltId])
            data.append(TiltData(matrix, shift, coordinates2d))
        
        positions, effectiveCounts = self.coordinateBackProjection(
            data=data,
            nCoords=nCoords,
            boxSize=boxSize,
            sigma=sigma
        )
        
        outputMd = xmippLib.MetaData()
        for position, effectiveCount in zip(positions, effectiveCounts):
            x, y, z = position
            objId = outputMd.addObject()
            outputMd.setValue(xmippLib.MDL_X, x, objId)
            outputMd.setValue(xmippLib.MDL_Y, y, objId)
            outputMd.setValue(xmippLib.MDL_Z, z, objId)
            outputMd.setValue(xmippLib.MDL_LL, float(effectiveCount), objId)
        outputMd.write(outputMetadataFn)
        
    def readTiltSeriesProjectionTransforms(self, filename):
        result = dict()
        
        md = xmippLib.MetaData(filename)
        for objId in md:
            #rot = np.radians(md.getValue(xmippLib.MDL_ANGLE_ROT, objId) or 0.0)
            tilt = np.radians(md.getValue(xmippLib.MDL_ANGLE_TILT, objId) or 0.0)
            #psi = np.radians(md.getValue(xmippLib.MDL_ANGLE_PSI, objId) or 0.0)
            shiftX = md.getValue(xmippLib.MDL_SHIFT_X, objId) or 0.0
            shiftY = md.getValue(xmippLib.MDL_SHIFT_Y, objId) or 0.0
            tiltId = md.getValue(xmippLib.MDL_IMAGE_IDX, objId) or objId

            # TODO consider tilt and psi
            matrix = np.zeros((2, 3))
            matrix[1, 1] = 1
            matrix[0, 0] = np.cos(tilt)
            matrix[0, 2] = np.sin(tilt)

            shift = np.array((shiftX, shiftY))

            result[tiltId] = (matrix, shift)

        return result
    
    def readTiltSeriesCoordinates(self, filename):
        result = dict()
        
        md = xmippLib.MetaData(filename)
        for objId in md:
            x = md.getValue(xmippLib.MDL_X, objId)
            y = md.getValue(xmippLib.MDL_Y, objId)
            tiltId = md.getValue(xmippLib.MDL_IMAGE_IDX, objId)
            
            coord2d = (x, y)
            if tiltId in result:
                result[tiltId].append(coord2d)
            else:
                result[tiltId] = [coord2d]
                
        return result
    
    def coordinateBackProjection(self,
                                 data: List[TiltData], 
                                 nCoords: int,
                                 sigma: float,
                                 boxSize: Tuple[int, int, int] ) -> Tuple[np.ndarray, np.ndarray]:
        EPS = 1e-12
        
        boundary = np.array(boxSize) / 2
        positions = np.random.uniform(
            low=-boundary,
            high=boundary,
            size=(nCoords, 3)
        )

        weights = np.full(nCoords, 1/nCoords)
        sigma2 = np.full(nCoords, np.square(sigma))

        responsibilities = None
        deltas = None
        backprojectedDeltas = None
        oldLogLikelihood = -np.inf
        while True:
            deltas, backprojectedDeltas = _computeProjectionDeltas(
                positions, 
                data, 
                outProjected=deltas,
                outBackprojected=backprojectedDeltas
            )
            
            responsibilities, logLikelihood = _computeGmmResponsibilities(
                deltas, sigma2, weights, 
                returnLogLikelihood=True,
                out=responsibilities
            )
            
            n = np.sum(responsibilities, axis=0)
            backprojectedDeltas *= responsibilities[:,:,None]
            positions += np.sum(backprojectedDeltas, axis=0) / np.maximum(n[:,None], EPS)
            positions = np.clip(positions, -boundary, boundary)
            
            weights = n / responsibilities.shape[0]
            weights = np.maximum(weights, EPS)

            if oldLogLikelihood + 0.1 > logLikelihood:
                break # No improvement
            oldLogLikelihood = logLikelihood
            
        return positions, n
        
if __name__=="__main__":
    ScriptCoordinateBackProjection().tryRun()
