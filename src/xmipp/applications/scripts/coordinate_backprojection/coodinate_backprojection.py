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

from typing import Tuple, List, NamedTuple

import random
import numpy as np
import scipy.spatial
import scipy.optimize

from xmipp_base import XmippScript
import xmippLib

class TiltData(NamedTuple):
    projectionMatrix: np.ndarray
    shift: np.ndarray
    coordinates2d: np.ndarray

class ScriptCoordinateBackProjection(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Coordinate backprojection')
        self.addParamsLine('-i <inputMetadata>          : Path to a metadata file with all the 2D coordinates')
        self.addParamsLine('-a <tsMetadata>             : Path to a metadata file with the tilt-series alignment information')
        self.addParamsLine('-o <outputMetadata>         : Path to a metadata with the reconstructed 3D coordinates')
        self.addParamsLine('-n <numberOfCoords=1000>    : Number of reconstructed 3D coordinates')
        self.addParamsLine('--lr <learningRate=0.3>     : Learning rate')
        self.addParamsLine('--batch <miniBatchSize=8>   : Number of tilts used in minibatches')
        self.addParamsLine('--box <x=1000> <y=1000> <z=300> : Box size')
        self.addParamsLine('[--loss <outputLossFunction>] : Path where loss function is written')
        
    def run(self):
        inputMetadataFn = self.getParam('-i')
        inputTsAliMetadataFn = self.getParam('-a')
        outputMetadataFn = self.getParam('-o')
        nCoords = self.getIntParam('-n')
        lr = self.getDoubleParam('--lr')
        miniBatchSize = self.getIntParam('--batch')
        boxSize = (
            self.getIntParam('--box', 0),
            self.getIntParam('--box', 1),
            self.getIntParam('--box', 2)
        )
        lossFn = None
        if self.checkParam('--loss'):
            lossFn = self.getParam('--loss')
        
        
        tiltSeriesCoordinates = self.readTiltSeriesCoordinates(inputMetadataFn)
        transforms = self.readTiltSeriesProjectionTransforms(inputTsAliMetadataFn)
        tiltIds = set(tiltSeriesCoordinates.keys()) & set(transforms.keys())
        
        data = []
        for tiltId in tiltIds:
            matrix, shift = transforms[tiltId]
            coordinates2d = np.array(tiltSeriesCoordinates[tiltId])
            data.append(TiltData(matrix, shift, coordinates2d))
        
        positions, count, losses = self.coordinateBackProjection(
            data=data,
            nCoords=nCoords,
            nBatch=miniBatchSize,
            lr=lr,
            boxSize=boxSize
        )
        
        outputMd = xmippLib.MetaData()
        for position, count in zip(positions, count):
            x, y, z = position
            objId = outputMd.addObject()
            outputMd.setValue(xmippLib.MDL_X, x, objId)
            outputMd.setValue(xmippLib.MDL_Y, y, objId)
            outputMd.setValue(xmippLib.MDL_Z, z, objId)
            outputMd.setValue(xmippLib.MDL_COUNT, int(count), objId)
        outputMd.write(outputMetadataFn)

        if lossFn is not None:
            np.save(lossFn, np.array(losses))
        
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
    
    def pointMatching(self, x, y):
        cost_matrix = np.square(scipy.spatial.distance.cdist(x, y))
        return scipy.optimize.linear_sum_assignment(cost_matrix)
    
    def countContributions(self,
                           positions: np.ndarray,
                           data: List[TiltData] ):
        count = np.zeros(len(positions), dtype=np.int64)

        for tilt in data:
            projectionMatrix = tilt.projectionMatrix
            coordinates2d = tilt.coordinates2d
            coordinates2dProj = (projectionMatrix @ positions.T).T
            
            _, positionIndices = self.pointMatching(
                coordinates2d, 
                coordinates2dProj
            )
            
            count[positionIndices] += 1

        return count
            
    def coordinateBackProjection(self,
                                 data: List[TiltData], 
                                 nCoords: int,
                                 nBatch: int,
                                 lr: float,
                                 boxSize: Tuple[int, int, int],
                                 patience: int = 1024 ) -> Tuple[np.ndarray, np.ndarray]:
        
        positions = np.random.uniform(
            low=np.array(boxSize)/2,
            high=np.array(boxSize)/2,
            size=(nCoords, 3)
        )

        iterationsSinceLastImprovement = 0
        bestLoss = np.inf
        losses = []
        while iterationsSinceLastImprovement < patience:
            batch = random.sample(data, nBatch)
            gradient = np.zeros_like(positions)
            loss = 0.0
            count = 0
            for tilt in batch:
                coordinates = tilt.coordinates2d
                projectionMatrix = tilt.projectionMatrix

                projections = (projectionMatrix @ positions.T).T
                sampleIndices, positionIndices = self.pointMatching(coordinates, projections)

                coordinates = coordinates[sampleIndices]
                projections = projections[positionIndices]

                residual2d = coordinates - projections
                residual3d = (projectionMatrix.T @ residual2d.T).T

                gradient[positionIndices] += residual3d
                loss += np.sum(np.square(residual2d))
                count += len(positionIndices)

            loss /= count
            gradient /= len(batch)
            losses.append(loss)

            positions += lr*gradient

            if loss < bestLoss:
                bestLoss = loss
                iterationsSinceLastImprovement = 0
            else:
                iterationsSinceLastImprovement += 1

        count = self.countContributions(positions, data)
        return positions, count, losses
        
if __name__=="__main__":
    ScriptCoordinateBackProjection().tryRun()
