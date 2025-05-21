#!/usr/bin/env python3
""""
**************************************************************************
*
* Authors:  Oier Lauzirika Zarrabeitia (olauzirika@cnb.csic.es)
*
* Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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

from typing import Tuple, Dict

import random
import numpy as np
import scipy.spatial
import scipy.optimize

from xmipp_base import XmippScript
import xmippLib


class ScriptCoordinateBackProjection(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Coordinate backprojection')
        ## params
        self.addParamsLine('-i <inputMetadata>          : Path to a metadata file with all the 2D coordinates')
        self.addParamsLine('--ts <tsMetadata>           : Path to a metadata file with the tilt-series information')
        self.addParamsLine('-o <outputMetadata>         : Path to a metadata with the reconstructed 3D coordinates')
        self.addParamsLine('-n <numberOfCoords=500>     : Number of reconstructed 3D coordinates')
        self.addParamsLine('--lr <learningRate=0.01>    : Learning rate')
        self.addParamsLine('--iter <iterations=1024>    : Number of iterations')
        self.addParamsLine('--box <x=1000> <y=1000>, <z=300> : Box size')
        
    def run(self):
        inputMetadataFn = self.getParam('-i')
        inputTsMetadataFn = self.getParam('--ts')
        outputMetadataFn = self.getParam('-o')
        nCoords = self.getIntParam('-n')
        nu = self.getFloatParam('--lr')
        nIter = self.getIntParam('--iter')
        boxSize = (
            self.getIntParam('--box', 0),
            self.getIntParam('--box', 1),
            self.getIntParam('--box', 2)
        )
        
        self.coordinateBackProjection(
            inputMetadataFn=inputTsMetadataFn,
            inputTsMetadataFn=inputTsMetadataFn,
            outputMetadataFn=outputMetadataFn,
            nCoords=nCoords,
            nu=nu,
            nIter=nIter,
            boxSize=boxSize
        )

    def readTiltSeriesProjectionTransforms(self, filename):
        result = dict()
        
        md = xmippLib.MetaData(filename)
        for objId in md:
            rot = md.getValue(xmippLib.MDL_ANGLE_ROT, objId) or 0.0
            tilt = md.getValue(xmippLib.MDL_ANGLE_TILT, objId) or 0.0
            psi = md.getValue(xmippLib.MDL_ANGLE_PSI, objId) or 0.0
            shiftX = md.getValue(xmippLib.MDL_SHIFTX, objId) or 0.0
            shiftY = md.getValue(xmippLib.MDL_SHIFTY, objId) or 0.0
            tiltId = md.getValue(xmippLib.MDL_TILT_ID, objId) or objId

            # TODO matrix
            matrix = np.zeros((2, 3))
            matrix[0, 0] = 1
            matrix[1, 1] = 1
            
            shift = (shiftX, shiftY)

            result[tiltId] = (matrix, shift)

        return result
    
    def readCoordinateProjections(self, filename):
        result = dict()
        
        md = xmippLib.MetaData(filename)
        for objId in md:
            x = md.getValue(xmippLib.MDL_X, objId)
            y = md.getValue(xmippLib.MDL_Y, objId)
            tiltId = md.getValue(xmippLib.MDL_TILT_ID, objId)
            
            coord2d = (x, y)
            if tiltId in result:
                result[tiltId].append(coord2d)
            else:
                result[tiltId] = [coord2d]
                
        for key in result.keys():
            result[key] = np.array(result[key])
        
        return result
    
    def pointMatching(self, x, y):
        cost_matrix = np.square(scipy.spatial.distance.cdist(x, y))
        return scipy.optimize.linear_sum_assignment(cost_matrix)
    
    def countContributions(self,
                           positions,
                           keys,
                           trasforms,
                           tiltSeriesCoordinates ):
        count = np.zeros(len(positions))

        for key in keys:
            matrix, shift = trasforms[key]
            coordinates2d = tiltSeriesCoordinates[key]
            coordinates2dProj = (matrix @ positions.T).T + shift
            
            _, positionIndices = self.pointMatching(
                coordinates2d, 
                coordinates2dProj
            )
            
            count[positionIndices] += 1

        return count
            
    def coordinateBackProjection(self,
                                 inputMetadataFn: str,
                                 inputTsMetadataFn: str,
                                 outputMetadataFn: str,
                                 nCoords: int,
                                 nu: float,
                                 nIter: int,
                                 boxSize: Tuple[int, int, int]):
        
        tiltSeriesCoordinates = self.readCoordinateProjections(inputMetadataFn)
        transforms = self.readTiltSeriesProjectionTransforms(inputTsMetadataFn)
        keys = set(tiltSeriesCoordinates.keys()) & set(transforms.keys())

        positions = np.random.uniform(
            low=-2*np.array(boxSize)/2,
            high=+2*np.array(boxSize)/2,
            size=(nCoords, 3)
        )

        for _ in range(nIter):
            key = random.choice(keys)
            
            matrix, shift = transforms[key]
            tiltCoordinates = tiltSeriesCoordinates[key]

            projections = (matrix @ positions.T).T + shift
            sampleIndices, positionIndices = self.pointMatching(
                tiltCoordinates, 
                projections
            )

            tiltCoordinates = tiltCoordinates[sampleIndices]
            projections = projections[positionIndices]

            residual2d = tiltCoordinates - projections
            residual3d = (matrix.T @ residual2d.T).T

            positions[positionIndices] += nu*residual3d

        counts = self.countContributions(
            positions, 
            keys, 
            transforms, 
            tiltSeriesCoordinates
        )
        
        resultMd = xmippLib.Metadata()
        for coordinate3d, score in zip(positions, counts):
            objId = resultMd.addObject()
            resultMd.setValue(xmippLib.MDL_X, float(coordinate3d[0]), objId)
            resultMd.setValue(xmippLib.MDL_Y, float(coordinate3d[1]), objId)
            resultMd.setValue(xmippLib.MDL_Z, float(coordinate3d[2]), objId)
            resultMd.setValue(xmippLib.MDL_SCORE, float(score), objId)
        
        resultMd.write(outputMetadataFn)
        
if __name__=="__main__":
    ScriptCoordinateBackProjection().tryRun()
