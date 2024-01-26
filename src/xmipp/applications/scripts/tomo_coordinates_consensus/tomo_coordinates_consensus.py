#!/usr/bin/env python3
# **************************************************************************
# *
# * Authors:    Mikel Iceta Tena (miceta@cnb.csic.es)
# *
# * Unidad de Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
# *  e-mail address 'coss@cnb.csic.es'
# *
# *
# * Neural Network utils for the deep picking consensus protocol found in
# * scipion-em-xmipptomo
# *
# * Initial release: june 2023
# **************************************************************************

import sys

import numpy as np
from typing import NamedTuple
import xmippLib
from xmipp_base import XmippScript

COORD_CONS_CENTROID = 1

def distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a-b)

class ScriptCoordsConsensusTomo(XmippScript):
    inputFile : str
    outputFile : str
    outputFileDoubt : str
    outputFilePos : str
    outputFileNeg : str
    boxSize : int
    consensusRadius : float
    consensusThreshold : float
    consensus : list # Here will lie the full list

    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Coordinates Consensus Tomo - Unify coordinates for a single tomogram\n'
                          'This program accepts an XMD file containing the following information:\n'
                          '- 3D coordinates in separate columns (X, Y, Z)\n'
                          '- Integer identifier of the picker who saw it'
                          )
        
        self.addParamsLine('--input <path> : input')
        self.addParamsLine('--outputPos <path> : output path for positive subtomos')
        self.addParamsLine('--outputDoubt <path> : output path for doubtful subtomos')
        self.addParamsLine('--outputAll <path> : output path for all subtomos')
        self.addParamsLine('--boxsize <int> : boxsize')
        self.addParamsLine('--samplingrate <double> : sampling rate')
        self.addParamsLine('--radius <double> : radius')
        self.addParamsLine('--number <int> : number of pickers that need to reference a coordinate for it to be POS')
        self.addParamsLine('--constype <int> : type of consensus (0 for first, 1 for centroid)')
        self.addParamsLine('[ --inputTruth <path> ] : Optional path to XMD file containing truthful coordinates. Added to POS automatically.')
        self.addParamsLine('[ --inputLie <path> ] : Optional path to XMD file containing lie(negative) coordinates. Added to NEG automatically.')
        self.addParamsLine('--outputNeg <path> : Output path for negative subtomos')
        self.addParamsLine('--startingId <int> : Initial number to use for MDL_PARTICLE_ID field in XMD.')

    def run(self):
        # Aux class
        class Coordinate(NamedTuple):
            xyz : np.ndarray
            pickers : set
            allcoords : list

        # Preassign to false optional input flags
        self.hasPositive = False
        self.hasNegative = False

        # Read args
        self.inputFile = self.getParam('--input')
        self.outputFile = self.getParam('--outputAll')
        self.outputFilePos = self.getParam('--outputPos')
        self.outputFileDoubt = self.getParam('--outputDoubt')
        self.outputFileNeg = self.getParam('--outputNeg')
        if self.checkParam('--inputTruth'):
            self.inputTruthFile = self.getParam('--inputTruth')
            self.hasPositive = True
        if self.checkParam('--inputLie'):
            self.inputLieFile = self.getParam('--inputLie')
            self.hasNegative = True
        self.boxSize = self.getIntParam('--boxsize')
        self.samplingrate = self.getDoubleParam('--samplingrate')
        self.consensusRadius = float(self.getDoubleParam('--radius'))
        self.consensusThreshold = int(self.getIntParam('--number'))
        self.consensusType = int(self.getIntParam('--constype'))
        self.startingSubtomoId = int(self.getIntParam('--startingId'))
        self.distancethreshold = float(self.boxSize * self.consensusRadius)

        # Initialize as empty list
        consensus = []

        # Read from Xmipp MD
        print("Starting 3D coordinates consensus")
        md = xmippLib.MetaData(self.inputFile)

        # Interested in the tomogram of origin for later
        self.tomoReference = md.getValue(xmippLib.MDL_TOMOGRAM_VOLUME, 1)

        # Do a check for each existing coordinate (cost n^2)
        for row_id in md:

            # Load the coordinates from the XMD into an ndarray
            coords = np.empty(3, dtype=int)
            coords[0] = md.getValue(xmippLib.MDL_XCOOR, row_id)
            coords[1] = md.getValue(xmippLib.MDL_YCOOR, row_id)
            coords[2] = md.getValue(xmippLib.MDL_ZCOOR, row_id)
            picker_id = int(md.getValue(xmippLib.MDL_REF, row_id))
            
            item : Coordinate
            for item in consensus:
                # Check the distance between this and all other coordinates
                if float(distance(coords, item.xyz)) < self.distancethreshold and picker_id not in item.pickers:
                    # print("Assimilated coords at distance %d" % distance(coords, item.xyz), flush=True)
                    item.pickers.add(picker_id)
                    # Also add coordinates for later centroid calculation
                    item.allcoords.append(coords)
                    break
            else:
                # If item does not match any other, make it be its own new structure
                # print("Distance not enough for threshold %s, creating new cluster" % str(self.distancethreshold), flush=True)
                consensus.append(Coordinate(coords, {picker_id}, [coords]))
        
        print("Went from %d to %d coordinates after consensus." % (md.size(), len(consensus)), flush=True)

        # If centroid is the method, calculate it for every set thanks to the added allcoords field
        if self.consensusType == COORD_CONS_CENTROID:
            for item in consensus:
                x = 0
                y = 0
                z = 0
                for coordinates in item.allcoords:
                    x += coordinates[0]
                    y += coordinates[1]
                    z += coordinates[2]
                item.xyz[0] = x // len(item.allcoords)
                item.xyz[1] = y // len(item.allcoords)
                item.xyz[2] = z // len(item.allcoords)
        else:
            # The first entered particle already is representing the cluster, nothing else needed here
            pass
        
        outMd = xmippLib.MetaData() # MD handle for all
        outMdDoubt = xmippLib.MetaData() # MD handle for unsure = all - {positive}
        outMdPos = xmippLib.MetaData() # MD handle for positive
        outMdNeg = xmippLib.MetaData() # MD handle for negative

        # Set the initial particle id for this execution
        partId = self.startingSubtomoId
        # Initial amount of true positions
        truthsize = 0

        # Manage if truth file was present
        if self.hasPositive:
            mdTrue = xmippLib.MetaData(self.inputTruthFile)
            for mdtrue_id in mdTrue:
                coords = np.empty(3, dtype=int)
                coords[0] = md.getValue(xmippLib.MDL_XCOOR, mdtrue_id)
                coords[1] = md.getValue(xmippLib.MDL_YCOOR, mdtrue_id)
                coords[2] = md.getValue(xmippLib.MDL_ZCOOR, mdtrue_id)
                item : Coordinate
                for item in consensus:
                    # Check the distance between this and all other coordinates
                    if distance(coords, item.xyz) < self.distancethreshold:
                        item.pickers.add(99)
                        break
                else:
                    # If item does not match any other, make it be its own new structure
                    consensus.append(Coordinate(coords, {99}))
            
                # Always write to positive list
                row_id = outMdPos.addObject()
                outMdPos.setValue(xmippLib.MDL_XCOOR, int(coords[0]), row_id)
                outMdPos.setValue(xmippLib.MDL_YCOOR, int(coords[1]), row_id)
                outMdPos.setValue(xmippLib.MDL_ZCOOR, int(coords[2]), row_id)
                outMdPos.setValue(xmippLib.MDL_PICKING_PARTICLE_SIZE, self.boxSize, row_id)
                outMdPos.setValue(xmippLib.MDL_SAMPLINGRATE, self.samplingrate, row_id)
                outMdPos.setValue(xmippLib.MDL_COUNT, 0, row_id)
                outMdPos.setValue(xmippLib.MDL_PARTICLE_ID, int(partId), row_id)
                outMdPos.setValue(xmippLib.MDL_TOMOGRAM_VOLUME, self.tomoReference, row_id)
                # Write to general
                row_idg = outMd.addObject()
                outMd.setValue(xmippLib.MDL_XCOOR, int(coords[0]), row_idg)
                outMd.setValue(xmippLib.MDL_YCOOR, int(coords[1]), row_idg)
                outMd.setValue(xmippLib.MDL_ZCOOR, int(coords[2]), row_idg)
                outMd.setValue(xmippLib.MDL_PICKING_PARTICLE_SIZE, self.boxSize, row_idg)
                outMd.setValue(xmippLib.MDL_SAMPLINGRATE, self.samplingrate, row_idg)
                outMd.setValue(xmippLib.MDL_COUNT, 0, row_idg)
                outMd.setValue(xmippLib.MDL_PARTICLE_ID, int(partId), row_idg)
                outMd.setValue(xmippLib.MDL_TOMOGRAM_VOLUME, self.tomoReference, row_idg)

                truthsize += 1
                partId += 1
            # print("Writing %d items from TRUTH to disk" % truthsize)
        
        falsesize = 0
        if self.hasNegative:
            # outMdNeg = xmippLib.MetaData(self.inputLieFile)
            for mdfalse_id in outMdNeg:
                coords = np.empty(3, dtype=int)
                coords[0] = md.getValue(xmippLib.MDL_XCOOR, mdfalse_id)
                coords[1] = md.getValue(xmippLib.MDL_YCOOR, mdfalse_id)
                coords[2] = md.getValue(xmippLib.MDL_ZCOOR, mdfalse_id)
            
                # Only write to negative list
                row_id = outMdNeg.addObject()
                outMdNeg.setValue(xmippLib.MDL_XCOOR, int(coords[0]), row_id)
                outMdNeg.setValue(xmippLib.MDL_YCOOR, int(coords[1]), row_id)
                outMdNeg.setValue(xmippLib.MDL_ZCOOR, int(coords[2]), row_id)
                outMdNeg.setValue(xmippLib.MDL_PICKING_PARTICLE_SIZE, self.boxSize, row_id)
                outMdNeg.setValue(xmippLib.MDL_SAMPLINGRATE, self.samplingrate, row_id)
                outMdNeg.setValue(xmippLib.MDL_COUNT, 0, row_id)
                outMdNeg.setValue(xmippLib.MDL_TOMOGRAM_VOLUME, self.tomoReference, row_id)
                falsesize += 1
            print("Writing %d items from NOISE to disk" % falsesize)

        consize = 0
        doubtsize = 0
        
        for item in consensus:
            consize += 1
            neg = False
            if len(item.pickers) > 2:
                variableMdPointer = outMdPos
                truthsize += 1
            elif len(item.pickers) == 2:
                variableMdPointer = outMdDoubt
                doubtsize += 1
            else:
                variableMdPointer = outMdNeg
                falsesize += 1
                neg = True

            # Write to specific
            row_id = variableMdPointer.addObject()
            variableMdPointer.setValue(xmippLib.MDL_XCOOR, int(item.xyz[0]), row_id)
            variableMdPointer.setValue(xmippLib.MDL_YCOOR, int(item.xyz[1]), row_id)
            variableMdPointer.setValue(xmippLib.MDL_ZCOOR, int(item.xyz[2]), row_id)
            variableMdPointer.setValue(xmippLib.MDL_PICKING_PARTICLE_SIZE, self.boxSize, row_id)
            variableMdPointer.setValue(xmippLib.MDL_SAMPLINGRATE, self.samplingrate, row_id)
            variableMdPointer.setValue(xmippLib.MDL_COUNT, len(item.pickers), row_id)
            variableMdPointer.setValue(xmippLib.MDL_PARTICLE_ID, int(partId), row_id)
            variableMdPointer.setValue(xmippLib.MDL_TOMOGRAM_VOLUME, self.tomoReference, row_id)
            
            # if not neg:
            # Write to general
            row_idg = outMd.addObject()
            outMd.setValue(xmippLib.MDL_XCOOR, int(item.xyz[0]), row_idg)
            outMd.setValue(xmippLib.MDL_YCOOR, int(item.xyz[1]), row_idg)
            outMd.setValue(xmippLib.MDL_ZCOOR, int(item.xyz[2]), row_idg)
            outMd.setValue(xmippLib.MDL_PICKING_PARTICLE_SIZE, self.boxSize, row_idg)
            outMd.setValue(xmippLib.MDL_SAMPLINGRATE, self.samplingrate, row_idg)
            outMd.setValue(xmippLib.MDL_COUNT, len(item.pickers), row_idg)
            outMd.setValue(xmippLib.MDL_PARTICLE_ID, int(partId), row_idg)
            outMd.setValue(xmippLib.MDL_TOMOGRAM_VOLUME, self.tomoReference, row_idg)

            partId += 1
        print("Writing %d items from CONS to disk" % consize)
        print("Now there are %d positive, %d doubt and %d negative particles" % (truthsize, doubtsize, falsesize))

        # Write everything to XMD files
        outMd.write(self.outputFile) 
        print("Written all subtomos to " + self.outputFile)   
        if outMdDoubt.size() > 0:
            outMdDoubt.write(self.outputFileDoubt)
            print("Written doubtful subtomo coords to " + self.outputFileDoubt)
        if outMdPos.size() > 0:
            outMdPos.write(self.outputFilePos)
            print("Written positive subtomo coords to " + self.outputFilePos)
        if outMdNeg.size() > 0:
            outMdNeg.write(self.outputFileNeg)
            print("Written negative subtomo coords to " + self.outputFileNeg)


if __name__ == '__main__':
    exitCode = ScriptCoordsConsensusTomo().tryRun()
    sys.exit(exitCode)
