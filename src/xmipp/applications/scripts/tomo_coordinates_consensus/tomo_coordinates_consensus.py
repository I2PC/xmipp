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
# * Initial release: June 2023
# * Updated: March 2024
# **************************************************************************

import sys

from math import sqrt
from typing import NamedTuple, Tuple, Iterable
import xmippLib
from xmipp_base import XmippScript

COORD_CONS_CENTROID = 1
COORD_CONS_CLIQUE = 2

# def distance(x1:int,y1:int,z1:int,x2:int,y2:int,z2:int) -> float:

#     pass
# def distance(a: np.ndarray, b: np.ndarray) -> float:
#     return np.linalg.norm(a-b)

def distance(a: Iterable[int], b: Iterable[int]) -> float:
    """
    Ace cencia
    """
    norm2 : float = 0.0

    for a_elem,b_elem in zip(a,b):
        norm2 += (a_elem-b_elem)**2

    return sqrt(norm2)

# Aux class
class MyCoordinate(NamedTuple):
    xyz : Tuple[int,int,int]
    pickers : set
    allcoords : list

class ScriptCoordsConsensusTomo(XmippScript):
    inputFile : str
    outputFileAll : str
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
        """
        Defines the Xmipp Script input parameters
        [] -> optional
        no [] -> mandatory
        """
        self.addUsageLine('MyCoordinates Consensus Tomo - Unify coordinates for a single tsId.\n'
                          'This program accepts an XMD file containing the following information:\n'
                          '- 3D coordinates in separate columns (X, Y, Z) (MDL_XCOOR, YCOOR, ZCOOR)\n'
                          '- Integer identifier of the picker who saw it (MDL_ITEM_ID).'
                          )
        
        self.addParamsLine('--infile <path> : XMD file containing the coordinates to be consensuated.')
        self.addParamsLine('--tsid <string> : name of the tsId. Used for I/O operations.')
        self.addParamsLine('--outall<path> : folder to which write XMD files containing the consensus coords.')
        self.addParamsLine('--outpos<path> : folder to which write XMD files containing the POS consensus coords.')
        self.addParamsLine('--outdoubt<path> : folder to which write XMD files containing the DOUBT consensus coords.')
        self.addParamsLine('--outneg<path> : folder to which write XMD files containing the NEG consensus coords.')
        self.addParamsLine('--boxsize <int> : boxsize.')
        self.addParamsLine('--samplingrate <double> : sampling rate.')
        self.addParamsLine('--radius <double> : radius for coordinate assimilation.')
        self.addParamsLine('--number <int> : number of pickers that need to reference a coordinate for it to be POS.')
        self.addParamsLine('--constype <int> : type of consensus (0 for first, 1 for centroid).')
        self.addParamsLine('[ --inTruth <path> ] : Optional path to XMD file containing truthful coordinates. Added to POS automatically.')
        self.addParamsLine('[ --inLie <path> ] : Optional path to XMD file containing lie(negative) coordinates. Added to NEG automatically.')

    def gatherParams(self):
        """
        Parses the input CMD lines parameters
        """
        # Preassign to false optional input flags
        self.hasPositive = False
        self.hasNegative = False

        # Input/Output management
        self.inputFile = self.getParam('--infile')
        self.outputFolder = self.getParam('--outall')
        self.tsId = self.getParam('--tsid')
        self.outputFileAll = self.getParam('--outall')
        self.outputFilePos = self.getParam('--outpos')
        self.outputFileDoubt = self.getParam('--outdoubt')
        self.outputFileNeg = self.getParam('--outneg')
        if self.checkParam('--inTruth'):
            self.inputTruthFile = self.getParam('--inputTruth')
            self.hasPositive = True
        if self.checkParam('--inLie'):
            self.inputLieFile = self.getParam('--inputLie')
            self.hasNegative = True

        # Operational parameters
        self.boxSize = self.getIntParam('--boxsize')
        self.samplingrate = self.getDoubleParam('--samplingrate')
        self.consensusRadius = float(self.getDoubleParam('--radius'))
        self.consensusThreshold = int(self.getIntParam('--number'))
        self.consensusType = int(self.getIntParam('--constype'))
        self.distancethreshold = float(self.boxSize * self.consensusRadius)

    def run(self):
        """
        Application's main flow
        """

        self.gatherParams()        
        # Initialize as empty list
        consensus = []

        # Read from XMD for this tsId
        print(f"Launched 3D coordinates consensus for {self.tsId}", flush=True)
        md = xmippLib.MetaData(self.inputFile)

        # Do a check for each existing coordinate (cost in the range [nlogn - n^2])
        for row_id in md:
            # Load the coordinates from the XMD into an ndarray
            coords = (
                md.getValue(xmippLib.MDL_XCOOR, row_id),
                md.getValue(xmippLib.MDL_YCOOR, row_id),
                md.getValue(xmippLib.MDL_ZCOOR, row_id)
            )
            picker_id = int(md.getValue(xmippLib.MDL_REF, row_id))
            
            item : MyCoordinate = None
            for item in consensus:
                # Check the distance between this and all other coordinates
                if float(distance(coords, item.xyz)) < self.distancethreshold and picker_id not in item.pickers:
                    item.pickers.add(picker_id)
                    # Also add coordinates for later centroid calculation
                    item.allcoords.append(coords)
                    break
            else:
                # If item does not match any other, make it be its own new structure
                consensus.append(MyCoordinate(coords, {picker_id}, [coords]))
        
        # print("Went from %d to %d coordinates after consensus." % (md.size(), len(consensus)), flush=True)

        # If centroid is the method, calculate it for every set thanks to the added allcoords field
        if self.consensusType == COORD_CONS_CENTROID:
            for item in consensus:
                x : int = 0
                y : int = 0
                z : int = 0
                for coordinates in item.allcoords:
                    x += coordinates[0]
                    y += coordinates[1]
                    z += coordinates[2]
                x = x // len(item.allcoords)
                y = y // len(item.allcoords)
                z = z // len(item.allcoords)
                item._replace(xyz=(x,y,z))
        else:
            # The first entered particle already is representing the cluster, nothing else needed here
            pass
        
        outMdAll = xmippLib.MetaData() # MD handle for all
        outMdPos = xmippLib.MetaData() # MD handle for positive
        outMdDoubt = xmippLib.MetaData() # MD handle for unsure = all - {positive}
        outMdNeg = xmippLib.MetaData() # MD handle for negative

        # Initial amount of true positions
        truthsize = 0
        # Manage if truth file was present
        # if self.hasPositive:
        #     pass
        #     mdTrue = xmippLib.MetaData(self.inputTruthFile)
        #     for mdtrue_id in mdTrue:
        #         coords = np.empty(3, dtype=int)
        #         coords[0] = md.getValue(xmippLib.MDL_XCOOR, mdtrue_id)
        #         coords[1] = md.getValue(xmippLib.MDL_YCOOR, mdtrue_id)
        #         coords[2] = md.getValue(xmippLib.MDL_ZCOOR, mdtrue_id)
        #         item : MyCoordinate
        #         for item in consensus:
        #             # Check the distance between this and all other coordinates
        #             if distance(coords, item.xyz) < self.distancethreshold:
        #                 item.pickers.add(99)
        #                 break
        #         else:
        #             # If item does not match any other, make it be its own new structure
        #             consensus.append(MyCoordinate(coords, {99}))
            
        #         # Always write to positive list
        #         row_id = outMdPos.addObject()
        #         outMdPos.setValue(xmippLib.MDL_XCOOR, int(coords[0]), row_id)
        #         outMdPos.setValue(xmippLib.MDL_YCOOR, int(coords[1]), row_id)
        #         outMdPos.setValue(xmippLib.MDL_ZCOOR, int(coords[2]), row_id)
        #         outMdPos.setValue(xmippLib.MDL_PICKING_PARTICLE_SIZE, self.boxSize, row_id)
        #         outMdPos.setValue(xmippLib.MDL_SAMPLINGRATE, self.samplingrate, row_id)
        #         outMdPos.setValue(xmippLib.MDL_COUNT, 0, row_id)
        #         # outMdPos.setValue(xmippLib.MDL_PARTICLE_ID, int(partId), row_id)
        #         outMdPos.setValue(xmippLib.MDL_TOMOGRAM_VOLUME, self.tomoReference, row_id)
        #         # Write to general
        #         row_idg = outMd.addObject()
        #         outMd.setValue(xmippLib.MDL_XCOOR, int(coords[0]), row_idg)
        #         outMd.setValue(xmippLib.MDL_YCOOR, int(coords[1]), row_idg)
        #         outMd.setValue(xmippLib.MDL_ZCOOR, int(coords[2]), row_idg)
        #         outMd.setValue(xmippLib.MDL_PICKING_PARTICLE_SIZE, self.boxSize, row_idg)
        #         outMd.setValue(xmippLib.MDL_SAMPLINGRATE, self.samplingrate, row_idg)
        #         outMd.setValue(xmippLib.MDL_COUNT, 0, row_idg)
        #         # outMd.setValue(xmippLib.MDL_PARTICLE_ID, int(partId), row_idg)
        #         outMd.setValue(xmippLib.MDL_TOMOGRAM_VOLUME, self.tomoReference, row_idg)

        #         truthsize += 1
                # partId += 1
            # print("Writing %d items from TRUTH to disk" % truthsize)
        
        falsesize = 0
        # if self.hasNegative:
        #     pass
        #     # outMdNeg = xmippLib.MetaData(self.inputLieFile)
        #     for mdfalse_id in outMdNeg:
        #         coords = np.empty(3, dtype=int)
        #         coords[0] = md.getValue(xmippLib.MDL_XCOOR, mdfalse_id)
        #         coords[1] = md.getValue(xmippLib.MDL_YCOOR, mdfalse_id)
        #         coords[2] = md.getValue(xmippLib.MDL_ZCOOR, mdfalse_id)
            
        #         # Only write to negative list
        #         row_id = outMdNeg.addObject()
        #         outMdNeg.setValue(xmippLib.MDL_XCOOR, int(coords[0]), row_id)
        #         outMdNeg.setValue(xmippLib.MDL_YCOOR, int(coords[1]), row_id)
        #         outMdNeg.setValue(xmippLib.MDL_ZCOOR, int(coords[2]), row_id)
        #         outMdNeg.setValue(xmippLib.MDL_PICKING_PARTICLE_SIZE, self.boxSize, row_id)
        #         outMdNeg.setValue(xmippLib.MDL_SAMPLINGRATE, self.samplingrate, row_id)
        #         outMdNeg.setValue(xmippLib.MDL_COUNT, 0, row_id)
        #         outMdNeg.setValue(xmippLib.MDL_TOMOGRAM_VOLUME, self.tomoReference, row_id)
        #         falsesize += 1
        #     print("Writing %d items from NOISE to disk" % falsesize)

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
            variableMdPointer.setValue(xmippLib.MDL_COUNT, len(item.pickers), row_id)
            
            # if not neg:
            # Write to general
            row_idg = outMdAll.addObject()
            outMdAll.setValue(xmippLib.MDL_XCOOR, int(item.xyz[0]), row_idg)
            outMdAll.setValue(xmippLib.MDL_YCOOR, int(item.xyz[1]), row_idg)
            outMdAll.setValue(xmippLib.MDL_ZCOOR, int(item.xyz[2]), row_idg)
            outMdAll.setValue(xmippLib.MDL_COUNT, len(item.pickers), row_idg)
            # partId += 1
        print("Writing %d items from CONS to disk" % consize)
        print("Now there are %d positive, %d doubt and %d negative particles" % (truthsize, doubtsize, falsesize))

        # Write everything to XMD files
        outMdAll.write(self.outputFileAll) 
        print("Written all subtomos to " + self.outputFileAll)   
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
