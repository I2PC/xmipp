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

def distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a-b)

class ScriptCoordsConsensusTomo(XmippScript):
    inputFile : str
    outputFile : str
    outputFileDoubt : str
    outputFilePos : str
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
        self.addParamsLine('--number <int> : number')
        self.addParamsLine('--constype <int> : type of consensus (0 for first, 1 for centroid)')

    def run(self):
        # Aux class
        class Coordinate(NamedTuple):
            xyz : np.ndarray
            pickers : set

        # Read args
        self.inputFile = self.getParam('--input')
        self.outputFile = self.getParam('--outputAll')
        self.outputFilePos = self.getParam('--outputPos')
        self.outputFileDoubt = self.getParam('--outputDoubt')
        self.boxSize = self.getIntParam('--boxsize')
        self.samplingrate = self.getDoubleParam('--samplingrate')
        self.consensusRadius = self.getDoubleParam('--radius')
        self.consensusThreshold = self.getIntParam('--number')
        self.consensusType = self.getIntParam('--constype')
        distancethreshold = self.boxSize * self.consensusRadius

        # Initialize as empty list
        consensus = []

        # Read from Xmipp MD
        print("Starting 3D coordinates consensus")
        md = xmippLib.MetaData(self.inputFile)
        for row_id in md:

            coords = np.empty(3, dtype=int)
            coords[0] = md.getValue(xmippLib.MDL_XCOOR, row_id)
            coords[1] = md.getValue(xmippLib.MDL_YCOOR, row_id)
            coords[2] = md.getValue(xmippLib.MDL_ZCOOR, row_id)
            picker_id = md.getValue(xmippLib.MDL_REF, row_id)
            
            item : Coordinate
            for item in consensus:
                if distance(coords, item.xyz) < distancethreshold and picker_id not in item.pickers:
                    item.pickers.add(picker_id)
                    break
            else:
                consensus.append(Coordinate(coords, {picker_id}))

        
        outMd = xmippLib.MetaData() # MD handle for all
        outMdDoubt = xmippLib.MetaData() # MD handle for unsure = all - {positive}
        outMdPos = xmippLib.MetaData() # MD handle for positive
        

        for item in consensus:
            if len(item.pickers) >= self.consensusThreshold:
                nun = outMdPos
            else:
                nun = outMdDoubt

            # Write to specific
            row_id = nun.addObject()
            nun.setValue(xmippLib.MDL_XCOOR, int(item.xyz[0]), row_id)
            nun.setValue(xmippLib.MDL_YCOOR, int(item.xyz[1]), row_id)
            nun.setValue(xmippLib.MDL_ZCOOR, int(item.xyz[2]), row_id)
            nun.setValue(xmippLib.MDL_COUNT, len(item.pickers), row_id)
            
            # Write to general
            row_idg = outMd.addObject()
            outMd.setValue(xmippLib.MDL_XCOOR, int(item.xyz[0]), row_idg)
            outMd.setValue(xmippLib.MDL_YCOOR, int(item.xyz[1]), row_idg)
            outMd.setValue(xmippLib.MDL_ZCOOR, int(item.xyz[2]), row_idg)
            outMd.setValue(xmippLib.MDL_COUNT, len(item.pickers), row_idg)

        # Write everything to XMD files
        outMd.write(self.outputFile) 
        print("Written all subtomos to " + self.outputFile)   
        outMdDoubt.write(self.outputFileDoubt)
        print("Written doubtful subtomos to " + self.outputFileDoubt)
        outMdPos.write(self.outputFilePos)
        print("Written positive subtomos to " + self.outputFilePos)


if __name__ == '__main__':
    exitCode = ScriptCoordsConsensusTomo().tryRun()
    sys.exit(exitCode)
    
    

                    
                    

            


            




        




