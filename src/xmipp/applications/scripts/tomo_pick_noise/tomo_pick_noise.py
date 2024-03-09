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
# * Initial release: aug 2023
# **************************************************************************

import sys
import numpy as np
from typing import Iterable
from math import sqrt

import multiprocessing as mp
from multiprocessing import Process, Queue

from xmipp_base import XmippScript
import xmippLib

MAX_ITERS_PER_THREAD = 10000
DEFAULT_BOX_PERCENTAGE = 0.5

def distance(a: Iterable[int], b: Iterable[int]) -> float:
    norm2 : float = 0.0
    for a_elem,b_elem in zip(a,b):
        norm2 += (a_elem - b_elem)**2
    return sqrt(norm2)

class ScriptPickNoiseTomo(XmippScript):

    inputFn : str
    outputFn : str
    tsId : str
    boxSize : int
    boxPercentage : float
    distance : int
    tomoSize : list
    nrPositive : int
    nThreads : int
    nrPositivePerThread : int
    allCoords: list

    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Culo')

        self.addParamsLine('--infile <inputFile> : Path to the XMD file containing positive MDL_X MDL_Y and MDL_Z')
        self.addParamsLine('--output <outputFile> : Path for the resulting XMD file to be written.')
        self.addParamsLine('--tsid <TSID> : String for the TSID name')
        self.addParamsLine('--boxsize <boxsize> : Size in pixels of the square box for subtomograms')
        self.addParamsLine('--tomosize <x> <y> <z> : Triplet representing the tomogram size')
        self.addParamsLine('--threads <int> : Number of threads to parallelize the search')
    
    def gatherParams(self):
        # Parse input arguments
        self.inputFn = self.getParam('--infile')
        self.outputFn = self.getParam('--output')
        self.boxSize = self.getIntParam('--boxsize')
        self.boxPercentage = DEFAULT_BOX_PERCENTAGE
        self.distance = int(self.boxPercentage * self.boxSize)
        self.tomoSize = [-1, -1, -1]
        self.tomoSize[0] = self.getIntParam('--tomosize', 0)
        self.tomoSize[1] = self.getIntParam('--tomosize', 1)
        self.tomoSize[2] = self.getIntParam('--tomosize', 2)
        self.nThreads = self.getIntParam('--threads')
        self.tsId = self.getParam('--tsid')        

    def run(self):

        self.gatherParams()
        print(f"Launched 3D noise picking script for {self.tsId}", flush=True)

        # Load all of the coordinates of the tomo into memory
        md = xmippLib.MetaData(self.inputFn)
        self.nrPositive = md.size()
        self.nrPositivePerThread = self.nrPositive // self.nThreads
        self.allCoords = []
        for row_id in md:
            coords = (
                md.getValue(xmippLib.MDL_XCOOR, row_id),
                md.getValue(xmippLib.MDL_YCOOR, row_id),
                md.getValue(xmippLib.MDL_ZCOOR, row_id))
            self.allCoords.append(coords)

        res = []
        for _ in range(self.nThreads):
            res += self.pickFun()
        # Write the results
        print("PickNoiseTomo found %d noise volumes" %(len(res)))
        outMd = xmippLib.MetaData()

        for elem in res:
            row_id = outMd.addObject()
            outMd.setValue(xmippLib.MDL_XCOOR, int(elem[0]), row_id)
            outMd.setValue(xmippLib.MDL_YCOOR, int(elem[1]), row_id)
            outMd.setValue(xmippLib.MDL_ZCOOR, int(elem[2]), row_id)

        outMd.write(self.outputFn)

    def pickFun(self):
        lres = []

        maxDistance = self.distance

        for i in range(MAX_ITERS_PER_THREAD):
            # Generate a random coordinate
            candidate = (np.random.rand(3) * self.tomoSize).astype(int)

            # Validate
            for existingCoord in self.allCoords:
                if distance(candidate, existingCoord) < maxDistance:
                    break
            else:
                # No particle collides with this, adding to noise
                # print("Found OK candidate: " + str(candidate))
                lres.append(candidate)
            
            if (len(lres) >= self.nrPositivePerThread):
                # The goal is achieved, no more needed
                break

            if i == (MAX_ITERS_PER_THREAD - 1):
                print("Sera verdad no more iters", flush=True)

        if lres == []:
            print("NoisePick process ended with no results, perhaps too many random pickings in your input?", flush=True)
        # q.put_nowait(lres)
        return lres

if __name__ == '__main__':
    exitCode = ScriptPickNoiseTomo().tryRun()
    sys.exit(exitCode)