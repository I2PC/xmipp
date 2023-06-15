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

import sys, os
import numpy as np
import pandas as pd

import multiprocessing as mp

from xmipp_base import XmippScript
import xmippLib

ITERS_PER_THREAD = 20000

def distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a-b)

class ScriptPickNoiseTomo(XmippScript):

    inputFn : str
    radius : float
    boxSize : int
    samplingRate : float
    limit : float
    limitPerThread : int
    nThreads : int
    static : bool

    def __init__(self):
        XmippScript.__init__(self)
        self.static = False

    def defineParams(self):
        self.addUsageLine('PickNoiseTomo - picks random coordinates for DeepConsensusTomo\n'
                          'This program takes an XMD formatted file containing coordinates for picked\n'
                          'subtomos and uses brute force to obtain coordinates representing "noise"\n'
                          'for use as negative input in the DeepConsensusTomo NN training.')

        self.addParamsLine('--input <path> : Path to the XMD file containing MDL_X MDL_Y and MDL_Z')
        self.addParamsLine('--output <path> : Path for the resulting XMD file to be written.')
        self.addParamsLine('--radius <float> : Radius (in fraction of boxsize) for thresholding the distances')
        self.addParamsLine('--boxsize <int> : Size in pixels of the square box for subtomograms')
        self.addParamsLine('--samplingrate <float> : Sampling rate in Angstroms/pixel')
        self.addParamsLine('--limit <float> : Amount (in fraction of total input coords) of pickings to do. Default=0.7')
        self.addParamsLine('--threads <int> : Number of threads to parallelize the search. Default=4')
        self.addParamsLine('--static : flag to stop the algorithm from reducing the radius')

        self.addExampleLine('xmipp_pick_noise_tomo --input tomo05_coords.xmd --boxsize 200 --samplingrate 2.17')

    def run(self):
        
        # Parse input arguments
        self.inputFn = self.getParam('--input')
        self.outputFn = self.getParam('--output')
        self.radius = self.getDoubleParam('--radius')
        self.boxSize = self.getIntParam('--boxsize')
        self.samplingRate = self.getDoubleParam('--samplingrate')
        
        # Default limit if not specified
        if self.checkParam('--limit'):
            self.limit = self.getDoubleParam('--limit')
        else:
            self.limit = 0.7

        # Default threads to 4 if not specified
        if self.checkParam('--threads'):
            self.nThreads = self.getIntParam('--threads')
        else:
            self.nThreads = min(4, mp.cpu_count())

        # Set the static flag if needed
        if self.checkParam('--static'):
            self.static = True

        assert os.path.isdir(self.inputFn), "Provided input file name is not a directory"

        allCoords : set  = {}

        # Load all of the coordinates of the tomo into memory
        tomoMd = xmippLib.MetaData(self.inputFn)
        for row_id in tomoMd:
            coords = np.empty(3)
            coords[0] = tomoMd.getValue(xmippLib.MDL_X, row_id)
            coords[1] = tomoMd.getValue(xmippLib.MDL_Y, row_id)
            coords[2] = tomoMd.getValue(xmippLib.MDL_Z, row_id)
            allCoords.add(coords)

        # Generate iterspace for MP
        self.limitPerThread = int(len(allCoords) * self.limit / self.nThreads)


        # Launch the search in parallel
        print("Launching search for %d coordinates on %d cores" % self.limitPerThread*self.nThreads, self.nThreads)
        res : list = []
        with mp.Pool(processes=self.nThreads) as pool:
            res.append (pool.apply(self.pickFun, args=(allCoords, self.limitPerThread, self.boxSize, self.radius)))
        # Write the results
        outMd = xmippLib.MetaData()
        for elem in res:
            row_id = outMd.addObject()
            outMd.setValue(xmippLib.MDL_X, elem[0], row_id)
            outMd.setValue(xmippLib.MDL_Y, elem[1], row_id)
            outMd.setValue(xmippLib.MDL_Z, elem[2], row_id)
        outMd.write(self.outputFn)

    def pickFun(coordsList : set, limit: int, boxsize: int, radius: float) -> list :
        res = []

        maxDistance : float = radius * boxsize

        for i in range(ITERS_PER_THREAD):
            # Generate a random coordinate
            candidate = np.empty(3)
            candidate[0] = np.random.choice([-1, 1]) * np.random.rand(1,3) * boxsize
            candidate[1] = np.random.choice([-1, 1]) * np.random.rand(1,3) * boxsize
            candidate[2] = np.random.choice([-1, 1]) * np.random.rand(1,3) * boxsize

            for existingCoord in coordsList:
                if distance(candidate, existingCoord) < maxDistance:
                    break
            else:
                # No particle collides with this, adding to noise
                res.append(candidate)
            
            if len(res) >= limit:
                # The goal is achieved, no more needed
                break

        return res

if __name__ == '__main__':
    exitCode = ScriptPickNoiseTomo().tryRun()
    sys.exit(exitCode)