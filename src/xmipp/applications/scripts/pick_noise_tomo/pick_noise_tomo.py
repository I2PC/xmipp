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

from typing import List

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

        self.addParamsLine('--input <inputFile> : Path to the XMD file containing MDL_X MDL_Y and MDL_Z')
        self.addParamsLine('--output <outputFile> : Path for the resulting XMD file to be written.')
        self.addParamsLine('--radius <radius> : Radius (in fraction of boxsize) for thresholding the distances')
        self.addParamsLine('--boxsize <boxsize> : Size in pixels of the square box for subtomograms')
        self.addParamsLine('--samplingrate <srate> : Sampling rate in Angstroms/pixel')
        self.addParamsLine('--size <x> <y> <z> : Triplet representing the tomo size')
        self.addParamsLine('[ --limit <limit=0.7> ] : Amount (in fraction of total input coords) of pickings to do. Default=0.7')
        self.addParamsLine('[ --threads <int=4> ] : Number of threads to parallelize the search. Default=4')
        self.addParamsLine('[ --static ] : flag to stop the algorithm from reducing the radius')

        self.addExampleLine('xmipp_pick_noise_tomo --input tomo05_coords.xmd --boxsize 200 --samplingrate 2.17')

    def run(self):
        
        print("Started 3D noise picking script", flush=True)
        # Parse input arguments
        self.inputFn = self.getParam('--input')
        self.outputFn = self.getParam('--output')
        self.radius = self.getDoubleParam('--radius')
        self.boxSize = self.getIntParam('--boxsize')
        self.samplingRate = self.getDoubleParam('--samplingrate')
        self.tomoSize = np.empty(3, dtype=int)
        self.tomoSize[0] = self.getIntParam('--size', 0)
        self.tomoSize[1] = self.getIntParam('--size', 1)
        self.tomoSize[2] = self.getIntParam('--size', 2)
        print ("Picking noise for %d X %d X %d tomogram" %(self.tomoSize[0],self.tomoSize[1],self.tomoSize[2]))
        
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

        if not os.path.isfile(self.inputFn):
            raise RuntimeError("Provided input file name does not exist")

        # Load all of the coordinates of the tomo into memory
        print("PickNoiseTomo reading coords into memory", flush=True)
        tomoMd = xmippLib.MetaData(self.inputFn)

        total = tomoMd.size()
        indizeak = ['xyz', 'tomo_id']
        self.allCoords = pd.DataFrame(index=range(total),columns=indizeak)

        for row_id in tomoMd:
            coords = np.empty(3, dtype=int)
            coords[0] = tomoMd.getValue(xmippLib.MDL_XCOOR, row_id)
            coords[1] = tomoMd.getValue(xmippLib.MDL_YCOOR, row_id)
            coords[2] = tomoMd.getValue(xmippLib.MDL_ZCOOR, row_id)
            tomoid = tomoMd.getValue(xmippLib.MDL_TOMOGRAM_VOLUME, row_id)
            self.allCoords.loc[row_id, 'xyz'] = coords
            self.allCoords.loc[row_id, 'tomo_id'] = tomoid            

        # Generate iterspace for MP
        self.limitPerThread = int(len(self.allCoords) * self.limit / self.nThreads)

        # Launch the search in parallel
        print("Launching search for %d coordinates on %d cores" % (self.limitPerThread*self.nThreads, self.nThreads))
        
        # Generate a queue to queue the output data from the worker threads
        # queue = mp.Queue()
        # workers = []
        # Launch the worker
        # for _ in range(self.nThreads):
        #     p = mp.Process(target=self.pickFun, args=(queue, allCoords, self.limitPerThread, self.boxSize, self.radius))
        #     p.start()
        #     workers.append(p)
        
        # res : list = []
        # process: mp.Process
        # for process in workers:
        #     process.join() # Wait for completion
        #     res += queue.get() # Add to final res

        # Lo mesmo pero en secoencial
        res = []
        for _ in range(self.nThreads):
            res += self.pickFun()
        # Write the results
        print("PickNoiseTomo found %d noise volumes" %(len(res)))
        outMd = xmippLib.MetaData()
        print("Writing to file...")
        for elem in res:
            row_id = outMd.addObject()
            outMd.setValue(xmippLib.MDL_XCOOR, int(elem[0]), row_id)
            outMd.setValue(xmippLib.MDL_YCOOR, int(elem[1]), row_id)
            outMd.setValue(xmippLib.MDL_ZCOOR, int(elem[2]), row_id)
        outMd.write(self.outputFn)

    def pickFun(self) -> list:
        res = []

        maxDistance : float = self.radius * self.boxSize

        for i in range(ITERS_PER_THREAD):
            # Generate a random coordinate
            candidate = (np.random.rand(3) * self.tomoSize).astype(int)

            # Validate
            for existingCoord in self.allCoords['xyz']:
                if distance(candidate, existingCoord) < maxDistance:
                    break
            else:
                # No particle collides with this, adding to noise
                # print("Found OK candidate: " + str(candidate))
                res.append(candidate)
            
            if len(res) >= self.limitPerThread:
                # The goal is achieved, no more needed
                break

        #queue.put(res)
        return res

if __name__ == '__main__':
    exitCode = ScriptPickNoiseTomo().tryRun()
    sys.exit(exitCode)