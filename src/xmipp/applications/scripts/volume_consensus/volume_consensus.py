#!/usr/bin/env python3
""""
**************************************************************************
*
* Authors:  Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
*           Ruben Sanchez (rsanchez@cnb.csic.es)
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

import shutil
import numpy as np
import mrcfile
import pywt
import pywt.data
from xmipp_base import XmippScript
import xmippLib


class ScriptVolumeConsensus(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Volume consensus')
        ## params
        self.addParamsLine('-i <inputFile>   : A .txt file that contains the path of input volumes')
        self.addParamsLine('-o  <outputFile>   : Consensus volume filename')
        ## examples
        self.addExampleLine('xmipp_volume_consensus -i1 path/to/inputs/file.txt -o path/to/outFile.mrc')

    def run(self):
        inputFile = self.getParam('-i')
        outVolFn = self.getParam('-o')
        self.computeVolumeConsensus(inputFile, outVolFn)

    def computeVolumeConsensus(self, inputFile, outVolFn, wavelet='sym11'):
        wtVols = []  # list of wt transform of all volumes (list of dicts with len = #inputVols)
        outputWt = None
        outputMin = None
        nlevel = 3
        with open(inputFile) as f:
            for line in f:
                # vol = self.loadVol(line.split()[0])
                fileName = line.split()[0]
                if fileName.endswith('.mrc'):
                    fileName += ':mrc'
                V = xmippLib.Image(line.split()[0])
                vol = V.getData()
                wt = pywt.swtn(vol, wavelet, nlevel)  # compute wt of each volume (list of dicts with len = 1 )
                if outputWt == None:
                    outputWt = wt
                    outputMin = wt[0]['aaa']*0
                else:
                    for level in range(0, nlevel):
                        wtLevel = wt[level]
                        outputWtLevel = outputWt[level]
                        for key in wtLevel:
                            outputWtLevel[key] = np.where(np.abs(outputWtLevel[key]) > np.abs(wtLevel[key]),
                                                          outputWtLevel[key], wtLevel[key])
                            outputMin = np.max(outputMin, np.abs((np.abs(outputWtLevel[key]) - np.abs(wtLevel[key])) /
                                                                np.abs(outputWtLevel[key])))

            f.close()

        consensus = pywt.iswtn(outputWt, wavelet)  # compute inverse of the new wt ==> vol fusion
        V = xmippLib.Image()
        V.setData(consensus)
        V.write(outVolFn)
        V.setData(outputMin)
        V.write("kkk.mrc")
        return consensus

    def saveVol(self, data, fname, fnameToCopyHeader):
        shutil.copyfile(fnameToCopyHeader, fname)
        with mrcfile.open(fname, "w+", permissive=True) as f:
            f.data[:] = data

    def loadVol(self, fname):
        with mrcfile.open(fname, permissive=True) as f:
            return f.data.astype(np.float32)


if __name__=="__main__":
    ScriptVolumeConsensus().tryRun()
