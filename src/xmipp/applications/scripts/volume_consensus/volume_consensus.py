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
        # inputVols = []
        # with open(inputFile) as f:
        #     for line in f:
        #         inputVols.append(self.loadVol(line.split()[0]))
        # f.close()
        self.computeVolumeConsensus(inputFile, outVolFn)

    def computeVolumeConsensus(self, inputFile, outVolFn, wavelet='sym11'):
        coeffDictList = []
        # for vol in inputVols.iterItems():
        with open(inputFile) as f:
            for line in f:
                vol = self.loadVol(line.split()[0])
            coeffDict = pywt.swtn(vol, wavelet, 1)
            coeffDictList.append(coeffDict)
        f.close()
        newDict = {}
        for key, _ in enumerate(coeffDict):
            coeffInPosKey = []
            for coeffDict in coeffDictList:
                coeffInPosKey.append(coeffDict[key])
            newDict[key] = max(coeffInPosKey)
            # print("-------------------", coeffDict[key].shape)
        consensus = pywt.iswtn(newDict, wavelet)
        self.saveVol(consensus, outVolFn, line)
        return consensus

    def saveVol(self, data, fname, fnameToCopyHeader):
        shutil.copyfile(fnameToCopyHeader, fname)
        with mrcfile.open(fname, "r+", permissive=True) as f:
            f.data[:] = data

    def loadVol(self, fname):
        with mrcfile.open(fname, permissive=True) as f:
            return f.data.astype(np.float32)


if __name__=="__main__":
    ScriptVolumeConsensus().tryRun()
