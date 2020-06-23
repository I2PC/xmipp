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
        self.addParamsLine('--i1 <inputFile1>   : Volume 1 filename')
        self.addParamsLine('--i2 <inputFile2>   : Volume 2 filename')
        self.addParamsLine('-o  <outputFile>   : Consensus volume filename')
        ## examples
        self.addExampleLine('xmipp_volume_consensus -i1 path/to/inputs/file1.mrc -i2 path/to/inputs/file2.mrc '
                            '-o path/to/outFile.mrc')

    def run(self):
        vol1fn = self.getParam('--i1')
        vol2fn = self.getParam('--i2')
        outVol = self.getParam('-o')
        vol1 = self.loadVol(vol1fn)
        vol2 = self.loadVol(vol2fn)
        outVolData = self.computeVolumeConsensus(vol1, vol2)
        self.saveVol(outVolData, outVol, vol1fn)

    def computeVolumeConsensus(self, vol1, vol2, wavelet='sym11'):
        coeffDict1 = pywt.swtn(vol1, wavelet, 1)
        coeffDict2 = pywt.swtn(vol2, wavelet, 1)
        newDict={}
        print("---------hola----------")
        for key in coeffDict1:
            newDict[key] = np.where(np.abs(coeffDict1[key]) > np.abs(coeffDict2[key]), coeffDict1[key], coeffDict2[key])
            print("-------------------", coeffDict1[key].shape)
        consensus = pywt.iswtn(newDict, wavelet)
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
