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
from os.path import splitext
import numpy as np
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
        outputWt = None
        outputMin = None
        fnCoef = splitext(outVolFn)[0] + '_coef.txt'
        fhCoef = open(fnCoef, 'w')
        with open(inputFile) as f:
            for line in f:
                fileName = line.split()[0]
                if fileName.endswith('.mrc'):
                    fileName += ':mrc'
                V = xmippLib.Image(line.split()[0])
                vol = V.getData()
                nlevel = pywt.swt_max_level(len(vol))
                wt = pywt.swtn(vol, wavelet, nlevel, 0)
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
                            fhCoef.write(str(outputWtLevel[key]))
                            diff = np.abs(np.abs(outputWtLevel[key]) - np.abs(wtLevel[key]))
                            outputMin = np.where(outputMin > diff, outputMin, diff)
            f.close()
        fhCoef.close()
        consensus = pywt.iswtn(outputWt, wavelet)
        V = xmippLib.Image()
        V.setData(consensus)
        V.write(outVolFn)
        V.setData(outputMin)
        outVolFn2 = splitext(outVolFn)[0] + '_diff.mrc'
        V.write(outVolFn2)


if __name__=="__main__":
    ScriptVolumeConsensus().tryRun()
