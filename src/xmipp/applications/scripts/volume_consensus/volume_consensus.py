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
import math
import numpy as np
import pywt
import pywt.data
from scipy.ndimage import zoom
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
        xdim2 = None
        xdimOrig = None
        with open(inputFile) as f:
            for line in f:
                fileName = line.split()[0]
                if fileName.endswith('.mrc'):
                    fileName += ':mrc'
                V = xmippLib.Image(fileName)
                vol = V.getData()
                if xdimOrig is None:
                    xdimOrig = vol.shape[0]
                    xdim2 = 2**(math.ceil(math.log(xdimOrig, 2))) # Next power of 2
                    ydimOrig = vol.shape[1]
                    ydim2 = 2 ** (math.ceil(math.log(ydimOrig, 2)))  # Next power of 2
                    zdimOrig = vol.shape[2]
                    zdim2 = 2 ** (math.ceil(math.log(zdimOrig, 2)))  # Next power of 2
                if xdimOrig!=xdim2 or ydimOrig!=ydim2 or zdimOrig!=zdim2:
                    vol = zoom(vol, (xdim2/xdimOrig,ydim2/ydimOrig,zdim2/zdimOrig))
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
        if xdimOrig!=xdim2 or ydimOrig!=ydim2 or zdimOrig!=zdim2:
            consensus = zoom(consensus, (xdimOrig/xdim2,ydimOrig/ydim2,zdimOrig/zdim2))
        V = xmippLib.Image()
        V.setData(consensus)
        V.write(outVolFn)
        V.setData(outputMin)
        outVolFn2 = splitext(outVolFn)[0] + '_diff.mrc'
        V.write(outVolFn2)


if __name__=="__main__":
    ScriptVolumeConsensus().tryRun()
