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
        xdim2 = None
        xdimOrig = None
        image = xmippLib.Image()
        with open(inputFile) as f:
            for line in f:
                fileName = line.split()[0]
                if fileName.endswith('.mrc'):
                    fileName += ':mrc'
                    
                image.read(fileName)
                volume = image.getData()
                
                if xdimOrig is None:
                    xdimOrig = volume.shape[0]
                    xdim2 = 2**(math.ceil(math.log2(xdimOrig))) # Next power of 2
                    ydimOrig = volume.shape[1]
                    ydim2 = 2**(math.ceil(math.log2(ydimOrig)))  # Next power of 2
                    zdimOrig = volume.shape[2]
                    zdim2 = 2**(math.ceil(math.log2(zdimOrig)))  # Next power of 2
                    
                if xdimOrig!=xdim2 or ydimOrig!=ydim2 or zdimOrig!=zdim2:
                    volume = zoom(volume, (xdim2/xdimOrig,ydim2/ydimOrig,zdim2/zdimOrig))
                
                nlevel = pywt.dwtn_max_level(volume.shape, wavelet=wavelet)
                wt = pywt.wavedecn(
                    data=volume, 
                    wavelet=wavelet, 
                    level=nlevel
                )
                
                if outputWt == None:
                    outputWt = wt
                    #outputMin = np.zeros_like(wt[1]['aaa'])
                else:
                    outputWt[0] = np.where(
                        np.abs(wt[0]) > np.abs(outputWt[0]),
                        wt[0], outputWt[0]
                    )
                            
                    for level in range(1, nlevel+1):
                        wtLevel = wt[level]
                        outputWtLevel = outputWt[level]
                        for detail in wtLevel:
                            wtLevelDetail = wtLevel[detail]
                            outputWtLevelDetail = outputWtLevel[detail]
                            
                            outputWtLevelDetail[...] = np.where(
                                np.abs(wtLevelDetail) > np.abs(outputWtLevelDetail),
                                wtLevelDetail, outputWtLevelDetail
                            )
                            
                            """
                            diff = np.abs(np.abs(wtLevelDetail) - np.abs(outputWtLevelDetail))
                            np.maximum(
                                diff, outputMin,
                                out=outputMin
                            )
                            """
     
            f.close()
        consensus = pywt.waverecn(outputWt, wavelet)
        if xdimOrig!=xdim2 or ydimOrig!=ydim2 or zdimOrig!=zdim2:
            consensus = zoom(consensus, (xdimOrig/xdim2,ydimOrig/ydim2,zdimOrig/zdim2))
        image.setData(consensus)
        image.write(outVolFn)
        #image.setData(outputMin)
        #outVolFn2 = splitext(outVolFn)[0] + '_diff.mrc'
        #image.write(outVolFn2)


if __name__=="__main__":
    ScriptVolumeConsensus().tryRun()
