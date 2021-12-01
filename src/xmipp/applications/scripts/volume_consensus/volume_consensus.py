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
        print("----------0----------")
        outputWt = None
        print("----------1----------")
        outputMin = None
        print("----------2----------")
        fnCoef = splitext(outVolFn)[0] + '_coef.txt'
        print("----------3----------")
        fhCoef = open(fnCoef, 'w')
        print("----------4----------")
        with open(inputFile) as f:
            for line in f:
                print("----------5----------")
                fileName = line.split()[0]
                print("----------6----------")
                if fileName.endswith('.mrc'):
                    fileName += ':mrc'
                print("----------7----------")
                V = xmippLib.Image(line.split()[0])
                print("----------8----------")
                vol = V.getData()
                print("----------9----------")
                nlevel = pywt.swt_max_level(len(vol))
                print("----------10----------")
                wt = pywt.swtn(vol, wavelet, nlevel, 0)
                print("----------11----------")
                if outputWt == None:
                    print("----------12----------")
                    outputWt = wt
                    print("----------13----------")
                    outputMin = wt[0]['aaa']*0
                    print("----------14----------")
                else:
                    print("----------15----------")
                    for level in range(0, nlevel):
                        print("----------16----------")
                        wtLevel = wt[level]
                        print("----------17----------")
                        outputWtLevel = outputWt[level]
                        print("----------18----------")
                        for key in wtLevel:
                            print("----------19----------")
                            outputWtLevel[key] = np.where(np.abs(outputWtLevel[key]) > np.abs(wtLevel[key]),
                                                          outputWtLevel[key], wtLevel[key])
                            print("----------20----------")
                            fhCoef.write(str(outputWtLevel[key]))
                            print("----------21----------")
                            diff = np.abs(np.abs(outputWtLevel[key]) - np.abs(wtLevel[key]))
                            print("----------22----------")
                            outputMin = np.where(outputMin > diff, outputMin, diff)
                            print("----------23----------")

            f.close()
            print("----------24----------")
        fhCoef.close()
        print("----------25----------")
        consensus = pywt.iswtn(outputWt, wavelet)
        print("----------26----------")
        V = xmippLib.Image()
        print("----------27----------")
        V.setData(consensus)
        print("----------28----------")
        V.write(outVolFn)
        print("----------29----------")
        V.setData(outputMin)
        print("----------30----------")
        outVolFn2 = splitext(outVolFn)[0] + '_diff.mrc'
        print("----------31----------")
        V.write(outVolFn2)
        print("----------32----------")


if __name__=="__main__":
    ScriptVolumeConsensus().tryRun()
