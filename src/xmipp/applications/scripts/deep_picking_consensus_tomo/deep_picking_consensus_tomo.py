#!/usr/bin/env python3
""""
**************************************************************************
*
* Authors:  Mikel Iceta Tena (miceta@cnb.csic.es)
* 
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
* Initial version: nov 2022
**************************************************************************
"""
import multiprocessing
import sys

#class ScriptDeepConsensus3D(XmippScript):
#    _conda_env= CondaEnvManager.CONDA_DEFAULT_ENVIRON
#    def __init__(self):
#        XmippScript.__init__(self)

class ScriptDeepConsensus3D():

    def defineParams(self):
        self.addUsageLine()

        # Application parameters
        self.addParamsLine('[ -g <gpuId> ]      : (ONLY ONE) GPU Id. (Default: no GPU acceleration)')
        self.addParamsLine('[ -t <numThreads>   : Number of threads (Default: 4) ]')
        self.addParamsLine('[ --mode <mode>     : training|scoring ]')

        # Train parameters

        # Use examples

    def run(self):
        numThreads = None
        gpu = None

        # Parse and set parameters regarding CPU and GPU
        if self.checkParam('t'):
            numThreads = self.getIntParam('t')
        else:
            sysThreads = multiprocessing.cpu_count()
            numThreads = min(sysThreads, 4)

        if self.checkParam('-g'):
            gpu = self.getIntParam('-g')

        gpuString = ("and GPU (id: " + gpu + ")") if gpu is not None else ""
        print("Starting execution with " + numThreads + gpuString + ".")
        
        return 0

if __name__ == '__main__':
    exitCode = ScriptDeepConsensus3D().run()
    sys.exit(exitCode)
        

        
