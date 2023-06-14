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

from xmipp_base import XmippScript
import xmippLib

class ScriptDeepConsensus3D(XmippScript):
    _conda_env = 'xmipp_DLTK_v1.0'

    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('DeepConsensus3D. Launches a CNN to process cryoET (tomo) subvolumes.\n'
                          'It can be used in these cases:\n'
                          '1) Train network from scratch\n'
                          '2) Load a network and keep training\n'
                          '3) Load a network and score\n'
                          'Keep in mind that training and scoring are separated, so two separate\n'
                          'calls to this program are needed for a train+score scenario')

        # Application parameters
        self.addParamsLine('==== Application ====')
        self.addParamsLine('[ -g <gpuId> ]           : GPU Id. Set to -1 to use all CUDA_VISIBLE_DEVICES (Default: no GPU)')
        self.addParamsLine('[ -t <numThreads> ]      : Number of threads (Default: 4)')
        self.addParamsLine('[ --mode <mode> }        : training | scoring')
        self.addParamsLine('[ --netpath <path> ]     : path for network models read/write (needed in any case)')

        # Tomo
        self.addParamsLine('==== Tomo ====')
        self.addParamsLine('[ --consboxsize <boxsize> ]     : desired box size (int)')
        self.addParamsLine('[ --conssamprate <samplingrate>]: desired sampling rate (float)')
        self.addParamsLine('[]:')

        # Score parameters
        self.addParamsLine('==== Scoring mode ====')
        self.addParamsLine('[ -i <path> ]   : path to the metadata of the input subtomos (xmd)')
        self.addParamsLine('[ -o <path> ]   : path for the program to write the scored coordinates (xmd)')

        # Train parameters
        self.addParamsLine('==== Training mode ====')
        self.addParamsLine('[ --truevols <path> ]  : path to the positive subtomos (xmd)')
        self.addParamsLine('[ --falsevols <path>  ]  : path to the negative subtomos (xmd)')
        self.addParamsLine('[ --trueweights <path> ]  : path to positive subtomos weights (xmd) '
                            'Default: none - program autoweights')
        self.addParamsLine('[ --falseweights <path> ]  : path to negative subtomos weights (xmd) '
                           'Default: none - program autoweights')
        self.addParamsLine('[ -e <numberOfEpochs> ]  : Number of training epochs (int). Default: 5')
        self.addParamsLine('[ -l <learningRate> ] : Learning rate (float). Default: 1e-4')
        self.addParamsLine('[ -r <regStrength> ] : L2 regularization level (float). Default: 1e-5')
        self.addParamsLine('[ -s <autoStop>]  : Autostop on convergency detection (boolean). Default: True')
        self.addParamsLine('[ --ensemble <numberOfModels> ] : If set, an ensemble of models will be used in a voting instead one.')

        # Use examples
        self.addExampleLine('Training the network from scratch\n'
                            'xmipp_deep_picking_consensus_tomo')
        
        self.addExampleLine('Keep training the network from previous run\n'
                            'xmipp_deep_picking_consensus_tomo')
        
        self.addExampleLine('Training the network from scratch\n'
                            'xmipp_deep_picking_consensus_tomo -')

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
        

        
