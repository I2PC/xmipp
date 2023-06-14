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
import os
import numpy as np

from xmipp_base import XmippScript
import xmippLib

MODEL_TRAIN_NEW = 0
MODEL_TRAIN_PRETRAIN = 1
MODEL_TRAIN_PREVRUN = 2
MODEL_TRAIN_TYPELIST= ["From scratch", "Existing model", "Previous run"]

class ScriptDeepConsensus3D(XmippScript):
    _conda_env = 'xmipp_DLTK_v1.0'

    # Type the parameters
    gpus : list # List of GPUs containing the ID of the ones to be used
    numThreads : int # Number of threads for parallel execution

    mode : str # Execution mode - train, score
    traintype : str # Type of training, for when mode is train
    truevolpaths : list # path/s to volumes deemed as  ground true
    falsevolpaths : list # path/s to volumes deemed as ground false
    trueweights : list # Weights for each of the files given as input
    falseweights : list # Weights for each of the files given as input
    posTrainDict : dict # Joining of file + weight
    negTrainDict : dict # Joining of file + weight

    inputfname : str # path for the MD file describing the volumes to be scored
    outputfname : str # desired output MD file name

    nepochs : int # Number of epoch for trainings
    learningrate : float # Learning rate for NN training
    regstrength : float # L2 regularisation
    autostop : bool # autostop flag for convergency stops
    ensemble : int # number of parallel models to run


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
        self.addParamsLine('[ -t <numThreads> ]      : Number of threads (Default: 4)')
        self.addParamsLine('[ -g <gpuId> ]           : space separated GPU Ids. Set to -1 to use all CUDA_VISIBLE_DEVICES (Default: no GPU)') 
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
        self.addParamsLine('[ --truevolfiles <path> ]  : ":" separated paths to the positive subtomos (xmd)')
        self.addParamsLine('[ --falsevolfiles <path>  ]  : ":" separated paths to the negative subtomos (xmd)')
        self.addParamsLine('[ --trueweights ]  : ":" separated positive subtomo file weights'
                            'Default: none - program autoweights -1:-1:-1...')
        self.addParamsLine('[ --falseweights]  : ":" separated negative subtomo file weights'
                           'Default: none - program autoweights -1:-1:-1...')
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

    def parseParams(self):
        """
        This function does the reading of input flags and parameters. It sanity-checks all
        inputs to make sure the program does not unnecesarily crash later.
        """
        
        # Default for CPU threads
        if self.checkParam('-t'):
            self.numThreads = self.getIntParam('-t')
        else:
            sysThreads = multiprocessing.cpu_count()
            self.numThreads = min(sysThreads, 4)

        # Default for GPU: no GPU
        if self.checkParam('-g'):
            # GPU acceleration
            gpustr : str = self.getParam('-g')
            self.gpus : list = [ int(item) for item in gpustr.split()]
            # -1 means all CUDA_VISIBLE_DEVICES must be used
            if -1 in self.gpus:
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    gpustr = os.environ.get('CUDA_VISIBLE_DEVICES')
                    self.gpus : list = [int(item) for item in gpustr.split(",")]
        else:
            # No GPU acceleration
            self.gpus = []

        # Netpath
        if self.checkParam('--netpath'):
            self.netpath = self.getParam('--netpath')
            if not os.path.isdir(self.netpath):
                print("Network path is not a valid path")
                sys.exit(-1)
        
        # Consensuated box size
        if self.checkParam('--consboxsize'):
            pass

        # Consensuated sampling rate
        if self.checkParam('--conssamprate'):
            pass
       
        # Mode
        if self.checkParam('--mode'):
            self.mode = str(self.getParam('--mode'))
            self.mode = self.mode.strip()

            # Define several words for the same 
            trainwords = ["train", "training", "t"]
            scorewords = ["score", "scoring", "s", "predict"]

            # The desired running mode is training
            if self.mode in trainwords:
                print("Execution mode is: TRAINING")
                self.mode = "train"

                # Determine the kind of training
                if self.checkParam('--ttype'):
                    self.traintype = int(self.getIntParam('--ttype'))
                    # Check if it's in the correct range and set to default if needed
                    if self.traintype not in [0, 1, 2]:
                        print("Training mode %d not recognized. Running a new model instead." % self.traintype)
                        self.traintype = MODEL_TRAIN_NEW
                    else:
                        print("Training in mode: " +  MODEL_TRAIN_TYPELIST[self.traintype])
                else: # If not specified, start from scratch
                    print("No training mode specified. Running a new model instead")
                    self.traintype = MODEL_TRAIN_NEW

                # General training checks
                # When training, you need to read the pos and neg paths as well as create
                # the needed weights (for the future) - for any of the cases
                truestring : str = self.getParam('--truevolfiles')
                falsestring : str = self.getParam('--falsevolfiles')
                self.truevolpaths = [ fn for fn in truestring.split(":") if os.path.isfile(fn) ]
                self.falsevolpaths = [ fn for fn in falsestring.split(":") if os.path.isfile(fn) ]
                
                # Assign weights to items if variable is set
                if self.checkParam('--trueweights'):
                    tw : str = self.getParam('--trueweights')
                    self.trueweights = np.array(tw.split(":"), dtype=int).tolist()
                else:
                    self.trueweights = [-1] * len(self.truevolpaths)
                if self.checkParam('--falseweights'):
                    tw : str = self.getParam('--falseweights')
                    self.falseweights = np.array(tw.split(":"), dtype=int).tolist()
                else:
                    self.falseweights = [-1] * len(self.falsevolpaths)
                
                # Check size
                msg_weights = "Error, the number of weights provided does not match the amount of files provided."+\
                            "Check --trueweights and --falseweights"
                assert len(self.falseweights) == len(self.truevolpaths), msg_weights
                assert len(self.trueweights) == len(self.falsevolpaths), msg_weights

                # Create the dictionaries
                self.posTrainDict = {path: weight for path, weight in zip(self.truevolpaths, self.trueweights)}
                self.negTrainDict = {path: weight for path, weight in zip(self.falsevolpaths, self.falseweights)}

                # Epochs for training
                if self.checkParam('-e'):
                    self.nepochs = int(self.getIntParam('-e'))
                else:
                    self.nepochs = 5

                # Learning rate
                if self.checkParam('-l'):
                    self.learningrate = float(self.getDoubleParam('-l'))
                else:
                    self.learningrate = 1.0e-4

                # Regularization strenght
                if self.checkParam('-r'):
                    self.regstrength= float(self.getDoubleParam('-r'))
                else:
                    self.regstrength = 1.0e-5

                # Auto stop feature
                if self.checkParam('-s'):
                    self.autostop = bool(self.getParam('-s'))
                else:
                    self.autostop = True

                # Ensemble amount
                if self.checkParam('--ensemble'):
                    self.ensemble = int(self.getIntParam('--ensemble'))
                else:
                    self.ensemble = 1


            # The desired mode is scoring
            elif self.mode in scorewords:
                print("Execution mode is: SCORING")
                self.mode = "score"

                if self.checkParam('-i'):
                    pass
                else:
                    print("No input volume files found")
                    sys.exit(-1)
                
                # Output naming
                if self.checkParam('-o'):
                    self.outputfname = self.getParam('-o')
                else:
                    print("No output name given. Defaulting to scored_output.xmd")
                    self.outputfname = "scored_output.xmd"
                

    def run(self):

        print("deep_picking_consensus_tomo.py run() is launched\nParsing input...")
        self.parseParams()
        print("Execution will be done using %d threads." % self.numThreads)
        gpustring = str(self.gpus)
        print("Execution will use GPUS with ID: " + gpustring)

        # TODO: generate args for train or score
        # TODO: launch train o test worker
        sys.exit(0)

    @staticmethod
    def trainingWorker():
        # TODO: definir entradas
        pass

    @staticmethod
    def scoringWorker():
        # TODO: definir entradas
        pass

if __name__ == '__main__':
    exitCode = ScriptDeepConsensus3D().run()
    sys.exit(exitCode)
        

        
