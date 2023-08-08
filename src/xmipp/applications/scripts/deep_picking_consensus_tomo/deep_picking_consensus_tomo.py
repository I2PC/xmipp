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
* Initial version: sept 2023
**************************************************************************
"""
import multiprocessing
import sys
import os
import numpy as np

from xmipp_base import XmippScript
import xmippLib

from xmippPyModules.deepPickingConsensusTomo.deepPickingConsensusTomo_networks import NetMan
from xmippPyModules.deepPickingConsensusTomo.deepPickingConsensusTomo_dataman import DataMan

MODEL_TRAIN_NEW         = 0
MODEL_TRAIN_PRETRAIN    = 1
MODEL_TRAIN_PREVRUN     = 2
MODEL_TRAIN_TYPELIST    = ["From scratch", "Existing model", "Previous run"]

NN_TRAINWORDS           = ["train", "training", "t"]
NN_SCOREWORDS           = ["score", "scoring", "s", "predict"]

DEFAULT_MP              = 8

class ScriptDeepConsensus3D(XmippScript):
    
    _conda_env = 'scipion3'

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
        self.addParamsLine('[ -t <numThreads=4> ] : Number of threads')
        self.addParamsLine(' -g <gpuId> : comma separated GPU Ids. Set to -1 to use all CUDA_VISIBLE_DEVICES') 
        self.addParamsLine(' --mode <execMode> : training or scoring')
        self.addParamsLine(' --netpath <netpath> : path for network models read/write (needed in any case)')

        # Tomo
        self.addParamsLine('==== Tomo ====')
        self.addParamsLine(' --consboxsize <boxsize> : desired box size (int)')
        self.addParamsLine(' --conssamprate <samplingrate> : desired sampling rate (float)')

        # Score parameters
        self.addParamsLine('==== Scoring mode ====')
        self.addParamsLine('[ --inputvolpath <path> ]   : path to the metadata files of the input doubtful subtomos (mrc)')
        self.addParamsLine('[ --outputfile <path> ]   : path for the program to write the scored coordinates (xmd)')

        # Train parameters
        self.addParamsLine('==== Training mode ====')
        self.addParamsLine(' --ttype <traintype=0> : train mode')
        self.addParamsLine(' --truevolpath <truevolpath> : path to the positive subtomos (mrc)')
        self.addParamsLine(' --falsevolpath <falsevolpath> : path to the negative subtomos (mrc)')
        self.addParamsLine('[ -e <numberOfEpochs=5> ]  : Number of training epochs (int).')
        self.addParamsLine('[ -l <learningRate=0.0001> ] : Learning rate (float).')
        self.addParamsLine('[ -r <regStrength=0.00001> ] : L2 regularization level (float).')
        self.addParamsLine('[ -s <AutoStop>] : Autostop on convergency detection.')
        self.addParamsLine('[ --ensemble <numberOfModels=1> ] : If set, an ensemble of models will be used in a voting instead one.')

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
            self.numThreads = DEFAULT_MP
        # GPU acceleration - assuming GPU is always used
        gpustr : str = self.getParam('-g')
        self.gpus : list = [ int(item) for item in gpustr.split(",")]
        # -1 means all CUDA_VISIBLE_DEVICES must be used
        if -1 in self.gpus:
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                gpustr = os.environ.get('CUDA_VISIBLE_DEVICES')
                self.gpus : list = [int(item) for item in gpustr.split(",")]
        # Execution mode
        self.execMode = str(self.getParam('--mode'))
        # Netpath
        self.netPath = self.getParam('--netpath')
        if not os.path.isdir(self.netPath):
            print("Network path is not a valid path")
            sys.exit(-1)
        # Consensuated boxsize and sampling ratesize
        self.consBoxSize : int = self.getIntParam('--consboxsize')
        self.consSampRate : float = self.getDoubleParam('--conssamprate')
       
        # The desired running mode is training
        if self.execMode.strip() in NN_TRAINWORDS:
            self.execMode = "train"
            print("Execution mode is: TRAINING")

            # Training type
            self.trainType = str(self.getParam('--ttype'))
            # Read paths
            self.posPath : str = self.getParam('--truevolpath')
            self.negPath : str = self.getParam('--falsevolpath')
            # Learning rate
            if self.checkParam('-l'):
                self.learningRate = float(self.getDoubleParam('-l'))
            else:
                self.learningRate = 1.0e-4
            # Epochs for training
            if self.checkParam('-e'):
                self.nEpochs = int(self.getIntParam('-e'))
            else:
                self.nEpochs = 5
            # Regularization strenght
            if self.checkParam('-r'):
                self.regStrength= float(self.getDoubleParam('-r'))
            else:
                self.regStrength = 1.0e-5
            # Auto stop feature
            self.autoStop = self.checkParam('-s')
            # Ensemble amount
            if self.checkParam('--ensemble'):
                self.ensemble = self.getIntParam('--ensemble')
            else:
                self.ensemble = 1
            # Check if it's in the correct range and set to default if needed
            if self.traintype not in [0, 1, 2]:
                print("Training mode %d not recognized. Running a new model instead." % self.traintype)
                self.traintype = MODEL_TRAIN_NEW
            else:
                print("Training in mode: " +  MODEL_TRAIN_TYPELIST[self.traintype])
                          
            # # Assign weights to items if variable is set
            # if self.checkParam('--trueweights'):
            #     tw : str = self.getParam('--trueweights')
            #     self.trueweights = np.array(tw.split(":"), dtype=int).tolist()
            # else:
            #     self.trueweights = [-1] * len(self.truevolpaths)
            # if self.checkParam('--falseweights'):
            #     tw : str = self.getParam('--falseweights')
            #     self.falseweights = np.array(tw.split(":"), dtype=int).tolist()
            # else:
            #     self.falseweights = [-1] * len(self.falsevolpaths)
            
            # Check size
            # msg_weights = "Error, the number of weights provided does not match the amount of files provided."+\
            #            "Check --trueweights and --falseweights"
            # assert len(self.falseweights) == len(self.truevolpaths), msg_weights
            # assert len(self.trueweights) == len(self.falsevolpaths), msg_weights

            # Create the dictionaries
            # self.posTrainDict = {path: weight for path, weight in zip(self.truevolpaths, self.trueweights)}
            # self.negTrainDict = {path: weight for path, weight in zip(self.falsevolpaths, self.falseweights)}

        # The desired mode is scoring
        elif self.execMode.strip() in NN_SCOREWORDS:
            print("Execution mode is: SCORING")
            self.execMode = "score"

            # Input/Output
            self.inputVolPath = str(self.getParam('--inputvolpath'))
            if not os.path.exists(self.inputVolPath):
                print("Path to input subtomograms does not exist. Exiting.")
                sys.exit(-1)
            self.outputFile = str(self.getParam('--outputfile'))
                
    def run(self):

        print("deep_picking_consensus_tomo.py is launched\nParsing input...")
        self.parseParams()
        print("Execution will be done using %d threads." % self.numThreads)
        gpustring = str(self.gpus)
        print("Execution will use GPUS with ID: " + gpustring)

        if self.mode == "train":
            dataMan = DataMan(self.posPath, self.negPath, self.consBoxSize)
            self.doTrain(dataMan)

        elif self.mode == "score":
            pass
            dataMan = DataMan()
            self.doScore()
        else:
            print("Execution mode not specified, exiting...")
            sys.exit(-1)


        sys.exit(0)

    def doTrain(self, dataMan):
        
        netMan = NetMan(self.numThreads, self.gpus, self.netpath)
        netMan.createNetwork(self.consBoxSize, self.consBoxSize, self.consBoxSize, 1)
        netMan.compiledNetwork.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        netMan.trainNetwork(self.nepochs, dataMan, self.learningrate, autoStop=True)

    def doScore():
        pass

if __name__ == '__main__':
    exitCode = ScriptDeepConsensus3D().tryRun()
    sys.exit(exitCode)
