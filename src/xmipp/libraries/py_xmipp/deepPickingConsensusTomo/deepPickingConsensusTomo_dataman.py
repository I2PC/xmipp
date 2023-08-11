#!/usr/bin/env python3
# **************************************************************************
# *
# * Authors:    Mikel Iceta Tena (miceta@cnb.csic.es)
# *
# * Unidad de Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'coss@cnb.csic.es'
# *
# *
# * Neural Network utils for the deep picking consensus protocol found in
# * scipion-em-xmipptomo
# *
# * Initial release: june 2023
# **************************************************************************

import numpy as np
import random
import scipy

import sys, os

import xmippLib


BATCH_SIZE = 128
N_BATCHES = 100
SAVE_AFTER = 25

class DataMan(object):
    """
    DataMan - Data Management objetc for deepPickingConsensus.
    This class can provide the NN with the required information for
    the NN to train and work.
    """

    valFrac : float # Fraction for validation steps
    nNeg : int # Total number of negative examples
    nPos : int # Total number of positive examples
    nDoubt : int # Total number of doubt files
    batchSize : int # Batch size for NN
    train : bool # To see if training or testing
    

    def __init__(self, boxSize: int, valFrac=0.15, batchSize = BATCH_SIZE, posPath: str = None, negPath: str = None, doubtPath:str = None):
        """
        boxSize: consensuated box size
        samplingRate: consensuated sampling rate
        valFrac: fraction of the total used for validation - Default 0.15
        pospath: positive pickings path - If None, scoring is assumed
        negpath: negative pickings path - If None, scoring is assumed
        doubtpath: doubtful pickings path - If None, training is assumed
        """

        self.train = (posPath is not None) and (negPath is not None)

        # Variables that are always needed
        self.boxSize = boxSize
        self.batchSize = batchSize

        if self.train:
            # TRAIN
            print("DataMan: training selected, loading pos+neg examples")
            # MD Loading
            self.posVolsFns = self.getFolderContent(posPath, ".mrc")
            self.nPos = len(self.posVolsFns)
            self.negVolsFns = self.getFolderContent(negPath, ".mrc")
            self.nNeg = len(self.negVolsFns)
            # Only interested in validation fraction if training
            self.valFrac = valFrac

            # Shuffle the input files and create separate sets with the filenames for later use
            if valFrac > 0:
                self.trainingFnsPos= random.choices(self.posVolsFns, k=int((1-valFrac)*self.nPos))
                self.validationFnsPos= list(set(self.posVolsFns).difference(self.trainingFnsPos))

                self.trainingFnsNeg = random.choices(self.negVolsFns, k=int((1-valFrac)*self.nNeg))
                self.validationFnsNeg = list(set(self.negVolsFns).difference(self.trainingFnsNeg))
            else:
                self.trainingFnsPos = self.posVolsFns
                self.validationFnsPos = None
                self.trainingFnsNeg = self.negVolsFns
                self.validationFnsPos = None
            
        else:
            # SCORE
            print("DataMan: scoring selected, loading doubt examples")
            # MD Loading
            self.doubtVolsFns = self.getFolderContent(doubtPath, ".mrc")
            self.nDoubt = len(self.doubtVolsFns)

    def getNBatchesPerEpoch(self):
        return (int((1-self.valFrac)*self.nPos*2./self.batchSize),
             int(self.valFrac*self.nPos*2./self.batchSize))

    def getFolderContent(self, path: str, filter: str) -> list :
        fns = os.listdir(path)
        return [ path + "/" + fn for fn in fns if filter in fn] 
    
    # def getDataIterator(self, stage, nEpochs=-1, nBatches=N_BATCHES):
    def getDataIterator(self, stage, nEpochs=-1, nBatches=getNBatchesPerEpoch):

        if nEpochs < 0:
            nEpochs = sys.maxsize
        for i in range(nEpochs):
            for batch in self._getOneEpochTrainOrValidate(stage = stage, nBatches = nBatches):
                yield batch

    def _getOneEpochTrainOrValidate(self, stage, nBatches):

        # Volumes and labels for this batch
        x = np.empty((self.batchSize, self.boxSize, self.boxSize, self.boxSize))
        y = np.empty((self.batchSize, 2))

        if stage == "train":
            posFns = self.trainingFnsPos
            negFns = self.trainingFnsNeg
            
        elif stage == "validate":
            posFns = self.validationFnsPos
            negFns = self.validationFnsNeg
        
        image = xmippLib.Image()
        for _ in range(nBatches):
            for i in range(self.batchSize):
                label = random.randrange(2)
                if label == 0:
                    filename = random.choice(negFns)
                else:
                    filename = random.choice(posFns)

                image.read(filename)
                xmippIm : np.ndarray = image.getData()
                x[i] = xmippIm
                y[i,0] = label
                y[i,1] = 1 - label
            yield x,y

    # DATA AUGMENTATION METHODS

    def _do_randomFlip_x(self, batch):
        output = []
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                output.add()
        return output         

    def _do_randomFlip_y(self, batch):
        output = []
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                output.add()
        return output
    
    def _do_randomFlip_z(self, batch):
        output = []
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                output.add()
        return output

    def _do_randomRot(self, max_angle, batch):
        output = []
        for i in range(len(batch)):
            # Random angle
            angle = random.uniform(-max_angle, max_angle)
            # Dim0
            if bool(random.getrandbits(1)):
                self.data.add()
            # Dim1
            if bool(random.getrandbits(1)):
                self.data.add()
        return output

    def _do_randomBlur(self, batch):
        output = []
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                self.data.add()
        return output

    # DATA AUGMENTATION
    def augmentBatch(self, batch):
        """
        When called, it will permorm random operations to generate more
        information based on the input dataset (data augmentation).
        """
        if bool(random.getrandbits(1)):
            self._do_randomFlip_x(batch)
        if bool(random.getrandbits(1)):
            self._do_randomFlip_y(batch)
        if bool(random.getrandbits(1)):
            self._do_randomFlip_z(batch)
        if bool(random.getrandbits(1)):
            self._do_randomRot(batch)
        if bool(random.getrandbits(1)):
            self._do_randomBlur(batch)