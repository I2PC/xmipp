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

from deepPickingConsensusTomo_networks import PREF_SIDE
from deepPickingConsensusTomo_dataman import Batch

BATCH_SIZE = 128
SAVE_AFTER = 25

class DataMan(object):
    """
    DataMan - Data Management objetc for deepPickingConsensus.
    This class can provide the NN with the required information for
    the NN to train and work.
    """

    def __init__(self, posDict, negDict, valFrac=0.15):
        """
        posDict: positive pickings
        negDict: negative pickings
        valFrac: fraction to use in the validation step
        """
        self.batch : Batch = None

        if valFrac > 0.3:
            valFrac = 0.3
            print("You must use less than 0.3 for validation")
        self.mdListFalse = None
        
        # Internal variables
        self.mdListTrue, self.fnMergedListTrue, self.weightListTrue, self.nTrue, self.shape = self.collectMD(posDict)
        self.nFalse = 0 #100% false subtomos
        self.batchSize = BATCH_SIZE
        self.split = self.batchSize // 2
        self.valFrac = valFrac
        # Know if MD has been loaded
        self.mdLoaded = False

        if valFrac !=0:
            assert 0 not in self.getNBatchesPerEpoch(), "Error, the number of positive particles for training is to small (%d). Must be >> %d"%(self.nTrue, BATCH_SIZE)
        else:
            assert self.getNBatchesPerEpoch()[0] != 0, "Error, the number of particles for testing is to small (%d). Must be >> %d"%(self.nTrue, BATCH_SIZE)
 
        if valFrac > 0:
            pass
        else:
            pass

        self.mdListFalse, self.fnMergedListFalse, self.weightListFalse, self.nFalse, shapeFalse = self.collectMD(negDict)
        assert shapeFalse == self.shape, "Not all images have the same shape!"
        # Randomly choose which 100% bad subtomos are for training
        self.trainIdsNeg = np.random.choice(self.nFalse, int((1-valFrac)*self.nFalse), False)
        # Randomly choose which 100% bad subtomos are for validating
        self.valIdsNeg = np.array()
    
    def collectMD(self, dataD):
        """
        Form the metadata structures needed for the NN to work
        dataD: dictionary to gather information about
        """

        # Generate empty structures
        MDList = []
        fnamesList_merged = []
        weightsList_merged = []
        nSubtomos = []
        subtomoShape = (None, None, None, 1)

        # Start reading with Xmipp image library
        for fname in sorted(dataD):
            weight = float(dataD[fname])
            mdObj = xmippLib.MetaData(fname)
            XI = xmippLib.Image()
            XI.read()
            # TODO: Equivalente en 3D?
            pass

        self.mdLoaded = True

    def getMetadata(self, which = None):
        """
        Return previously collected metadata
        """
        if not self.mdLoaded:
            return None
        if which is None:
            true = [t for t in self.mdListTrue]
            false = [f for f in self.mdListFalse]
            return true, false
        else:
            mdTrue = self.mdListTrue[which]
            mdFalse = self.mdListFalse[which]
            return mdTrue, mdFalse

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
    
    def _do_randomFlip_y(self, batch):
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