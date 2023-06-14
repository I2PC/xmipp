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

BATCH_SIZE = 128
SAVE_AFTER = 25

class DataMan(object):
    """
    DataMan - Data Management objetc for deepPickingConsensus.
    This class can provide the NN with the required information for
    the NN to train and work.
    """

    valFrac : float # Fraction for validation steps
    nFalse : int # Total number of negative examples
    nTrue : int # Total number of positive examples
    mdLoaded : bool # Is metadata loaded?
    splitPoint : int # Where to split the set
    batchSize : int # Batch size for NN
    

    def __init__(self, posfn: str, negfn:str, valFrac=0.15):
        """
        posfn: positive pickings XMD filename
        negfn: negative pickings XMD filename
        valFrac: fraction to use in the validation step
        """

        # MD Loading
        self.boxsize, self.posVolsFns = self.collectMD(posfn)
        _, self.negVolsFns = self.collectMD(negfn)
        self.batchSize = BATCH_SIZE
        self.splitPoint = self.batchSize // 2
        self.valFrac = valFrac
    
    def collectMD(self, filename: str):
        """
        Form the metadata structures needed for the NN to work
        dataD: dictionary to gather information about
        """

        md = xmippLib.MetaData(filename)
        md.getColumnValues(xmippLib.MDL_IMAGE, )


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