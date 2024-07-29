#!/usr/bin/env python3

# **************************************************************************
# *
# * Author:    Laura Baena Márquez
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
# **************************************************************************

#TODO: Change details, if any
#TODO: Check comments are used homogeneously
#TODO: Check if function names follow convention
#TODO: Eliminate library once the new generators are completed

#TODO: Should I use more self.stuff?

#TODO: eventually evaluate if test set is also going to be used Proposal (80%, 10% 10%)
#TODO: eventually evaluate the use of ensembles (add index to the model's fn name in bestModel)
#TODO: Evaluate if TensorBoard could be used to save info in a specific 

#TODO: Check imports

import numpy as np
import os
import sys
import xmippLib
from xmipp_base import XmippScript
from time import time

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Dense, concatenate, Activation, GlobalAveragePooling2D, Add
import keras
from keras.models import load_model
import tensorflow as tf

class ScriptDeepWrongAssignCheckScore(XmippScript):
    
    conda_env="xmipp_DLTK_v1.0" 
    
    def __init__(self):

        XmippScript.__init__(self)

    def defineParams(self):
        
        #TODO: define programUsage
        self.addUsageLine('')

        ## params to be read
        self.addParamsLine(' -i <fnResdInfer> : filename containg the unlabeled residuals (images) to be classified')
        
        self.addParamsLine(' -m <nnModel> : h5 filename where the model for inference is stored. In case of training, the final model will overwrite any existing data in this file.')

        self.addParamsLine(' -b <batchSize>: data`s subset size which will be fed to the network')

        self.addParamsLine(' -o <inferOutput> : filename where the inference results will be stored')

        #self.addParamsLine(' [ -t <ifTraining>]: (optional) write flag if training required')

        #self.addParamsLine(' [-r <fnResdTrain> ]: (optional) filename containg the residuals (images) to be used')

        #self.addParamsLine(' [ --pretrained ]: (optional) write flag if a pretained model will be used')

        #self.addParamsLine('[ -f <fnPretrainedModel> ]: (optional) filename of the pretrained model to be used instead of a new one')

        #TODO: Is this set in the program or in the protocol? (Also applied to learning rate and patience)
        #self.addParamsLine(' [-e <numEpoch> ]: (optional) number of epochs to train the model')
        
        #self.addParamsLine(' [ -l <learningRate=0.3> ]: (optional) learning rate used for the optimizer.')

        #self.addParamsLine(' [ -p <patience> ]: (optional) number of epochs with no improvement after which training will be stopped.')
        
        #TODO: Make sure how to use this beforehand
        self.addParamsLine(' [ --gpus <gpuId> ]: (optional) GPU ids to employ. Comma separated list. E.g. "0,1". Use -1 for CPU-only computation or -2 to use all devices found in CUDA_VISIBLE_DEVICES')

        ## examples ##TODO: modify example
        self.addExampleLine('deep_wrong_assign_check_sc -i path/to/inferenceResd -m path/to/finalModel -b $BATCH_SIZE -o path/to/programOutput')

    def run(self):

        #--------------- Initial comprobations and settings ----------------

        #TODO: Check if I can leave this like that
        from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed
        checkIf_tf_keras_installed()
        
        #If no specific GPUs are requested all available GPUs will be used
        if self.checkParam("--gpus"):
            os.environ["CUDA_VISIBLE_DEVICES"] = self.getParam("--gpus")

        #--------------- Function definitions ----------------
        
        #TODO: Write function definition
        # stores in memory the information contained in the images files
        def readFileInfo(fnImages):

            xDim, _, _, _, _ = xmippLib.MetaDataInfo(fnImages)
            mdAux = xmippLib.MetaData(fnImages)
            fnImgs = mdAux.getColumnValues(xmippLib.MDL_IMAGE)
        
            return xDim, fnImgs
            
        #TODO: Write function definition
        def getImage(fnImg, dim):

            img = np.reshape(xmippLib.Image(fnImg).getData(), (dim, dim, 1))
            return (img - np.mean(img)) / np.std(img)

        #TODO: Write function definition
        def dataManage(data,dim):
            
            for elem in data:
                yield getImage(elem,dim)
        
        #TODO: Write function definition + check general comments
        def performInference(inferData, model, batchSize, xDims):
            
            inferenceSet = tf.data.Dataset.from_generator(dataManage(inferData,None,xDims),tf.TensorSpec(shape = (), dtype = tf.variant))
            inferenceSet = inferenceSet.batch(batchSize, drop_reminder = False)
            
            # Returns a numPy array of predictions
            # if a value is 0.85, it means the model is 85% confident that the sample belongs to the positive class (1)
            # Values around 0.5 indicate uncertainty or low confidence in the prediction, suggesting the sample could belong to either class
            predictions = model.predict(x = inferenceSet)

            # Code to round into classes in case is needed (0.5 is upper int)
            # classPred = [round(x[0]) for x in predictions]

            return predictions
        
        #TODO: Write function definition
        def saveResults(fnInput, results, fnOutput):

            #TODO: check the use of self here

            outMd = xmippLib.MetaData(fnInput)
            for i, elem in enumerate(results):
                
                #TODO: Create label in core code + bindings (keep in mind recompile Xmipp)
                outMd.setValue(xmippLib.MDL_CLASS_PROBABILITY, float(elem), i+1)

            outMd.write(fnOutput)
            print("Written predictions to " + fnOutput)

            return

        #--------------- BASIC INPUT reading ----------------

        ## xmipp metadata of the residue images ready for inference (file name)
        fnXmdInfer = self.getParam("-i")
        if not os.path.isfile(fnXmdInfer):
            ## if the file doesn't exist the program will be interrupted
            print("Inference datafile does not exist inside path")
            sys.exit(-1)
        
        #TODO: When is this file created if empty? protocol?
        ## file name where the infernce model is stored either pretrained or from scratch in the program
        fnModel = self.getParam("-m")
        if not os.path.isfile(fnModel):
            ## if the file doesn't exist the program will be interrupted
            print(" Final model file does not exist inside path")
            sys.exit(-1)
        
        #TODO: When is this file created? protocol?
        ## file name where the inferece results will be stored at the end
        fnOutput = self.getParam("-o")
        if not os.path.isfile(fnOutput):
            ## if the file doesn't exist the program will be interrupted
            print("Output file does not exist inside path")
            sys.exit(-1)
        
        ## size of the batches to be used for the nn (both inference and training, if necessary)
        batchSz = int(self.getParam("-b"))

        #--------------- Executing core ----------------
        
        #TODO: Strategy?

        print("Starting inference procedure")

        ## inference is always executed
        xDim, inferFnImgs = readFileInfo(fnXmdInfer)
        inferenceStartTime = time()

        model = load_model(fnModel, compile = False)

        ## if the batch size is bigger than the data, only one batch is used
        if batchSz > len(inferFnImgs): batchSz = len(inferFnImgs)

        predictions = performInference(inferFnImgs, model, batchSz, xDim)
        
        inferenceElapsedTime = time() - inferenceStartTime 
        print("Time in infering results: %0.10f seconds." % inferenceElapsedTime)

        saveResults(fnXmdInfer, predictions, fnOutput)

        return

if __name__ == '__main__':
    exitCode = ScriptDeepWrongAssignCheckScore().tryRun()
    sys.exit(exitCode)

