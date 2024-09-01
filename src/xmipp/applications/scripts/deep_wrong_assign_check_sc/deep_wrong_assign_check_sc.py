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

#TODO: Should I use more self.stuff? (YES)

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

#TODO: Is any callback necessary?
#TODO: make sure we are not using Keras 3
import tensorflow as tf
from tensorflow.keras.models import load_model


class ScriptDeepWrongAssignCheckScore(XmippScript):
    
    conda_env="xmipp_DLTK_v1.0" 
    
    def __init__(self):

        XmippScript.__init__(self)

    def defineParams(self):
        
        #TODO: define programUsage
        self.addUsageLine('')

        #TODO: check names make sense

        ## params to be read
        self.addParamsLine(' -i <fnInferResd> : filename containg the unlabeled residuals (images) to be classified')
        self.addParamsLine(' -m <nnModel> : h5 filename where the model for inference is stored. In case of training, the final model will overwrite any existing data in this file.')
        self.addParamsLine(' -o <inferOutput> : filename where the inference results will be stored')
        self.addParamsLine(' -b <batchSize>: data`s subset size which will be fed to the network')
        
        #TODO: What should be the default behavior? (keep in mind any changes in the othe program)
        self.addParamsLine(' [ -t <nThreads=1> ]: (optional) number of threads to use in multiprocessing.')

        self.addParamsLine(' [ --gpus <gpuId> ]: (optional) GPU ids to employ. Comma separated list. E.g. "0,1". Use -1 for CPU-only computation or -2 to use all devices found in CUDA_VISIBLE_DEVICES')

        ## examples
        self.addExampleLine('xmipp_deep_wrong_assign_check_sc -i path/to/inferenceSet -m path/to/pretrainedModel -o path/to/outputFile -b $BATCH_SIZE ')

    #TODO: write function definition
    def convertInputs(self):

        ## Xmipp metadata of the residue images ready for inference (file name)
        self.fnInput = self.getParam("-i")
        if not os.path.isfile(self.fnInput):
            ## If the file doesn't exist the program will be interrupted
            print("Inference datafile does not exist inside path")
            sys.exit(-1)

        self.xDim, _, _, _, _ = xmippLib.MetaDataInfo(self.fnInput)

        self.inferFnImgs = self.readFileInfo(self.fnInput)
        
        ## File name where the infernce model is stored either pretrained or from scratch in the program
        self.fnModel = self.getParam("-m")
        if not os.path.isfile(self.fnModel):
            ## If the file doesn't exist the program will be interrupted
            print(" Final model file does not exist inside path")
            sys.exit(-1)
        
        ## File name where the inferece results will be stored at the end
        self.fnOutput = self.getParam("-o")
        if not os.path.isfile(self.fnOutput):
            ## If the file doesn't exist the program will be interrupted
            print("Output file does not exist inside path")
            sys.exit(-1)
        
        #TODO: Keep in mind the name changed, check the rest of the code
        ## Size of the batches to be used for the nn
        self.batchSize = int(self.getParam("-b"))

        ## If the batch size is bigger than the data or the user requested no batches (using 0), only one batch is used
        if self.batchSize > len(self.inferFnImgs) or self.batchSize == 0: self.batchSize = len(self.inferFnImgs)

        ## Number of threads to be used in multiprocessing
        self.nThreads = int(self.getParam("-t"))
        if self.nThreads < 1:
            self.nThreads = 1

        #If no specific GPUs are requested all available GPUs will be used
        if self.checkParam("--gpus"):
            os.environ["CUDA_VISIBLE_DEVICES"] = int(self.getParam("--gpus"))

    #TODO: Write function definition
    def readFileInfo(self,fnImages):

        mdAux = xmippLib.MetaData(fnImages)
        fnImgs = mdAux.getColumnValues(xmippLib.MDL_IMAGE)
    
        return fnImgs
    
    #TODO: Write function definition
    def getImage(self, fnImg):

        img = np.reshape(xmippLib.Image(fnImg).getData(), (self.xDim, self.xDim, 1))
        return (img - np.mean(img)) / np.std(img)

    #--------------- Neural Network Generators ----------------

    #TODO: Write function definition
    def manageData(self):
        
        for elem in self.inferFnImgs:

            yield self.getImage(elem)

    #--------------- Scoring function ----------------

    #TODO: Write function definition + check general comments
    def performInference(self):
        
        #TODO: evaluate use of tensorSpec
        inferenceSet = tf.data.Dataset.from_generator(self.manageData(),tf.TensorSpec(shape = (), dtype = tf.variant))
        inferenceSet = inferenceSet.batch(self.batchSize, drop_reminder = False)

        model = load_model(self.fnModel, compile = False)
        
        ## Returns a numPy array of predictions
        ## if a value is 0.85, it means the model is 85% confident that the sample belongs to class 1 
        ## Values around 0.5 indicate uncertainty or low confidence in the prediction, suggesting the sample could belong to either class
        predictions = model.predict(x = inferenceSet, workers = self.nThreads, use_multiprocessing = True)

        ## Code to round into binary classes in case is needed (0.5 is upper int)
        ## classPred = [round(x[0]) for x in predictions]

        return predictions
    
    #--------------- Output preparation function ----------------

    #TODO: Write function definition
    def saveResults(self):

        #TODO: check the use of self here

        #TODO: evaluate use setColumnValues instead of the loop
        outMd = xmippLib.MetaData(self.fnInput)
        for i, elem in enumerate(self.predictions):
            
            #TODO: Evaluate changing label name into something similar to score
            #TODO: Create label in core code + bindings (keep in mind recompile Xmipp)
            ## Keep in mind that XMD files start indexing in 1, therefore "i+1" to make them coincide
            outMd.setValue(xmippLib.MDL_CLASS_PROBABILITY, float(elem), i+1)

        outMd.write(self.fnOutput)
        print("Written predictions to " + self.fnOutput)
    
    #--------------- Program Execution Function ----------------

    def run(self):
        
        #TODO: Strategy?

        print("Starting inference")

        inferenceStartTime = time()

        self.convertInputs()
        self.predictions = self.performInference()
        
        inferenceElapsedTime = time() - inferenceStartTime 
        print("Time in infering results: %0.10f seconds." % inferenceElapsedTime)

        self.saveResults()


if __name__ == '__main__':
    exitCode = ScriptDeepWrongAssignCheckScore().tryRun()
    sys.exit(exitCode)

###############################Old#Code#Saved#In#Case#Of#Loss#######################################################

'''
#If no specific GPUs are requested all available GPUs will be used
if self.checkParam("--gpus"):
    os.environ["CUDA_VISIBLE_DEVICES"] = int(self.getParam("--gpus"))
'''
'''
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
def manageData(data,dim):
    
    for elem in data:
        yield self.getImage(elem,dim)

#TODO: Write function definition + check general comments
def performInference(self, inferData, model, batchSize, xDims):
    
    inferenceSet = tf.data.Dataset.from_generator(self.manageData(inferData,None,xDims),tf.TensorSpec(shape = (), dtype = tf.variant))
    inferenceSet = inferenceSet.batch(batchSize, drop_reminder = False)
    
    ## Returns a numPy array of predictions
    ## if a value is 0.85, it means the model is 85% confident that the sample belongs to class 1 
    ## Values around 0.5 indicate uncertainty or low confidence in the prediction, suggesting the sample could belong to either class
    predictions = model.predict(x = inferenceSet)

    ## Code to round into binary classes in case is needed (0.5 is upper int)
    ## classPred = [round(x[0]) for x in predictions]

    return predictions

#TODO: Write function definition
def saveResults(fnInput, results, fnOutput):

    #TODO: check the use of self here

    #TODO: evaluate use setColumnValues instead of the loop
    outMd = xmippLib.MetaData(fnInput)
    for i, elem in enumerate(results):
        
        #TODO: Create label in core code + bindings (keep in mind recompile Xmipp)
        ## Keep in mind that XMD files start indexing in 1, therefore "i+1" to make them coincide
        outMd.setValue(xmippLib.MDL_CLASS_PROBABILITY, float(elem), i+1)

    outMd.write(fnOutput)
    print("Written predictions to " + fnOutput)

    return
'''
'''
## Xmipp metadata of the residue images ready for inference (file name)
fnInput = self.getParam("-i")
if not os.path.isfile(fnInput):
    ## If the file doesn't exist the program will be interrupted
    print("Inference datafile does not exist inside path")
    sys.exit(-1)

## File name where the infernce model is stored either pretrained or from scratch in the program
fnModel = self.getParam("-m")
if not os.path.isfile(fnModel):
    ## If the file doesn't exist the program will be interrupted
    print(" Final model file does not exist inside path")
    sys.exit(-1)

## File name where the inferece results will be stored at the end
fnOutput = self.getParam("-o")
if not os.path.isfile(fnOutput):
    ## If the file doesn't exist the program will be interrupted
    print("Output file does not exist inside path")
    sys.exit(-1)

## Size of the batches to be used for the nn
batchSz = int(self.getParam("-b"))
'''
