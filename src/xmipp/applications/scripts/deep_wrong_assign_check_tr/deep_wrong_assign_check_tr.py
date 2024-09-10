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

#TODO: eventually evaluate if test set is also going to be used Proposal (80%, 10% 10%)
#TODO: eventually evaluate the use of ensembles (add index to the model's fn name in bestModel)
#TODO: Evaluate if TensorBoard could be used to save info in a specific file

#TODO: Check imports

import numpy as np
import os
import sys
import xmippLib
from xmipp_base import XmippScript
from time import time

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, concatenate, Activation, GlobalAveragePooling2D, Add, Flatten
#TODO: Make sure we are not using any Keras 3 functionality

from contextlib import redirect_stdout
import atexit

class ScriptDeepWrongAssignCheckTrain(XmippScript):
    
    conda_env="xmipp_DLTK_v1.0" 

    #TODO: Evaluate if anything else is necessary
    def __init__(self):

        XmippScript.__init__(self)

    def defineParams(self):

        #TODO: define programUsage
        self.addUsageLine('')

        ## params to be read
        
        #TODO: check names make sense 

        self.addParamsLine(' -c <fnCorrResd> : filename containg the properly assigned residuals for training. ')
        self.addParamsLine(' -w <fnWronResd> : filename containg the wrongly assigned residuals for training. ')
        self.addParamsLine(' -o <finalModel> : h5 filename where the final model will be stored.')
        #TODO: check other programs implementation of batches (should it be mandatory?)
        self.addParamsLine(' -b <batchSize>: data`s subset size which will be fed to the network.')
        self.addParamsLine(' [ -f <fnPretrainedModel> ]: (optional) filename of the pretrained model to be used instead of a new one.')
        #TODO: Set values here and in protocol (protocol?)
        self.addParamsLine(' [ -e <numEpoch=30> ]: (optional) number of epochs to train the model')
        self.addParamsLine(' [ -l <learningRate=0.3> ]: (optional) learning rate used for the optimizer.')
        #TODO: modify this value (adaptative?)
        self.addParamsLine(' [ -p <patience=10> ]: (optional) number of epochs with no improvement after which training will be stopped.')

        #TODO: What should be the default behavior? (keep in mind any changes in the other program)
        self.addParamsLine(' [ -t <nThreads=1> ]: (optional) number of threads to use in multiprocessing.')

        #TODO: Finish explanation, include file format
        self.addParamsLine(' [ -s <fnSaveInfo> ]: (optional) file where the neural network information will be stored. Must be in <> format.')

        #TODO: make sure help text reflects reality
        self.addParamsLine(' [ --gpus <gpuId> ]: (optional) GPU ids to employ. Comma separated list. E.g. "0,1". Use -1 for CPU-only computation or -2 to use all devices found in CUDA_VISIBLE_DEVICES.')

        ## examples ##TODO: keep in mind changes in params until completely stable
        self.addExampleLine('deep_wrong_assign_check_tr -c path/to/correctResiduals -w path/to/wrongResiduals -o path/to/outputModel.h5 -b $BATCH_SIZE')

    #TODO: Write function definition
    def convertInputs(self):

        ## Xmipp metadata filename containing the positive residues examples info (including references to images)
        self.fnXmdPos = self.getParam("-c")
        if not os.path.isfile(self.fnXmdPos):
            ## If the file doesn't exist the program will be interrupted
            print("Positive examples file does not exist inside path")
            sys.exit(-1)

        ## Xmipp metadata filename containing the negative residues examples info (including references to images)
        self.fnXmdNeg = self.getParam("-w")
        if not os.path.isfile(self.fnXmdNeg):
            ## If the file doesn't exist the program will be interrupted
            print("Negative examples file does not exist inside path")
            sys.exit(-1)

        ## Keep in mind that here we assume positive and negative sets have the same size
        self.xDim, _, _, _, _ = xmippLib.MetaDataInfo(self.fnXmdPos)

        trainInfoPos = self.readFileInfo(self.fnXmdPos,True)
        trainInfoNeg = self.readFileInfo(self.fnXmdNeg,False)

        self.trainInfo = trainInfoPos + trainInfoNeg
        
        ## Filename where the best model will be stored after training
        ## The program does not check if it exists since the file will be generated (written on) as a result of the training
        self.fnOutputModel = self.getParam("-o")

        ## Size of the batches to be used for the nn
        self.batchSize = int(self.getParam("-b"))

        ## If the batch size is bigger than the data or the user requested no batches (using 0), only one batch is used
        if self.batchSize > len(self.trainInfo) or self.batchSize == 0: self.batchSize = len(self.trainInfo)

        #TODO: Check comments for this section (am I over explaning?)
        ## Checking if a pretrained model will be used
        if self.checkParam("-f"):
            ## File name where the pre existing model is stored, only used for reading
            ## The freshly trained model is stored in fnModel
            self.fnPreModel = self.getParam("-f")
            if not os.path.isfile(self.fnPreModel):
                ## If the file doesn't exist the program will be interrupted
                print("Model file does not exist inside path")
                sys.exit(-1)
            self.isPretrained = True
        else:
            self.isPretrained = False

        ## Number of epochs for training
        self.numEpochs = int(self.getParam("-e"))
        ## Value to overwrite in the optimizer
        self.learningRate = float(self.getParam("-l"))
        ## Number of epochs with no improvement after which training will be stopped
        self.patience = int(self.getParam("-p"))

        ## Number of threads to be used in multiprocessing
        self.nThreads = int(self.getParam("-t"))
        if self.nThreads < 1:
            self.nThreads = 1

        #TODO: Check proper integration
        #TODO: Keep in mind file doesn't exist since it hasn't been written on
        if self.checkParam("-s"):
            self.pthSaveInfo = self.getParam("-s")
            self.fnSaveInfo = self.pthSaveInfo + "nnInfo.txt"
            self.doSaveInfo = True
        else:
            self.doSaveInfo = False

        ## If no specific GPUs are requested all available GPUs will be used
        if self.checkParam("--gpus"):
            os.environ["CUDA_VISIBLE_DEVICES"] = self.getParam("--gpus")

    #TODO: Write function definition
    def readFileInfo(self, fnImages, isCorrect):

        if isCorrect:
            value = 1
        else:
            value = 0
        
        mdAux = xmippLib.MetaData(fnImages)
        fnImgs = mdAux.getColumnValues(xmippLib.MDL_IMAGE_RESIDUAL)
        labels = [value] * len(fnImgs)

        final = []

        for img, lbl in zip(fnImgs,labels):

            aux = (img,lbl)
            final.append(aux)

        return final

    #TODO: Write function definition
    def getImage(self, fnImg):
        '''
        with open (self.fnSaveInfo, 'a') as f:
            print(fnImg, file = f)
            np.set_printoptions(threshold=np.inf)
            print(xmippLib.Image(fnImg).getData(), file = f)
            print("hello", file = f)

        print("ahh ahh ahh")
        '''
        img = np.reshape(xmippLib.Image(fnImg).getData(), (self.xDim, self.xDim, 1))
        '''
        with open (self.fnSaveInfo, 'a') as f:
            #print(fnImg, file = f)
            np.set_printoptions(threshold=np.inf)
            print(img, file = f)
            print("gusbai", file = f)

        print("ehhhhhhhh")
        '''
        #print(img)
        return (img - np.mean(img)) / np.std(img)
    
    #--------------- Neural Network Generators ----------------

    #TODO: Write function definition
    def manageTrain(self):

        for elem in self.randomizedData[:self.lenTrain]:
            
            input = self.getImage(elem[0])
            target = int(elem[1])

            yield (input,target)

    #TODO: write function definition
    def manageVal(self):
        
        for elem in self.randomizedData[self.lenTrain:]:
            
            input = self.getImage(elem[0])
            target = int(elem[1])

            yield (input,target)

    #--------------- Training related functions ----------------

    #TODO: Write function definition
    def conv_block(self, tensor, filters):
        # Convolutional block of RESNET
        x = Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(tensor)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=3)(x)

        x_res = Conv2D(filters, (1, 1), strides=(2, 2))(tensor)

        x = Add()([x, x_res])
        x = Activation('relu')(x)
        return x
    
    #TODO: Write function definition
    #TODO: Keep in mind Model (inputs, feature)
    def constructModel(self):
        #RESNET architecture

        inputLayer = Input(shape=(self.xDim, self.xDim, 1), name="input")

        x = self.conv_block(inputLayer, filters=64)

        x = self.conv_block(x, filters=128)

        x = self.conv_block(x, filters=256)

        x = self.conv_block(x, filters=512)

        #x = self.conv_block(x, filters=1024)

        x = GlobalAveragePooling2D()(x)

        x = Dense(1, name= "output", activation='hard_sigmoid')(x)

        return Model(inputLayer, x)
    
    def constructModel2(self):

        pass
    
    #TODO: Write function definition
    def performTrain(self):

        self.lenTrain = int(len(self.trainInfo)*0.8)

        #TODO: is this the best way to do this?
        randomizedIds = np.random.choice(len(self.trainInfo), len(self.trainInfo), replace = False)
        npyTrainData = np.array(self.trainInfo)
        randomizedNpData = npyTrainData[randomizedIds]
        self.randomizedData = randomizedNpData.tolist()

        #TODO: comments on shapes (TensorSpecs)(proper explanation)
        trainingSet = tf.data.Dataset.from_generator(self.manageTrain, output_signature = (tf.TensorSpec((self.xDim, self.xDim, 1), dtype =tf.float32), tf.TensorSpec((), dtype=tf.int32)))
        trainingSet = trainingSet.batch(self.batchSize, drop_remainder = False)

        validationSet = tf.data.Dataset.from_generator(self.manageVal, output_signature = (tf.TensorSpec((self.xDim, self.xDim, 1), dtype =tf.float32), tf.TensorSpec((), dtype=tf.int32)))
        validationSet = validationSet.batch(self.batchSize, drop_remainder = False)

        #TODO: Explain what this is (should I include this line elsewhere?)
        #TODO: Evaluate if there is a way to use strategy in the scoring program
        strategy = tf.distribute.MirroredStrategy()

        atexit.register(strategy._extended._collective_ops._pool.close) # type: ignore

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        trainingSet = trainingSet.with_options(options)
        validationSet = validationSet.with_options(options)

        with strategy.scope():

            if self.isPretrained:
                
                model = load_model(self.fnPreModel)
                
            else:
                
                from tensorflow.keras.applications import ResNet50, VGG16
                ResNet_50 = ResNet50(weights = None, input_shape=(40, 40, 1), classes = 1, classifier_activation = "relu")
                VGG16_net = VGG16(weights = None, input_shape = (self.xDim,self.xDim,1), classes = 1, classifier_activation = "relu")
                adam_opt = tf.keras.optimizers.Adam(learning_rate = self.learningRate)
                sgd_opt = tf.keras.optimizers.SGD(learning_rate = self.learningRate, momentum=0.9, decay=0.0001)
                other_opt = tf.keras.optimizers.RMSprop(learning_rate = self.learningRate)
                #ResNet_50.compile(optimizer=adam_opt, loss='binary_crossentropy', metrics = ['accuracy'])
                VGG16_net.compile(optimizer=adam_opt, loss='binary_crossentropy', metrics = ['accuracy'])
                '''
                model = self.constructModel()
                #TODO: Evaluate other optimizers
                adam_opt = tf.keras.optimizers.Adam(learning_rate = self.learningRate)
                # Configuring the model's metrics
                model.compile(optimizer=adam_opt, loss='binary_crossentropy', metrics = ['accuracy'])
                '''
        #TODO: evaluate callbacks
        #TODO: check notes on LearningRateScheduler + Cyclical Learning Rate (CLR)

        #TODO: Keep in mind the data is overwritten every time a "better" model is found (save_best_only=True) by now it saves the whole model
        bestModel = ModelCheckpoint(filepath = self.fnOutputModel, monitor='val_loss', save_best_only=True)
        #TODO: what if the patience is = to epochs or even > ?
        patienceCallBack = EarlyStopping(monitor='val_loss', patience = self.patience)

        #TODO: Evaluate if this works
        if self.doSaveInfo:
            
            #TODO: evaluate params once callback is tested
            logsCallback = TensorBoard(log_dir = self.pthSaveInfo + "/logs", histogram_freq = 1, write_images = True)

            # sess.graph contains the graph definition; that enables the Graph Visualizer.
            #TODO: investigate: tensorboard --logdir=/home/lbaena/ScipionUserData/projects/Example_10248_Scipion3/Runs/016102_XmippProtWrongAssignCheckTrain/016102_XmippProtWrongAssignCheckTrain
            # https://www.mat-d.com/site/how-to-use-tensorflow-and-the-tensorboard-summary-dashboard
            # https://www.tensorflow.org/guide/intro_to_graphs
            writer = tf.summary.create_file_writer(self.pthSaveInfo + "/logs")
            #file_writer = tf.summary.FileWriter('/path/to/logs', sess.graph) #TF1, not suported
            
            history = VGG16_net.fit(x = trainingSet, epochs = self.numEpochs, verbose = "auto",
                        callbacks = [bestModel, patienceCallBack, logsCallback], validation_data = validationSet, workers = self.nThreads, use_multiprocessing = True)
            '''
            history = ResNet_50.fit(x = trainingSet, epochs = self.numEpochs, verbose = "auto",
                        callbacks = [bestModel, patienceCallBack, logsCallback], validation_data = validationSet, workers = self.nThreads, use_multiprocessing = True)
            
            history = model.fit(x = trainingSet, epochs = self.numEpochs, verbose = "auto",
                        callbacks = [bestModel, patienceCallBack, logsCallback], validation_data = validationSet, workers = self.nThreads, use_multiprocessing = True)
            '''
            #loss, accuracy = model.evaluate(x= validationSet, verbose = 2)
            #loss, accuracy = ResNet_50.evaluate(x = validationSet, verbose = 2)
            loss, accuracy = VGG16_net.evaluate(x = validationSet, verbose = 2)
            print(accuracy)

        else:
        
            #TODO: Traditionally the steps per epoch is calculated has train_length // batch size
            history = model.fit(x = trainingSet, epochs = self.numEpochs, verbose = 0,
                            callbacks = [bestModel, patienceCallBack], validation_data = validationSet, workers = self.nThreads, use_multiprocessing = True)
        
        # prints a string summary of the network
        #model.summary()  

        #TODO: A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).)
        if self.doSaveInfo:
            with open (self.fnSaveInfo, 'a') as f:
                print(history.history, file = f)
                #f.write(history.history)
                with redirect_stdout(f):
                    #ResNet_50.summary()
                    model.summary()
                #f.close()

        #print(history.history)

    #--------------- Program Execution Function ----------------

    #TODO: write function definition?
    def run(self):

        print("Starting training process")

        trainingStartTime = time()

        self.convertInputs()
        
        #TODO: include saving metrics in file if flag activated
        self.performTrain()
        
        trainingElapsedTime = time() - trainingStartTime
        print("Time in training model: %0.10f seconds." % trainingElapsedTime)

if __name__ == '__main__':
    exitCode = ScriptDeepWrongAssignCheckTrain().tryRun()
    sys.exit(exitCode)

###############################In#Case#Of#Loss##Original#Code#Ahead######################################################

# Code Extracted from run, turned into individual functions
'''
#TODO: Write function definition
# stores in memory the information contained in the images files
def readFileInfo(fnImages, isCorrect):

    if isCorrect:
        value = 1

    else:
        value = 0
    
    xDim, _, _, _, _ = xmippLib.MetaDataInfo(fnImages)
    mdAux = xmippLib.MetaData(fnImages)
    fnImgs = mdAux.getColumnValues(xmippLib.MDL_IMAGE_RESIDUAL)
    labels = [value] * len(fnImgs)

    final = []

    for img, lbl in zip(fnImgs,labels):

        aux = (img,lbl)
        final.append(aux)

    return xDim, final
'''
'''
#TODO: Write function definition
def conv_block(tensor, filters):
    # Convolutional block of RESNET
    x = Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)

    x_res = Conv2D(filters, (1, 1), strides=(2, 2))(tensor)

    x = Add()([x, x_res])
    x = Activation('relu')(x)
    return x

#TODO: Write function definition
#TODO: Keep in mind Model (inputs, feature)
def constructModel(Xdim):
    #RESNET architecture

    inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")

    x = self.conv_block(inputLayer, filters=64)

    x = self.conv_block(x, filters=128)

    x = self.conv_block(x, filters=256)

    x = self.conv_block(x, filters=512)

    x = self.conv_block(x, filters=1024)

    x = GlobalAveragePooling2D()(x)

    x = Dense(1, name="output", activation="linear")(x)

    return Model(inputLayer, x)


#TODO: Write function definition
#TODO: Return? (Probably not, reevaluate once cleaned)
def performTrain(self, trainData, numEpochs, learningRate, patience, batchSize):

    self.lenTrain = int(len(trainData)*0.8)

    #TODO: is this the best way to do this?
    randomizedIds = np.random.choice(len(trainData), len(trainData), replace = False)
    npyTrainData = np.array(trainData)
    randomizedNpData = npyTrainData[randomizedIds]
    self.randomizedData = randomizedNpData.tolist()

    #TODO: comments on shapes (TensorSpecs)(proper explanation)
    trainingSet = tf.data.Dataset.from_generator(self.manageTrain, output_signature = (tf.TensorSpec((self.xDim, self.xDim, 1), dtype =tf.float32), tf.TensorSpec((), dtype=tf.int32)))
    trainingSet = trainingSet.batch(batchSize, drop_remainder = False)

    validationSet = tf.data.Dataset.from_generator(self.manageVal, output_signature = (tf.TensorSpec((self.xDim, self.xDim, 1), dtype =tf.float32), tf.TensorSpec((), dtype=tf.int32)))
    validationSet = validationSet.batch(batchSize, drop_remainder = False)

    ## Checking if a pretrained model will be used and if the corresponding file has been given
    if self.checkParam("--pretrained") and self.checkParam("-f"):

        ## File name where the pre existing model is stored, only used for reading
        ## The freshly trained model is stored in fnModel
        fnPreModel = self.getParam("-f")
        if not os.path.isfile(fnPreModel):
            ## If the file doesn't exist the program will be interrupted
            print("Model file does not exist inside path")
            sys.exit(-1)
        else:
            ## Compile is set to false so the existing data (weights) is properly maintained
            model = load_model(fnPreModel, compile=False)
    else:
    ## A new model will also be used if no pre exiting file was given despite the "pretrained" flag being present
        model = self.constructModel(self.xDim)
        #TODO: Evaluate other optimizers
        adam_opt = tf.keras.optimizers.Adam(learning_rate = learningRate)
        # Configuring the model's metrics
        model.compile(optimizer=adam_opt, loss='mean_squared_error')

    # prints a string summary of the network
    model.summary()        

    #TODO: evaluate callbacks
    #TODO: should I use tensorBoard for documentation?
    #TODO: check notes on LearningRateScheduler + Cyclical Learning Rate (CLR)

    #TODO: Keep in mind the data is overwritten every time a "better" model is found (save_best_only=True) by now it saves the whole model
    bestModel = ModelCheckpoint(filepath = self.fnOutputModel, monitor='val_loss', save_best_only=True)
    #TODO: what if the patience is = to epochs or even > ?
    patienceCallBack = EarlyStopping(monitor='val_loss', patience=patience)

    #TODO: This could be stored in extra for example, just in case
    #TODO: A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).)
    #TODO: Traditionally the steps per epoch is calculated has train_length // batch size
    history = model.fit(x=trainingSet, epochs= numEpochs, steps_per_epoch = 50,
                    callbacks = [bestModel, patienceCallBack], validation_data = validationSet)
    #TODO: use_multiprocessing = self.nThreads inside of fit args 

    return 
'''
'''
## Xmipp metadata of the residue images ready for inference (file name)
fnXmdOk = self.getParam("-c")
if not os.path.isfile(fnXmdOk):
    ## If the file doesn't exist the program will be interrupted
    print("Positive examples datafile does not exist inside path")
    sys.exit(-1)

## Xmipp metadata of the residue images ready for inference (file name)
fnXmdNok = self.getParam("-w")
if not os.path.isfile(fnXmdNok):
    ## If the file doesn't exist the program will be interrupted
    print("Negative examples datafile does not exist inside path")
    sys.exit(-1)

## File name where the infernce model is stored either pretrained or from scratch in the program
#TODO: check the use of self
self.fnOutputModel = self.getParam("-o")
if not os.path.isfile(self.fnOutputModel):
    ## If the file doesn't exist the program will be interrupted
    print("Final model file does not exist inside path")
    sys.exit(-1)

## Size of the batches to be used for the nn
batchSz = int(self.getParam("-b"))
## Number of epochs for training
numEpochs = int(self.getParam("-e"))
## Value to overwrite in the optimizer
learnRate = float(self.getParam("-l"))
## Number of epochs with no improvement after which training will be stopped
patienceValue = int(self.getParam("-p"))
'''
""" #Deprecated use of deprecated library
        # checks if a previously existing model will be used and if the user added the model itself
if self.checkParam("-t") and self.checkParam("-f"):
    fnPreModel = self.getParam("-f")
    # checks if the file exists, if not an error will arise
    if not os.path.isfile(fnPreModel):
        pass 
"""

"""
training_set = DataHandler([trainData[i] for i in random_sample[0:lenTrain]],
                                [labels[i] for i in random_sample[0:lenTrain]],
                                    batchSize, xDims)
validation_set = DataHandler([trainData[i] for i in random_sample[lenTrain:lenTrain+lenVal]],
                                    [labels[i] for i in random_sample[lenTrain:lenTrain+lenVal]], 
                                    batchSize, xDims)
"""

"""
#Name: batch_deep_global_assignment.py

if __name__ == "__main__":

    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    fnModel = sys.argv[2]
    sigma = float(sys.argv[3])
    numEpochs = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    gpuId = sys.argv[6]
    numModels = int(sys.argv[7])
    learning_rate = float(sys.argv[8])
    patience = int(sys.argv[9])
    pretrained = sys.argv[10]
    if pretrained == 'yes':
        fnPreModel = sys.argv[11]

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    from keras.callbacks import TensorBoard, ModelCheckpoint
    from keras.models import Model
    from keras.layers import Input, Conv2D, BatchNormalization, Dense, concatenate, \
        Activation, GlobalAveragePooling2D, Add
    import keras
    from keras.models import load_model
    import tensorflow as tf


    class DataGenerator(keras.utils.all_utils.Sequence):
        #Generates data for fnImgs

        def __init__(self, fnImgs, labels, sigma, batch_size, dim, shifts, readInMemory):
            #Initialization
            self.fnImgs = fnImgs
            self.labels = labels
            self.sigma = sigma
            self.batch_size = batch_size
            if self.batch_size > len(self.fnImgs):
                self.batch_size = len(self.fnImgs)
            self.dim = dim
            self.readInMemory = readInMemory
            self.on_epoch_end()
            self.shifts = shifts


            # Read all data in memory
            if self.readInMemory:
                self.Xexp = np.zeros((len(self.labels), self.dim, self.dim, 1), dtype=np.float64)
                for i in range(len(self.labels)):
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[i]).getData(), (self.dim, self.dim, 1))
                    self.Xexp[i,] = (Iexp - np.mean(Iexp)) / np.std(Iexp)

        def __len__(self):
            #Denotes the number of batches per epoch
            num_batches = int(np.floor((len(self.labels)) / self.batch_size))
            return num_batches

        def __getitem__(self, index):
            #Generate one batch of data
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            # Find list of IDs
            list_IDs_temp = []
            for i in range(int(self.batch_size)):
                list_IDs_temp.append(indexes[i])
            # Generate data
            Xexp, y = self.__data_generation(list_IDs_temp)

            return Xexp, y

        def on_epoch_end(self):
            #Updates indexes after each epoch
            self.indexes = [i for i in range(len(self.labels))]
            np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):
            #Generates data containing batch_size samples
            yvalues = np.array(itemgetter(*list_IDs_temp)(self.labels))
            yshifts = np.array(itemgetter(*list_IDs_temp)(self.shifts))

            # Functions to handle the data
            def get_image(fn_image):
                img = np.reshape(xmippLib.Image(fn_image).getData(), (self.dim, self.dim, 1))
                return (img - np.mean(img)) / np.std(img)

            def shift_image(img, shiftx, shifty, yshift):
                return shift(img, (shiftx-yshift[0], shifty-yshift[1], 0), order=1, mode='wrap')

            def rotate_image(img, angle):
                # angle in degrees
                return rotate(img, angle, order=1, mode='reflect', reshape=False)

            def R_rot(theta):
                return np.array([[1, 0, 0],
                                  [0, math.cos(theta), -math.sin(theta)],
                                  [0, math.sin(theta), math.cos(theta)]])

            def R_tilt(theta):
                return np.array([[math.cos(theta), 0, math.sin(theta)],
                                  [0, 1, 0],
                                  [-math.sin(theta), 0, math.cos(theta)]])

            def R_psi(theta):
                return np.array([[math.cos(theta), -math.sin(theta), 0],
                                  [math.sin(theta), math.cos(theta), 0],
                                  [0, 0, 1]])

            def euler_angles_to_matrix(angles, psi_rotation):
                Rx = R_rot(angles[0])
                Ry = R_tilt(angles[1] - math.pi / 2)
                Rz = R_psi(angles[2] + psi_rotation)
                return np.matmul(np.matmul(Rz, Ry), Rx)

            def matrix_to_rotation6d(mat):
                r6d = np.delete(mat, -1, axis=1)
                return np.array((r6d[0, 0], r6d[0, 1], r6d[1, 0], r6d[1, 1], r6d[2, 0], r6d[2, 1]))

            def euler_to_rotation6d(angles, psi_rotation):
                mat = euler_angles_to_matrix(angles, psi_rotation)
                return matrix_to_rotation6d(mat)

            def make_redundant(rep_6d):
                rep_6d = np.append(rep_6d, 2*rep_6d)
                for i in range(6):
                    j = (i+1) % 6
                    rep_6d = np.append(rep_6d, rep_6d[i]-rep_6d[j])
                for i in range(6):
                    j = (i + 3) % 6
                    rep_6d = np.append(rep_6d, rep_6d[i+6] - rep_6d[j])
                for i in range(6):
                    j = (i + 2) % 6
                    k = (i + 4) % 6
                    rep_6d = np.append(rep_6d, rep_6d[i]+rep_6d[j]-rep_6d[k])
                for i in range(6):
                    j = (i + 5) % 6
                    rep_6d = np.append(rep_6d, rep_6d[i] - rep_6d[j])
                for i in range(6):
                    j = (i + 4) % 6
                    rep_6d = np.append(rep_6d, rep_6d[i] - rep_6d[j])
                return rep_6d

            if self.readInMemory:
                Iexp = list(itemgetter(*list_IDs_temp)(self.Xexp))
            else:
                fnIexp = list(itemgetter(*list_IDs_temp)(self.fnImgs))
                Iexp = list(map(get_image, fnIexp))

            rX = self.sigma * np.random.uniform(-1, 1, size=self.batch_size)
            rY = self.sigma * np.random.uniform(-1, 1, size=self.batch_size)
            # Shift image a random amount of px in each direction
            Xexp = np.array(list((map(shift_image, Iexp, rX, rY, yshifts))))
            # Rotates image a random angle. Psi must be updated
            rAngle = 180 * np.random.uniform(-1, 1, size=self.batch_size)
            Xexp = np.array(list(map(rotate_image, Xexp, rAngle)))
            rAngle = rAngle * math.pi / 180
            yvalues = yvalues * math.pi / 180
            y_6d = np.array(list((map(euler_to_rotation6d, yvalues, rAngle))))
            y = np.array(list((map(make_redundant, y_6d))))

            return Xexp, y

    def conv_block(tensor, filters):
        # Convolutional block of RESNET
        x = Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(tensor)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=3)(x)

        x_res = Conv2D(filters, (1, 1), strides=(2, 2))(tensor)

        x = Add()([x, x_res])
        x = Activation('relu')(x)
        return x

    def constructModel(Xdim):
        #RESNET architecture
        inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")

        x = conv_block(inputLayer, filters=64)

        x = conv_block(x, filters=128)

        x = conv_block(x, filters=256)

        x = conv_block(x, filters=512)

        x = conv_block(x, filters=1024)

        x = GlobalAveragePooling2D()(x)

        x = Dense(5, name="output", activation="linear")(x)

        return Model(inputLayer, x)


    def get_labels(fnImages):
        #Returns dimensions, images, angles and shifts values from images files
        Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnImages)
        mdExp = xmippLib.MetaData(fnImages)
        fnImg = mdExp.getColumnValues(xmippLib.MDL_IMAGE)
        shiftX = mdExp.getColumnValues(xmippLib.MDL_SHIFT_X)
        shiftY = mdExp.getColumnValues(xmippLib.MDL_SHIFT_Y)
        rots = mdExp.getColumnValues(xmippLib.MDL_ANGLE_ROT)
        tilts = mdExp.getColumnValues(xmippLib.MDL_ANGLE_TILT)
        psis = mdExp.getColumnValues(xmippLib.MDL_ANGLE_PSI)

        label = []
        img_shift = []

        # For better performance, images are selected to be 'homogeneously' distributed in the sphere
        # 50 divisions with equal area
        numTiltDivs = 5
        numRotDivs = 10
        limits_rot = np.linspace(-180.01, 180, num=(numRotDivs+1))
        limits_tilt = np.zeros(numTiltDivs+1)
        limits_tilt[0] = -0.01
        for i in range(1, numTiltDivs+1):
            limits_tilt[i] = math.acos(1-2*(i/numTiltDivs))
        limits_tilt = limits_tilt*180/math.pi

        # Each particle is assigned to a division
        zone = [[] for _ in range((len(limits_tilt)-1)*(len(limits_rot)-1))]
        i = 0
        for r, t, p, sX, sY in zip(rots, tilts, psis, shiftX, shiftY):
            label.append(np.array((r, t, p)))
            img_shift.append(np.array((sX, sY)))
            region_rot = np.digitize(r, limits_rot, right=True) - 1
            region_tilt = np.digitize(t, limits_tilt, right=True) - 1
            # Region index
            region_idx = region_rot * (len(limits_tilt)-1) + region_tilt
            zone[region_idx].append(i)
            i += 1

        return Xdim, fnImg, label, zone, img_shift

    Xdims, fnImgs, labels, zones, shifts = get_labels(fnXmdExp)
    start_time = time()

    # Train-Validation sets
    if numModels == 1:
        lenTrain = int(len(fnImgs)*0.8)
        lenVal = len(fnImgs)-lenTrain
    else:
        lenTrain = int(len(fnImgs) / 3)
        lenVal = int(len(fnImgs) / 12)

    elements_zone = int((lenVal+lenTrain)/len(zones))

    for index in range(numModels):
        # chooses equal number of particles for each division
        random_sample = np.random.choice(range(0, len(fnImgs)), size=lenTrain+lenVal, replace=False)

        training_generator = DataGenerator([fnImgs[i] for i in random_sample[0:lenTrain]],
                                           [labels[i] for i in random_sample[0:lenTrain]],
                                           sigma, batch_size, Xdims, shifts, readInMemory=False)
        validation_generator = DataGenerator([fnImgs[i] for i in random_sample[lenTrain:lenTrain+lenVal]],
                                             [labels[i] for i in random_sample[lenTrain:lenTrain+lenVal]],
                                             sigma, batch_size, Xdims, shifts, readInMemory=False)

        if pretrained == 'yes':
            model = load_model(fnPreModel, compile=False)
        else:
            model = constructModel(Xdims)

        adam_opt = tf.keras.optimizers.Adam(lr=learning_rate)
        model.summary()

        model.compile(loss='mean_squared_error', optimizer=adam_opt)
        save_best_model = ModelCheckpoint(fnModel + str(index) + ".h5", monitor='val_loss',
                                          save_best_only=True)
        patienceCallBack = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)


        history = model.fit_generator(generator=training_generator, epochs=numEpochs,
                                      validation_data=validation_generator, callbacks=[save_best_model, patienceCallBack])

    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)
"""