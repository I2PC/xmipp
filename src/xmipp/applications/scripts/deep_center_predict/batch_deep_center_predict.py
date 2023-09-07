#!/usr/bin/env python3

import numpy as np
import os
import sys
import xmippLib
from time import time

maxSize = 32

if __name__ == "__main__":
    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    gpuId = sys.argv[2]
    outputDir = sys.argv[3]
    fnXmdImages = sys.argv[4]
    fnModel = sys.argv[5]
    numModels = int(sys.argv[6])
    tolerance = int(sys.argv[7])
    maxModels = int(sys.argv[8])
    fnPreModel = sys.argv[9]

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    import keras
    from keras.models import load_model



    class DataGenerator(keras.utils.all_utils.Sequence):
        """Generates data for fnImgs"""

        def __init__(self, fnImgs, maxSize, dim, readInMemory):
            """Initialization"""
            self.fnImgs = fnImgs
            self.maxSize = maxSize
            self.dim = dim
            self.readInMemory = readInMemory
            self.on_epoch_end()

            # Read all data in memory
            if self.readInMemory:
                self.Xexp = np.zeros((len(self.fnImgs), self.dim, self.dim, 1), dtype=np.float64)
                for i in range(len(self.fnImgs)):
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[i]).getData(), (self.dim, self.dim, 1))
                    self.Xexp[i,] = (Iexp - np.mean(Iexp)) / np.std(Iexp)

        def __len__(self):
            """Denotes the number of batches per predictions"""
            num = len(self.fnImgs) // maxSize
            if len(self.fnImgs) % maxSize > 0:
                num = num + 1
            return num

        def __getitem__(self, index):
            """Generate one batch of data"""
            # Generate indexes of the batch
            indexes = self.indexes[index * maxSize:(index + 1) * maxSize]
            # Find list of IDs
            list_IDs_temp = []
            for i in range(len(indexes)):
                list_IDs_temp.append(indexes[i])

            # Generate data
            Xexp = self.__data_generation(list_IDs_temp)

            return Xexp

        def on_epoch_end(self):
            self.indexes = [i for i in range(len(self.fnImgs))]

        def getNumberOfBlocks(self):
            self.st = len(self.fnImgs) // maxSize
            if len(self.fnImgs) % maxSize > 0:
                self.st = self.st + 1

        def __data_generation(self, list_IDs_temp):
            """Generates data containing batch_size samples"""
            # Initialization
            Xexp = np.zeros((len(list_IDs_temp), self.dim, self.dim, 1), dtype=np.float64)
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Read image
                if self.readInMemory:
                    Xexp[i, ] = self.Xexp[ID]
                else:
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[ID]).getData(), (self.dim, self.dim, 1))
                    Xexp[i, ] = (Iexp - np.mean(Iexp)) / np.std(Iexp)
            return Xexp

    def produce_output(mdExp, Y, distance, fnImg):
        ID = 0
        for objId in mdExp:
            # Set predictions in mdExp
            shiftX, shiftY = Y[ID]
            mdExp.setValue(xmippLib.MDL_SHIFT_X, float(shiftX), objId)
            mdExp.setValue(xmippLib.MDL_SHIFT_Y, float(shiftY), objId)
            mdExp.setValue(xmippLib.MDL_IMAGE, fnImg[ID], objId)
            if distance[ID] > tolerance:
                mdExp.setValue(xmippLib.MDL_ENABLED, -1, objId)
            ID += 1

    def average_of_shifts(predshift):
        """Consensus tool"""
        # Calculates average shift for each particle
        av_shift = np.average(predshift, axis=0)
        distancesxy = np.abs(av_shift-predshift)
        # min number of models
        minModels = np.shape(predshift)[0] - maxModels
        # Calculates norm 1 of distances
        distances = np.sum(distancesxy, axis=1)
        max_distance = np.max(distances)
        # max distance model to the average
        max_dif_model = np.argmax(distances)
        while (np.shape(predshift)[0] > minModels) and (max_distance > tolerance):
            # deletes predictions from the max_dif_model and recalculates averages
            predshift = np.delete(predshift, max_dif_model, axis=0)
            av_shift = np.average(predshift, axis=0)
            distancesxy = np.abs(av_shift - predshift)
            distances = np.sum(distancesxy, axis=1)
            max_distance = np.max(distances)
            max_dif_model = np.argmax(distances)
        return np.append(av_shift, max_distance)

    def compute_shift_averages(predshift):
        """Calls consensus tool"""
        averages_mdistance = np.array(list(map(average_of_shifts, predshift)))
        average = averages_mdistance[:, 0:2]
        mdistance = averages_mdistance[:, 2]
        return average, mdistance

    Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnXmdExp)

    mdExp = xmippLib.MetaData(fnXmdExp)
    fnImgs = mdExp.getColumnValues(xmippLib.MDL_IMAGE)

    mdExpImages = xmippLib.MetaData(fnXmdImages)
    fnImages = mdExpImages.getColumnValues(xmippLib.MDL_IMAGE)

    start_time = time()

    predictions = np.zeros((len(fnImgs), numModels, 2))
    ShiftManager = DataGenerator(fnImgs, maxSize, Xdim, readInMemory=False)
    for index in range(numModels):
        #ShiftModel = load_model(fnModel + str(index) + ".h5", compile=False)
        ShiftModel = load_model(fnPreModel + "/modelCenter" + "/modelCenter" + str(index) + ".h5", compile=False)
        ShiftModel.compile(loss="mean_squared_error", optimizer='adam')
        predictions[:, index, :] = ShiftModel.predict_generator(ShiftManager, ShiftManager.getNumberOfBlocks())
    Y, distance = compute_shift_averages(predictions)
    produce_output(mdExp, Y, distance, fnImages)

    mdExp.write(os.path.join(outputDir, "predict_results.xmd"))

    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)




