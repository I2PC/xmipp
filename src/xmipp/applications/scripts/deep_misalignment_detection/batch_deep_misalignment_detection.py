#!/usr/bin/env python3
import numpy as np

import sys, os
import xmippLib


from xmipp_base import XmippScript


class ScriptDeepMisalignmentDetection(XmippScript):
    # _conda_env="xmipp_pyTorch" *** generar nuestro propio enviroment??

    def __init__(self):

        XmippScript.__init__(self)


    #  --------------------- DEFINE PARAMS -----------------------------
    def defineParams(self):
        # Description
        self.addUsageLine('Detect artifacted tomographic reconstruction from extracted fiducial markers')
        
        # Params
        self.addParamsLine(' --inputModel1: path to model for strong misalignment detection')
        self.addParamsLine(' --inputModel2: path to model for waek misalignment detection')
        self.addParamsLine(' --subtomoFilePath: file path of the xmd file containg the extracted subtomos. ' +
                           'The extracted subtomo should be in the same folder with the same basename + "-[numberOfSubtomo]." ' +
                           'This is the output you get when extracting with xmipp_tomo_extract_subtomograms.')
        self.addParamsLine(' --misaliThr: Threshold to settle if a tomogram presents weak or strong misalignment. If this value is '
                           'not provided two output set of tomograms are generated, those discarded which present '
                           'strong misalignment and those which do not. If this value is provided the second group of '
                           'tomograms is splitted into two, using this threshold to settle if the tomograms present'
                           'or not a weak misalignment.')
        
        # Examples       
        self.addExampleLine('xmipp_deep_misalingment_detection -inputModel1 path/to/model1 --inputModel2 path/to/model2 ' +
                            '--subtomoFilePath path/to/xoords.xmd')

    
    #  --------------------- I/O FUNCTIONS -----------------------------
    def readInputParams(self):
        self.inputModel1 = self.getParam('--inputModel1')
        self.inputModel2 = self.getParam('--inputModel2')
        self.subtomoFilePath = self.getParam('--subtomoFilePath')
        self.misaliThrBool = self.checkParam('--misaliThr')

        if self.misaliThrBool:
            self.misaliThr = self.getDoubleParam('--misaliThr')        


    #  --------------------- MAIN FUNCTIONS -----------------------------
    def getSubtomoPathList(coordFilePath):
        coordFilePath_noExt = os.path.splitext(coordFilePath)[0]
        counter = 1

        subtomoPathList = []

        while True:
            subtomoPath = coordFilePath_noExt + '-' + str(counter) + '.mrc'

            if not os.path.exists(subtomoPath):
                break

            subtomoPathList.append(subtomoPath)
            counter += 1

        return subtomoPathList
    
    def makePrediction(self, subtomoPathList):
        """
        :param subtomoPathList: list to every subtomo extracted to be analyzed
        :return: overallPrediction: alignment statement for the whole tomograms obtained from the estimations of each
        subtomo:
            1: strong misalignment (first split negative)
            2: weak misalignment (second split negative). Implies the existence of an input alignment threshold
            3: alignment (second split positive)
        """
        ih = ImageHandler() # *** como se usa el IH desde aqui??

        numberOfSubtomos = len(subtomoPathList)

        subtomoArray = np.zeros((numberOfSubtomos, 32, 32, 32), dtype=np.float64)

        for index, subtomo in enumerate(subtomoPathList):
            subtomoDataTmp = ih.read(subtomo)
            subtomoDataTmp = subtomoDataTmp.getData()

            subtomoArray[index, :, :, :] = subtomoDataTmp[:, :, :]

        std = subtomoArray.std()
        mean = subtomoArray.mean()

        subtomoArray = (subtomoArray - mean) / std

        firstPredictionArray = self.firstModel.predict(subtomoArray)

        overallPrediction, predictionAverage = self.determineOverallPrediction(firstPredictionArray, overallCriteria=1)

        if not overallPrediction:
            overallPrediction = 1  # Strong misalignment

            # Set misalignment score to -1 if subtomos removed by the first network
            secondPredictionArray = np.full(firstPredictionArray.shape, -1)

        else:
            secondPredictionArray = self.secondModel.predict(subtomoArray)

            overallPrediction, predictionAverage = self.determineOverallPrediction(secondPredictionArray,
                                                                                   overallCriteria=1)

            if self.misaliThrBool:  # Using threshold

                if predictionAverage > self.misaliThr:
                    overallPrediction = 3  # Alignment
                else:
                    overallPrediction = 2  # Weak misalignment

        return overallPrediction, predictionAverage, firstPredictionArray, secondPredictionArray


    #  --------------------- UTILS FUNCTIONS -----------------------------
    def loadModels(self):
        self.firstModel = self.load_model(self.inputModel1)
        # print(self.firstModel.summary())

        self.secondModel = self.load_model(self.inputModel2)
        # print(self.secondModel.summary())

    def determineOverallPrediction(predictionList, overallCriteria):
        """
        This method return an overall prediction based on the different singular predictions for each gold bead. This
        can be estimated with a voting system (no considering the actual score value) or as the average of the obtained
        scores for each gold beads.
        :param predictionList: vector with the score values predicted for each gold bead
        :param overallCriteria: criteria to be used to calculate the overall prediction as the most voted option (0) or
        the average of all the scores (1)
        :return: bool indicating if the tomogram present misalignment or not
        :return average of the predicted scores
        """

        predictionAvg = np.average(predictionList)

        if overallCriteria == 0:
            predictionClasses = np.round(predictionList)

            overallPrediction = 0

            for prediction in predictionClasses:
                overallPrediction += prediction

            print("Subtomo analysis: " + str(overallPrediction) + " aligned vs " +
                  str(predictionList.size - overallPrediction) + "misaligned")

            overallPrediction = overallPrediction / predictionList.size

            # aligned (1) or misaligned (0)
            return (True if overallPrediction > 0.5 else False), predictionAvg

        elif overallCriteria == 1:
            print("prediction list:")
            print(predictionList)

            print("Subtomo analysis preditcion: " + str(predictionAvg))

            # aligned (1) or misaligned (0)
            return (True if predictionAvg > 0.5 else False), predictionAvg


    #  --------------------- RUN -----------------------------
    def run(self):
        # Read input params
        self.readInputParams()

        # Get subtomo path list from directory
        subtomoPathList = self.getSubtomoPathList(self.subtomoFilePath)

        # Make prediction from subtomos in list
        if len(subtomoPathList) != 0:
            totalNumberOfSubtomos = len(subtomoPathList)
            print("Total number of subtomos: " + str(totalNumberOfSubtomos))

            self.loadModels()

            overallPrediction, predictionAverage, firstPredictionArray, secondPredictionArray = \
                self.makePrediction(subtomoPathList)

            print("For volume id " + str(tsId) + " obtained prediction from " +
                    str(len(subtomoPathList)) + " subtomos is " + str(overallPrediction))
            
            
    
if __name__ == "__main__":
    exitCode = ScriptDeepMisalignmentDetection().tryRun()
    sys.exit(exitCode)
