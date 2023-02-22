"""
/***************************************************************************
 *
 * Authors:    Federico P. de Isidro Gomez			  fp.deisidro@cnb.csic.es
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
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/
"""


from xml.dom import XML_NAMESPACE
from xmipp_base import *
import xmippLib
  
import sys
import os
from math import sqrt
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from scipy.spatial import ConvexHull, QhullError
import numpy as np


class ScriptTomoResidualStatistics(XmippScript):

  def __init__(self):
    XmippScript.__init__(self)

    self.p = 0.5
    self.alpha = 0.05

    self.debug = False

    # Save landmark residuals
    self.residSize = {}

    self.residX = {}
    self.residY = {}

    self.residXAcc = {}
    self.residYAcc = {}
    self.residModuleAcc = {}

    self.nResidPosX = {}
    self.nResidPosY = {}

    self.coordsByLandmark = {}


    # Save image residuals
    self.imageSize = {}

    self.imageX = {}
    self.imageY = {}

    self.imageXAcc = {}
    self.imageYAcc = {}
    self.imageModuleAcc = {}

    self.nImagePosX = {}
    self.nImagePosY = {}

    self.coordsByImage = {}



  def defineParams(self):
    """
      Define program parameters
    """

    self.addUsageLine('Perform statistical test on a dataset of residual vectors comming from landmark \
      to check for possible patterns, correlations or some other non-random behaviour. ')

    ## params
    self.addParamsLine(' -i <inputMetadaFile>   : coordinate model metadata file path. This file contains the coordinate in slice, \
                        its assigned coordinate 3d and the residual vector between the first and the proyection of the second. This is generated by the \
                        xmipp_tomo_detect_misalignment_trajectory program.\n')

    self.addParamsLine(' -o <outputMetadaFile>   : output residual statistics file path. Location to save the calulated statistics from the residual model \n')
    self.addParamsLine(' [ --debug ]             : add this option to output extra information for debugging \n')


  def readResidInfo(self):
    """
      Read residual information from metadata
    """
    
    mdFilePath = self.getParam('-i')
    mData = xmippLib.MetaData(mdFilePath)

    for objId in mData:
      # Read landmark chains vectors
      id = mData.getValue(xmippLib.MDL_FRAME_ID, objId)

      if id in self.residX.keys():
        self.residX[id].append(mData.getValue(xmippLib.MDL_SHIFT_X, objId))
        self.residY[id].append(mData.getValue(xmippLib.MDL_SHIFT_Y, objId))

      else:
        self.coordsByLandmark[id] = [mData.getValue(xmippLib.MDL_XCOOR, objId),
                           mData.getValue(xmippLib.MDL_YCOOR, objId),
                           mData.getValue(xmippLib.MDL_ZCOOR, objId)]
        self.residX[id] = [mData.getValue(xmippLib.MDL_SHIFT_X, objId)]
        self.residY[id] = [mData.getValue(xmippLib.MDL_SHIFT_Y, objId)]

      #Read image vectors
      id = int(mData.getValue(xmippLib.MDL_Z, objId))

      if id in self.imageX.keys():
        self.imageX[id].append(mData.getValue(xmippLib.MDL_SHIFT_X, objId))
        self.imageY[id].append(mData.getValue(xmippLib.MDL_SHIFT_Y, objId))

      else:
        self.coordsByImage[id] = [mData.getValue(xmippLib.MDL_XCOOR, objId),
                           mData.getValue(xmippLib.MDL_YCOOR, objId),
                           mData.getValue(xmippLib.MDL_ZCOOR, objId)]
        self.imageX[id] = [mData.getValue(xmippLib.MDL_SHIFT_X, objId)]
        self.imageY[id] = [mData.getValue(xmippLib.MDL_SHIFT_Y, objId)]
      
    # Debug
    if self.checkParam('--debug'):
      self.debug=True

    print("Residual information read successfully!")
               

  def writeOutputStatsInfo(self, residualStats, filePath):
    """
      Write statistical information into metadata
    """

    print("Writting output stest at " + filePath)

    mData = xmippLib.MetaData()

    for i in range(len(residualStats)):
      id = mData.addObject()
      mData.setValue(xmippLib.MDL_ENABLED, residualStats[i][0], id)
      mData.setValue(xmippLib.MDL_MIN, residualStats[i][1], id)
      mData.setValue(xmippLib.MDL_MAX, residualStats[i][2], id)
      mData.setValue(xmippLib.MDL_IMAGE, residualStats[i][3], id)
      mData.setValue(xmippLib.MDL_XCOOR, residualStats[i][4], id)
      mData.setValue(xmippLib.MDL_YCOOR, residualStats[i][5], id)
      mData.setValue(xmippLib.MDL_ZCOOR, residualStats[i][6], id)

    mData.write(filePath)


  def generateSideInfo(self):
    """
      Generate residual side information to perform posterior tests
    """

    # Residual info vectors for landmarks
    print("Generate side information for landmarks...")
    print(self.residX.keys())

    for key in self.residX.keys():
      self.residSize[key] = len(self.residX[key])

      nPosX = 0
      nPosY = 0

      for i, r in enumerate(self.residX[key]):
        if r > 0:
          nPosX += 1

        if i == 0:
          self.residXAcc[key] = [r]
        else:
          self.residXAcc[key].append(r+self.residXAcc[key][i-1])

      for i, r in enumerate(self.residY[key]):
        if r > 0:
          nPosY += 1

        if i == 0:
          self.residYAcc[key] = [r]
        else:
          self.residYAcc[key].append(r+self.residYAcc[key][i-1])

      self.nResidPosX[key] = nPosX
      self.nResidPosY[key] = nPosY

      for i in range(self.residSize[key]):
        if i == 0:
          self.residModuleAcc[key] = [sqrt(self.residXAcc[key][i]*self.residXAcc[key][i] + self.residYAcc[key][i]*self.residYAcc[key][i])]
        else:
          self.residModuleAcc[key].append(sqrt(self.residXAcc[key][i]*self.residXAcc[key][i] + self.residYAcc[key][i]*self.residYAcc[key][i]))

    # Residual info vectors for images
    print("Generate side information for images...")
    print(self.imageX.keys())

    for key in self.imageX.keys():
      self.imageSize[key] = len(self.imageX[key])

      nPosX = 0
      nPosY = 0

      for i, r in enumerate(self.imageX[key]):
        if r > 0:
          nPosX += 1

        if i == 0:
          self.imageXAcc[key] = [r]
        else:
          self.imageXAcc[key].append(r+self.imageXAcc[key][i-1])

      for i, r in enumerate(self.imageY[key]):
        if r > 0:
          nPosY += 1

        if i == 0:
          self.imageYAcc[key] = [r]
        else:
          self.imageYAcc[key].append(r+self.imageYAcc[key][i-1])

      self.nImagePosX[key] = nPosX
      self.nImagePosY[key] = nPosY

      for i in range(self.imageSize[key]):
        if i == 0:
          self.imageModuleAcc[key] = [sqrt(self.imageXAcc[key][i]*self.imageXAcc[key][i] + self.imageYAcc[key][i]*self.imageYAcc[key][i])]
        else:
          self.imageModuleAcc[key].append(sqrt(self.imageXAcc[key][i]*self.imageXAcc[key][i] + self.imageYAcc[key][i]*self.imageYAcc[key][i]))


    print("Side information generated successfully!")


  def convexHull(self, vX, vY):
    """
      Calculate the convex hull from the residual vectors returning its area and perimeter
    """

    residualVectors = []

    for i in range(len(vX)):
      residualVectors.append([vX[i], vY[i]])

    try:
      convexHull = ConvexHull(residualVectors)

      hullPerimeter = 0

      # For 2-Dimensional convex hulls volume attribute equals to the area.
      hullArea = [convexHull.volume][0]

      hullVertices = []

      for position in convexHull.vertices:
          hullVertices.append(convexHull.points[position])

      for i in range(len(hullVertices)):
          shiftedIndex = (i + 1) % len(hullVertices)

          distanceVector = np.array(hullVertices[i]) - np.array(hullVertices[shiftedIndex])
          distanceVector = [i ** 2 for i in distanceVector]
          hullPerimeter += np.sqrt(sum(distanceVector))
      
    except QhullError:
      print("ERROR: Invalid geometry, lack of dimensionality (probably all residuals belong to a line). Only perieter calculated as the lenght of this line.")

      hullArea = 0.0

      xMin = min([item[0] for item in residualVectors])
      xMax = max([item[0] for item in residualVectors])
      yMin = min([item[1] for item in residualVectors])
      yMax = max([item[1] for item in residualVectors])

      xDiff = xMax-xMin
      yDiff = yMax-yMin
      hullPerimeter = xDiff if xDiff > yDiff else yDiff

    return hullArea, hullPerimeter


  def binomialTest(self, nPos, rs):
    """
      Binomial test for sign distribution
    """

    print("-----------------------------------------")
    print(nPos)
    print(rs)
    print(self.p)

    pValue = stats.binom_test(nPos, rs , self.p)

    # print("Binomial test p-value: " + str(pValue))

    return pValue


  def fTestVar(self, fStatistic, rs):
    """
      F-test of equality of variances
    """

    pValue = stats.f.cdf(fStatistic, rs-1, rs-1)

    # print("F test for variance p-value: " + str(pValue))

    return pValue


  def augmentedDickeyFullerTest(self, modAcc):
    """
      Augmented Dickey-Fuller test for random walk
    """

    if len(modAcc)<3:
      adfStatistic = 0.0
      pValue = 1.0
      criticalValues = None

      return adfStatistic, pValue, criticalValues 

    result = adfuller(modAcc)
    adfStatistic = result[0]
    pValue = result[1]
    criticalValues = result[4]

    # print("Augmented Dickey-Fuller test for random walk ADF statistic: " + str(adfStatistic))
    # print("Augmented Dickey-Fuller test for random walk p-value: " + str(pValue))
    # print("Augmented Dickey-Fuller test for random walk critical values: ")
    # for key, value in criticalValues.items():
    #   print('\t%s: %.3f' % (key, value))
    
    return adfStatistic, pValue, criticalValues



  def run(self):
    print("Running statistical analysis of misalingment residuals...")

    self.readResidInfo()
    self.generateSideInfo()

    # Generate residual information for landmarks
    pValues = []
    ch = []

    for key in self.residX.keys():
      rs = self.residSize[key]

      if self.debug:
        print("KEY--------->" + str(key))
        print("resid size " + str(self.residSize[key]))

      # Convex hull
      convexHullArea, convexHullPerimeter = self.convexHull(vX=self.residX[key], vY=self.residY[key])

      ch.append([1, convexHullArea,      convexHullArea,       str(key) + "_chArea",  self.coordsByLandmark[key][0], self.coordsByLandmark[key][1], self.coordsByLandmark[key][2]])
      ch.append([1, convexHullPerimeter, convexHullPerimeter,  str(key) + "_chPerim", self.coordsByLandmark[key][0], self.coordsByLandmark[key][1], self.coordsByLandmark[key][2]])


      # Variance distribution matrix
      varianceMatrix = np.zeros([2, 2])

      for i in range(len(self.residX[key])):
        rx = self.residX[key][i]
        ry = self.residY[key][i]

        rx2 = rx * rx
        ry2 = ry * ry
        rxy = rx * ry

        sumRadius = sqrt(rx2+ry2)

        if(sumRadius==0):
          varianceMatrix += np.matrix([[rx2, rxy], [rxy, ry2]])
        else:
          varianceMatrix += np.matrix([[rx2/sumRadius, rxy/sumRadius], [rxy/sumRadius, ry2/sumRadius]])

      [lambda1, lambda2], _ = np.linalg.eig(varianceMatrix)

      if self.debug:
        print("lambda1: " + str(lambda1))
        print("lambda2: " + str(lambda2))
        print("self.residModuleAcc")
        print(self.residModuleAcc[key])
        print("self.residYAcc")
        print(self.residYAcc[key])
        print("self.residXAcc")
        print(self.residXAcc[key])
        print("self.residY")
        print(self.residY[key])
        print("self.residX")
        print(self.residX[key]) 

      try:
        fTestStat = lambda1/lambda2
      except ZeroDivisionError:
        fTestStat = 1

      # Statistical tests
      pvBinX = self.binomialTest(self.nResidPosX[key], rs)
      pvBinY = self.binomialTest(self.nResidPosY[key], rs)
      pvF = self.fTestVar(fTestStat, rs)
      adfStatistic, pvADF, cvADF = self.augmentedDickeyFullerTest(self.residModuleAcc[key])

      if self.debug:
        print("self.nResidPosX[key]" + str(self.nResidPosX[key]))
        print("self.nResidPosY[key]" + str(self.nResidPosY[key]))

      pValues.append([pvBinX, str(key) + "_pvBinX", self.coordsByLandmark[key][0], self.coordsByLandmark[key][1], self.coordsByLandmark[key][2]])
      pValues.append([pvBinY, str(key) + "_pvBinY", self.coordsByLandmark[key][0], self.coordsByLandmark[key][1], self.coordsByLandmark[key][2]])
      pValues.append([pvF,    str(key) + "_pvF",    self.coordsByLandmark[key][0], self.coordsByLandmark[key][1], self.coordsByLandmark[key][2]])
      pValues.append([pvADF,  str(key) + "_pvADF",  self.coordsByLandmark[key][0], self.coordsByLandmark[key][1], self.coordsByLandmark[key][2]])

    pValues.sort()
 
    residualStats = ch
    firstFail = True

    if self.debug:
      print("Pvalues------")
      for p in pValues:
        print(p)

    for j, pv in enumerate(pValues):
      i = j + 1

      if pv[0] > self.alpha:
        residualStats.append([-1, pv[0], pv[0], pv[1], pv[2], pv[3], pv[4]])
        
        if firstFail:
          print("Failed test "+ str(pv[1]) + " with value " + str(pv[0]) + ". Test " + str(i) + "/" + str(len(pValues)))
          firstFail = False
      
      else:
        residualStats.append([1, pv[0], pv[0]*i, pv[1], pv[2], pv[3], pv[4]])


    mdFilePath = self.getParam('-o')
    mdFileName, mdFileExt = os.path.splitext(mdFilePath)
    mdFilePath = mdFileName + "_resid" + mdFileExt

    self.writeOutputStatsInfo(residualStats, mdFilePath)

    print("\n")


    # Generate residual information for images
    pValues = []
    ch = []

    for key in self.imageX.keys():
      rs = self.imageSize[key]

      if self.debug:
        print("KEY--------->" + str(key))
        print("resid size " + str(self.imageSize[key]))

      # Convex hull
      convexHullArea, convexHullPerimeter = self.convexHull(vX=self.imageX[key], vY=self.imageY[key])

      print("a")

      ch.append([1, convexHullArea,      convexHullArea,       str(key) + "_chArea",  self.coordsByImage[key][0], self.coordsByImage[key][1], self.coordsByImage[key][2]])
      ch.append([1, convexHullPerimeter, convexHullPerimeter,  str(key) + "_chPerim", self.coordsByImage[key][0], self.coordsByImage[key][1], self.coordsByImage[key][2]])

      print("b")

      # Variance distribution matrix
      sumRadius = 0
      varianceMatrix = np.zeros([2, 2])

      print("c")

      for i in range(len(self.imageX[key])):
        rx = self.imageX[key][i]
        ry = self.imageY[key][i]

        rx2 = rx * rx
        ry2 = ry * ry
        rxy = rx * ry

        sumRadius = sqrt(rx2+ry2)

        if(sumRadius == 0):
          varianceMatrix += np.matrix([[rx2, rxy], [rxy, ry2]])
        else:
          varianceMatrix += np.matrix([[rx2/sumRadius, rxy/sumRadius], [rxy/sumRadius, ry2/sumRadius]])

      print("d")

      [lambda1, lambda2], _ = np.linalg.eig(varianceMatrix)

      if self.debug:
        print("lambda1: " + str(lambda1))
        print("lambda2: " + str(lambda2))
        print("self.imageModuleAcc")
        print(self.imageModuleAcc[key])
        print("self.imageYAcc")
        print(self.imageYAcc[key])
        print("self.imageXAcc")
        print(self.imageXAcc[key])
        print("self.imageY")
        print(self.imageY[key])
        print("self.imageX")
        print(self.imageX[key]) 
        print("self.nImagePosX[key]")
        print(self.nImagePosX[key])
        print("self.nImagePosY[key]")
        print(self.nImagePosY[key])

      try:
        fTestStat = lambda1/lambda2
      except ZeroDivisionError:
        fTestStat = 1

      # Statistical tests
      pvBinX = self.binomialTest(self.nImagePosX[key], rs)
      pvBinY = self.binomialTest(self.nImagePosY[key], rs)
      pvF = self.fTestVar(fTestStat, rs)
      adfStatistic, pvADF, cvADF = self.augmentedDickeyFullerTest(self.imageModuleAcc[key])

      pValues.append([pvBinX, str(key) + "_pvBinX", self.coordsByImage[key][0], self.coordsByImage[key][1], self.coordsByImage[key][2]])
      pValues.append([pvBinY, str(key) + "_pvBinY", self.coordsByImage[key][0], self.coordsByImage[key][1], self.coordsByImage[key][2]])
      pValues.append([pvF,    str(key) + "_pvF",    self.coordsByImage[key][0], self.coordsByImage[key][1], self.coordsByImage[key][2]])
      pValues.append([pvADF,  str(key) + "_pvADF",  self.coordsByImage[key][0], self.coordsByImage[key][1], self.coordsByImage[key][2]])

    pValues.sort()
 
    residualStats = ch
    firstFail = True

    if self.debug:
      print("Pvalues------")
      for p in pValues:
        print(p)

    for j, pv in enumerate(pValues):
      i = j + 1

      if pv[0] > self.alpha:
        residualStats.append([-1, pv[0], pv[0], pv[1], pv[2], pv[3], pv[4]])
        
        if firstFail:
          print("Failed test "+ str(pv[1]) + " with value " + str(pv[0]) + ". Test " + str(i) + "/" + str(len(pValues)))
          firstFail = False
      
      else:
        residualStats.append([1, pv[0], pv[0]*i, pv[1], pv[2], pv[3], pv[4]])

    mdFilePath = self.getParam('-o')
    mdFileName, mdFileExt = os.path.splitext(mdFilePath)
    mdFilePath = mdFileName + "_image" + mdFileExt

    self.writeOutputStatsInfo(residualStats, mdFilePath)

    print("\n")



if __name__ == '__main__':

  exitCode=ScriptTomoResidualStatistics().tryRun()
  sys.exit(exitCode)
