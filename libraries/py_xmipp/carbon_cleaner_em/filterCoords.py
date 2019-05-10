import sys, os
import numpy as np
import pandas as pd
from .filesManager import getCoordsColNames

def filterCoords( inputCoords, predictedMask, boxSize, deepThr, sizeThr):

  intBoxSize= int(boxSize)
  h,w = predictedMask.shape[:2]
  inputCoordsMat= inputCoords[getCoordsColNames(inputCoords)].values
  keepIndices=[]
  scores=[]
  totalSurface= np.sum(predictedMask)/np.prod(predictedMask.shape)
  if totalSurface>sizeThr and deepThr is not None:
    print("Predictions were ignored due to failure size threshold. Score set to -1")
    scores= [-1]*len(inputCoords)
    keepIndices= range(len(inputCoords))
  else:
    for i, (x,y) in enumerate(inputCoordsMat):
      x= int(round(x-boxSize/2.))
      y= int(round(y-boxSize/2.))
      y_0, y_1= max(0,y), min(y+intBoxSize,h)
      x_0, x_1= max(0,x), min(x+intBoxSize,w)
      scoreBox= predictedMask[tuple([tuple(range(y_0, y_1)), tuple(range(x_0, x_1)) ])]
      meanScoreBox= np.mean(scoreBox)
      if deepThr is None or meanScoreBox< deepThr:
        keepIndices.append(i)
        scores.append(meanScoreBox)
        
  if deepThr is not None:
    filteredCoords= inputCoords.iloc[keepIndices,:]
  else:
    filteredCoords= inputCoords
  filteredCoords= filteredCoords.assign(deepSegGoodRegionScore=1-np.array(scores))
  return filteredCoords
  
