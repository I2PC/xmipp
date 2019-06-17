import sys, os
import numpy as np
import pandas as pd
from .filesManager import getCoordsColNames
from .config import DESIRED_PARTICLE_SIZE

def filterCoords( inputCoords, predictedMask, deepThr, sizeThr, boxSize=DESIRED_PARTICLE_SIZE):

  intBoxSize= int(boxSize)
  h,w = predictedMask.shape[:2]
  inputCoordsMat= inputCoords[getCoordsColNames(inputCoords)].values
  keepIndices=[]
  scores=[]
  totalSurface= np.sum(predictedMask)/np.prod(predictedMask.shape)
  
  if totalSurface>sizeThr and deepThr is not None:
    print("Predictions were ignored due to failure size threshold. Score set to +2")
    scores= [+2]*len(inputCoords)
    keepIndices= range(len(inputCoords))
  else:
    for i, (x,y) in enumerate(inputCoordsMat):
      x= int(round(x-boxSize/2.))
      y= int(round(y-boxSize/2.))
      y_0, y_1= min(max(0,y), h), max(min(y+intBoxSize,h), 0)
      x_0, x_1= min(max(0,x), w), max(min(x+intBoxSize,w), 0)
      scoreBox= predictedMask[y_0:y_1, x_0:x_1 ]
      meanScoreBox= np.mean(scoreBox)
      if deepThr is None or meanScoreBox< deepThr:
        keepIndices.append(i)
        scores.append(meanScoreBox)
        
  if deepThr is not None:
    filteredCoords= inputCoords.iloc[keepIndices,:]
  else:
    filteredCoords= inputCoords
  filteredCoords= filteredCoords.assign(goodRegionScore=1-np.array(scores))
#  print(filteredCoords.sort_values("goodRegionScore"))
  return filteredCoords
  
