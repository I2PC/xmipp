import sys, os
import numpy as np
from threading import Lock


LOCK = Lock()
MASK_PREDICTOR_HANDLER=None

def cleanOneMic(micFname, inputCoordsFname, outCoordsFname, predictedMaskFname, deepLearningModel, boxSize,
                downFactor=1, deepThr=0.5, sizeThr=0.8, gpus="0"):
  from .filesManager import loadMic, loadCoords, writeMic, writeCoords
  from .predictMask import MaskPredictor, normalizeImg
  from .filterCoords import filterCoords

  global MASK_PREDICTOR_HANDLER
  with LOCK:
    if MASK_PREDICTOR_HANDLER is None:
      MASK_PREDICTOR_HANDLER= MaskPredictor(deepLearningModel, boxSize, gpus)
      
  maskPredictor= MASK_PREDICTOR_HANDLER

  if predictedMaskFname is not None and os.path.isfile(predictedMaskFname):
    print("WARNING: mask already predicted for %s. Using it instead computing a new predicted mask"%(micFname))
    predictedMask= loadMic( predictedMaskFname)
    internalDownFactor= maskPredictor.getDownFactor()
  else:
    inputMic= loadMic( micFname )
    predictedMask, internalDownFactor= maskPredictor.predictMask(inputMic)
    if predictedMaskFname is not None:
      writeMic(predictedMaskFname, predictedMask)
  
  if inputCoordsFname is not None:
    downFactor= float(internalDownFactor* downFactor)
    boxSize= boxSize/downFactor
    inputCoords= loadCoords(inputCoordsFname, downFactor)
    filteredCoords= filterCoords( inputCoords, predictedMask, boxSize, deepThr, sizeThr)

    writeCoords(outCoordsFname, filteredCoords, downFactor)

