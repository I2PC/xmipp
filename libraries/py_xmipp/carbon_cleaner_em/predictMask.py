import os, sys
import numpy as np
import keras
from skimage.util import view_as_windows
from .preprocessMic import preprocessMic, padToRegularSize, getDownFactor, normalizeImg
from skimage.transform import resize

from .config import MODEL_IMG_SIZE

BATCH_SIZE=12

class MaskPredictor(object):
  def __init__(self, deepLearningModelFname, boxSize, gpus):
    self.model= keras.models.load_model(deepLearningModelFname, {})
    self.boxSize= boxSize
    if gpus is not None:
      if len(gpus)>1:
        self.model= keras.utils.multi_gpu_model(self.model, gpus=len(gpus))
      
  def getDownFactor(self):
    return getDownFactor(self.boxSize)
    
  def predictFake(self, arrayOfPatches):
    for i in range(arrayOfPatches.shape[0]):
      arrayOfPatches[i][arrayOfPatches[i]>0.5]=1
      arrayOfPatches[i][arrayOfPatches[i]<=0.5]=0
    return arrayOfPatches
    
  def predictMask(self, inputMic, strideFactor=2):
    
    inputMic, downFactor= preprocessMic( inputMic, self.boxSize )
    
    mask= np.zeros_like(inputMic)

    mask+=  self.predictOnePose( inputMic, strideFactor)
    mask+=  np.rot90( self.predictOnePose( np.rot90(inputMic, 1), strideFactor) , 4-1)
    mask/=2.
    
#    import matplotlib.pyplot as plt
#    fig=plt.figure();fig.add_subplot(121); plt.imshow(np.squeeze(inputMic), cmap="gray"); fig.add_subplot(122); plt.imshow(np.squeeze(mask), cmap="gray", vmin=0., vmax=1.); plt.show()
     
    return mask, downFactor

  def predictOnePose(self, inputMic, strideFactor, verbose=False):
  
    paddedMic, paddingTuples = padToRegularSize(inputMic, MODEL_IMG_SIZE, strideFactor)
    
##    import matplotlib.pyplot as plt
##    fig=plt.figure();fig.add_subplot(121); plt.imshow(np.squeeze(paddedMic), cmap="gray");plt.show()
     
    micShape= paddedMic.shape
    windows= view_as_windows(paddedMic, (MODEL_IMG_SIZE, MODEL_IMG_SIZE), step=MODEL_IMG_SIZE//strideFactor)
    originalWinShape= windows.shape
    windowsOri= windows
    windows= windows.reshape((-1, MODEL_IMG_SIZE, MODEL_IMG_SIZE, 1))
    #Check it out if normalization helps    
    
    micro_preds= self.model.predict(windows, batch_size=BATCH_SIZE, verbose=1 if verbose else 0)
#    micro_preds= self.predictFake(windows)

    mask= self.return_as_oneMic( micro_preds.reshape(originalWinShape), micShape, paddingTuples, MODEL_IMG_SIZE, strideFactor )
    
    return mask

  def return_as_oneMic(self, predWindows, micShape, paddingTuples, patchSize, strideFactor):
    stride= patchSize//strideFactor
    endPoint_height, endPoint_width= micShape[:2]
    endPoint_height-= patchSize -1
    endPoint_width-=  patchSize -1
    micro_out= np.zeros(micShape)
    micro_weights= np.zeros(micShape)
    for i, i_inMat in enumerate(range(0, endPoint_height, stride)):
      for j, j_inMat in enumerate(range(0, endPoint_width, stride)):
        micro_out[i_inMat: i_inMat+patchSize, j_inMat: j_inMat+patchSize]+= predWindows[i,j,...]
        micro_weights[i_inMat: i_inMat+patchSize, j_inMat: j_inMat+patchSize]+=1.
    
    micro_out= micro_out/ micro_weights
    micro_out= micro_out[paddingTuples[0][0]:-paddingTuples[0][1], paddingTuples[1][0]:-paddingTuples[1][1],...]
    return micro_out

        
