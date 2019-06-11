import numpy as np

from skimage.util import pad
from skimage.transform import resize

from .config import MODEL_IMG_SIZE, DESIRED_PARTICLE_SIZE

try:
  from scipy.stats import iqr
except ImportError:
  def iqr(x, rng=(25,75)):
    q1_x = np.percentile(x, rng[0])
    q3_x = np.percentile(x, rng[1])
    return q3_x - q1_x

def normalizeImg(img, squeezeToRange=False, sigmoidInsteadTanh=True, iqrRange=(25,75)):
  '''
  Proposed better normalization. Testing better normalization
  '''
  iqr_val= iqr(img, rng= iqrRange )
  if iqr_val==0:
      print("warning, bad iqr")
      iqr_val= (np.max(img)-np.min(img)) + 1e-12
  newImg=(img- np.median(img))/iqr_val
  if squeezeToRange:
    if sigmoidInsteadTanh:
      newImg=1. / (1 + np.exp(-newImg))
    else:
      newImg= np.tanh(newImg)
  return newImg


def padToRegularSize(inputMic, windowSide, strideDiv, fillWith0=True ):
  stride= windowSide//strideDiv
  height, width= inputMic.shape[:2]
  paddingHeight= (0, stride- height%stride )
  paddingWidth=  (0, stride- width%stride  )
  
  paddingValues= [paddingHeight, paddingWidth]
  if fillWith0:
    paddedMic= pad(inputMic, paddingValues, mode="constant", constant_values= np.min(inputMic) )
  else:
    paddedMic= pad(inputMic, paddingValues, mode="wrap" )
  return paddedMic, paddingValues
  
def getDownFactor(particleSize):
  return particleSize/float(DESIRED_PARTICLE_SIZE)
  
def preprocessMic(mic, particleSize):

  mic= normalizeImg(mic, squeezeToRange=False, iqrRange=(25,75))
  downFactor= getDownFactor(particleSize)
  mic= resize(mic, tuple([int(s/downFactor) for s in mic.shape]), preserve_range=True, 
                  anti_aliasing=True, mode='reflect')
  mic= normalizeImg(mic, squeezeToRange=True, sigmoidInsteadTanh=True, iqrRange=(10, 90))
  return mic, downFactor
  
