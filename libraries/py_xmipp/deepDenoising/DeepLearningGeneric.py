import sys, os
import keras
import math
import numpy as np
from matplotlib import pyplot as plt

from dataGenerator import getDataGenerator,  BATCH_SIZE
from augmentators import generateReverseNormalizationFunction


class DeepLearningModel():
  def __init__(self, boxSize, saveModelFname, gpuList, batchSize, generatorLoss, trainingDataMode, 
                      modelDepth=4, regularizationStrength=1e-5):
  
    self.saveModelFname = saveModelFname
    self.gpuList= gpuList
    self.batchSize= batchSize
    self.modelDepth= modelDepth
    self.regularizationStrength= regularizationStrength
    self._setShape(boxSize)
    self.addSyntheticEmpty= True if trainingDataMode=="ParticlesAndSyntheticNoise" else False
    
    if generatorLoss=="MSE":
      self.generatorLoss= keras.losses.mse
    elif generatorLoss=="PerceptualLoss":
      self.generatorLoss= generatePerceptualLoss(self.img_shape)
    elif generatorLoss=="Both":
      self.generatorLoss= generateBothLoss(self.img_shape)
    else:
      raise ValueError("Unrecognized loss type %s"%(generatorLoss))  

  def _setShape(self, boxSize):
    raise ValueError("Not implemented yet")

  def getRandomRows(self, x, n):
    return x[np.random.choice(x.shape[0], n),... ]
      
  def train(self, learningRate, nEpochs, xmdParticles, xmdProjections, xmdEmptyParts=None):        
    raise ValueError("Not implemented yet")
                        

  def createSaveImgsPath(self, xmdParticles):
    saveImagesPath= os.path.split(xmdParticles)[0]
    saveImagesPath= os.path.join(saveImagesPath, "batchImages")
    if not os.path.exists(saveImagesPath):
      os.mkdir(saveImagesPath)
    return saveImagesPath
    
  def yieldPredictions(self, xmdParticles, xmdProjections=None):
    keras.backend.clear_session()
    if xmdProjections is None:
      xmdProjections= xmdParticles
    print("loading saved model")
    model = keras.models.load_model(self.saveModelFname, custom_objects={self.generatorLoss.__name__: self.generatorLoss})
    print("model loaded")
    trainIterator, stepsPerEpoch= getDataGenerator(xmdParticles, xmdProjections, isTrain=False, valFraction=0, 
                                                   augmentData=False, nEpochs=1, batchSize= self.batchSize)
    normalizePredFun= generateReverseNormalizationFunction(self.img_shape, radiusFraction=0.8)
    for noisyParticles, projections in trainIterator:
      preds= model.predict(noisyParticles, batch_size=BATCH_SIZE)
      preds= normalizePredFun(preds, noisyParticles)
      yield preds, noisyParticles, projections

  def clean(self):
    keras.backend.clear_session()


  def save_imgs(self, imgs, titles, saveImagesPath,  epoch, plotInstead=False, nImagesToPlot=8): #For debugging purposes
#    save_imgs( imgs, titles, saveImagesPath,  epoch, plotInstead, nImagesToPlot)
    return
    
  def takeListMean(self, x):
    if len(x)==0:
      return np.nan
    else:
      return np.nanmean(x)
      
      
      
def save_imgs( imgs, titles, saveImagesPath,  epoch, plotInstead=False, nImagesToPlot=8):

  nTypes= len(imgs)
  plt.switch_backend('agg')
  fig, axs = plt.subplots(nImagesToPlot, nTypes)
  assert nTypes==len(titles)
  for i in range( nTypes ):
    axs[0, i].set_title(titles[i])
  for i in range(nImagesToPlot):
    for j in range(nTypes): 
      axs[i, j].imshow(np.squeeze(imgs[j][i]), cmap='gray')
      axs[i, j].axis('off')

  fname= os.path.join(saveImagesPath, "denoise_%d.png"% epoch)
  if os.path.exists(fname):
    try:
      os.remove(fname)
    except IOError, OSError:
      pass
  if not plotInstead:
    plt.savefig( fname)
    plt.close()
  else:
    plt.show()
  plt.switch_backend('TkAgg')   
      
def generatePerceptualLoss(image_shape):
  import xmipp3
  from keras.layers import Lambda, Input
  from keras.models import load_model, Model
  import keras.backend as K
  effectiveSize=int(5e4)
  ignoreCTF=True
  modelTypeDir= "keras_models/%sPhaseFlip_Invert/nnetData_%d/tfchkpoints_0" % (
                      "no" if ignoreCTF else "", effectiveSize)
  modelTypeDir= xmipp3.Plugin.getModel("deepConsensus", modelTypeDir)
  modelFname= os.path.join(modelTypeDir, "deepModel.hdf5")
  
  def perceptual_loss(y_true, y_pred):
    input_tensor= Input(image_shape[:-1]+(1,))
    evalModel= load_model(modelFname)

    targetLayer='activation_4'
    out= input_tensor
    for layer in evalModel.layers[1:]:
#      print(layer.name)
      out= layer(out)
      if layer.name == targetLayer: break
    
    loss_model = Model(inputs=input_tensor, outputs=out)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))
    
  return perceptual_loss

def generateBothLoss(image_shape):
  import keras.backend as K
  def bothLoss(y_true, y_pred):
    return generatePerceptualLoss(image_shape)(y_true, y_pred) + K.mean(keras.metrics.mean_squared_error(y_true, y_pred))
  return bothLoss
