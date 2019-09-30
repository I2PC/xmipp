import sys, os
import keras
import numpy as np

from skimage.io import imsave
from dataGenerator import getDataGenerator, BATCH_SIZE
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
    self.addSyntheticEmpty= (trainingDataMode=="ParticlesAndSyntheticNoise")
    
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


  def save_imgs(self, imgs, saveImagesPath, epoch, nImagesToPlot=8): #For debugging purposes
    '''

    :param imgs: A list of different types of matching images. E.g. [noisyStack, denoisedStack]
    :param saveImagesPath: path where images will be saved
    :param epoch: the current epoch during training
    :param nImagesToPlot: number of images to plot
    :return: None
    '''
    save_imgs(imgs, saveImagesPath, epoch, basenameTemplate="denoise_%d.png", nImagesToPlot=nImagesToPlot)

  def takeListMean(self, x):
    if len(x)==0:
      return np.nan
    else:
      return np.nanmean(x)
      
def save_imgs(imgs, saveImagesPath, epoch, basenameTemplate, nImagesToPlot=8):

  nTypes= len(imgs)
  imgSize= np.squeeze(imgs[0][0]).shape
  out=np.zeros( (nImagesToPlot*imgSize[0], nTypes*imgSize[1]), dtype=np.uint8)
  for i in range(nImagesToPlot):
    for j in range(nTypes):
      img=  np.squeeze(imgs[j][i])
      img= (255*(img-np.min(img))/(np.max(img)-np.min(img))).astype(dtype=np.uint8)
      out[i*imgSize[0]:(i+1)*imgSize[0], j*imgSize[1]:(j+1)*imgSize[1]] = np.squeeze(img)

  fname= os.path.join(saveImagesPath, basenameTemplate% epoch)
  if os.path.exists(fname):
    try:
      os.remove(fname)
    except (IOError, OSError):
      pass
  imsave(fname, out)

      
def generatePerceptualLoss(image_shape):
  import xmipp3
  from keras.layers import Input
  from keras.models import load_model, Model
  import keras.backend as K
  effectiveSize=50000
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
