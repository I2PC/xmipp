import importlib
import os

from .config import CUBES_PATH_GENERIC, NNET_CHECKPOINT_TEMPLATE, JSON_TRAIN_VAL_TEST_SPLIT_TEMPLATE, N_GPUs, \
  BATCH_SIZE, DEBUG_MODE
from .utils.pathUtils import tryToCreateDir

REGRESSION_FLAGS=["pdb", "locscale", "improveRes"]
CLASSIFICATION_FLAGS=["tightMask", "mask"]

ALL_PROTOCOLS_FLAGS= REGRESSION_FLAGS+CLASSIFICATION_FLAGS

class ConfigManager(object):
  processing=None

  unet_module=None

  n_gpus=N_GPUs if not DEBUG_MODE else 0

  @staticmethod
  def setProcessingType( processing):
    if ConfigManager.processing is not None:
      raise Exception("Error, processing has already been set")
    else:
      ConfigManager.processing= processing

  @staticmethod
  def getUnetModule():
    if ConfigManager.processing is None:
      raise Exception("Error, processing has not been set")

    if ConfigManager.unet_module is None:
      print("Using default unet")
      ConfigManager.setUnetModule(unetAsStr=None)
    return ConfigManager.unet_module

  @staticmethod
  def setUnetModule(unetAsStr=None):
    if ConfigManager.processing is None:
      raise Exception("Error, processing has not been set")

    if ConfigManager.unet_module is not None:
      raise Exception("Error, setUnetModule has already been set")

    if ConfigManager.unet_module is None:
      if unetAsStr:
        pass
      elif ConfigManager.processing in CLASSIFICATION_FLAGS:
        unetAsStr="unet_tightMask_v6"
      elif ConfigManager.processing in REGRESSION_FLAGS:
        unetAsStr="unet_v28" #"unet_v22"
      else:
        raise ValueError("Not supported processing option: " + str(ConfigManager.processing))

      ConfigManager.unet_module = importlib.import_module('devel_code.trainNet.architectures.'+unetAsStr)


  @staticmethod
  def getCubesPath( tryToCreate=False):
    processing= ConfigManager.processing
    if processing=="locscale":
      cubes_path =  CUBES_PATH_GENERIC + "_locscale"
    elif processing=="improveRes":
      cubes_path =  CUBES_PATH_GENERIC + "_improveRes"
    elif ConfigManager.processing in REGRESSION_FLAGS or \
         ConfigManager.processing=="pdb":
      cubes_path =  CUBES_PATH_GENERIC + "_pdb"
    elif "ask" in processing: #masking network
      cubes_path = CUBES_PATH_GENERIC + "_mask"
    else:
      raise ValueError("Error, not recognized processing option: %s" % processing)

    if tryToCreate: tryToCreateDir(cubes_path)
    return cubes_path

  @staticmethod
  def getModelPath(useProduction=False, useMasked=False):

    if useProduction:
      productionTag="production_"
    else:
      productionTag=""
    processing= ConfigManager.processing
    if processing in CLASSIFICATION_FLAGS+REGRESSION_FLAGS:
      if useMasked:
        processing += "_masked"
      modelName = NNET_CHECKPOINT_TEMPLATE %(productionTag, processing)
    else:
      raise ValueError("Error, not recognized processing option: %s"%processing)

    if not useProduction:
      tryToCreateDir( os.path.split(modelName)[0])
    return modelName

  @staticmethod
  def getLastActivationAndLoss():
    processing= ConfigManager.processing

    from keras.losses import binary_crossentropy, mean_absolute_error
    from .trainNet.utilsTraining.metrics import weighted_mae, mae_and_edge

    if processing in REGRESSION_FLAGS:
      activ = "linear"
      loss = mean_absolute_error #mae_and_edge #mean_absolute_error #, mean_absolute_error # mean_squared_error #,weighted_mae
    elif processing in CLASSIFICATION_FLAGS:
      activ, loss = "sigmoid", binary_crossentropy # weighted_bce, #binary_crossentropy
    else:
      raise ValueError("Error, not recognized processing option: %s" % processing)

    input("ConfigManager loss:, "+str(loss)+" press enter to continue")
    return activ, loss

  @staticmethod
  def getInputNormalization(useMask=False):

    from .utils.normalization import inputNormalization_3, inputNormalizationWithMask_2, inputNormalization_classification

    if useMask is True:
      print("WARNING: trying input inputNormalizationWithMask_2.")
      return inputNormalizationWithMask_2

    if ConfigManager.processing in [ "locscale"]:
      print("WARNING: trying input normalization v3")
      return inputNormalization_3

    elif ConfigManager.processing in ["pdb"]:
      print("WARNING: trying input normalization v3")
      return inputNormalization_3

    elif ConfigManager.processing == "improveRes":
      raise NotImplementedError("Normalization not implemented for "+ConfigManager.processing)

    elif ConfigManager.processing in CLASSIFICATION_FLAGS:
      print("WARNING: trying input inputNormalization_classification.")
      return inputNormalization_classification
    else:
      raise ValueError("Error, not recognized processing option: %s" % ConfigManager.processing)


  @staticmethod
  def getTrainValJson():
    processing= ConfigManager.processing
    return JSON_TRAIN_VAL_TEST_SPLIT_TEMPLATE % processing

  @staticmethod
  def getAugmentations():
    from .trainNet.utilsTraining.augmentators  import randomRotation90, random_noise_addition, random_blur, random_blur_patches, random_mask_out, random_mask_noise_distroy
    if ConfigManager.processing == "locScale":
      myRandomBlur = lambda x, y: random_blur(x, y, sigmaRange=(0.5, 2))
    else:
      myRandomBlur = lambda x, y: random_blur(x, y, sigmaRange=(0.5, 1))

    augments= [(randomRotation90, 0), (random_noise_addition, 0), (myRandomBlur, 0)]
    augments+=[(random_mask_out,0), (random_blur_patches,0), (random_mask_noise_distroy, 0)]
    return augments

  @staticmethod
  def setN_GPUs(nGpus):
    if nGpus is not None:
        ConfigManager.n_gpus= nGpus
    else:
      raise Exception("Error, nGpus must not be None")

  @staticmethod
  def getN_GPUs():
    if ConfigManager.n_gpus is None:
      raise Exception("Error, nGpus has not being been set")
    return ConfigManager.n_gpus

  @staticmethod
  def getBatchSize():
    if ConfigManager.n_gpus > 0:
      return BATCH_SIZE *ConfigManager.n_gpus
    else:
      return BATCH_SIZE

  @staticmethod
  def getOptimizer(learningRate):
    from keras.optimizers import SGD, Adam
    from keras_radam import RAdam
    # opt = RAdam(lr=learningRate, min_lr=1e-2 * learningRate, total_steps=0, warmup_proportion=0.01)  # len(train_dataManager)*N_EPOCHS )
    # opt = SGD(lr=learningRate)
    opt = Adam(lr=learningRate)
    return opt
if __name__=="__main__":
  mod= importlib.import_module('devel_code.trainNet.architectures.unet_v22_phaseReg')
  print(mod)