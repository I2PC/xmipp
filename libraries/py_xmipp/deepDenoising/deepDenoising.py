import os


from dataGenerator import normalization
from ..error_msgs import BAD_IMPORT_TENSORFLOW_KERAS_MSG

def updateEnviron(gpuNum):
    """ Create the needed environment for TensorFlow programs. """
    print("updating environ to select gpu %s" % (gpuNum))
    if gpuNum == '':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNum)

def getModelClass(modelTypeName, gpuList):
  '''
  modelTypeName is one of ["GAN", "U-Net"]
  '''
  updateEnviron( gpuList )
  try:
    import tensorflow as tf
    import keras
  except ImportError as e:
    print(e)
    raise ValueError(BAD_IMPORT_TENSORFLOW_KERAS_MSG)
    
  if modelTypeName=="U-Net":
    from unet import UNET as modelClass
  elif modelTypeName=="GAN":
    from gan import GAN as modelClass
  else:
    raise ValueError('modelTypeName must be one of one of ["GAN", "U-Net"]') 
    
  return modelClass



