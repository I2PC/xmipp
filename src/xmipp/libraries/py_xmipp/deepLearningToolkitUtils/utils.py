import os
import traceback

BAD_IMPORT_MSG='''
Error, tensorflow/keras is probably not installed. Install it with:\n  ./scipion installb deepLearningToolkit
If gpu version of tensorflow desired, install cuda 8.0 or cuda 9.0
We will try to automatically install cudnn, if unsucesfully, install cudnn and add to LD_LIBRARY_PATH
add to SCIPION_DIR/config/scipion.conf
CUDA = True
CUDA_VERSION = 8.0  or 9.0
CUDA_HOME = /path/to/cuda-%(CUDA_VERSION)
CUDA_BIN = %(CUDA_HOME)s/bin
CUDA_LIB = %(CUDA_HOME)s/lib64
CUDNN_VERSION = 6 or 7
'''


def checkIf_tf_keras_installed():
  try:
    import tensorflow, keras
  except ImportError as e:
    print(e)
    raise ValueError(BAD_IMPORT_MSG)


def updateEnviron(gpus=None):
  """ Create the needed environment for TensorFlow programs. """
  print("updating environ to select gpus: %s"%(gpus) )
  if gpus is None:
    os.environ['CUDA_VISIBLE_DEVICES']="-1"
    return None
  else:
    if isinstance(gpus, str):
      if gpus.startswith("all"):
        return "all"
      elif gpus is not "":
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        if "," in gpus:
          return [int(elem) for elem in gpus.split(",")]
        else:
          return [int(elem) for elem in gpus]
      else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
        return None
    else:
      os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
      if isinstance(gpus, list):
        return [int(elem) for elem in gpus]
      else:
        return [int(gpus)]
  return None

