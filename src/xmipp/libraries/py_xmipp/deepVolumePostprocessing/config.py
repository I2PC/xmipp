import os

#These are the parameters that where used to train the network
RESIZE_VOL_TO= 1.
NNET_INPUT_SIZE=64
NNET_INPUT_STRIDE= NNET_INPUT_SIZE//4 #TODO: REMOVE all this parameters as they are included in the model
MAX_VAL_AFTER_NORMALIZATION=200

DOWNLOAD_MODEL_URL = 'http://biocomp.cnb.csic.es/deepVol_postprocess/deepVolPostModels.zip'
DEFAULT_MODEL_DIR = os.path.expanduser("~/.local/share/volume_post_processer/production_checkpoints")

VALID_MODELS={"maskAndSharp":["bestCheckpoint_locscale.hd5", "bestCheckpoint_tightMask.hd5"], #Only used "bestCheckpoint_locscale.hd5"
              "mask":["bestCheckpoint_tightMask.hd5"], "tightMask":["bestCheckpoint_tightMask.hd5"],
              "sharp":["bestCheckpoint_locscale.hd5"]}


NNET_NJOBs= 8
N_GPUs=1 #ONLY ONE GPU is currently used
BATCH_SIZE=4



def GET_CUSTOM_OBJECTS():
  from keras_contrib.layers.normalization import groupnormalization
  return {  "GroupNormalization":groupnormalization.GroupNormalization }


######################################################
BATCH_SIZE*= N_GPUs


#######################################
# Remove the following when configManager removed
#######################################

NNET_CHECKPOINT_TEMPLATE= os.path.join(DEFAULT_MODEL_DIR, "%scheckpoints", "bestCheckpoint_%s.hd5") #TODO: REMOVE THIS OLD DEPENDENCIES
CUBES_PATH_GENERIC=None
DEBUG_MODE=None
JSON_TRAIN_VAL_TEST_SPLIT_TEMPLATE=None
VOLUMES_PATH=None