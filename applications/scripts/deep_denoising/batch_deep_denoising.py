#!/usr/bin/env python2
"""/***************************************************************************
 *
 * Authors:    Ruben Sanchez Garcia
 *
 * CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/
"""

import sys, os
import xmipp_base
import xmippLib

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
from xmippPyModules.deepDenoising.deepDenoising import getModelClass

try:
  from xmippPyModules.deepDenoising.deepDenoising import getModelClass
except ImportError as e:
  print(e)
  raise ValueError(BAD_IMPORT_MSG)

class ScriptDeepDenoising(xmipp_base.XmippScript):
  def __init__(self):
    xmipp_base.XmippScript.__init__(self)

  def defineParams(self):
    self.addUsageLine('Denoise particles Deep Learning. It works in two modes:\n'
                      '* mode 1: Train a neural network using a training dataset of particles and projections. '
                      '* mode 2: Denoise a set of particles using an already trained network')
    ## params
    self.addParamsLine(' -n <netDataPath>               : A path where the networks will be saved or loaded. '
                       'If there is an already created  network in the path, it will be loaded and 2 options can be done:'
                       '(1) the training continues '
                       '(2) the putative set of particles can be scored')

    self.addParamsLine(' --mode <mode> : "training"|"denoising". Select training or denoising mode')

    self.addParamsLine(' -i <noisyParticles> : Noisy particles to denoise or train network')
    self.addParamsLine(' [ -p <noisyParticles> ] : Projections to train network (mandatory) or to evaluate denoising')
    self.addParamsLine(' [ --empty_particles <emptyParticles> ] : Empty particles to make training more robust, optional')

    self.addParamsLine('[ -c <pathToNetworkConfJson>   ]      : A path to a json file that contains the '
                       'arguments required to create/use the network'
                       'see scipion-em-xmipp/xmipp/protocols/protocol_deep_denoising.')

    self.addParamsLine('[ -g <gpuId>  ]               : GPU Ids. By default no gpu will be used. Comma separated')
    self.addParamsLine('[ -t <numThreads>  <N=2>  ]   : Number of threads')

    self.addParamsLine("== arguments ==")


    ## examples
    self.addExampleLine('trainNet net:  xmipp_deep_denoising -n ./netData -i ./params.json')

  def run(self):
    import json

    numberOfThreads = self.getIntParam('-t')
    gpuToUse = None
    if self.checkParam('-g'):
      gpuToUse = self.getIntParam('-g')
      numberOfThreads = None

    gpuList= updateEnviron(gpuToUse)
    netArgsFname = self.getParam('-c')
    assert os.path.isfile(netArgsFname), "Error, %s is not a file"%netArgsFname
    with open(netArgsFname) as f:
      args= json.load(f)

    dataPathParticles = self.getParam('-i')
    dataPathProjections=None
    dataPathEmpty=None
    if self.checkParam('-p'):
      dataPathProjections = self.getParam('-p')

    if self.checkParam('--empty_particles'):
      dataPathEmpty = self.getParam('--empty_particles')
    else:
      dataPathEmpty = None

    mode = self.getParam('--mode')
    trainKeyWords = ["train", "training"]
    predictKeyWords = ["predict", "denoising"]
    if mode in trainKeyWords:
      print("mode 1: training")
      assert dataPathProjections is not None, "Error, projections must be provided to train the network"
      ModelClass = getModelClass(args["builder"]["modelType"], gpuList)
      del args["builder"]["modelType"]
      model = ModelClass(**args["builder"])

      model.train(args["running"]["lr"], args["running"]["nEpochs"], dataPathParticles,
                  dataPathProjections, dataPathEmpty)
      model.clean()
      del model
    elif mode in predictKeyWords:
      model= loadModel()
    else:
      raise Exception("Error, --mode must be training or denoising")

def updateEnviron(gpus=None):
  """ Create the needed environment for TensorFlow programs. """
  print("updating environ to select gpus: %s"%(gpus) )
  if gpus is not None and gpus.startswith("all"): return None
  if gpus is not None or gpus is not "":
    os.environ['CUDA_VISIBLE_DEVICES']=gpus
    return [int(elem) for elem in gpus]
  else:
    os.environ['CUDA_VISIBLE_DEVICES']="-1"
    return None

if __name__ == '__main__':
  exitCode = ScriptDeepDenoising().tryRun()
  sys.exit(exitCode)

