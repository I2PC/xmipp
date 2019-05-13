from __future__ import absolute_import, division, print_function
import sys, os
import glob
from joblib import Parallel, delayed

DOWNLOAD_MODEL_URL="http://campins.cnb.csic.es/carbon_cleaner/defaultModel.keras.gz"
DEFAULT_MODEL_PATH=os.path.expanduser("~/.local/share/carbon_cleaner_em/models/")
def main(inputMicsPath, inputCoordsDir, outputCoordsDir, deepLearningModel, boxSize, downFactor, deepThr,
         sizeThr, predictedMaskDir, gpus="0"):

  selectGpus(gpus)
  micsFnames=getFilesInPath(inputMicsPath, ["mrc", "tif"])
  inputCoordsFnames=getFilesInPath(inputCoordsDir, ["txt", "tab", "pos"])
  coordsExtension= inputCoordsFnames[0].split(".")[-1] if inputCoordsFnames is not None else None
  matchingFnames= getMatchingFiles(micsFnames, inputCoordsDir, outputCoordsDir, predictedMaskDir, coordsExtension)
  assert len(matchingFnames)>0, "Error, there are no matching coordinate-micrograph files"
  from .cleanOneMic import cleanOneMic
  Parallel(n_jobs=1)( delayed(cleanOneMic)( * multipleNames+( deepLearningModel, 
                                              boxSize, downFactor, deepThr,sizeThr, gpus) )
                                          for multipleNames in matchingFnames.values() )
def selectGpus(gpusStr):
  print("updating environ to select gpus %s" % (gpusStr))
  if gpusStr == '':
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpusStr).replace(" ", "")
                                         
def getFilesInPath(pathsList, extensions):
  if pathsList is None:
    return None
  if isinstance(pathsList, str) or len(pathsList)==1:
    if not isinstance(pathsList, str) and len(pathsList)==1:
      pathsList= pathsList[0]
    if os.path.isdir(pathsList):
      pathsList= os.path.join(pathsList, "*")
    fnames=glob.glob(pathsList)
    assert len(fnames)>=1 and not os.path.isdir(pathsList), "Error, %s path not found or incorrect"%(pathsList)
    errorPath= pathsList
  else:
    fnames= pathsList
    errorPath= os.path.split(pathsList[0])[0]
  extensions= set(extensions)
  fnames= [ fname for fname in fnames if fname.split(".")[-1] in extensions ]
  assert len(fnames)>0, "Error, there are no < %s > files in path %s"%(" - ".join(extensions), errorPath)
  return fnames

def getMatchingFiles(micsFnames, inputCoordsDir, outputCoordsDir, predictedMaskDir, coordsExtension):
  def getMicName(fname):
    return ".".join( os.path.basename( fname).split(".")[:-1]  )
    
  matchingFnames={}
  for fname in micsFnames:
    micName= getMicName(fname)
    print(micName)
    if inputCoordsDir is not None:
      inCoordsFname= os.path.join(inputCoordsDir, micName+"."+coordsExtension)
      if os.path.isfile(inCoordsFname):
        outCoordsFname= os.path.join(outputCoordsDir, micName+"."+coordsExtension)
        if predictedMaskDir is not None:
          predictedMaskFname= os.path.join(predictedMaskDir, micName+".mrc")
        else:
          predictedMaskFname=None
        matchingFnames[micName]= (fname, inCoordsFname, outCoordsFname, predictedMaskFname)  
      else:
        print("Warning, no coordinates for micrograph %s"%(fname))
    else:
        predictedMaskFname= os.path.join(predictedMaskDir, micName+".mrc")
        matchingFnames[micName]= (fname, None, None, predictedMaskFname)  
  return matchingFnames
    
def parseArgs():
  import argparse
  
  example_text = '''examples:
  
  + Donwload deep learning model
cleanMics --download
    
  + Compute masks from imput micrographs and store them
cleanMics  -c path/to/inputCoords/ -b $BOX_SIXE  -i  /path/to/micrographs/ --predictedMaskDir path/to/store/masks
   
  + Rule out input bad coordinates (threshold<0.5) and store them into path/to/outputCoords
cleanMics  -c path/to/inputCoords/ -o path/to/outputCoords/ -b $BOX_SIXE -s $DOWN_FACTOR  -i  /path/to/micrographs/ --deepThr 0.5

+ Compute goodness scores from input coordinates and store them into path/to/outputCoords
  cleanMics  -c path/to/inputCoords/ -o path/to/outputCoords/ -b $BOX_SIXE -s $DOWN_FACTOR  -i  /path/to/micrographs/ --deepThr 0.5
     
'''
 
  parser = argparse.ArgumentParser(description='Compute goodness score for picked coordinates. Rule out bad coordinates',
                                   epilog=example_text, formatter_class=argparse.RawDescriptionHelpFormatter)
  def getRestricetedFloat(minVal=0, maxVal=1):
    def restricted_float(x):
      x = float(x)
      if x < minVal or x > maxVal:
        raise argparse.ArgumentTypeError("%r not in range [%f, %f]"%(x,minVal, maxVal))
      return x
    return restricted_float
  def file_choices(choices,fname):
    '''
    str
    '''
    ext = os.path.splitext(fname)[1][1:]
    if ext not in choices:
       parser.error("file %s extension not allowed: %s"%( (fname,)+(" ".join(choices),) ))
    return os.path.abspath(os.path.expanduser(fname))
    
  parser.add_argument('-i', '--inputMicsPath',  metavar='MIC_FNAME', type=str,  nargs='+', required=True,
                      help='micrograph(s) filenames where coordinates were picked (.mrc or .tif).\n'+
                      'Linux wildcards or several files are allowed.')

  parser.add_argument('-c', '--inputCoordsDir', type=str, required=False,
                      help='input coordinates directory (.pos or tab separated x y). Filenames '+
                           'must agree with input micrographs except for file extension.')

  parser.add_argument('-o', '--outputCoordsDir', type=str,  required=False,
                      help='output coordinates directory.')

  parser.add_argument('-d', '--deepLearningModel', metavar='MODEL_PATH', type=str,  nargs='?', required=False,
                      help=('(optional) deep learning model filename. If not provided, model at %s '+
                           'will be employed')%(DEFAULT_MODEL_PATH))
                                                             
  parser.add_argument('-b', '--boxSize', metavar='PXLs', type=int,  required=True,
                      help='particles box size in pixels')
                      
  parser.add_argument('-s', '--downFactor', type=float, nargs='?', required=False, default=1,
                      help='(optional) micrograph downsampling factor to scale coordinates, Default no scaling')
                      
  parser.add_argument('--deepThr', type=getRestricetedFloat(), nargs='?', default=None, required=False,
                      help='(optional) deep learning threshold to rule out coordinates (coord_score<=deepThr-->accepted). '+
                           'The smaller the treshold '+
                           'the more coordinates will be ruled out. Ranges 0..1. Recommended 0.3')
                           
  parser.add_argument('--sizeThr', type=getRestricetedFloat(0,1.), nargs='?', default=0.8, required=False,
                      help='Failure threshold. Fraction of the micrograph predicted as contamination to ignore predictions. '+
                           'Ranges 0..1. Default 0.8')
                           
  parser.add_argument('--predictedMaskDir', type=str, nargs='?', required=False,
                      help='directory to store the predicted masks. If a given mask already existed, it will be used instead'+
                           ' of a new prediction')

  parser.add_argument('-g', '--gpus', metavar='GPU_Ids', type=str,  required=False, default="0",
                      help='GPU ids to employ. Comma separated list. E.g. "0,1". Default 0. use -1 for CPU-only computation')
                      
  class _DownloadModel(argparse.Action):
      def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super(_DownloadModel, self).__init__( option_strings=option_strings, dest=dest, default=default,
                                              nargs=0,  help=help)

      def __call__(self, parser, namespace, values, option_string=None):
          import requests, gzip
          from io import BytesIO
          r = requests.get(DOWNLOAD_MODEL_URL)
          if r.status_code!=200:
            raise Exception("It was not possible to download model")
          if not os.path.exists(DEFAULT_MODEL_PATH):
            os.makedirs(DEFAULT_MODEL_PATH) 
          deepLearningModelPath= os.path.join(DEFAULT_MODEL_PATH, "defaultModel.keras")
          print("DOWNLAODING MODEL at %s"%(DEFAULT_MODEL_PATH) )
          with open(deepLearningModelPath , 'wb') as f:
            content= gzip.GzipFile(fileobj=BytesIO(r.content) )
            f.write(content.read())
          print("DOWNLOADED!!")
          parser.exit()

  parser.add_argument('--download', action=_DownloadModel,
                      help='Download default carbon_cleaner_em model. It will be saved at %s'%(DEFAULT_MODEL_PATH) )
                      
  args = vars(parser.parse_args())
#  print(args)
  deepLearningModelPath=args["deepLearningModel"]
  if deepLearningModelPath is None:
    if not os.path.exists(DEFAULT_MODEL_PATH):
      os.makedirs(DEFAULT_MODEL_PATH)
    deepLearningModelPath= os.path.join(DEFAULT_MODEL_PATH, "defaultModel.keras")
  args["deepLearningModel"]= deepLearningModelPath
  if not  os.path.isfile(deepLearningModelPath):
    print(("Deep learning model not found at %s. Downloading default model with --download or "+
          "indicate its location with --deepLearningModel.")%DEFAULT_MODEL_PATH )
    sys.exit(1)

  if args["inputCoordsDir"] is None and args["predictedMaskDir"] is None:
    raise Exception("Either inputCoordsDir or predictedMaskDir (or both) must be provided")
    parser.print_help()
  if args["inputCoordsDir"] is None and args["outputCoordsDir"] is not None:
    raise Exception("Error, if inputCoordsDir provided, then outputCoordsDir must also be provided")
    parser.print_help()
    
  if args["outputCoordsDir"] is None and args["inputCoordsDir"] is not None:
    raise Exception("Error, if outputCoordsDir provided, then inputCoordsDir must also be provided")
    parser.print_help()

  if "-1" in args["gpus"]:
    args["gpus"]=""
  return args

def commanLineFun():
  main( ** parseArgs() )

if __name__=="__main__":
  '''
LD_LIBRARY_PATH=/home/rsanchez/app/cuda-9.0/lib64:$LD_LIBRARY_PATH

python -m  carbon_cleaner_em.cleanMics  -c /home/rsanchez/ScipionUserData/projects/2dAverages_embeddings/Runs/008337_XmippParticlePickingAutomatic/extra/ -o ~/tmp/carbon_cleaner_em/coordsCleaned/ -b 180 -s 1   --inputMicsPath  /home/rsanchez/ScipionUserData/projects/2dAverages_embeddings/Runs/002321_ProtImportMicrographs/extra/stack_0021_2x_SumCorr.mrc

  '''
  commanLineFun()
  
