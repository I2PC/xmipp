import sys,os
from subprocess import check_call

MODELS_URL= "http://campins.cnb.csic.es/deep_cons/"

def retriveOneItem(baseName):
    cmd= "wget -N %s/%s"%(MODELS_URL, baseName)
    print(cmd)
    check_call(cmd, shell=True)
    cmd= "tar -zxvf %s; rm %s"%(baseName, baseName)
    print(cmd)
    check_call(cmd, shell=True)
    
def retrieveModelsForDeepConsensus(destDir):
  curDir= os.getcwd()
  os.chdir( os.path.expanduser(destDir) )
  for suffix in ["no", ""]:
    baseName= "negativeTrain_%sPhaseFlip_Invert.mrcs.tar.gz"%(suffix)
    retriveOneItem(baseName)
  baseName= "keras_models.tar.gz"
  retriveOneItem(baseName)


if __name__=="__main__":
  '''
    usage.
    python build/pylib/xmippPyModules/deepConsensusWorkers/updateModels.py src/xmipp/models/deepConsensus
  '''
  retrieveModelsForDeepConsensus(sys.argv[1])
  
