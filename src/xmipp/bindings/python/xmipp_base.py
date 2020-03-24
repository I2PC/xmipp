import re

from xmippLib import *
import os
import sys
import subprocess
import json


def xmippExists(path):
    return FileName(path).exists()

def getXmippPath(*paths):
    ''' Return the path of the Xmipp installation folder
        if a subfolder is provided, will be concatenated to the path
    '''
    if 'XMIPP_HOME' in os.environ:
        return os.path.join(os.environ['XMIPP_HOME'], *paths)
    else:
        # xmipp   =     build      <   bindings    <     python    <     xmipp_base.py
        xmippHome = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        if os.path.isfile(os.path.join(xmippHome, 'lib', 'libXmipp.so')):
            print("Warning: XMIPP_HOME not found.\n"
                  "Using an auto-detected directory: " + xmippHome)
            return os.path.join(xmippHome, *paths)
        else:
            raise Exception('Error: XMIPP_HOME environment variable not set')

def getMatlabEnviron(*toolPaths):
    """ Return an Environment prepared for launching Matlab
    scripts using the Xmipp binding.
    """
    env = getEnviron()
    env.set('PATH', os.environ['MATLAB_BINDIR'], Environ.BEGIN)
    env.set('LD_LIBRARY_PATH', os.environ['MATLAB_LIBDIR'], Environ.BEGIN)
    for toolpath in toolPaths:
        env.set('MATLABPATH', toolpath, Environ.BEGIN)
    env.set('MATLABPATH', os.path.join(os.environ['XMIPP_HOME'], 'libraries', 'bindings', 'matlab'),
            Environ.BEGIN)
    
    return env


class XmippScript:
    ''' This class will serve as wrapper around the XmippProgram class
    to have same facilities from Python scripts'''
    def __init__(self, runWithoutArgs=False):
        self._prog = Program(runWithoutArgs)
        
    def defineParams(self):
        ''' This function should be overwrited by subclasses for 
        define its own parameters'''
        pass
    
    def readParams(self):
        ''' This function should be overwrited by subclasses for 
        and take desired params from command line'''
        pass
    
    def checkParam(self, param):
        return self._prog.checkParam(param)
    
    def getParam(self, param, index=0):
        return self._prog.getParam(param, index)
    
    def getIntParam(self, param, index=0):
        return int(self._prog.getParam(param, index))
    
    def getDoubleParam(self, param, index=0):
        return float(self._prog.getParam(param, index))
    
    def getListParam(self, param):
        return self._prog.getListParam(param)
    
    def addUsageLine(self, line, verbatim=False):
        self._prog.addUsageLine(line, verbatim)

    def addExampleLine(self, line, verbatim=True):
        self._prog.addExampleLine(line, verbatim)
        
    def addParamsLine(self, line):
        self._prog.addParamsLine(line)
    
    def run(self):
        # type: () -> object
        ''' This function should be overwrited by subclasses and
        it the main body of the script'''   
        pass
     
    def tryRun(self):
        ''' This function should be overwrited by subclasses and
        it the main body of the script'''
        try:
            print("WARNING: This is xmipp_base implementation for script")
            self.defineParams()
            doRun = self._prog.read(sys.argv)
            if doRun:
                self.readParams()
                self.run()
            return 0
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return 1

    @classmethod
    def runCondaCmd(cls, program, arguments, **kwargs):
       '''
       This class method is used to run programs that are independent of xmipp but employ conda. The class should
       possess a _conda_env attribute to be used. Otherwise  CONDA_DEFAULT_ENVIRON is used. To use xmipp dependent
       programs, runCondaJob within a XmippProtocol is preferred.
       :param program:str. A program/pythonScript to execute (included in environment bin or full path)
       :param arguments: str. The arguments for the program
       :param kwargs: options
       :return:
       '''
       if "env" not in kwargs:
           kwargs["env"]= os.environ
       if (hasattr(cls, "_conda_env")):
           condaEnvName = cls._conda_env
       else:
           condaEnvName = CONDA_DEFAULT_ENVIRON
           # raise Exception("Error, protocols using runCondaJob must define the variable _conda_env")
       program, arguments, kwargs = prepareRunConda(program, arguments, condaEnvName, **kwargs)
       print(program + " " + arguments)
       try:
           subprocess.check_call(program + " " + arguments, shell=True, **kwargs)
       except subprocess.CalledProcessError as e:
           subprocess.check_call(program + " " + arguments, shell=True, **kwargs)

def prepareRunConda(program, arguments, condaEnvName, **kwargs):
    '''
    Used to get the arguments for the methods runCondaJob or runCondaCmd
    :param program: string
    :param arguments: string
    :param condaEnvName: string
    :param kwargs:
    :return: (program, arguments, kwargs). Updated values
    '''

    kwargs['env'] = CondaEnvManager.modifyEnvToUseConda(kwargs['env'],
                                                               condaEnvName)

    kwargs['env']['PYTHONWARNINGS'] = 'ignore::FutureWarning'
    usePython = False
    programName = os.path.join(getXmippPath('bin'), program)
    try:
        with open(programName) as f:
            if "python" in f.read(32):
                usePython = True
    except (OSError, IOError):
        pass
    if usePython:
        return ("python", programName + " " + arguments, kwargs)
    else:
        return (program, arguments, kwargs)

class CondaEnvManager(object):
    CONDA_DEFAULT_ENVIRON = "xmipp_DLTK_v0.3" #'xmipp_DLTK_v0.2'
    from condaEnvsDef import DEFAULT_ENVS
    DICT_OF_CONDA_ENVIRONS =DEFAULT_ENVS

    @staticmethod
    def getCoondaRoot(env=None):
        """ Tries to find the conda root path given an environment
            :param: env. An environ obtaining using Plugin.getEnviron()
            :return: None if conda not found or CONDA_ROOT_PATH (could be defined into config file??)
        """
        if env == None:
            env = {}
        if "CONDA_HOME" in env:  # TODO. Allow CONDA_HOME to be in config file
            condaRoot = "CONDA_HOME"
        if "CONDA_ACTIVATION_CMD" in env:
            condaRoot = os.path.split(os.path.split(os.path.split(env["CONDA_ACTIVATION_CMD"])[0])[0])[0]
            condaRoot = re.sub("^\.", "", condaRoot).strip()
        else:
            if "CONDA_EXE" in env:
                condaRoot = env["CONDA_EXE"]
                success = True
            else:
                try:
                    condaRoot = subprocess.check_output(["which", "conda"]).decode('utf-8')
                    success = True
                except subprocess.CalledProcessError:
                    success = False

            if success:
                condaRoot = os.path.split(os.path.split(condaRoot)[0])[0]

        assert condaRoot is not None, "Error, conda was not found"+str(env)
        return os.path.expanduser(condaRoot)

    @staticmethod
    def getCondaPathInEnv(condaRoot, condaEnv, condaSubDir=""):
        """ :param condaRoot: The path where conda is installed. E.g.  ~/tools/miniconda3
            :param condaEnv: The name (prefix) of the conda environment that will be used
            :param condaSubDir: The path of the subdirectory within conda env. E.g. condaSubDir="bin -> path/to/env/bin
            :return: the path to a subdirectory within condaEnv
        """
        return os.path.join(condaRoot, "envs", condaEnv, condaSubDir)

    @staticmethod
    def modifyEnvToUseConda(env, condaEnv):
        env.update({"PATH": CondaEnvManager.getCondaPathInEnv(CondaEnvManager.getCoondaRoot(env),
                                                              condaEnv, "bin")+":"+env["PATH"]})

        newPythonPath= CondaEnvManager.getCondaPathInEnv(CondaEnvManager.getCoondaRoot(env),
                                                         condaEnv, "lib/python*/site-packages/")
        if CondaEnvManager.DICT_OF_CONDA_ENVIRONS[condaEnv]["xmippEnviron"]:
            newPythonPath+=":"+env["PYTHONPATH"]
        env.update({"PYTHONPATH": newPythonPath})
#        print(env["PYTHONPATH"])
        return env

    @staticmethod
    def getCondaActivationCmd():
        condaActivationCmd = os.environ.get('CONDA_ACTIVATION_CMD', "")
        if not condaActivationCmd:
            condaRoot = CondaEnvManager.getCoondaRoot()
            condaActivationCmd = ". "+os.path.join(condaRoot, "etc/profile.d/conda.sh")
        condaActivationCmd = condaActivationCmd.strip()
        if condaActivationCmd[-1] == ";":
            condaActivationCmd = condaActivationCmd[:-1]+" &&  "
        elif condaActivationCmd.endswith("&&"):
            pass  # condaActivationCmd is already well ended
        elif condaActivationCmd.endswith("&"):
            condaActivationCmd += "&"
        else:
            condaActivationCmd += " && "
        return condaActivationCmd

    @staticmethod
    def yieldInstallAllCmds( useGpu):
        installCmdOptions={}
        if useGpu:
            installCmdOptions["gpuTag"]="-gpu"
        else:
            installCmdOptions["gpuTag"]=""

        for envName in CondaEnvManager.DICT_OF_CONDA_ENVIRONS:
            yield CondaEnvManager.installEnvironCmd(envName, installCmdOptions=installCmdOptions,
                                                    **CondaEnvManager.DICT_OF_CONDA_ENVIRONS[envName])

    @staticmethod
    def installEnvironCmd( environName, pythonVersion, dependencies, channels,
                           defaultInstallOptions, pipPackages=[], installCmdOptions=None,
                           **kwargs):

        cmd="export PYTHONPATH=\"\" && conda create -q --force --yes -n "+environName+" python="+pythonVersion+" "
        cmd += " "+ " ".join([dep for dep in dependencies])
        if len(channels)>0:
            cmd += " -c "+ " -c ".join([chan for chan in channels])
        if installCmdOptions is not None:
            try:
                cmd= cmd%installCmdOptions
            except KeyError:
                pass
        else:
            try:
                cmd= cmd%defaultInstallOptions
            except KeyError:
                pass
        cmd += " && " + CondaEnvManager.getCondaActivationCmd() + " conda activate " + environName
        # cmd += " && export PATH="+ os.path.join(CondaEnvManager.getCoondaRoot(), "bin")+":$PATH &&  conda activate "+environName

        if len(pipPackages)>0:
            cmd += " && pip install  "
            cmd += " " + " ".join([dep for dep in pipPackages])
        cmd += " && conda env export > "+environName+".yml"
        return cmd, environName


def createMetaDataFromPattern(pattern, isStack=False, label="image"):
    ''' Create a metadata from files matching pattern'''
    import glob
    if isinstance(pattern, list):
        files = []
        for pat in pattern:
            files += glob.glob(pat)
    else:
        files = glob.glob(pattern)
    files.sort()

    label = str2Label(label) #Check for label value
    
    mD = MetaData()
    inFile = FileName()
    
    nSize = 1
    for file in files:
        fileAux=file
        if isStack:
            if file.endswith(".mrc"):
                fileAux=file+":mrcs"
            x, x, x, nSize = getImageSize(fileAux)
        if nSize != 1:
            counter = 1
            for jj in range(nSize):
                inFile.compose(counter, fileAux)
                objId = mD.addObject()
                mD.setValue(label, inFile, objId)
                mD.setValue(MDL_ENABLED, 1, objId)
                counter += 1
        else:
            objId = mD.addObject()
            mD.setValue(label, fileAux, objId)
            mD.setValue(MDL_ENABLED, 1, objId)
    return mD


