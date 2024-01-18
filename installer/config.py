# ***************************************************************************
# * Authors:		Alberto García (alberto.garcia@cnb.csic.es)
# *					Martín Salinas (martin.salinas@cnb.csic.es)
# *
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307 USA
# *
# * All comments concerning this program package may be sent to the
# * e-mail address 'scipion@cnb.csic.es'
# ***************************************************************************/

"""
This module contains the necessary functions to run the config command.
"""

import shutil, glob, sys
from os import path, environ
from os.path import isdir, join, isfile

from .constants import (SCONS_MINIMUM, CONFIG_FILE, GCC_MINIMUM,
                        GPP_MINIMUM, MPI_MINIMUM, PYTHON_MINIMUM, NUMPY_MINIMUM,
                        CXX_FLAGS, PATH_TO_FIND, INC_PATH, INC_HDF5_PATH,
                        CONFIG_DICT, CXX_FLAGS,INTERNAL_FLAGS,
                        OK, UNKOW_ERROR, SCONS_VERSION_ERROR, SCONS_ERROR,
                        GCC_VERSION_ERROR, CC_NO_EXIST_ERROR, CXX_NO_EXIST_ERROR, CXX_VERSION_ERROR,
                        MPI_VERSION_ERROR, MPI_NOT_FOUND_ERROR, PYTHON_VERSION_ERROR ,
                        PYTHON_NOT_FOUND_ERROR , NUMPY_NOT_FOUND_ERROR ,NVCC_CXXFLAGS_ERROR,
                        JAVA_HOME_PATH_ERROR, MATLAB_WARNING , MATLAB_HOME_WARNING,
                        CUDA_VERSION_WARNING , CUDA_WARNING , HDF5_ERROR, LINKFLAGS,
                        MPI_COMPILLATION_ERROR, MPI_RUNNING_ERROR, OPENCV_WARNING,
                        JAVAC_DOESNT_WORK_ERROR, JAVA_INCLUDE_ERROR, CMAKE_MINIMUM,
                        CMAKE_VERSION_ERROR, CMAKE_ERROR, cmakeInstallURL, SCONS_MINIMUM,
                        CC, CXX, MPI_CC, MPI_CXX, MPI_RUN, OPENCV_CUDA_WARNING,
                        STARPU_INCLUDE_WARNING, STARPU_LIB_WARNING, STARPU_LIBRARY_WARNING,
                        STARPU_RUN_WARNING, STARPU_CUDA_WARNING, HDF5_MINIMUM,
                        HDF5_VERSION_ERROR, TIFF_ERROR, FFTW3_ERROR, PATH_TO_FIND_H,
                        TIFF_H_ERROR, FFTW3_H_ERROR, FFTW_MINIMUM, FFTW3_VERSION_ERROR,
                        WARNING_CODE, GIT_MINIMUM, GIT_VERSION_ERROR, PYTHONINCFLAGS_ERROR,
                        RSYNC_MINIMUM, RSYNC_VERSION_ERROR, HDF5_NOT_FOUND_ERROR)
from .utils import (red, green, yellow, blue, runJob, existPackage,
                    getPackageVersionCmd,JAVAVersion, printWarning,
                    whereIsPackage, findFileInDirList, getINCDIRFLAG,
                    getCompatibleGCC, CXXVersion, gitVersion,
                    get_Hdf5_name, printError, MPIVersion, installScons, versionToNumber,
                    HDF5Version, opencvVersion, TIFFVersion, printMessage, FFTW3Version,
                    updateXmippEnv)

from .versions import (getOSReleaseName, getArchitectureName, getCUDAVersion,
                                getCmakeVersion, getGPPVersion, getGCCVersion,
                       getSconsVersion, getRsyncVersion)
from .versions import getCUDAVersion
from datetime import datetime
from sysconfig import get_paths


def config(debugP:bool=True):
    global debugPrints
    debugPrints = debugP
    """check the config if exist else create it and check it"""
    # printMessage('LD_LIBRARY_PATH: ', debug=debugPrints)
    # runJob('echo $LD_LIBRARY_PATH', showOutput=True)
    if not existConfig():
        printMessage(text='Generating config file xmipp.conf', debug=True)
        dictPackages = getSystemValues()
        dictInternalFlags = getInternalFlags(dictPackages)
        writeConfig(dictPackages, dictInternalFlags)
        return dictPackages
    else:
        dictPackages, dictInternalFlags = readConfig()
    dictNoChecked = dictPackages.copy()
    checkConfig(dictPackages, dictInternalFlags)
    dictInternalFlags2 = getInternalFlags(dictPackages)#if checkConfig change any parameter...
    if dictPackages != dictNoChecked or dictInternalFlags != dictInternalFlags2:
        writeConfig(dictP=dictPackages, dictInt=dictInternalFlags2)

    # printMessage('LD_LIBRARY_PATH: ', debug=debugPrints)
    # runJob('echo $LD_LIBRARY_PATH', showOutput=True)
    return dictPackages


def getSystemValues():
    """
    Retrieves system information related to various packages and configurations.

    Returns:
    - dict: Dictionary containing system package information.
    """
    printMessage(text='- Getting system libraries...', debug=True)
    dictPackages = {'INCDIRFLAGS': '-I../ ',
                    'LIBDIRFLAGS': ''}
    getCC(dictPackages)
    getCXX(dictPackages)
    getMPI(dictPackages)
    getJava(dictPackages)
    getTIFF(dictPackages)
    getFFTW3(dictPackages)
    getHDF5(dictPackages)
    getINCDIRFLAGS(dictPackages)
    getLIBDIRFLAGS(dictPackages)
    getOPENCV(dictPackages)
    getCUDA(dictPackages)
    getSTARPU(dictPackages)
    getMatlab(dictPackages)
    printMessage(text=green('Done'), debug=True)

    return dictPackages

def getInternalFlags(dictPackages, debug: bool=False):
    printMessage(text='\n- Getting internal flags for config file...', debug=True)
    dictInternalFlags = INTERNAL_FLAGS
    #CCFLAGS
    dictInternalFlags['CCFLAGS'] = '-std=c99'
    #CCXXFLAGS
    dictInternalFlags['CXXFLAGS'] = CXX_FLAGS
    if debug:
        dictInternalFlags['CXXFLAGS'] += ' -O0 -g'
    else:
        dictInternalFlags['CXXFLAGS'] += ' -O3 -g'
    #PYTHONINCFLAGS
    try:
        from sysconfig import get_paths
        import numpy
        info = get_paths()
        incDirs = [info['include'], numpy.get_include()]
        dictInternalFlags['PYTHONINCFLAGS'] = ' '.join(["-I%s" % iDir for iDir in incDirs])
    except Exception as e:
        printError(errorMsg=str(e), retCode=PYTHONINCFLAGS_ERROR)
    #PYTHON_LIB
    malloc = "m" if sys.version_info.minor < 8 else ""
    dictInternalFlags['PYTHON_LIB'] = 'python%s.%s%s' % (sys.version_info.major, sys.version_info.minor, malloc)
    #LINKERFORPROGRAMS
    dictInternalFlags['LINKERFORPROGRAMS'] = dictPackages['CXX']
    #MPI_LINKERFORPROGRAMS
    dictInternalFlags['MPI_LINKERFORPROGRAMS'] = dictPackages['MPI_CXX']
    # NVCC_CXXFLAGS
    if dictPackages['CUDA'] == 'True':
        try:
            if versionToNumber(getCUDAVersion(dictPackages)) < versionToNumber('11.0'):
                dictInternalFlags['NVCC_CXXFLAGS'] = \
                         "--x cu -D_FORCE_INLINES -Xcompiler -fPIC "\
                         "-ccbin {} -std=c++11 --expt-extended-lambda "\
                         "-gencode=arch=compute_35,code=compute_35 "\
                         "-gencode=arch=compute_50,code=compute_50 "\
                         "-gencode=arch=compute_60,code=compute_60 "\
                         "-gencode=arch=compute_61,code=compute_61".format(dictPackages['CUDACXX'])
            else:
                dictInternalFlags['NVCC_CXXFLAGS'] = \
                         "--x cu -D_FORCE_INLINES -Xcompiler -fPIC "\
                         "-ccbin {} -std=c++14 --expt-extended-lambda "\
                         "-gencode=arch=compute_60,code=compute_60 "\
                         "-gencode=arch=compute_61,code=compute_61 "\
                         "-gencode=arch=compute_75,code=compute_75 "\
                         "-gencode=arch=compute_86,code=compute_86".format(dictPackages['CUDACXX'])
        except Exception as e:
            printError(errorMsg=str(e), retCode=NVCC_CXXFLAGS_ERROR)
    # LINKFLAGS_NVCC
    dictHomeCUDA= dictPackages['CUDA_HOME'].split('bin/nvcc')[0]
    paths = [join(dictHomeCUDA, 'lib'),
             join(dictHomeCUDA, 'lib64')]
    for route in paths:
        if isfile(join(route, 'libcudart.so')):
            LINKFLAGS_NVCC = '-L{}'.format(route)
            updateXmippEnv(LD_LIBRARY_PATH=route)
            stubroute = join(route, 'stubs')
            if path.exists(stubroute):
                LINKFLAGS_NVCC += ' -L{}'.format(stubroute)
                updateXmippEnv(LD_LIBRARY_PATH=stubroute)
    dictInternalFlags['LINKFLAGS_NVCC'] = LINKFLAGS_NVCC
    #JAVAS
    dictInternalFlags['JAVA_BINDIR'] = join(dictPackages['JAVA_HOME'], 'bin')
    dictInternalFlags['JAVAC'] = join(dictInternalFlags['JAVA_BINDIR'], 'javac')
    dictInternalFlags['JAR'] = join(dictInternalFlags['JAVA_BINDIR'], 'jar')
    dictInternalFlags['JNI_CPPPATH'] = (join(dictPackages['JAVA_HOME'], 'include') +
                        ':' + join(dictPackages['JAVA_HOME'], 'include/linux'))

    dictInternalFlags['LINKFLAGS'] = LINKFLAGS

    printMessage(text=green('Done'), debug=True)

    return dictInternalFlags


def readConfig():
    """Check if valid all the flags of the config file"""
    dictPackages = {}
    internalFlags = {}

    with open(CONFIG_FILE, 'r') as f:
        config = f.read()
    for key, _ in CONFIG_DICT.items():
        idx = config.find(key+'=')
        idx2 = config[idx:].find('=') + 1
        value = config[idx+idx2:idx + idx2 + config[idx+idx2:].find('\n')]
        dictPackages[key] = value
    for key, _ in INTERNAL_FLAGS.items():
        idx = config.find(key+'=')
        idx2 = config[idx:].find('=') + 1
        value = config[idx+idx2:idx + idx2 + config[idx+idx2:].find('\n')]
        internalFlags[key] = value
    return dictPackages, internalFlags

def checkConfig(dictPackages, dictInternalFlags):
    """
    Checks the configurations of various packages.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    """
    checkPackagesStatus = []
    printMessage(text='\n- Checking libraries from config file...', debug=True)
    checkCC(dictPackages)
    checkCXX(dictPackages)
    checkMPI(dictPackages, dictInternalFlags)
    checkJava(dictPackages)
    if dictPackages['MATLAB'] == 'True':
        checkMatlab(dictPackages, checkPackagesStatus)
    if dictPackages['OPENCV'] == 'True':
        checkOPENCV(dictPackages, checkPackagesStatus)
    if dictPackages['CUDA'] == 'True':
        checkCUDA(dictPackages, checkPackagesStatus)
    if dictPackages['STARPU'] == 'True':
        checkSTARPU(dictPackages, checkPackagesStatus)
    checkGit()
    checkHDF5(dictPackages)
    checkTIFF(dictPackages)
    checkFFTW3(dictPackages)
    checkScons()
    checkCMake()
    checkRsync()

    if checkPackagesStatus != []:
        for pack in checkPackagesStatus:
            printWarning(text=pack[0], warningCode=pack[0], debug=True)

    printMessage(text=green('Done'), debug=True)


def existConfig():
    """ Checks if the config file exist.Return True or False """
    if path.exists(CONFIG_FILE):
        return True
    else:
        return False

def writeConfig(dictP: dict, dictInt: dict):
    """Write the config file"""
    printMessage(text='\n- Writting config file...', debug=True)

    with open(CONFIG_FILE, 'w') as f:
        f.write('[USER FLAGS]\n')
        for key, value in dictP.items():
            f.write('{}={}\n'.format(key, value))
        f.write('\n\n[INTERNAL FLAGS]\n')
        for key, value in dictInt.items():
            f.write('{}={}\n'.format(key, value))
        f.write('\n\n[DATE]\n')
        f.write('Config file written: {} \n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    printMessage(text=green('Done'), debug=True)

def parseConfig():
    """Read and save on configDic all flags of config file"""
    dictPackages = {}
    with open(CONFIG_FILE, 'r'):
        pass
    return dictPackages


#PACKAGES
def getCC(dictPackages):
    """
    Retrieves information about the CC (GCC) package and updates the dictionary accordingly.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    Modifies:
    - dictPackages: Updates the 'CC' key based on the availability of 'gcc'.
    """
    if existPackage('gcc'):
        dictPackages['CC'] = 'gcc'
    else:
        dictPackages['CC'] = ''

def checkCC(dictPackages):
    """
    Checks the GCC (CC) package at the specified path for version compatibility.

    Params:
    - packagePath (str): Path to the GCC (CC) package directory.

    Returns:
    - int: Error code.
        - 1: Success.
        - 4: gcc version is lower than the required version.
        - 5: GCC package path does not exist.
    """
    if existPackage(dictPackages[CC]):
        version = getGCCVersion(dictPackages)
        if versionToNumber(version) >= versionToNumber(GCC_MINIMUM):
            printMessage(text=green('gcc {} found'.format(version)), debug=debugPrints)
            return OK
        printError(retCode=GCC_VERSION_ERROR, errorMsg='gcc {} lower than required ({})'.format(version, GCC_MINIMUM))
    else:
        printError(retCode=CC_NO_EXIST_ERROR, errorMsg='GCC package path: {} does not exist'.format(dictPackages[CC]))

def getCXX(dictPackages):
    """
    Retrieves information about the CXX package and updates the dictionary accordingly.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    Modifies:
    - dictPackages: Updates the 'CXX' key based on the availability of 'g++'.
    """
    if existPackage('g++'):
        dictPackages['CXX'] = 'g++'
    else:
        dictPackages['CXX'] = ''

def checkCXX(dictPackages):
    """
    Checks the CXX package at the specified path for version compatibility.

    Params:
    - packagePath (str): Path to the CXX package directory.

    Returns:
    - int: Error code.
        - 1: Success.
        - 6: CXX package path does not exist.
        - 7: g++ version is lower than the required version.
    """
    if existPackage(dictPackages['CXX']):
        version = getGPPVersion(dictPackages)
        if versionToNumber(version) >= versionToNumber(GCC_MINIMUM):
            printMessage(text=green('g++ {} found'.format(version)), debug=debugPrints)
            return OK
        printError(retCode=CXX_VERSION_ERROR, errorMsg='g++ {} lower than required ({})'.format(version, GPP_MINIMUM))
    else:
        printError(retCode=CXX_NO_EXIST_ERROR, errorMsg='CXX package path: {} does not exist'.format(dictPackages[CXX]))

def getMPI(dictPackages):
    """
    Retrieves information about the MPI package components and updates the dictionary accordingly.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    Modifies:
    - dictPackages: Updates keys 'MPI_CC', 'MPI_CXX', and 'MPI_RUN' based on MPI component availability.
    """
    if existPackage('mpicc'):
        dictPackages['MPI_CC'] = 'mpicc'
    else:
        dictPackages['MPI_CC'] = ''
    if existPackage('mpicxx'):
        dictPackages['MPI_CXX'] = 'mpicxx'
    else:
        dictPackages['MPI_CXX'] = ''
    if existPackage('mpirun'):
        dictPackages['MPI_RUN'] = 'mpirun'
        updateXmippEnv(PATH=whereIsPackage('mpirun'))
    else:
        dictPackages['MPI_RUN'] = ''

def checkMPI(dictPackages, dictInternalFlags):
    """
    Checks the MPI packages for compatibility and performs additional checks.

    Params:
    - dictPackages (dict): Dictionary containing MPI package information.

    Returns:
    - int: Error code.
        - 1: Success.
        - 6: MPI version lower than required.
        - 7: MPI package not found.
        - 8: Error during compilation.
        - 9: Error while running MPI jobs.
    """
    for pack in [MPI_CC, MPI_RUN, MPI_CXX]:
        if existPackage(dictPackages[pack]):
            if pack == MPI_RUN:
                strVersion = getPackageVersionCmd(dictPackages[pack])
                version = MPIVersion(strVersion)
                if versionToNumber(version) >= versionToNumber(MPI_MINIMUM):
                    printMessage(text=green('{} {} found'.format(pack, version)), debug=debugPrints)
                else:
                    printError(retCode=MPI_VERSION_ERROR, errorMsg='mpi {} lower than required ({})'.format(version, GPP_MINIMUM))
        else:
            printError(retCode=MPI_NOT_FOUND_ERROR, errorMsg='MPI package: {} does not exist'.format(dictPackages[pack]))

    #More checks

    cppProg = """
    #include <mpi.h>
    int main(){}
    """
    with open("xmipp_mpi_test_main.cpp", "w") as cppFile:
        cppFile.write(cppProg)
    cmd = ("%s -c -w %s %s xmipp_mpi_test_main.cpp -o xmipp_mpi_test_main.o"
           % (dictPackages["MPI_CXX"], dictPackages["LIBDIRFLAGS"],CXX_FLAGS))
    status, output = runJob(cmd)
    if status != 0:
        printError(retCode=MPI_RUNNING_ERROR, errorMsg='Fails running the command: \n{}\nError message: {}'.format(cmd, output))

    libhdf5 = get_Hdf5_name(dictPackages["LIBDIRFLAGS"])
    cmd = (("%s %s %s xmipp_mpi_test_main.o -o xmipp_mpi_test_main -lfftw3"
           " -lfftw3_threads -l%s  -lhdf5_cpp -ltiff -ljpeg -lsqlite3 -lpthread ")
           % (dictInternalFlags["MPI_LINKERFORPROGRAMS"],
              dictInternalFlags["LINKFLAGS"],
              dictPackages["LIBDIRFLAGS"],
              libhdf5
              ))

    status, output = runJob(cmd)
    if status != 0:
        printError(retCode=MPI_COMPILLATION_ERROR, errorMsg='Fails running the command: \n{}\n\nError message:\n{}'.format(cmd, output))

    runJob("rm xmipp_mpi_test_main*", showOutput=False,showCommand=False)

    processors = 2
    output = runJob('{} -np {} echo {}'.format(dictPackages['MPI_RUN'], processors, 'Running'))[1]
    if output.count('Running') != processors:
        output = runJob('{} -np 2 --allow-run-as-root echo {}'.format(dictPackages['MPI_RUN'], processors,  'Running'))[1]
        if output.count('Running') != processors:
            printError(retCode=MPI_RUNNING_ERROR,  errorMsg='mpirun or mpiexec have failed.')

def getJava(dictPackages):
    """
    Retrieves information about the Java package and updates the dictionary accordingly.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    Modifies:
    - dictPackages: Updates the 'JAVA_HOME' key based on the Java installation path.
    """
    javaProgramPath = whereIsPackage('javac')
    if not javaProgramPath:
        javaProgramPath = findFileInDirList('javac', ['/usr/lib/jvm/java-*/bin'])
    if javaProgramPath:
        javaHomeDir = javaProgramPath.replace("/jre/bin", "")
        javaHomeDir = javaHomeDir.replace("/bin", "")
        dictPackages['JAVA_HOME'] = javaHomeDir
        updateXmippEnv(PATH=javaProgramPath)
    else:
        dictPackages['JAVA_HOME'] = ''

def checkJava(dictPackages):
    """
    Checks the Java installation and configuration.

    Params:
    - dictPackages (dict): Dictionary containing Java package information.

    Returns:
    - int: Error code.
        - 1: Success.
    """
    if isfile(join(dictPackages['JAVA_HOME'], 'bin/jar')) and \
            whereIsPackage(join(dictPackages['JAVA_HOME'], 'bin/javac')) and\
            isdir(join(dictPackages['JAVA_HOME'], 'include')) and existPackage('java'):
        printMessage(text=green('java {} found'.format(JAVAVersion(getPackageVersionCmd('java')))), debug=debugPrints)
    else:
        printError('JAVA_HOME path: {} does not work'.format(dictPackages['JAVA_HOME']), JAVA_HOME_PATH_ERROR)

    #JAVA Version
    version = JAVAVersion(getPackageVersionCmd('java'))
    #Other check
    javaProg = """
        public class Xmipp {
        public static void main(String[] args) {}
        }
    """
    with open("Xmipp.java", "w") as javaFile:
        javaFile.write(javaProg)
    cmd= "%s Xmipp.java" % join(dictPackages['JAVA_HOME'], 'bin/javac')
    retCode, outputStr = runJob(cmd, showError=True)
    if retCode!= 0:
        printError(retCode=JAVAC_DOESNT_WORK_ERROR, errorMsg=cmd)
    runJob("rm Xmipp.java Xmipp.class", showError=True)

    #Other check 2
    if isdir(join(dictPackages['JAVA_HOME'], 'include')):
        incJ = join(dictPackages['JAVA_HOME'], 'include')
    if isdir(join(dictPackages['JAVA_HOME'], 'include', 'linux')):
        if incJ != '':
            incJ += ':' + join(dictPackages['JAVA_HOME'], 'include', 'linux')
        else:
            incJ = join(dictPackages['JAVA_HOME'], 'include', 'linux')
    cppProg = """
        #include <jni.h>
        int dummy(){return 0;}
        """
    with open("xmipp_jni_test.cpp", "w") as cppFile:
        cppFile.write(cppProg)
    incs = ""
    for x in incJ.split(':'):
        incs += " -I"+x
    cmd = "%s -c -w %s %s xmipp_jni_test.cpp -o xmipp_jni_test.o" %(dictPackages['CXX'], incs, dictPackages["INCDIRFLAGS"])
    status, output = runJob(cmd)
    if status != 0:
        printError(retCode=JAVA_INCLUDE_ERROR, errorMsg=output)
    runJob("rm xmipp_jni_test*", showError=True)

def getMatlab(dictPackages):
    """
    Retrieves information about the MATLAB package and updates the dictionary accordingly.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    Modifies:
    - dictPackages: Updates keys 'MATLAB' and 'MATLAB_HOME' based on MATLAB availability.
    """
    matlabProgramPath = whereIsPackage('matlab')
    if matlabProgramPath:
        dictPackages['MATLAB'] = True
        dictPackages['MATLAB_HOME'] = matlabProgramPath.replace("/bin", "")
        printMessage(text=green('MATLAB_HOME detected at {}'.format(dictPackages['MATLAB_HOME'])), debug=debugPrints)
        updateXmippEnv(MATLAB_BIN_DIR=matlabProgramPath)
    else:
        dictPackages['MATLAB'] = False
        dictPackages['MATLAB_HOME'] = ''

def checkMatlab(dictPackages, checkErrors):
    """
    Checks for the existence of MATLAB package and verifies if a specified path is a directory.

    Params:
    - packagePath (str): Path to the package directory.

    Returns:
    - int: Error code.
        - 15: MATLAB package does not exist.
        - 16: Specified path is not a directory.
        - 1: Success.
        - 10: JAVA_HOME path is not configured correctly.
        - 11: 'javac' compiler error.
        - 12: Error in including Java libraries.
    """
    if not isdir(dictPackages['MATLAB_HOME']):
        checkErrors.append([MATLAB_HOME_WARNING, 'MATLAB_HOME={} does not exist'.format(dictPackages['MATLAB_HOME'])])

    cppProg = """
    #include <mex.h>
    int dummy(){return 0;}
    """
    with open("xmipp_mex.cpp", "w") as cppFile:
        cppFile.write(cppProg)

    cmd = " {} -silent xmipp_mex.cpp".format(join(dictPackages["MATLAB_HOME"], 'bin', 'mex'))
    status, output = runJob(cmd, showError=True)
    if status != 0:
        checkErrors.append([MATLAB_HOME_WARNING, output])
        dictPackages['MATLAB'] = 'False'
    else:
        printMessage(text=green('Matlab installation found'), debug=debugPrints)
    runJob("rm xmipp_mex*")


def getOPENCV(dictPackages):
    opencvPath = ['opencv2', 'opencv4/opencv2']
    filePath = ['core.hpp', 'core/core.hpp']
    for p in INC_PATH:
        for oP in opencvPath:
            for f in filePath:
                if path.isfile(join(p, oP, f)):
                    try:
                        dictPackages['INCDIRFLAGS'] += ' -I' + p + '/' + oP.split('/')[0]
                    except KeyError:
                        dictPackages['INCDIRFLAGS'] = ' -I' + p + '/' + oP.split('/')[0]
                    dictPackages['OPENCV'] = True
                    dictPackages['OPENCVCUDASUPPORTS'] = True
                    break

def checkOPENCV(dictPackages, checkErrors):
    """
    Checks OpenCV installation, version, and CUDA support.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    Returns:
    - int: Error code.
        - 1: Success.
        - [Potential custom error codes based on specific checks]
    """
    cppProg = "#include <opencv2/core/core.hpp>\n"
    cppProg += "int main(){}\n"
    with open("xmipp_test_opencv.cpp", "w") as cppFile:
        cppFile.write(cppProg)

    status, output = runJob("%s -c -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv.o %s" % (dictPackages['CXX'], CXX_FLAGS, dictPackages['INCDIRFLAGS']), showError=True)
    if status != 0:
        checkErrors.append([OPENCV_WARNING, 'OpenCV set as True but {}'.format(output)])
        dictPackages['OPENCV'] = ''
    else:
        printMessage(text=green('OPENCV {} found'.format(opencvVersion(dictPackages, CXX_FLAGS))), debug=debugPrints)

    # Check CUDA Support
    if dictPackages['OPENCVCUDASUPPORTS'] == 'True':
        cppProg = "#include <opencv2/core/version.hpp>\n"
        if opencvVersion(dictPackages, CXX_FLAGS) < 3:
            cppProg += "#include <opencv2/core/cuda.hpp>\n"
        else:
            cppProg += "#include <opencv2/cudaoptflow.hpp>\n"
        cppProg += "int main(){}\n"
        with open("xmipp_test_opencv.cpp", "w") as cppFile:
            cppFile.write(cppProg)
        status, output = runJob("%s -c -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv.o %s" % (dictPackages['CXX'], CXX_FLAGS, dictPackages['INCDIRFLAGS']))
        if status != 0:
            checkErrors.append([OPENCV_CUDA_WARNING, 'OpenCV CUDA suport set as True but is not ready on your computer'])
            dictPackages['OPENCVCUDASUPPORTS'] = False

    runJob("rm xmipp_test_opencv*", showError=True)

def getCUDA(dictPackages):
    """
     Retrieves information about the CUDA package and updates the dictionary accordingly.

     Params:
     - dictPackages (dict): Dictionary containing package information.

     Modifies:
     - dictPackages: Updates keys 'CUDA', 'CUDA_HOME', and 'CUDACXX' based on CUDA package availability.
     """
    if not existPackage('nvcc'):
        dictPackages['CUDA'] = 'False'
        dictPackages['CUDA_HOME'] = ''
        dictPackages['CUDACXX'] = ''
        updateXmippEnv(CUDA=False)
    else:
        dictPackages['CUDA'] = 'True'
        dictPackages['CUDA_HOME'] = shutil.which('nvcc')
        dictPackages['CUDACXX'] = dictPackages['CXX']
        updateXmippEnv(CUDA=True)

def checkCUDA(dictPackages, checkPackagesStatus):
    """
    Checks the compatibility of CUDA with the current g++ compiler version and updates the dictionary accordingly.

    Params:
    - dictPackages (dict): Dictionary containing OpenCV package information.

    Returns:
    - int: Error code.
        - 1: Success.
        - 17: CUDA not compatible with the current g++ compiler version.
        - 18: CUDA version information not available.
    """

    nvcc_version = getCUDAVersion(dictPackages)
    if nvcc_version != 'Unknow' or nvcc_version != None:
        gppVersion = getGPPVersion(dictPackages)
        gxx_version = '.'.join(gppVersion.split('.')[:2])
        candidates, resultBool = getCompatibleGCC(nvcc_version)
        if resultBool == True and gxx_version in candidates:
            printMessage(text=green(
                'CUDA {} found'.format(nvcc_version)), debug=debugPrints)
        else:
            checkPackagesStatus.append([CUDA_VERSION_WARNING, 'CUDA {} not compatible with the current g++ compiler version {}\n'
                      'Compilers candidates for your CUDA: {}'.format(
                nvcc_version, gxx_version, candidates)])
            dictPackages['CUDA'] = 'False'
            updateXmippEnv(CUDA=False)
    else:
        checkPackagesStatus.append([CUDA_VERSION_WARNING, 'CUDA version not found{}\n'])
        dictPackages['CUDA'] = 'False'
        updateXmippEnv(CUDA=False)

def getSTARPU(dictPackages):
    """
    Retrieves information about the STARPU package and updates the dictionary accordingly.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    Modifies:
    - dictPackages: Updates keys related to STARPU package information.
    """
    if whereIsPackage("starpu_sched_display"):
        dictPackages["STARPU"] = 'True'
        starpuBinDir = whereIsPackage("starpu_sched_display")
        dictPackages["STARPU_HOME"] = starpuBinDir.replace("/bin", "")
        dictPackages["STARPU_INCLUDE"] = "%(STARPU_HOME)s/include/starpu/1.3"
        dictPackages["STARPU_LIB"] = "%(STARPU_HOME)s/lib"
        dictPackages["STARPU_LIBRARY"] = "libstarpu-1.3"
        printMessage(text=green('STARPU detected at {}'.format(dictPackages['STARPU_HOME'])), debug=debugPrints)

    else:
        dictPackages["STARPU"] = False
        dictPackages["STARPU_HOME"] = ''
        dictPackages["STARPU_INCLUDE"] = ''
        dictPackages["STARPU_LIB"] = ''
        dictPackages["STARPU_LIBRARY"] = ''

def checkSTARPU(dictPackages, checkPackagesStatus):
    """
    Checks the configuration of the STARPU package and CUDA compatibility, printing error messages if necessary.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    Returns:
    - int: Error code.
        - 1: Success.
    """
    #TODO check behaviour in a system with starpu installed
    if dictPackages["CUDA"] != "True":
        ans = False
        checkPackagesStatus.append([STARPU_CUDA_WARNING, ''])
    if dictPackages["STARPU_INCLUDE"] == "" or not isdir(
            dictPackages["STARPU_INCLUDE"]):
        ans = False
        checkPackagesStatus.append([STARPU_INCLUDE_WARNING, "Check the STARPU_INCLUDE directory: " +
                  dictPackages["STARPU_INCLUDE"]])
    if dictPackages["STARPU_LIB"] == "" or not isdir(
            dictPackages["STARPU_LIB"]):
        ans = False
        checkPackagesStatus.append([STARPU_LIB_WARNING, "Check the STARPU_LIB directory: " +
                  dictPackages["STARPU_LIB"]])
    if dictPackages["STARPU_LIBRARY"] == "":
        ans = False
        checkPackagesStatus.append([STARPU_LIBRARY_WARNING])
    if ans:
        with open("xmipp_starpu_config_test.cpp", "w") as cppFile:
            cppFile.write("""
            #include <starpu.h>
            int dummy(){return 0;}
            """)

        if runJob(
                "%s -c -w %s %s -I%s -L%s -l%s xmipp_starpu_config_test.cpp -o xmipp_starpu_config_test.o" %
                (dictPackages["NVCC"], dictPackages["NVCC_CXXFLAGS"],
                 dictPackages["INCDIRFLAGS"],
                 dictPackages["STARPU_INCLUDE"],
                 dictPackages["STARPU_LIB"],
                 dictPackages["STARPU_LIBRARY"]))[0] != 0:
            checkPackagesStatus.append([STARPU_LIBRARY_WARNING])
    else:
        dictPackages['STARPU'] = 'False'
    runJob("rm -f xmipp_starpu_config_test*")

# def checkPYTHONINCFLAGS(incPath):
#     includes = incPath.split(' ')
#     pythonPath = includes[0].replace('-I', '')
#     numpyPath = includes[1].replace('-I', '')
#     if existPackage(pythonPath):
#         strVersion = getPackageVersionCmd(pythonPath)
#         idx = strVersion.find('\n')
#         idx2 = strVersion[idx].rfind(' ')
#         version = strVersion[idx - idx2:idx]
#         if versionToNumber(version) < versionToNumber(PYTHON_MINIMUM):
#             printMessage(text=red('python {} lower than required ({})'.format(version,
#                                                                PYTHON_MINIMUM)), debug=debugPrints)
#             return 10
#
#     #NUMPY
#     import sys
#     sys.path.append('/path/to/directory')
#     if existPackage(numpyPath):
#         strVersion = getPackageVersionCmd(pythonPath)
#         idx = strVersion.find('\n')
#         idx2 = strVersion[idx].rfind(' ')
#         version = strVersion[idx - idx2:idx]
#         if versionToNumber(version) < versionToNumber(PYTHON_MINIMUM):
#             prinprintMessaget(text=red('python {} lower than required ({})'.format(version,
#                                                                PYTHON_MINIMUM)), debug=True)
#             return 10
#
def getHDF5(dictPackages):
    """
    This function searches for HDF5 library ('libhdf5*') in specified directories.
    If found, updates 'LIBDIRFLAGS' in 'dictPackages' with the HDF5 library path.
    If not found, prints a message indicating HDF5 is not detected.
    Updates 'LIBDIRFLAGS' in 'dictPackages' based on HDF5 library detection.

    Params:
    - dictPackages (dict): Dictionary with package information.
        Expected keys: 'LIBDIRFLAGS'.
    """
    #get hdf5 libdir
    PATH_TO_FIND.append(join(get_paths()['data'].replace(' ', ''), 'lib'))
    for path in PATH_TO_FIND:
        hdf5PathFound = findFileInDirList("libhdf5*", path)
        if hdf5PathFound:
            dictPackages['LIBDIRFLAGS'] += " -L%s" % hdf5PathFound
            dictPackages['HDF5_HOME'] = hdf5PathFound
            updateXmippEnv(LD_LIBRARY_PATH=hdf5PathFound)
            break
    if hdf5PathFound == '':
        printMessage(text=red('HDF5 nod found'), debug=debugPrints)

def getTIFF(dictPackages):
    for path in PATH_TO_FIND:
        libtiffPathFound = findFileInDirList("libtiff.so", path)
        if libtiffPathFound:
            dictPackages['LIBDIRFLAGS'] += " -L%s" % libtiffPathFound
            dictPackages['TIFF_SO'] = join(libtiffPathFound, 'libtiff.so')
            break
    if libtiffPathFound == '':
        printError(errorMsg='TIFF library not found at {}'.format(PATH_TO_FIND), retCode=TIFF_ERROR)

    patron = '/**/tiffio.h'
    for path in PATH_TO_FIND_H:
        pathTIFF_H = glob.glob(f'''{path}/{patron}''')
        if pathTIFF_H:
            dictPackages['TIFF_H'] = pathTIFF_H[0]
    if dictPackages['TIFF_H'] == '':
        printError(retCode=TIFF_H_ERROR, errorMsg='')

def getFFTW3(dictPackages):
    for path in PATH_TO_FIND:
        libfftw3PathFound = findFileInDirList("libfftw3f.so", path)
        if libfftw3PathFound:
            dictPackages['LIBDIRFLAGS'] += " -L%s" % libfftw3PathFound
            dictPackages['FFTW3_SO'] = join(libfftw3PathFound, 'libfftw3.so')
            break
    if libfftw3PathFound == '':
        printError(errorMsg='FFTW3 library not found at {}'.format(PATH_TO_FIND), retCode=FFTW3_ERROR)

    patron = 'fftw3.h'
    for path in PATH_TO_FIND_H:
        pathFFTW3_H = glob.glob(join(path, patron))
        if pathFFTW3_H:
            dictPackages['FFTW3_H'] = pathFFTW3_H[0]
    if dictPackages['FFTW3_H'] == '':
        printError(retCode=FFTW3_H_ERROR, errorMsg='')

def getINCDIRFLAGS(dictPackages):
    """
    This function checks for HDF5 ('hdf5.h') in a specified directory list.
    If found, updates 'INCDIRFLAGS' in 'dictPackages' with the HDF5 include path.
    If not found, prints a message indicating HDF5 installation is required.

    Updates 'INCDIRFLAGS' in 'dictPackages' based on HDF5 presence.

    Params:
    - dictPackages (dict): Dictionary with package information.
        Expected keys: 'INCDIRFLAGS'.

    """
    pathHdf5 = findFileInDirList('hdf5.h', INC_HDF5_PATH)
    if pathHdf5:
        try:
            dictPackages['INCDIRFLAGS'] += ' -I' + pathHdf5
        except KeyError:
            dictPackages['INCDIRFLAGS'] = ' -I' + pathHdf5
    else:
        printError(retCode=HDF5_NOT_FOUND_ERROR, errorMsg='HDF5 not detected but required, please install it')

    #TIFF
    if path.exists(dictPackages['TIFF_H']):
        dictPackages['INCDIRFLAGS'] += ' -I' + path.dirname(dictPackages['TIFF_H'])

    #FFTW3
    if path.exists(dictPackages['FFTW3_H']):
        dictPackages['INCDIRFLAGS'] += ' -I' + path.dirname(dictPackages['FFTW3_H'])

def getLIBDIRFLAGS(dictPackages):
    localLib = "%s/lib" % get_paths()['data']
    dictPackages["LIBDIRFLAGS"] += " -L%s" % localLib
    updateXmippEnv(LD_LIBRARY_PATH=localLib)

def checkGit():
    version = gitVersion()
    if versionToNumber(version) < versionToNumber(GIT_MINIMUM):
        printError(retCode=GIT_VERSION_ERROR, errorMsg='GIT version {} lower than minimum: {}'.
                   format(version, GIT_MINIMUM))
    else:
        printMessage(text=green('git {} found'.format(version)), debug=debugPrints)


def checkHDF5(dictPackages):
    """
    Checks HDF5 library configuration based on provided package information.

    Params:
    - dictPackages (dict): Dictionary with package information.
        Keys: "LIBDIRFLAGS", "LINKFLAGS".

    Returns:
    - tuple: Success status (bool) and error code (int).
        False, 6: HDF5 configuration failed.
        True, 0: HDF5 configuration successful.
    """
    version = HDF5Version(dictPackages['HDF5_HOME'])
    if versionToNumber(version) < versionToNumber(HDF5_MINIMUM):
        printError('HDF5 {} version minor than {}'.format(version, HDF5_MINIMUM), HDF5_VERSION_ERROR)
    cppProg = ("""
               #include <hdf5.h>
               \n int main(){}\n
               """)
    with open("xmipp_test_main.cpp", "w") as cppFile:
        cppFile.write(cppProg)
    cmd = ("%s %s %s xmipp_test_main.cpp -o xmipp_test_main" %
           (dictPackages['CXX'], LINKFLAGS, dictPackages["INCDIRFLAGS"]))
    status, output = runJob(cmd)
    if status != 0:
        printError(retCode=HDF5_ERROR, errorMsg=output)

    runJob("rm xmipp_test_main*", showError=True)
    printMessage(text=green('HDF5 {} found'.format(version)), debug=debugPrints)


def checkTIFF(dictPackages):
    if path.exists(dictPackages['TIFF_H']):
        printMessage(text=green('TIFF {} found'.format(TIFFVersion(dictPackages['TIFF_SO']))), debug=debugPrints)
    else:
        printError(retCode=TIFF_H_ERROR, errorMsg='{} file does not exist'.format(dictPackages['TIFF_H']))

def checkFFTW3(dictPackages):
    if path.exists(dictPackages['FFTW3_H']):
        version = FFTW3Version(dictPackages['FFTW3_SO'])
        if versionToNumber(version) >= versionToNumber(FFTW_MINIMUM):
            printMessage(text=green('FFTW3 {} found'.format(version)), debug=debugPrints)
        else:
            printError(retCode=FFTW3_VERSION_ERROR, errorMsg=green('FFTW3 version {} lower than minimum: {}'.format(version, FFTW_MINIMUM)))
    else:
        printError(retCode=TIFF_H_ERROR, errorMsg='{} file does not exist'.format(dictPackages['FFTW3_H']))


def checkCMake():
    """
    ### This function checks if the current installed version, if installed, is above the minimum required version.
    ### If no version is provided it just checks if CMake is installed.

    #### Params:
    minimumRequired (str): Optional. Minimum required CMake version.

    #### Returns:
    An error message in color red in a string if there is a problem with CMake, None otherwise.
    """
    try:
        cmakVersion = getCmakeVersion()
        # Checking if installed version is below minimum required
        if versionToNumber(cmakVersion) < versionToNumber(CMAKE_MINIMUM):
            printError('Your CMake version ({cmakVersion}) is below {CMAKE_MINIMUM}', CMAKE_VERSION_ERROR)
    except FileNotFoundError:
        printError('CMake is not installed', CMAKE_ERROR)
    except Exception:
        printError('Can not get the cmake version', CMAKE_ERROR)

    printMessage(text=green('cmake {} found'.format(cmakVersion)), debug=debugPrints)

def checkScons():
    sconsV = getSconsVersion()
    if sconsV is not None:
        if versionToNumber(sconsV) < versionToNumber(SCONS_MINIMUM):
          status = installScons()
          if status[0]:
            sconsV = getSconsVersion()
            printMessage(text=green('Scons {} installed on scipion3 enviroment'.format(sconsV)), debug=True)
          else:
            printError('scons found {}, required {}\n{}'.
              format(sconsV, SCONS_MINIMUM, status[1]), SCONS_VERSION_ERROR)
        else:
          printMessage(text=green('SCons {} found'.format(sconsV)), debug=debugPrints)
    else:
        status = installScons()
        if status[0]:
          sconsV = getSconsVersion()
          printMessage(text=green('Scons {} installed on scipion3 enviroment'.format(sconsV)), debug=True)
        else:
          printError('Scons not found. {}'.format(status[1]), SCONS_ERROR)

def checkRsync():
    rsyncV = getRsyncVersion()
    if rsyncV is None:
        if versionToNumber(rsyncV) < versionToNumber(RSYNC_MINIMUM):
            printError('rsync found {}, required {}'.format(rsyncV, RSYNC_MINIMUM), RSYNC_VERSION_ERROR)
    printMessage(text=green('rsync {} found'.format(rsyncV)), debug=debugPrints)

