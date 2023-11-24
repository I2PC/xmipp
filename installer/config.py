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

from os import path, remove, environ
from os.path import isdir, join, isfile

from .constants import (SCONS_MINIMUM, CONFIG_FILE, GCC_MINIMUM,
                        GPP_MINIMUM, MPI_MINIMUM, PYTHON_MINIMUM, NUMPY_MINIMUM,
                        CXX_FLAGS, PATH_TO_FIND_HDF5, INC_PATH, INC_HDF5_PATH,
                        CONFIG_DICT,CXX_FLAGS,
                        OK,UNKOW_ERROR,SCONS_INSTALLATION_ERROR,NO_SCONS_NO_SCIPION_ERROR,
                        GCC_VERSION_ERROR,CC_NO_EXIST_ERROR,CXX_NO_EXIST_ERROR,CXX_VERSION_ERROR,
                        MPI_VERSION_ERROR,MPI_NOT_FOUND_ERROR,PYTHON_VERSION_ERROR ,
                        PYTHON_NOT_FOUND_ERROR ,NUMPY_NOT_FOUND_ERROR ,
                        JAVA_HOME_PATH_ERROR, MATLAB_ERROR ,MATLAB_HOME_ERROR,
                        CUDA_VERSION_ERROR ,CUDA_ERROR ,HDF5_ERROR, LINK_FLAGS,
                        MPI_COMPILLATION_ERROR, MPI_RUNNING_ERROR,
                        JAVAC_DOESNT_WORK_ERROR, JAVA_INCLUDE_ERROR)
from .utils import (red, green, yellow, blue, runJob, versionToNumber, existPackage,
                    versionPackage,
                    whereIsPackage, findFileInDirList, getINCDIRFLAG, pathPackage,
                    getCompatibleGCC, CXXVersion, findFileInDirList, checkLib,
                    get_Hdf5_name, showError, MPIVersion,CUDAVersion)
from datetime import datetime
from sysconfig import get_paths


def config():
    """check the config if exist else create it and check it"""
    if not existConfig():
        writeConfig(getSystemValues())

    #checkConfig(readConfig())


def getSystemValues():
    """Collect all the required package details of the system"""
    dictPackages = {'INCDIRFLAGS': '-I../ '}
    getCC(dictPackages)
    getCXX(dictPackages)
    getMPI(dictPackages)
    getJava(dictPackages)
    getOPENCV(dictPackages)
    getCUDA(dictPackages)
    getSTARPU(dictPackages)
    getMatlab(dictPackages)
    getLIBDIRFLAGS(dictPackages)
    getINCDIRFLAGS(dictPackages)
    return dictPackages

def readConfig():
    """Check if valid all the flags of the config file"""
    dictPackages = {}
    with open(CONFIG_FILE, 'r') as f:
        config = f.read()
    for key, _ in CONFIG_DICT.items():
        idx = config.find(key+'=')
        idx2 = config[idx:].find('=') + 1
        value = config[idx+idx2:idx + idx2 + config[idx+idx2:].find('\n')]
        dictPackages[key] = value
    return dictPackages

def checkConfig(dictPackages):
    checkCC(dictPackages) #TODO extra check, run a compillation?
    checkCXX(dictPackages) #TODO extra check, run a compillation?
    checkMPI(dictPackages)
    checkJava(dictPackages)
    if dictPackages['MATLAB'] == 'True':
        checkMatlab(dictPackages)
    if dictPackages['OPENCV'] == 'True':
        checkOPENCV(dictPackages)
    if dictPackages['CUDA'] == 'True':
        checkCUDA(dictPackages)
    if dictPackages['STARPU'] == 'True':
        checkSTARPU(dictPackages)
    checkHDF5(dictPackages)


def existConfig():
    """ Checks if the config file exist.Return True or False """
    if path.exists(CONFIG_FILE):
        return True
    else:
        return False

def writeConfig(dictPackages):
    """Write the config file"""

    with open(CONFIG_FILE, 'a') as f:
        for key, value in dictPackages.items():
            f.write('{}={}\n'.format(key, value))

        f.write('\n')
        f.write('Date written: {} \n'.format(datetime.today()))

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
    if existPackage(dictPackages['CC']):
        strVersion = versionPackage(dictPackages['CC'])
        version = CXXVersion(strVersion)
        if versionToNumber(version) >= versionToNumber(GCC_MINIMUM):
            print(green('gcc {} found'.format(version)))
            return OK
        showError('gcc {} lower than required ({})'.format(version, GCC_MINIMUM), GCC_VERSION_ERROR)
    else:
        showError('GCC package path: {} does not exist'.format(dictPackages['CC'], CC_NO_EXIST_ERROR))


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
        strVersion = versionPackage(dictPackages['CXX'])
        version = CXXVersion(strVersion)
        if versionToNumber(version) >= versionToNumber(GCC_MINIMUM):
            print(green('g++ {} found'.format(version)))
            return OK
        showError('g++ {} lower than required ({})'.format(version, GPP_MINIMUM), CXX_VERSION_ERROR)
    else:
        showError('CXX package path: {} does not exist'.format(dictPackages['CXX']), CXX_NO_EXIST_ERROR)

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
    else:
        dictPackages['MPI_RUN'] = ''

def checkMPI(dictPackages):
    """
    Checks the MPI package at the specified path for version compatibility.

    Params:
    - packagePath (str): Path to the MPI package directory.

    Returns:
    - int: Error code.
        - 1: Success.
        - 8: MPI version is lower than the required version.
        - 9: MPI package does not exist.
    """
    for pack in [dictPackages['MPI_CC'], dictPackages['MPI_CXX'], dictPackages['MPI_RUN']]:
        if existPackage(pack):
            strVersion = versionPackage(pack)
            version = MPIVersion(strVersion)
            if versionToNumber(version) >= versionToNumber(MPI_MINIMUM):
                print(green('{} {} found'.format(pack, version)))
            else:
                showError('mpi {} lower than required ({})'.format(version, GPP_MINIMUM), MPI_VERSION_ERROR)
        else:
            showError('MPI package: {} does not exist'.format(pack), MPI_NOT_FOUND_ERROR)

    #More checks
    MPI_CXXFLAGS = ''
    mpiLib_env = environ.get('MPI_LIBDIR', '')
    if mpiLib_env:
        MPI_CXXFLAGS += ' -L'+mpiLib_env

    mpiInc_env = environ.get('MPI_INCLUDE', '')
    if mpiInc_env:
        MPI_CXXFLAGS += ' -I'+mpiInc_env

    cppProg = """
    #include <mpi.h>
    int main(){}
    """
    with open("xmipp_mpi_test_main.cpp", "w") as cppFile:
        cppFile.write(cppProg)
    cmd = ("%s -c -w %s %s %s xmipp_mpi_test_main.cpp -o xmipp_mpi_test_main.o"
           % (dictPackages["MPI_CXX"], dictPackages["INCDIRFLAGS"],CXX_FLAGS, MPI_CXXFLAGS))
    if not runJob(cmd, showOutput=False, showCommand=False):
        showError('Fails running this command: {}'.format(cmd), MPI_COMPILLATION_ERROR)

    libhdf5 = get_Hdf5_name(dictPackages["LIBDIRFLAGS"])
    cmd = (("%s %s  %s xmipp_mpi_test_main.o -o xmipp_mpi_test_main -lfftw3"
           " -lfftw3_threads -l%s  -lhdf5_cpp -ltiff -ljpeg -lsqlite3 -lpthread")
           % (dictPackages["MPI_CXX"], LINK_FLAGS, dictPackages["LIBDIRFLAGS"], libhdf5))
    if not runJob(cmd, showOutput=False, showCommand=False):
        showError('Fails running this command: {}'.format(cmd), MPI_COMPILLATION_ERROR)

    runJob("rm xmipp_mpi_test_main*", showOutput=False,showCommand=False)

    log = []
    processors = 2
    runJob('{} -np {} echo {}'.format(dictPackages['MPI_RUN'], processors, 'Running'),
           showCommand=False, logOut=log, showOutput=False)
    if log[0].count('Running') != processors:
        log = []
        runJob('{} -np 2 --allow-run-as-root echo {}'.format(dictPackages['MPI_RUN'], processors,  'Running'),
               showCommand=False, logOut=log, showOutput=False)
        if log[0].count('Running') != processors:
            print(red("mpirun or mpiexec have failed."))
            showError('', MPI_RUNNING_ERROR)
    return OK

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
    else:
        dictPackages['JAVA_HOME'] = ''


def checkJava(dictPackages):
    """
    Checks the existence and structure of a Java package at a specified path.

    Params:
    - packagePath (str): Path to the Java package directory.

    Returns:
    - int: Error code.
        - 13: Java package does not exist.
        - 14: Java package structure is incorrect.
        - 1: Success.
    """
    if isfile(join(dictPackages['JAVA_HOME'], 'bin/jar')) and \
            whereIsPackage(join(dictPackages['JAVA_HOME'], 'bin/javac')) and\
            isdir(join(dictPackages['JAVA_HOME'], 'include')) and existPackage('java'):
        print(green('java installation found'))
    else:
        showError('JAVA_HOME path: {} does not work'.format(dictPackages['JAVA_HOME']), JAVA_HOME_PATH_ERROR)

    #Other check
    javaProg = """
        public class Xmipp {
        public static void main(String[] args) {}
        }
    """
    with open("Xmipp.java", "w") as javaFile:
        javaFile.write(javaProg)
    cmd= "%s Xmipp.java" % join(dictPackages['JAVA_HOME'], 'bin/javac')
    if not runJob(cmd, showCommand=False, showOutput=False):
        showError(cmd, JAVAC_DOESNT_WORK_ERROR)
    runJob("rm Xmipp.java Xmipp.class",showCommand=False,showOutput=False)

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
    logE=[]
    if not runJob(cmd, showCommand=False,showOutput=False, logErr=logE):
        showError(logE[0], JAVA_INCLUDE_ERROR)
    runJob("rm xmipp_jni_test*", showCommand=False,showOutput=False)
    return OK

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
        print(green('MATLAB_HOME detected at {}'.format(dictPackages['MATLAB_HOME'])))
    else:
        dictPackages['MATLAB'] = False
        dictPackages['MATLAB_HOME'] = ''

def checkMatlab(dictPackages):
    """
    Checks for the existence of MATLAB package and verifies if a specified path is a directory.

    Params:
    - packagePath (str): Path to the package directory.

    Returns:
    - int: Error code.
        - 15: MATLAB package does not exist.
        - 16: Specified path is not a directory.
        - 1: Success.
    """
    if not isdir(dictPackages['MATLAB_HOME']):
        showError('', MATLAB_HOME_ERROR)
    if not whereIsPackage('matlab'):
        showError('', MATLAB_ERROR)

    cppProg = """
    #include <mex.h>
    int dummy(){return 0;}
    """
    with open("xmipp_mex.cpp", "w") as cppFile:
        cppFile.write(cppProg)

    cmd = " {} -silent xmipp_mex.cpp".format(join(dictPackages["MATLAB_HOME"], 'bin', 'mex'))
    logE = []
    if not runJob(cmd, showCommand=False,showOutput=False, logErr=logE):
        showError(logE[0], MATLAB_HOME_ERROR)
        runJob("rm xmipp_mex*")
    runJob("rm xmipp_mex*")
    return OK


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
                    print(green('OPENCV detected at {}'.format(join(p.split('/')[0]))))
                    break


def checkOPENCV(dictPackages):
    """
    Checks the OpenCV package and its CUDA support, updating the dictionary accordingly.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    Returns:
    - int: Error code.
        - 1: Success.
    """
    log = []
    cppProg = "#include <opencv2/core/core.hpp>\n"
    cppProg += "int main(){}\n"
    with open("xmipp_test_opencv.cpp", "w") as cppFile:
        cppFile.write(cppProg)

    if not runJob("%s -c -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv.o %s"
                  % (dictPackages['CXX'], CXX_FLAGS, dictPackages['INCDIRFLAGS']),
                  showCommand=False, showOutput=False, logErr=log):
        print(red('OpenCV set as True but {}'.format(log)))
        dictPackages['OPENCV'] = ''

    # Check version
    with open("xmipp_test_opencv.cpp", "w") as cppFile:
        cppFile.write('#include <opencv2/core/version.hpp>\n')
        cppFile.write('#include <fstream>\n')
        cppFile.write('int main()'
                      '{std::ofstream fh;'
                      ' fh.open("xmipp_test_opencv.txt");'
                      ' fh << CV_MAJOR_VERSION << std::endl;'
                      ' fh.close();'
                      '}\n')
    if not runJob("%s -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv %s "
                  % (dictPackages['CXX'],
                     CXX_FLAGS, dictPackages['INCDIRFLAGS']),
                  showCommand=False, showOutput=False):
        openCV_Version = 2
    else:
        runJob("./xmipp_test_opencv", showCommand=False, showOutput=False)
        f = open("xmipp_test_opencv.txt")
        versionStr = f.readline()
        f.close()
        version = int(versionStr.split('.', 1)[0])
        openCV_Version = version

    # Check CUDA Support
    cppProg = "#include <opencv2/core/version.hpp>\n"
    if openCV_Version < 3:
        cppProg += "#include <opencv2/core/cuda.hpp>\n"
    else:
        cppProg += "#include <opencv2/cudaoptflow.hpp>\n"
    cppProg += "int main(){}\n"
    with open("xmipp_test_opencv.cpp", "w") as cppFile:
        cppFile.write(cppProg)
    log = []
    if not runJob(
            "%s -c -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv.o %s" %
            (dictPackages['CXX'], CXX_FLAGS, dictPackages['INCDIRFLAGS']),
            showOutput=False, logErr=log, showCommand=False, showError=False):
        print(red('OPENCVSUPPORTSCUDA set as True but {}'.format(log)))
        dictPackages['OPENCVSUPPORTSCUDA'] = ''

    runJob("rm xmipp_test_opencv*", showCommand=False, showOutput=False)

    return OK


def getCUDA(dictPackages):
    """
     Retrieves information about the CUDA package and updates the dictionary accordingly.

     Params:
     - dictPackages (dict): Dictionary containing package information.

     Modifies:
     - dictPackages: Updates keys 'CUDA', 'CUDA_HOME', and 'CUDA_CXX' based on CUDA package availability.
     """
    if not existPackage('nvcc'):
        dictPackages['CUDA'] = False
        dictPackages['CUDA_HOME'] = ''
        dictPackages['CUDA_CXX'] = ''
    else:
        dictPackages['CUDA'] = True
        dictPackages['CUDA_HOME'] = pathPackage('nvcc')
        dictPackages['CUDA_CXX'] = dictPackages['CXX']
        print(green('CUDA nvcc detected at {}'.format(dictPackages['CUDA_HOME'])))



def checkCUDA(dictPackages):
    """
    Checks the compatibility of CUDA with the current g++ compiler version and updates the dictionary accordingly.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    Returns:
    - int: Error code.
        - 1: Success.
        - 17: CUDA not compatible with the current g++ compiler version.
        - 18: CUDA version information not available.
    """
    strversion = versionPackage(dictPackages['CUDA_HOME'])
    nvcc_version = CUDAVersion(strversion)
    if nvcc_version != '':
        gxx_version = versionPackage(dictPackages['CXX'])
        gxx_version = CXXVersion(gxx_version)
        candidates, resultBool = getCompatibleGCC(nvcc_version)
        if resultBool == True and gxx_version in candidates:
            return OK
        else:
            showError('CUDA {} not compatible with the current g++ compiler version {}\n'
                      'Compilers candidates for your CUDA: {}'.format(
                nvcc_version, gxx_version, candidates), CUDA_VERSION_ERROR)
    else:
        return CUDA_ERROR


def getSTARPU(dictPackages):
    """
    Retrieves information about the STARPU package and updates the dictionary accordingly.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    Modifies:
    - dictPackages: Updates keys related to STARPU package information.
    """
    if whereIsPackage("starpu_sched_display"):
        dictPackages["STARPU"] = True
        starpuBinDir = whereIsPackage("starpu_sched_display")
        dictPackages["STARPU_HOME"] = starpuBinDir.replace("/bin", "")
        dictPackages["STARPU_INCLUDE"] = "%(STARPU_HOME)s/include/starpu/1.3"
        dictPackages["STARPU_LIB"] = "%(STARPU_HOME)s/lib"
        dictPackages["STARPU_LIBRARY"] = "libstarpu-1.3"
        print(green('STARPU detected at {}'.format(dictPackages['STARPU_HOME'])))

    else:
        dictPackages["STARPU"] = False
        dictPackages["STARPU_HOME"] = ''
        dictPackages["STARPU_INCLUDE"] = ''
        dictPackages["STARPU_LIB"] = ''
        dictPackages["STARPU_LIBRARY"] = ''


def checkSTARPU(dictPackages):
    """
    Checks the configuration of the STARPU package and CUDA compatibility, printing error messages if necessary.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    Returns:
    - int: Error code.
        - 1: Success.
    """
    if dictPackages["CUDA"] != "True":
        ans = False
        print(red("CUDA must be enabled together with STARPU"))
    if dictPackages["STARPU_INCLUDE"] == "" or not isdir(
            dictPackages["STARPU_INCLUDE"]):
        ans = False
        print(red("Check the STARPU_INCLUDE directory: " +
                  dictPackages["STARPU_INCLUDE"]))
    if dictPackages["STARPU_LIB"] == "" or not isdir(
            dictPackages["STARPU_LIB"]):
        ans = False
        print(red("Check the STARPU_LIB directory: " +
                  dictPackages["STARPU_LIB"]))
    if dictPackages["STARPU_LIBRARY"] == "":
        ans = False
        print(red("STARPU_LIBRARY must be specified (link library name)"))

    if ans:
        with open("xmipp_starpu_config_test.cpp", "w") as cppFile:
            cppFile.write("""
            #include <starpu.h>
            int dummy(){return 0;}
            """)

        if not runJob(
                "%s -c -w %s %s -I%s -L%s -l%s xmipp_starpu_config_test.cpp -o xmipp_starpu_config_test.o" %
                (dictPackages["NVCC"], dictPackages["NVCC_CXXFLAGS"],
                 dictPackages["INCDIRFLAGS"],
                 dictPackages["STARPU_INCLUDE"],
                 dictPackages["STARPU_LIB"],
                 dictPackages["STARPU_LIBRARY"])):
            print(red("Check STARPU_* settings"))
        runJob("rm xmipp_starpu_config_test*")

    return OK



# def checkPYTHONINCFLAGS(incPath):
#     includes = incPath.split(' ')
#     pythonPath = includes[0].replace('-I', '')
#     numpyPath = includes[1].replace('-I', '')
#     if existPackage(pythonPath):
#         strVersion = versionPackage(pythonPath)
#         idx = strVersion.find('\n')
#         idx2 = strVersion[idx].rfind(' ')
#         version = strVersion[idx - idx2:idx]
#         if versionToNumber(version) < versionToNumber(PYTHON_MINIMUM):
#             print(red('python {} lower than required ({})'.format(version,
#                                                                PYTHON_MINIMUM)))
#             return 10
#
#     #NUMPY
#     import sys
#     sys.path.append('/path/to/directory')
#     if existPackage(numpyPath):
#         strVersion = versionPackage(pythonPath)
#         idx = strVersion.find('\n')
#         idx2 = strVersion[idx].rfind(' ')
#         version = strVersion[idx - idx2:idx]
#         if versionToNumber(version) < versionToNumber(PYTHON_MINIMUM):
#             print(red('python {} lower than required ({})'.format(version,
#                                                                PYTHON_MINIMUM)))
#             return 10
#
def getLIBDIRFLAGS(dictPackages):
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
    PATH_TO_FIND_HDF5.append(join(get_paths()['data'].replace(' ', ''), 'lib'))#TODO review path con /lib
    for path in PATH_TO_FIND_HDF5:
        hdf5PathFound = findFileInDirList("libhdf5*", path)
        if hdf5PathFound:
            dictPackages['LIBDIRFLAGS'] = " -L%s" % hdf5PathFound
            print(green('HDF5 detected at {}'.format(hdf5PathFound)))
            break
    if hdf5PathFound == '':
        print(red('HDF5 nod found'))


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
        print(red('HDF5 not detected but required, please install it'))

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
    libhdf5 = get_Hdf5_name(dictPackages['LIBDIRFLAGS'])#TODO review behave
    logE = []
    cppProg = ("""
               #include <hdf5.h>
               \n int main(){}\n
               """)
    with open("xmipp_test_main.cpp", "w") as cppFile:
        cppFile.write(cppProg)
    cmd = ("%s %s %s xmipp_test_main.o -o xmipp_test_main -lfftw3 -lfftw3_threads -l%s  -lhdf5_cpp -ltiff -ljpeg -lsqlite3 -lpthread" %
           (dictPackages['CXX'], LINK_FLAGS, dictPackages["LIBDIRFLAGS"], libhdf5))
    if not runJob(cmd, showCommand=False, showOutput=False, showError=False, logErr=logE):
        showError(logE[0], HDF5_ERROR)

    runJob("rm xmipp_test_main*", showCommand=False, showOutput=False)
    return OK
