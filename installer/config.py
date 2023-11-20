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

from os import path, remove
from os.path import isdir, join

from .constants import (SCONS_MINIMUM, CONFIG_FILE, GCC_MINIMUM,
                        GPP_MINIMUM, MPI_MINIMUM, PYTHON_MINIMUM, NUMPY_MINIMUM,
                        CXX_FLAGS, PATH_TO_FIND_HDF5)
from .utils import (red, green, yellow, runJob, versionToNumber, existPackage, versionPackage,
                    whereIsPackage, findFileInDirList, getINCDIRFLAG, pathPackage,
                    getCompatibleGCC, CXXVersion, findFileInDirList, checkLib)
from datetime import datetime
from sysconfig import get_paths


def config():
    """check the config if exist else create it and check it"""
    if not existConfig():
        writeConfig(getSystemValues())

    checkConfig(parseConfig())


def getSystemValues():
    """Collect all the required package details of the system"""
    dictPackages = {}
    getCC(dictPackages)
    getCXX(dictPackages)
    getMPI(dictPackages)
    getJava(dictPackages)
    getOPENCV(dictPackages)
    getCUDA(dictPackages)
    getSTARPU(dictPackages)
    getMatlab(dictPackages)
    return dictPackages

def checkConfig(dictPackages):
    """Check if valid all the flags of the config file"""
    pass


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
        f.write('Date written: {} '.format(datetime.today()))

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


def checkCC(packagePath):
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
    if existPackage(packagePath):
        strVersion = versionPackage(packagePath)
        idx = strVersion.find('\n')
        idx2 = strVersion[idx].rfind(' ')
        version = strVersion[idx - idx2:idx]
        if versionToNumber(version) >= versionToNumber(GCC_MINIMUM):
            return 1
        print(red('gcc {} lower than required ({})'.format(version, GCC_MINIMUM)))
        return 4
    else:
        print(red('GCC package path: {} does not exist'.format(packagePath)))
        return 5

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

def checkCXX(packagePath):
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
    if existPackage(packagePath):
        strVersion = versionPackage(packagePath)
        version = CXXVersion(strVersion)
        if versionToNumber(version) >= versionToNumber(GCC_MINIMUM):
            return 1
        print(red('g++ {} lower than required ({})'.format(version, GPP_MINIMUM)))
        return 7
    else:
        print(red('CXX package path: {} does not exist'.format(packagePath)))
        return 6

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

def checkMPI(packagePath):
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
    if existPackage(packagePath):
        strVersion = versionPackage(packagePath)
        idx = strVersion.find('\n')
        idx2 = strVersion[idx].rfind(' ')
        version = strVersion[idx - idx2:idx]
        if versionToNumber(version) >= versionToNumber(MPI_MINIMUM):
            return 1
        print(red('mpi {} lower than required ({})'.format(version, GPP_MINIMUM)))
        return 8
    else:
        print(red('MPI package: {} does not exist'.format(packagePath)))
        return 9

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


def checkJava(packagePath):
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
    if not existPackage('java'):
        return 13
    if isdir(join(packagePath, 'bin/jar')) and \
        isdir(join(packagePath, 'bin/javac')) and \
        isdir(join(packagePath, 'include')) and existPackage('java'):
        return 1
    else:
        return 14

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

def checkMatlab(packagePath):
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
    if not existPackage('matlab'):
        return 15
    if not isdir(packagePath):
        return 16
    return 1


def getOPENCV(dictPackages):
    """
    Retrieves information about the OpenCV package and its capabilities, and updates the dictionary accordingly.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    Modifies:
    - dictPackages: Updates keys 'OPENCV', 'OPENCVSUPPORTSCUDA' based on OpenCV availability and CUDA support.
    """
    cppProg = "#include <opencv2/core/core.hpp>\n"
    cppProg += "int main(){}\n"
    with open("xmipp_test_opencv.cpp", "w") as cppFile:
        cppFile.write(cppProg)

    if not runJob("%s -c -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv.o %s"\
        % (dictPackages['CXX'], CXX_FLAGS, getINCDIRFLAG()), showCommand=False,
        showOutput=False, showError=False):
        dictPackages['OPENCV'] = False
        dictPackages["OPENCVSUPPORTSCUDA"] = ''
    else:
        dictPackages['OPENCV'] = True
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
                      % (dictPackages['CXX'], CXX_FLAGS,
                         getINCDIRFLAG()), showCommand=False, showOutput=False):
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
        if runJob("%s -c -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv.o %s" %
                  (dictPackages['CXX'], CXX_FLAGS,
              getINCDIRFLAG()), showOutput=False, log=[], showCommand=False):
            dictPackages["OPENCVSUPPORTSCUDA"] = True
        else:
            dictPackages["OPENCVSUPPORTSCUDA"] = ''
        print(green("OPENCV-%s detected %s CUDA support"
                    % (version, 'with' if dictPackages["OPENCVSUPPORTSCUDA"] else 'without')))

    runJob("rm -v xmipp_test_opencv*", showOutput=False, showCommand=False)


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
                  % (dictPackages['CXX']),
                  CXX_FLAGS,
                  getINCDIRFLAG(), showCommand=False, showOutput=False, logErr=log):
        print(red('OpenCV set as True but {}'.format(log)))
        dictPackages['OPENCV'] = ''

    if dictPackages['OPENCVSUPPORTSCUDA'] == True:
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
                         CXX_FLAGS,
                         getINCDIRFLAG()), showCommand=False,
                      showOutput=False):
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
        if runJob(
                "%s -c -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv.o %s" %
                (dictPackages['CXX'], CXX_FLAGS,
                 getINCDIRFLAG()), showOutput=False, logErr=log,
                showCommand=False):
            print(red('OPENCVSUPPORTSCUDA set as True but {}'.format(log)))
            dictPackages['OPENCVSUPPORTSCUDA'] = ''

    return 1


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
    nvcc_version = versionPackage(dictPackages['CUDA_HOME'])
    if nvcc_version != '':
        if nvcc_version.find('release') != -1:
            idx = nvcc_version.find('release ')
            nvcc_version = nvcc_version[idx + len('release '):
                                        idx + nvcc_version[idx:].find(',')]
        gxx_version = versionPackage(dictPackages['CXX'])
        gxx_version = CXXVersion(gxx_version)
        candidates, resultBool = getCompatibleGCC(nvcc_version)
        if resultBool == True and gxx_version in candidates:
            return 1
        else:
            print(red('CUDA {} not compatible with the current g++ compiler version {}\n'
                      'Compilers candidates for your CUDA: {}'.format(nvcc_version, gxx_version, candidates)))
            return 17

    else:
        return 18


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

    return 1



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
    #get hdf5 libdir
    PATH_TO_FIND_HDF5.append("%s/lib" % get_paths()['data'])#TODO review path con /lib
    for path in PATH_TO_FIND_HDF5:
        hdf5PathFound = findFileInDirList("libhdf5*", path)
        if hdf5PathFound:
            dictPackages["LIBDIRFLAGS"] += " -L%s" % hdf5PathFound
            print(green('HDF5  detected at {}'.format(dictPackages['hdf5PathFound'])))
            break
    if hdf5PathFound == '':
        print(red('HDF5 nod found'))

    #get opencv libdir
    if dictPackages['OPENCV']:
        if findFileInDirList("opencv4/opencv2/core/core.hpp", ["/usr/include"]):
            dictPackages["INCDIRFLAGS"] += " -I%s" % "/usr/include/opencv4"


def getINCDIRFLAGS(dictPackages):#TODO
    pass


# def check_hdf5(self):#TODO
#     print("Checking hdf5 configuration")
#     libhdf5 = get_Hdf5_name(self.configDict["LIBDIRFLAGS"])
#     if not runJob("%s %s %s xmipp_test_main.o -o xmipp_test_main -lfftw3 -lfftw3_threads -l%s  -lhdf5_cpp -ltiff -ljpeg -lsqlite3 -lpthread" %
#                   (self.get(Config.KEY_LINKERFORPROGRAMS), self.configDict["LINKFLAGS"],
#                    self.configDict["LIBDIRFLAGS"], libhdf5),
#                   show_command=False, show_output=False):
#         return False, 6
#     runJob("rm xmipp_test_main*", show_command=False, show_output=False)
#     print(green('Done ' + (' ' * 70)))
#     return True, 0