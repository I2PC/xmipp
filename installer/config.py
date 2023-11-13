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

from os import path
from os.path import isdir, join

from .constants import (SCONS_MINIMUM, CONFIG_FILE, GCC_MINIMUM,
                        GPP_MINIMUM, MPI_MINIMUM, PYTHON_MINIMUM, NUMPY_MINIMUM)
from .utils import (red, green, yellow, runJob, versionToNumber, existPackage, versionPackage,
                    whereIsPackage, findFileInDirList, getINCDIRFLAG)

def config():
    """check the config if exist else create it and check it"""
    if not existConfig():
        writeConfig(getSystemValues())

    parseConfig()
    checkConfig()


def getSystemValues():
    """Collect all the required package details of the system"""
    dictPackages = {}

    getCC(dictPackages)


def checkConfig():
    """Check if valid all the flags of the config file"""
    pass


def existConfig():
    """ Checks if the config file exist.Return True or False """
    if path.exists(CONFIG_FILE):
        return True
    else:
        return False

def writeConfig():
    """Write the config file"""
    pass

def parseConfig():
    """Read and save on configDic all flags of config file"""
    pass


#PACKAGES
def getCC(dictPackages):
    if existPackage('gcc'):
        dictPackages['CC'] = 'gcc'
    else:
        dictPackages['CC'] = None


def checkCC(packagePath):
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
    if existPackage('g++'):
        dictPackages['CXX'] = 'g++'
    else:
        dictPackages['CXX'] = None

def checkCXX(packagePath):
    if existPackage(packagePath):
        strVersion = versionPackage(packagePath)
        idx = strVersion.find('\n')
        idx2 = strVersion[idx].rfind(' ')
        version = strVersion[idx - idx2:idx]
        if versionToNumber(version) >= versionToNumber(GCC_MINIMUM):
            return 1
        print(red('g++ {} lower than required ({})'.format(version, GPP_MINIMUM)))
        return 7
    else:
        print(red('CXX package path: {} does not exist'.format(packagePath)))
        return 6

def getMPI(dictPackages):
    if existPackage('mpicc'):
        dictPackages['MPI_CC'] = 'mpicc'
    else:
        dictPackages['MPI_CC'] = None
    if existPackage('mpicxx'):
        dictPackages['MPI_CXX'] = 'mpicxx'
    else:
        dictPackages['MPI_CXX'] = None
    if existPackage('mpirun'):
        dictPackages['MPI_RUN'] = 'mpirun'
    else:
        dictPackages['MPI_RUN'] = None

def checkMPI(packagePath):
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
    javaProgramPath = whereIsPackage('javac')
    if not javaProgramPath:
        javaProgramPath = findFileInDirList('javac', ['/usr/lib/jvm/java-*/bin'])
        if javaProgramPath:
            javaHomeDir = javaProgramPath.replace("/jre/bin", "")
            javaHomeDir = javaHomeDir.replace("/bin", "")
    if javaProgramPath:
        dictPackages['JAVA_HOME'] = javaHomeDir
    else:
        dictPackages['JAVA_HOME'] = None


def checkJava(packagePath):
    if not existPackage('java'):
        return 13
    if isdir(join(packagePath, 'bin/jar')) and \
        isdir(join(packagePath, 'bin/javac')) and \
        isdir(join(packagePath, 'include')) and existPackage('java'):
        return 1
    else:
        return 14

def getMatlab(dictPackages):
    matlabProgramPath = whereIsPackage('matlab')
    if matlabProgramPath:
        dictPackages['MATLAB'] = 'True'
        dictPackages['MATLAB_HOME'] = matlabProgramPath.replace("/bin", "")
    else:
        dictPackages['MATLAB'] = 'False'
        dictPackages['MATLAB_HOME'] = None

def checkMatlab(packagePath):
    if not existPackage('matlab'):
        return 15
    if not isdir(packagePath):
        return 16
    return 1


def getOPENCV(dictPackages):
    cppProg = "#include <opencv2/core/core.hpp>\n"
    cppProg += "int main(){}\n"
    with open("xmipp_test_opencv.cpp", "w") as cppFile:
        cppFile.write(cppProg)

    if not runJob("%s -c -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv.o %s"
        % (dictPackages['CXX']), ' -mtune=native -march=native -flto -std=c++17 -O3',
        getINCDIRFLAG(), showCommand=False, showOutput=False):
        dictPackages['OPENCV'] = False
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
                      % (dictPackages['CXX'], ' -mtune=native -march=native -flto -std=c++17 -O3',
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
                  (dictPackages['CXX'], ' -mtune=native -march=native -flto -std=c++17 -O3',
              getINCDIRFLAG()), showOutput=False, log=[], showCommand=False):
            dictPackages["OPENCVSUPPORTSCUDA"] = True
        else:
            dictPackages["OPENCVSUPPORTSCUDA"] = False
        print(green("OPENCV-%s detected %s CUDA support"
                    % (version, 'with' if dictPackages["OPENCVSUPPORTSCUDA"] else 'without')))

    runJob("rm -v xmipp_test_opencv*", showOutput=False, showCommand=False)


def checkOPENCV(dictPackages):
    log = []
    cppProg = "#include <opencv2/core/core.hpp>\n"
    cppProg += "int main(){}\n"
    with open("xmipp_test_opencv.cpp", "w") as cppFile:
        cppFile.write(cppProg)

    if not runJob("%s -c -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv.o %s"
                  % (dictPackages['CXX']),
                  ' -mtune=native -march=native -flto -std=c++17 -O3',
                  getINCDIRFLAG(), showCommand=False, showOutput=False, logErr=log):
        print(yellow('OpenCV set as True but {}'.format(log)))
        dictPackages['OPENCV'] = False

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
                         ' -mtune=native -march=native -flto -std=c++17 -O3',
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
                (dictPackages['CXX'],
                 ' -mtune=native -march=native -flto -std=c++17 -O3',
                 getINCDIRFLAG()), showOutput=False, logErr=log,
                showCommand=False):
            print(yellow('OPENCVSUPPORTSCUDA set as True but {}'.format(log)))
            dictPackages['OPENCVSUPPORTSCUDA'] = False

    return 1


def getCUDA(dictPackages):
    pass

def checkCUDA(packagePath):
    pass


def getSTARPU(dictPackages):
    pass

def checkSTARPU(packagePath):
    pass

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
    pass
def getINCDIRFLAGS(dictPackages):
    pass

