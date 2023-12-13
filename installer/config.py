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
                        OK,UNKOW_ERROR,SCONS_VERSION_ERROR,SCONS_ERROR,
                        GCC_VERSION_ERROR,CC_NO_EXIST_ERROR,CXX_NO_EXIST_ERROR,CXX_VERSION_ERROR,
                        MPI_VERSION_ERROR,MPI_NOT_FOUND_ERROR,PYTHON_VERSION_ERROR ,
                        PYTHON_NOT_FOUND_ERROR ,NUMPY_NOT_FOUND_ERROR ,
                        JAVA_HOME_PATH_ERROR, MATLAB_ERROR ,MATLAB_HOME_ERROR,
                        CUDA_VERSION_ERROR ,CUDA_ERROR ,HDF5_ERROR, LINK_FLAGS,
                        MPI_COMPILLATION_ERROR, MPI_RUNNING_ERROR,
                        JAVAC_DOESNT_WORK_ERROR, JAVA_INCLUDE_ERROR, CMAKE_MINIMUM,
                        CMAKE_VERSION_ERROR, CMAKE_ERROR, cmakeInstallURL, SCONS_MINIMUM,
                        VERSION_PACKAGES, CC, CXX, MPI_CC, MPI_CXX, MPI_RUN, JAVA, MATLAB,
                        OPENCV, CUDA, STARPU, HDF5, SCONS, CMAKE)
from .utils import (red, green, yellow, blue, runJob, existPackage,
                    getPackageVersionCmd,
                    whereIsPackage, findFileInDirList, getINCDIRFLAG, pathPackage,
                    getCompatibleGCC, CXXVersion, findFileInDirList, checkLib,
                    get_Hdf5_name, printError, MPIVersion, installScons)

from .versions import (getOSReleaseName, getArchitectureName, getCUDAVersion,
                                cmakeVersion, gppVersion, gccVersion, sconsVersion)
from .versions import getCUDAVersion, versionToNumber
from datetime import datetime
from sysconfig import get_paths


def config():
    """check the config if exist else create it and check it"""
    if not existConfig():
        print('Generating config file xmipp.conf')
        writeConfig(getSystemValues())
    dictConfig = readConfig()
    checkConfig(dictConfig)
    return dictConfig


def getSystemValues():
    """
    Retrieves system information related to various packages and configurations.

    Returns:
    - dict: Dictionary containing system package information.
    """
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
    """
    Checks the configurations of various packages.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    """
    checkErrors = []
    versionsPackages = VERSION_PACKAGES

    checkCC(dictPackages, checkErrors, versionsPackages) #TODO extra check, run a compillation?
    checkCXX(dictPackages, checkErrors, versionsPackages) #TODO extra check, run a compillation?
    checkMPI(dictPackages, checkErrors, versionsPackages)
    checkJava(dictPackages, checkErrors, versionsPackages)
    if dictPackages['MATLAB'] == 'True':
        checkMatlab(dictPackages, checkErrors, versionsPackages)
    if dictPackages['OPENCV'] == 'True':
        checkOPENCV(dictPackages, checkErrors, versionsPackages)
    if dictPackages['CUDA'] == 'True':
        checkCUDA(dictPackages, checkErrors, versionsPackages)
    if dictPackages['STARPU'] == 'True':
        checkSTARPU(dictPackages, checkErrors, versionsPackages)
    checkHDF5(dictPackages, checkErrors, versionsPackages)
    checkScons(checkErrors, versionsPackages)
    checkCMake(checkErrors, versionsPackages)


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

def checkCC(dictPackages, checkErrors, versionsPackages):
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
        version = gccVersion(dictPackages)
        versionsPackages[CC] = version
        if versionToNumber(version) >= versionToNumber(GCC_MINIMUM):
            print(green('gcc {} found'.format(version)))
            return OK
        checkErrors.append([GCC_VERSION_ERROR, 'gcc {} lower than required ({})'.format(version, GCC_MINIMUM)])
    else:
        checkErrors.append([CC_NO_EXIST_ERROR, 'GCC package path: {} does not exist'.format(dictPackages[CC])])

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

def checkCXX(dictPackages, checkErrors, versionsPackages):
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
        version = gppVersion(dictPackages)
        versionsPackages[CC] = version
        if versionToNumber(version) >= versionToNumber(GCC_MINIMUM):
            print(green('g++ {} found'.format(version)))
            return OK
        checkErrors.append([CXX_VERSION_ERROR, 'g++ {} lower than required ({})'.format(version, GPP_MINIMUM)])
    else:
        checkErrors.append([CXX_NO_EXIST_ERROR, 'CXX package path: {} does not exist'.format(dictPackages[CXX])])

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

def checkMPI(dictPackages, checkErrors, versionsPackages):
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
                versionsPackages[pack] = version
                if versionToNumber(version) >= versionToNumber(MPI_MINIMUM):
                    print(green('{} {} found'.format(pack, version)))
                checkErrors.append([MPI_VERSION_ERROR, 'mpi {} lower than required ({})'.format(version, GPP_MINIMUM)])
        else:
            checkErrors.append([MPI_NOT_FOUND_ERROR,'MPI package: {} does not exist'.format(pack)])

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
    status, output = runJob(cmd, showError=True)
    if status != None:
        checkErrors.append([MPI_RUNNING_ERROR, 'Fails running this command: {}\nError message: {}'.format(cmd, output)])

    libhdf5 = get_Hdf5_name(dictPackages["LIBDIRFLAGS"])
    cmd = (("%s %s  %s xmipp_mpi_test_main.o -o xmipp_mpi_test_main -lfftw3"
           " -lfftw3_threads -l%s  -lhdf5_cpp -ltiff -ljpeg -lsqlite3 -lpthread")
           % (dictPackages["MPI_CXX"], LINK_FLAGS, dictPackages["LIBDIRFLAGS"], libhdf5))

    status, output = runJob(cmd)
    if status != None:
        checkErrors.append([MPI_COMPILLATION_ERROR, 'Fails running this command: {}\nError message: {}'.format(cmd, output)])

    runJob("rm xmipp_mpi_test_main*", showOutput=False,showCommand=False)

    processors = 2
    output = runJob('{} -np {} echo {}'.format(dictPackages['MPI_RUN'], processors, 'Running'), showError=True)[1]
    if output.count('Running') != processors:
        output = runJob('{} -np 2 --allow-run-as-root echo {}'.format(dictPackages['MPI_RUN'], processors,  'Running'), showError=True)[1]
        if output.count('Running') != processors:
            checkErrors.append([MPI_RUNNING_ERROR,  'mpirun or mpiexec have failed.'])

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

def checkJava(dictPackages, checkErrors, versionsPackages):
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
        print(green('java installation found'))
    else:
        printError('JAVA_HOME path: {} does not work'.format(dictPackages['JAVA_HOME']), JAVA_HOME_PATH_ERROR)

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
    if retCode!= None:
        checkErrors.append([JAVAC_DOESNT_WORK_ERROR, cmd])
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
    if status != None:
        checkErrors.append([JAVA_INCLUDE_ERROR, output])
    runJob("rm xmipp_jni_test*", showError=True)
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
        - 10: JAVA_HOME path is not configured correctly.
        - 11: 'javac' compiler error.
        - 12: Error in including Java libraries.
    """
    #TODO check behaviour in a system with matlab installed
    if not isdir(dictPackages['MATLAB_HOME']):
        printError('MATLAB_HOME={} does not exist'.format(dictPackages['MATLAB_HOME']), MATLAB_HOME_ERROR)

    cppProg = """
    #include <mex.h>
    int dummy(){return 0;}
    """
    with open("xmipp_mex.cpp", "w") as cppFile:
        cppFile.write(cppProg)

    cmd = " {} -silent xmipp_mex.cpp".format(join(dictPackages["MATLAB_HOME"], 'bin', 'mex'))
    status, output = runJob(cmd, showError=True)
    if status != None:
        printError(output, MATLAB_HOME_ERROR)
        runJob("rm xmipp_mex*")
    runJob("rm xmipp_mex*")
    print(green('Matlab installation found'))
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
    if status != None:
        printError('OpenCV set as True but {}'.format(output))
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
    if runJob("%s -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv %s " % (dictPackages['CXX'], CXX_FLAGS, dictPackages['INCDIRFLAGS']), showError=True)[0] != 0:
        openCV_Version = 2
    else:
        runJob("./xmipp_test_opencv", showError=True)
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
    status, output = runJob("%s -c -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv.o %s" % (dictPackages['CXX'], CXX_FLAGS, dictPackages['INCDIRFLAGS']))
    if status != None:
        print(red('OPENCVSUPPORTSCUDA set as True but is not available'))
        dictPackages['OPENCVSUPPORTSCUDA'] = ''

    runJob("rm xmipp_test_opencv*", showError=True)

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
        dictPackages['CUDA_HOME'] = pathPackage('nvcc').replace('/bin/nvcc', '')
        dictPackages['CUDA_CXX'] = dictPackages['CXX']
        print(green('CUDA nvcc detected at {}'.format(dictPackages['CUDA_HOME'])))

def checkCUDA(dictPackages):
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
    if nvcc_version != 'Unknow':
        gxx_version = gppVersion(dictPackages)
        candidates, resultBool = getCompatibleGCC(nvcc_version)
        if resultBool == True and gxx_version in candidates:
            print(green('CUDA {} found'.format(nvcc_version)))
            return OK
        else:
            printError('CUDA {} not compatible with the current g++ compiler version {}\n'
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
    #TODO check behaviour in a system with starpu installed
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

        if runJob(
                "%s -c -w %s %s -I%s -L%s -l%s xmipp_starpu_config_test.cpp -o xmipp_starpu_config_test.o" %
                (dictPackages["NVCC"], dictPackages["NVCC_CXXFLAGS"],
                 dictPackages["INCDIRFLAGS"],
                 dictPackages["STARPU_INCLUDE"],
                 dictPackages["STARPU_LIB"],
                 dictPackages["STARPU_LIBRARY"]))[0] != 0:
            print(red("Check STARPU_* settings"))
        runJob("rm -f xmipp_starpu_config_test*")

    return OK



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
#             print(red('python {} lower than required ({})'.format(version,
#                                                                PYTHON_MINIMUM)))
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
    PATH_TO_FIND_HDF5.append(join(get_paths()['data'].replace(' ', ''), 'lib'))
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
    cppProg = ("""
               #include <hdf5.h>
               \n int main(){}\n
               """)
    with open("xmipp_test_main.cpp", "w") as cppFile:
        cppFile.write(cppProg)
    cmd = ("%s %s %s xmipp_test_main.cpp -o xmipp_test_main" %
           (dictPackages['CXX'], LINK_FLAGS, dictPackages["INCDIRFLAGS"]))
    status, output = runJob(cmd)
    if status != None:
        printError(output, HDF5_ERROR)

    runJob("rm xmipp_test_main*", showError=True)
    print(green('HDF5 installation found'))
    return OK


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
        cmakVersion = cmakeVersion()
        # Checking if installed version is below minimum required
        if versionToNumber(cmakVersion) < versionToNumber(CMAKE_MINIMUM):
            printError('Your CMake version ({cmakVersion}) is below {CMAKE_MINIMUM}', CMAKE_VERSION_ERROR)
    except FileNotFoundError:
        printError('CMake is not installed', CMAKE_ERROR)
    except Exception:
        printError('Can not get the cmake version', CMAKE_ERROR)

    print(green('cmake {} found'.format(cmakVersion)))

def checkScons():
	sconsV = sconsVersion()
	if sconsV is not None:
		if versionToNumber(sconsV) < versionToNumber(SCONS_MINIMUM):
			status = installScons()
			if status[0]:
				sconsV = sconsVersion()
				print(green('Scons {} installed on scipion3 enviroment'.format(sconsV)))
			else:
				printError('scons found {}, required {}\n{}'.
					format(sconsV, SCONS_MINIMUM, status[1]), SCONS_VERSION_ERROR)
		else:
			print(green('SCons {} found'.format(sconsV)))
	else:
		status = installScons()
		if status[0]:
			sconsV = sconsVersion()
			print(green('Scons {} installed on scipion3 enviroment'.format(sconsV)))
		else:
			printError('Scons not found. {}'.format(status[1]), SCONS_ERROR)
