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
from os.path import isdir, join, isfile
from os import path

from ..utils import (existPackage, versionToNumber, printMessage, green,
                     getPackageVersionCmd, updateXmippEnv, whereIsPackage,
                     getPackageVersionCmdReturn, get_Hdf5_name, runJob,
                     installScons, getCompatibleGCC, printWarning)
from ..versions import (getGPPVersion, getGCCVersion, MPIVersion, JAVAVersion,
                        getRsyncVersion, getSconsVersion,gitVersion, opencvVersion,
                        getCmakeVersion, FFTW3Version, getCUDAVersion, HDF5Version,
                        TIFFVersion)
from .main import exitError
from ..constants import *



def checkConfig(dictPackages, dictInternalFlags, scratch, dPrints):
    """
    Checks the configurations of various packages.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    """
    global tarPost
    tarPost = scratch
    global debugPrints
    debugPrints = dPrints

    checkPackagesStatus = []
    checkCC(dictPackages)
    checkCXX(dictPackages)
    checkMPI(dictPackages, dictInternalFlags)
    checkJava(dictPackages)
    if dictPackages['MATLAB'] == True:
        checkMatlab(dictPackages, checkPackagesStatus)
    if dictPackages['OPENCV'] == True:
        checkOPENCV(dictPackages, checkPackagesStatus)
    if dictPackages['CUDA'] == True:
        checkCUDA(dictPackages, checkPackagesStatus)
    if dictPackages['STARPU'] == True:
        checkSTARPU(dictPackages, checkPackagesStatus)
    checkGit()
    checkHDF5(dictPackages)
    checkTIFF(dictPackages)
    checkFFTW3(dictPackages)
    checkScons(dictPackages)
    checkCMake()
    checkRsync()

    if checkPackagesStatus != []:
        for pack in checkPackagesStatus:
            printWarning(text=pack[0], warningCode=pack[0], debug=True)


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
        exitError(retCode=GCC_VERSION_ERROR, output='gcc {} lower than required ({})'.format(version, GCC_MINIMUM), dictPackages=dictPackages)
    else:
        exitError(retCode=CC_NO_EXIST_ERROR, output='GCC package path: {} does not exist'.format(dictPackages[CC]), dictPackages=dictPackages)


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
        exitError(retCode=CXX_VERSION_ERROR, output='g++ {} lower than required ({})'.format(version, GPP_MINIMUM), dictPackages=dictPackages)

    else:
        exitError(retCode=CXX_NO_EXIST_ERROR, output='CXX package path: {} does not exist'.format(dictPackages[CXX]), dictPackages=dictPackages)

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
    for pack in ['mpicc', 'mpirun', 'mpicxx']:
        if existPackage(pack):
            if pack == 'mpirun':
                strVersion = getPackageVersionCmd(pack)
                version = MPIVersion(strVersion)
                if versionToNumber(version) >= versionToNumber(MPI_MINIMUM):
                    dictPackages['MPI_RUN'] = 'mpirun'
                    updateXmippEnv(PATH=whereIsPackage('mpirun'))
                    printMessage(text=green('{} {} found'.format(pack, version)), debug=debugPrints)
                else:
                    exitError(retCode=MPI_VERSION_ERROR,
                              output='mpi {} lower than required ({})'.format(version, GPP_MINIMUM),
                              dictPackages=dictPackages)
            elif pack == 'mpicc':
                dictPackages['MPI_CC'] = 'mpicc'
            else:
                dictPackages['MPI_CXX'] = 'mpicxx'
        else:
            if getPackageVersionCmd(pack) == None:
                output, retCode = getPackageVersionCmdReturn(pack)
                exitError(retCode=MPI_NOT_FOUND_ERROR,
                      output=f'{pack} package error:\n {output}', dictPackages=dictPackages)

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
        exitError(retCode=MPI_RUNNING_ERROR,
                  output='Fails running the command: \n{}\nError message: {}'.format(cmd, output),
                  dictPackages=dictPackages)

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
        exitError(retCode=MPI_COMPILLATION_ERROR,
                  output='Fails running the command: \n{}\n\nError message:\n{}'.format(cmd, output),
                  dictPackages=dictPackages)

    runJob("rm xmipp_mpi_test_main*", showOutput=False,showCommand=False)

    processors = 2
    output = runJob('{} -np {} echo {}'.format(dictPackages['MPI_RUN'], processors, 'Running'))[1]
    if output.count('Running') != processors:
        output = runJob('{} -np 2 --allow-run-as-root echo {}'.format(dictPackages['MPI_RUN'], processors,  'Running'))[1]
        if output.count('Running') != processors:
            exitError(retCode=MPI_RUNNING_ERROR,
                      output='mpirun or mpiexec have failed.',
                      dictPackages=dictPackages)

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
        exitError(retCode=JAVA_HOME_PATH_ERROR,
                  output='JAVA_HOME path: {} does not work'.format(dictPackages['JAVA_HOME']),
                  dictPackages=dictPackages)

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
        exitError(retCode=JAVAC_DOESNT_WORK_ERROR,
                  output=cmd,
                  dictPackages=dictPackages)
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
        exitError(retCode=JAVA_INCLUDE_ERROR,
                  output=output,
                  dictPackages=dictPackages)
    runJob("rm xmipp_jni_test*", showError=True)

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

    runJob("rm xmipp_test_opencv*", showError=False)

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
    if not path.exists(dictPackages['HDF5_HOME']):
        exitError(retCode=HDF5_NOT_FOUND_ERROR, output='HDF5 nod found', dictPackages=dictPackages)
    version = HDF5Version(dictPackages['HDF5_HOME'])
    if versionToNumber(version) < versionToNumber(HDF5_MINIMUM):
        exitError(retCode=HDF5_VERSION_ERROR, output='HDF5 {} version minor than {}'.format(version, HDF5_MINIMUM), dictPackages=dictPackages)
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
        exitError(retCode=HDF5_ERROR, output=output, dictPackages=dictPackages)
    runJob("rm xmipp_test_main*", showError=True)
    printMessage(text=green('HDF5 {} found'.format(version)), debug=debugPrints)

def checkTIFF(dictPackages):
    if path.exists(dictPackages['TIFF_H']):
        printMessage(text=green('TIFF {} found'.format(TIFFVersion(dictPackages['TIFF_SO']))), debug=debugPrints)
    else:
        exitError(retCode=TIFF_H_ERROR, output='TIFF library not found', dictPackages=dictPackages)
    if path.exists(dictPackages['TIFF_SO']) == False:
        exitError(retCode=TIFF_ERROR, output='libtiff.so not found', dictPackages=dictPackages)

def checkFFTW3(dictPackages):
    if path.exists(dictPackages['FFTW3_H']):
        if path.exists(dictPackages['FFTW3_SO']):
            version = FFTW3Version(dictPackages['FFTW3_SO'])
            if versionToNumber(version) >= versionToNumber(FFTW_MINIMUM):
                printMessage(text=green('FFTW3 {} found'.format(version)), debug=debugPrints)
            else:
                exitError(retCode=FFTW3_VERSION_ERROR, output='FFTW3 version {} lower than minimum: {}'.format(version, FFTW_MINIMUM),
                          dictPackages=dictPackages)
        else:
            exitError(retCode=FFTW3_ERROR, output='libfftw3.so does not exist',
                      dictPackages=dictPackages)
    else:
        exitError(retCode=FFTW3_H_ERROR, output='FFTW3 does not exist', dictPackages=dictPackages)

def checkGit():
    version = gitVersion()
    if versionToNumber(version) < versionToNumber(GIT_MINIMUM):
        exitError(retCode=GIT_VERSION_ERROR,
                  output='GIT version {} lower than minimum: {}'.
                   format(version, GIT_MINIMUM))
    else:
        printMessage(text=green('git {} found'.format(version)), debug=debugPrints)

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
            exitError(retCode=CMAKE_VERSION_ERROR, output='Your CMake version {} is below {}'.format(cmakVersion, CMAKE_MINIMUM))
    except FileNotFoundError:
        exitError(retCode=CMAKE_ERROR, output='CMake is not installed')
    except Exception:
        exitError(retCode=CMAKE_ERROR, output='Can not get the cmake version')

    printMessage(text=green('cmake {} found'.format(cmakVersion)), debug=debugPrints)

def checkScons(dictPackages:dict):
    sconsV = getSconsVersion(dictPackages)
    if sconsV is not None:
        if versionToNumber(sconsV) < versionToNumber(SCONS_MINIMUM):
            status = installScons()
            if status is False:
                exitError(retCode=SCONS_VERSION_ERROR,output='scons found {}, required {}'.format(sconsV, SCONS_MINIMUM))
            else:
                retCode, outputStr = runJob('which scons')
                if retCode == 0:
                    dictPackages['SCONS'] = outputStr
        else:
          printMessage(text=green('SCons {} found'.format(sconsV)), debug=debugPrints)
    else:
        status = installScons()
        if status is False:
          exitError(retCode=SCONS_ERROR, output='Scons not found.')
        else:
            retCode, outputStr = runJob('which scons')
            if retCode == 0 and outputStr != '':
                dictPackages['SCONS'] = outputStr
            else:
                exitError(retCode=SCONS_ENV_ERROR, output='Scons not found.')

def checkRsync():
    rsyncV = getRsyncVersion()
    if rsyncV is None:
        if versionToNumber(rsyncV) < versionToNumber(RSYNC_MINIMUM):
            exitError(retCode=RSYNC_VERSION_ERROR,
                      output='rsync found {}, required {}'.format(rsyncV, RSYNC_MINIMUM))
    printMessage(text=green('rsync {} found'.format(rsyncV)), debug=debugPrints)

