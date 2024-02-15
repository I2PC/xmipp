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
from os.path import join
import shutil
import glob
from os import path, environ
from sysconfig import get_paths

from ..utils import (existPackage, printMessage, green, updateXmippEnv, whereIsPackage,
                      runJob, findFileInDirList, printWarning)
from ..constants import (INC_PATH, PATH_TO_FIND, PATH_TO_FIND_H, INC_HDF5_PATH,
                         HEADER2, DONE2, CUDA_NOT_IN_PATH_WARNING)

def getSystemValues(scratch, debugP):
    """
    Retrieves system information related to various packages and configurations.

    Returns:
    - dict: Dictionary containing system package information.
    """
    global tarPost
    tarPost = scratch
    global debugPrints
    debugPrints = debugP

    printMessage(text=f'{HEADER2}  Getting system libraries...', debug=True)
    dictPackages = {'INCDIRFLAGS': '-I../ ',
                    'LIBDIRFLAGS': ''}
    getCC(dictPackages)
    getCXX(dictPackages)
    getJava(dictPackages)
    getTIFF(dictPackages)
    getFFTW3(dictPackages)
    getHDF5(dictPackages)
    getScons(dictPackages)
    getINCDIRFLAGS(dictPackages)
    getLIBDIRFLAGS(dictPackages)
    getOPENCV(dictPackages)
    getCUDA(dictPackages)
    getSTARPU(dictPackages)
    getMatlab(dictPackages)
    getAnonDataCol(dictPackages)
    printMessage(text=green(DONE2), debug=True)

    return dictPackages

def getCC(dictPackages):
    """
    Retrieves information about the CC (GCC) package and updates the dictionary accordingly.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    Modifies:
    - dictPackages: Updates the 'CC' key based on the availability of 'gcc'.
    """
    ccVar = environ.get('CC', '')
    if ccVar != '':
        dictPackages['CC'] = ccVar
    elif existPackage('gcc'):
        dictPackages['CC'] = 'gcc'
    else:
        dictPackages['CC'] = ''


def getCXX(dictPackages):
    """
    Retrieves information about the CXX package and updates the dictionary accordingly.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    Modifies:
    - dictPackages: Updates the 'CXX' key based on the availability of 'g++'.
    """
    ccVar = environ.get('CXX', '')
    if ccVar != '':
        dictPackages['CXX'] = ccVar
    elif existPackage('g++'):
        dictPackages['CXX'] = 'g++'
    else:
        dictPackages['CXX'] = ''

def getAnonDataCol(dictPackages):
    aDCVar = environ.get('ANON_DATA_COLLECT', '')
    if aDCVar == 'False':
        dictPackages['ANON_DATA_COLLECT'] = aDCVar
    else:
        dictPackages['ANON_DATA_COLLECT'] = 'True'

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


def getCUDA(dictPackages):
    """
     Retrieves information about the CUDA package and updates the dictionary accordingly.

     Params:
     - dictPackages (dict): Dictionary containing package information.

     Modifies:
     - dictPackages: Updates keys 'CUDA', 'CUDA_HOME', and 'CUDACXX' based on CUDA package availability.
     """
    if not existPackage('nvcc'):#not exist or not in PATH
        nvcc_loc_candidates = ['/usr/local/cuda/bin', '/usr/local/cuda*/bin']
        for path in nvcc_loc_candidates:
            if existPackage('nvcc', path2Find=path):
                 nvccPath = shutil.which('nvcc', path=path)
                 printWarning(text=f'nvcc found in {nvccPath}',
                              warningCode=CUDA_NOT_IN_PATH_WARNING, debug=True)
        dictPackages['CUDA'] = 'False'
        dictPackages['CUDA_HOME'] = ''
        dictPackages['CUDACXX'] = ''
        updateXmippEnv(CUDA=False)
    else:
        dictPackages['CUDA_HOME'] = shutil.which('nvcc')
        dictPackages['CUDA'] = 'True'
        dictPackages['CUDACXX'] = dictPackages['CXX']
        updateXmippEnv(CUDA=True)

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

def getScons(dictPackages):
    retCode, outputStr = runJob('which scons')
    if retCode == 0:
        dictPackages['SCONS'] = outputStr
    else:
        dictPackages['SCONS'] = ''


def getTIFF(dictPackages):
    for path in PATH_TO_FIND:
        libtiffPathFound = findFileInDirList("libtiff.so", path)
        if libtiffPathFound:
            dictPackages['LIBDIRFLAGS'] += " -L%s" % libtiffPathFound
            dictPackages['TIFF_SO'] = join(libtiffPathFound, 'libtiff.so')
            break
    patron = '/**/tiffio.h'
    for path in PATH_TO_FIND_H:
        pathTIFF_H = glob.glob(f'''{path}/{patron}''')
        if pathTIFF_H:
            dictPackages['TIFF_H'] = pathTIFF_H[0]


def getFFTW3(dictPackages):
    for path in PATH_TO_FIND:
        libfftw3PathFound = findFileInDirList("libfftw3f.so", path)
        if libfftw3PathFound:
            dictPackages['LIBDIRFLAGS'] += " -L%s" % libfftw3PathFound
            dictPackages['FFTW3_SO'] = join(libfftw3PathFound, 'libfftw3.so')
            break
    patron = 'fftw3.h'
    for path in PATH_TO_FIND_H:
        pathFFTW3_H = glob.glob(join(path, patron))
        if pathFFTW3_H:
            dictPackages['FFTW3_H'] = pathFFTW3_H[0]


def getINCDIRFLAGS(dictPackages):
    """
    This function checks for HDF5 ('hdf5.h') in a specified directory list.
    If found, updates 'INCDIRFLAGS' in 'dictPackages' with the HDF5 include path.
    Updates 'INCDIRFLAGS' in 'dictPackages' based on HDF5 presence.

    Params:
    - dictPackages (dict): Dictionary with package information.
        Expected keys: 'INCDIRFLAGS'.
    """
    #HDF5
    pathHdf5 = findFileInDirList('hdf5.h', INC_HDF5_PATH)
    if pathHdf5:
        try:
            dictPackages['INCDIRFLAGS'] += ' -I' + pathHdf5
        except KeyError:
            dictPackages['INCDIRFLAGS'] = ' -I' + pathHdf5

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
