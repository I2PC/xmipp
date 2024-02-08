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

import sys
import os

from ..utils import ( blue,  printWarning)
from datetime import datetime
from .configChecks import *
from .configGets import *
from .main import readConfig


def config(debugP:bool=True, scratch:bool=False, tarAndPost:bool=True):
    global tarPost
    setDebugPrint(debugP)
    tarPost = tarAndPost
    """check the config if exist else create it and check it"""
    # printMessage('LD_LIBRARY_PATH: ', debug=debugPrints)
    # runJob('echo $LD_LIBRARY_PATH', showOutput=True)
    if not existConfig() or scratch:
        try:
            os.remove(CONFIG_FILE)
        except FileNotFoundError:
            pass
        printMessage(text='\n-- Generating config file xmipp.conf...', debug=True)
        dictPackages = getSystemValues()
        dictInternalFlags = getInternalFlags(dictPackages)
        writeConfig(dictPackages, dictInternalFlags)
        printMessage(text=green('-- Done'), debug=True)
    else:
        dictPackages, dictInternalFlags = readConfig()
    dictNoChecked = dictPackages.copy()
    printMessage(text='\n-- Checking libraries from config file...', debug=True)
    checkConfig(dictPackages, dictInternalFlags)
    dictInternalFlags2 = getInternalFlags(dictPackages)#if checkConfig change any parameter...
    if dictPackages != dictNoChecked or dictInternalFlags != dictInternalFlags2:
        writeConfig(dictP=dictPackages, dictInt=dictInternalFlags2)
    printMessage(text=green('-- Done'), debug=True)
    # printMessage('LD_LIBRARY_PATH: ', debug=debugPrints)
    # runJob('echo $LD_LIBRARY_PATH', showOutput=True)
    return dictPackages

def getSystemValues():
    """
    Retrieves system information related to various packages and configurations.

    Returns:
    - dict: Dictionary containing system package information.
    """
    printMessage(text='-  Getting system libraries...', debug=True)
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
    printMessage(text=green('-  Done'), debug=True)

    return dictPackages

def getInternalFlags(dictPackages, debug: bool=False):
    printMessage(text='\n-  Getting internal flags for config file...', debug=True)
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
        exitError(retCode=PYTHONINCFLAGS_ERROR, output=str(e), dictPackages=dictPackages)

    #PYTHON_LIB
    malloc = "m" if sys.version_info.minor < 8 else ""
    dictInternalFlags['PYTHON_LIB'] = 'python%s.%s%s' % (sys.version_info.major, sys.version_info.minor, malloc)
    #LINKERFORPROGRAMS
    dictInternalFlags['LINKERFORPROGRAMS'] = dictPackages['CXX']
    #MPI_LINKERFORPROGRAMS
    dictInternalFlags['MPI_LINKERFORPROGRAMS'] = 'mpicxx'
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
            exitError(retCode=NVCC_CXXFLAGS_ERROR, output=str(e),
                      dictPackages=dictPackages)

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

    printMessage(text=green('-  Done'), debug=True)

    return dictInternalFlags


#CONFIG FILE

def existButOld():
    with open(CONFIG_FILE, 'r') as f:
        config = f.read()
        if config.find('BUILD') != -1:
            return True

def checkConfig(dictPackages, dictInternalFlags):
    """
    Checks the configurations of various packages.

    Params:
    - dictPackages (dict): Dictionary containing package information.

    """
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

    printMessage(text=green('-- Done'), debug=True)

def existConfig():
    """ Checks if the config file exist.Return True or False """
    if path.exists(CONFIG_FILE):
        if existButOld():
            runJob('mv {} {}'.format(CONFIG_FILE, OLD_CONFIG_FILE))
            print(blue('Old xmipp.conf detected, saved as to xmipp_old.conf'))
            return False
        return True

def writeConfig(dictP: dict, dictInt: dict):
    """Write the config file"""
    printMessage(text='\n-  Writting config file...', debug=True)

    with open(CONFIG_FILE, 'w') as f:
        f.write('[USER FLAGS]\n')
        for key, value in dictP.items():
            f.write('{}={}\n'.format(key, value))
        f.write('\n\n[INTERNAL FLAGS]\n')
        for key, value in dictInt.items():
            f.write('{}={}\n'.format(key, value))
        f.write('\n\n[DATE]\n')
        f.write('Config file written: {} \n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    printMessage(text=green('-  Done'), debug=True)

def parseConfig():
    """Read and save on configDic all flags of config file"""
    dictPackages = {}
    with open(CONFIG_FILE, 'r'):
        pass
    return dictPackages

