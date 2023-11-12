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

from os import path

from .constants import (SCONS_MINIMUM, CONFIG_FILE, PACKAGES_DICT, GCC_MINIMUM,
                        GPP_MINIMUM, MPI_MINIMUM, PYTHON_MINIMUM, NUMPY_MINIMUM)
from .utils import red, runJob, versionToNumber
from sysconfig import get_paths

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
    if existPackage('mpicxx'):
        dictPackages['MPI_CXX'] = 'mpicxx'
    if existPackage('mpirun'):
        dictPackages['MPI_RUN'] = 'mpirun'

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




#UTILS
def versionPackage(package):
    """Return the version of the package if found, else return False"""
    str = []
    if runJob('{} --version'.format(package), showOutput=False, logOut=str):
        if str[0].find('not found') != -1:
            return str[0]
    return ''


def existPackage(packageName):
    """Return True if packageName exist, else False"""
    path = pathPackage(packageName)
    if path != '' and versionPackage(path) != '':
        return True
    return False

def pathPackage(packageName):
    path = []
    runJob('which {}'.format(packageName), showOutput=False, logOut=path)
    return path[0]


def existPath(path):
    """Return True if path exist, else False"""
    pass


