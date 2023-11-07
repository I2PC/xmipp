#!/usr/bin/env python3
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

from .constants import SCONS_MINIMUM, CONFIG_FILE, PACKAGES_DICT
from utils import red


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

    for package in PACKAGES.items():
        print('Collecting {} info...'.format(package.key()))
        status, path = existPackage(package.value())
        if status == True:
            dictPackages[package.key()] = path
        else:
            print(red('Package {} not found on the system'.format(package.value())))


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
    path = existPackage('gcc')
    dictPackages['CC'] = path

def getCXX(dictPackages):
    path = existPackage('g++')
    dictPackages['CXX'] = path

def getPYTHONINCFLAGS(dictPackages):
    pass

def getLIBDIRFLAGS(dictPackages):
    pass
#UTILS
def versionPackage(packageName):
    """Return the version of the package if found, else return False"""
    pass


def existPackage(packageName):
    """Return True if packageName exist, else False"""

    pass

def existPath(path):
    """Return True if path exist, else False"""
    pass


