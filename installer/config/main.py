# ***************************************************************************
# * Authors:		Alberto García (alberto.garcia@cnb.csic.es)
# *							Martín Salinas (martin.salinas@cnb.csic.es)
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
from os import getcwd

from ..constants import CONFIG_FILE, CONFIG_DICT, INTERNAL_FLAGS
from ..utils import printError
from ..exit import exitXmipp

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



def exitError(output:str='', retCode:int=0, dictPackages:dict={}):
    printError(errorMsg=output, retCode=retCode)
    print(getcwd())
    if not dictPackages:
        dictPackages, _ = readConfig()
    exitXmipp(retCode=retCode, dictPackages=dictPackages, tarPost=tarPost)