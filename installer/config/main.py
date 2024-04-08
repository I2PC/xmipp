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

# General imports
from typing import Dict, Tuple

# Installer imports
from ..constants import CMAKE_MINIMUM
from .query import getCMake
from .versions import getCmakeVersion

def __compareVersionTuple(version1: Tuple[int], version2: Tuple[int]) -> int:
	"""
	### This function compares the two given int tuples.

	#### Params:
	- version1 (tuple(int)): First tuple.
	- version2 (tuple(int)): Second tuple.

	#### Returns:
	- (int): -1 if version1 is greater than version2, 0 if they are equal, or 1 otherwise. 
	"""
	# Result constants
	version1Greater = -1
	version2Greater = 1
	equals = 0

	# Zip the two versions together and compare each corresponding pair of elements
	for v1, v2 in zip(version1, version2):
		if v1 > v2:
			return version1Greater
		elif v1 < v2:
			return version2Greater

	# If we haven't returned yet, the versions are equal up to the length of the shorter one.
	# The version with more elements is considered greater.
	if len(version1) > len(version2):
		return version1Greater
	elif len(version1) < len(version2):
		return version2Greater
	else:
		return equals  # The versions are exactly equal

def __createTupleFromVersionStr(versionStr: str, delimiter: str='.') -> Tuple[int]:
	"""
	### This function creates an int tuple from a version string.

	#### Params:
	- versionStr (str): Version string to obtain the tuple from.
	- delimiter (str): Optional. Character that separates each version number.

	#### Returns:
	- (tuple(int)): Version as an int tuple.
	"""
	versionTuple = tuple()
	versionList = versionStr.split(delimiter)
	if versionList:
		versionList = [int(number) for number in versionList]
		versionTuple = tuple(versionList)
	return versionTuple

def isCMakeValid(configDict: Dict={}) -> bool:
	"""
	### This function checks if the provided CMake version is valid.

	#### Params:
	- configDict (dict): Optional. Dictionary containing all the variables from the config file.

	#### Returns:
	- (bool): True if the current CMake version is valid, False otherwise.
	"""
	cmakeExec = getCMake(configDict)
	if not cmakeExec:
		return False
	
	# Check if current version is greater or equal to minimum required
	currentVersion = __createTupleFromVersionStr(getCmakeVersion(cmakeExec))
	return __compareVersionTuple(__createTupleFromVersionStr(CMAKE_MINIMUM), currentVersion) > 0
