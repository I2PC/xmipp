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

"""
This module contains functions to collect the versions of
os, architecture, cuda, cmake, g++, and gcc.
"""

# General imports
from typing import Optional
import re

# Installer imports
from ..utils import runJob, getPackageVersionCmd
from ..constants import UNKNOWN_VALUE

####################### AUX FUNCTIONS #######################
def __parseCompilerVersion(versionCmdStr: Optional[str]) -> Optional[str]:
	"""
	### Parses the string output of the command that extracts the version of the given compiler.

	#### Params:
	- versionCmdStr (str): Output string of the --version command of the given compiler.

	#### Returns:
	- (str): Compiler's version.
	"""
	# Initialize default value
	compilerVersion = None

	# If the command string exists, get the first line
	if versionCmdStr is not None:
		versionStr = versionCmdStr.splitlines()[0]
		
		# From the first line, get the last word (version number string)
		if versionStr:
			compilerVersion = versionStr.split()[-1]

	# Returning compiler version
	return compilerVersion

####################### PACKAGE SPECIFIC FUNCTIONS #######################
def getOSReleaseName() -> str:
	"""
	### This function returns the name of the current system OS release.

	#### Returns:
	- (str): OS release name.
	"""
	# Initializing default release name 
	releaseName = UNKNOWN_VALUE
	
	# Text around release name
	textBefore = 'PRETTY_NAME="'
	textAfter = '"\n'

	# Obtaining os release name
	retCode, name = runJob('cat /etc/os-release', logOutput=False)

	# Look for release name if command did not fail
	if retCode == 0:
		# Find release name's line in command output
		targetStart = name.find(textBefore)
		if targetStart != 1:
			# Search end of release name's line
			nameEnd = name[targetStart:].find(textAfter)

			# Calculate release name's start index
			nameStart = targetStart + len(textBefore)
			if nameEnd != -1 and nameStart != nameEnd:
				# If everything was correctly found and string is 
				# not empty, extract release name
				releaseName = name[nameStart:nameEnd]

	# Return release name
	return releaseName

def getArchitectureName() -> str:
	"""
	### This function returns the name of the system's architecture name.

	#### Returns:
	- (str): Architecture name.
	"""
	# Initializing to unknown value
	archName = UNKNOWN_VALUE

	# Obtaining architecture name
	retCode, architecture = runJob('cat /sys/devices/cpu/caps/pmu_name', logOutput=False)

	# If command worked and returned info, extract it
	if retCode == 0 and architecture:
		archName = architecture
	
	# Returing architecture name
	return archName

def getCUDAVersion(nvccExecutable: Optional[str]) -> Optional[str]:
	"""
	### Extracts the NVCC (NVIDIA CUDA Compiler) version from the PATH or the config file, the last one having a higher priority.

	#### Params:
	- nvccExecutable (str): Optional. Dictionary containing packages found in the config file.

	#### Returns:
	- (str | None): CUDA version or None if there were any errors.
	"""
	if not nvccExecutable:
		return None

	# Extracting version command string
	versionCmdStr = getPackageVersionCmd(nvccExecutable)

	# Check if there were any errors
	if not versionCmdStr:
		return None

	r = re.compile(r'release (\d+\.\d+)\,')
	match = r.search(versionCmdStr)
  
	if not match:
		return None

	return match.group(1)

def getCmakeVersion(cmakeExecutable: str) -> Optional[str]:
	"""
	### Extracts the CMake version from the PATH or the config file, the last one having a higher priority.

	#### Params:
	- dictPackages (dict): Optional. Dictionary containing packages found in the config file.

	#### Returns:
	- (str | None): CMake version, or None if there were any errors.
	"""
	if not cmakeExecutable:
		return None

	# Extracting version command string
	versionCmdStr = getPackageVersionCmd(cmakeExecutable)

	# Check if there were any errors
	if not versionCmdStr:
		return None

	r = re.compile(r'cmake version (\d+\.\d+\.\d+)')
	match = r.search(versionCmdStr)
  
	if not match:
		return None

	return match.group(1)

def getGXXVersion(gxxExecutable: str) -> Optional[str]:
	"""
	### Extracts g++'s version string from the PATH or the config file, the last one having a higher priority.

	#### Params:
	- dictPackages (dict): Optional. Dictionary containing packages found in the config file.

	#### Returns:
	- (str | None): g++'s version or None if there were any errors.
	"""
	# Return g++ version
	return __parseCompilerVersion(getPackageVersionCmd(gxxExecutable))

def getGCCVersion(gccExecutable: str) -> Optional[str]:
	"""
	### Extracts gcc's version string from the PATH or the config file, the last one having a higher priority.

	#### Params:
	- dictPackages (dict): Optional. Dictionary containing packages found in the config file.

	#### Returns:
	- (str | None): gcc's version or None if there were any errors.
	"""
	# Return gcc version
	return __parseCompilerVersion(getPackageVersionCmd(gccExecutable))
