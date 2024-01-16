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
os, architecture, cuda, cmake, gpp, gcc and scons.
"""

# General imports
from typing import Dict, Union
import os
# Installer imports
from .utils import (runJob, getPackageVersionCmd, getPythonPackageVersion,
										getCurrentBranch, isBranchUpToDate, MPIVersion, JAVAVersion,
										HDF5Version, TIFFVersion, FFTW3Version, opencvVersion)
from .constants import UNKNOWN_VALUE, CC, CXX, CMAKE, CUDA, CXX_FLAGS

def collectAllVersions(dictPackages: dict):
  return {'branch': getCurrentBranch(),
					'isUpdated': isBranchUpToDate(),
					'gccV': getGCCVersion(),
					'gppV': getGPPVersion(),
					'cudaV': getCUDAVersion(),
					'sconsV': getSconsVersion(),
					'cmakeV': getCmakeVersion(),
					'rsyncV': getRsyncVersion(),
					'mpiV': MPIVersion(getPackageVersionCmd(dictPackages['MPI_RUN'])),
					'javaV': JAVAVersion(getPackageVersionCmd('java')),
					'hdf5V': HDF5Version(dictPackages['HDF5_HOME']),
					'TIFFVn': TIFFVersion(dictPackages['TIFF_SO']),
					'FFTW3V': FFTW3Version(dictPackages['FFTW3_SO']),
					'opencvV': opencvVersion(dictPackages, CXX_FLAGS)}


####################### AUX FUNCTIONS #######################
def parseCompilerVersion(versionCmdStr: Union[str, None]) -> Union[str, None]:
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
	retCode, name = runJob('cat /etc/os-release')

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
	retCode, architecture = runJob('cat /sys/devices/cpu/caps/pmu_name')

	# If command worked and returned info, extract it
	if retCode == 0 and architecture:
		archName = architecture
	
	# Returing architecture name
	return archName

def getCUDAVersion(dictPackages: Dict=None) -> Union[str, None]:
	"""
	### Extracts the NVCC (NVIDIA CUDA Compiler) version from the PATH or the config file, the last one having a higher priority.

	#### Params:
	- dictPackages (dict): Optional. Dictionary containing packages found in the config file.

	#### Returns:
	- (str | None): CUDA version or None if there were any errors.
	"""
	# Initializing default version
	nvccVersion = None

	# If CUDA is set to False, don't fetch version
	if dictPackages is not None and CUDA in dictPackages and dictPackages[CUDA] == 'False':
		return None

	# Get the nvcc to extract
	nvccExecutable = dictPackages['CUDA_HOME'] if dictPackages is not None and CUDA in dictPackages else 'nvcc'

	# Extracting version command string
	versionCmdStr = getPackageVersionCmd(nvccExecutable)

	# Check if there were any errors
	if versionCmdStr is None:
		return None

	# Defining text around version number
	textBefore = 'release '
	textAfter = ','

	# Finding the text before the version to obtain its starting index
	textBeforeStart = versionCmdStr.find(textBefore)
	if textBeforeStart != -1:
		# Calculating location of version string start
		# if the text before was found
		versionStart = textBeforeStart + len(textBefore)

		# If exists, getting location of text after version
		versionEnd = versionCmdStr[versionStart:].find(textAfter)

		if versionEnd != -1 and versionStart != versionEnd:
			# If everything was found and string is not empty, extracting version
			nvccVersion = versionCmdStr[versionStart:versionStart + versionEnd]
	
	# Returning resulting version
	return nvccVersion

def getCmakeVersion(dictPackages: Dict=None) -> str:
	"""
	### Extracts the CMake version from the PATH or the config file, the last one having a higher priority.

	#### Params:
	- dictPackages (dict): Optional. Dictionary containing packages found in the config file.

	#### Returns:
	- (str | None): CMake version, or None if there were any errors.
	"""
	# Initializing default version
	cmakeVersion = None

	# Get the cmake to extract
	cmakeExecutable = dictPackages[CMAKE] if dictPackages is not None and CMAKE in dictPackages else 'cmake'

	# Extracting version command string
	versionCmdStr = getPackageVersionCmd(cmakeExecutable)

	# Version number is the last word of the first line of the output text
	if versionCmdStr is not None:
		# Only extract if command output string is not empty
		cmakeVersion = versionCmdStr.splitlines()[0].split()[-1]

	# Return cmake version
	return cmakeVersion

def getGPPVersion(dictPackages: Dict=None) -> Union[str, None]:
	"""
	### Extracts g++'s version string from the PATH or the config file, the last one having a higher priority.

	#### Params:
	- dictPackages (dict): Optional. Dictionary containing packages found in the config file.

	#### Returns:
	- (str | None): g++'s version or None if there were any errors.
	"""
	# Get the g++ to extract
	gppExecutable = dictPackages[CXX] if dictPackages is not None and CXX in dictPackages else 'g++'

	# Return g++ version
	return parseCompilerVersion(getPackageVersionCmd(gppExecutable))

def getGCCVersion(dictPackages: Dict=None) -> Union[str, None]:
	"""
	### Extracts gcc's version string from the PATH or the config file, the last one having a higher priority.

	#### Params:
	- dictPackages (dict): Optional. Dictionary containing packages found in the config file.

	#### Returns:
	- (str | None): gcc's version or None if there were any errors.
	"""
	
	# Get the gcc to extract
	gccExecutable = dictPackages[CC] if dictPackages is not None and CC in dictPackages else 'gcc'

	# Return gcc version
	return parseCompilerVersion(getPackageVersionCmd(gccExecutable))

def getSconsVersion() -> Union[str, None]:
	"""
	### Extracts scons's version string.

	#### Returns:
	- (str | None): scons's version or None if there were any errors.
	"""
	# Attepmt to get Scons version through python package
	version = getPythonPackageVersion('scons')

	# If Scons is not installed via Python, try standalone method
	if version is None:
		version = getPackageVersionCmd('scons')

		if version is not None:
			# Defining text before version number
			textBefore = 'SCons: v'

			# Searching for text before version number
			textBeforeStart = version.find(textBefore)

			if textBeforeStart != -1:
				# If text was found, we need to get the first 3 numbers
				versionStart = textBeforeStart + len(textBefore)
				numbers = version[versionStart:].splitlines()[0].split('.')

				# Only extract macro, minor, and micro version numbers
				if len(numbers) >= 3:
					# Make sure last number stops when it shoulds
					micro = numbers[2].split(',')[0]
					version = f'{numbers[0]}.{numbers[1]}.{micro}'
				
	# Returning extracted version
	return version


def getRsyncVersion():
	"""
	### Extracts rsync's version string.

	#### Returns:
	- (str | None): rsync's version or None if there were any errors.
	"""
	version = getPackageVersionCmd('rsync')
	if version is not None:
			textBefore = 'rsync  version '
			textBeforeStart = version.find(textBefore)
			if textBeforeStart != -1:
				versionStart = textBeforeStart + len(textBefore)
				numbers = version[versionStart:].splitlines()[0].split('.')
				if len(numbers) >= 2:
					version = f'{numbers[0]}.{numbers[1]}'
	return version