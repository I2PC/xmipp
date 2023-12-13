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

# Installer imports
from .utils import runJob
from .constants import UNKNOWN_VALUE

####################### GENERAL FUNCTIONS #######################
def getPackageVersionCmd(packageName: str) -> Union[str, None]:
	"""
	### Retrieves the version of a package or program by executing '[packageName] --version' command.

	Params:
	- packageName (str): Name of the package or program.

	Returns:
	- (str | None): Version information of the package or None if not found or errors happened.
	"""
	# Running command
	retCode, output = runJob(f'{packageName} --version', showError=True)

	# Check result if there were no errors
	return output if retCode == 0 else None

def getPythonPackageVersion(packageName: str) -> Union[str, None]:
	"""
	### Retrieves the version of a Python package.

	Params:
	- packageName (str): Name of the Python package.

	Returns:
	- (str | None): Version string of the Python package or None if not found or errors happened.
	"""
	# Running command
	retCode, output = runJob(f'pip show {packageName}')

	# Extract variable if there were no errors
	if retCode == 0 and output:
		# Split output into lines and select the one which starts with 'Version:'
		output = output.splitlines()

		for line in output:
			if line.startswith('Version'):
				# If that line was found, return last word of such line
				return line.split()[-1]

####################### SPECIFIC FUNCTIONS #######################
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
	### Extracts the NVCC (NVIDIA CUDA Compiler) version.

	#### Returns:
	- (str | None): CUDA version or None if there were any errors.
	"""
	# Initializing default version
	nvccVersion = None

	# Extracting version command string
	versionCmdStr = getPackageVersionCmd('nvcc')

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

def cmakeVersion() -> str:
	"""
	### Extracts the CMake version.

	#### Returns:
	- (str | None): CMake version, or None if there were any errors.
	"""
	# Initializing default version
	cmakeVersion = None

	# Extracting version command string
	versionCmdStr = getPackageVersionCmd('cmake')

	# Version number is the last word of the first line of the output text
	if versionCmdStr is not None:
		# Only extract if command output string is not empty
		cmakeVersion = versionCmdStr.splitlines()[0].split()[-1]

	# Return cmake version
	return cmakeVersion

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

def gppVersion(dictPackages: Dict) -> Union[str, None]:
	"""
	### Extracts g++'s version string.

	#### Params:
	- dictPackages (dict): Dictionary containing all found packages.

	#### Returns:
	- (str | None): g++'s version or None if there were any errors.
	"""
	return parseCompilerVersion(getPackageVersionCmd(dictPackages['CXX']))

def gccVersion(dictPackages: Dict) -> Union[str, None]:
	"""
	### Extracts gcc's version string.

	#### Params:
	- dictPackages (dict): Dictionary containing all found packages.

	#### Returns:
	- (str | None): gcc's version or None if there were any errors.
	"""
	return parseCompilerVersion(getPackageVersionCmd(dictPackages['CC']))

def sconsVersion():
	"""
	### Extracts scons's version string.

	#### Returns:
	- (str | None): scons's version or None if there were any errors.
	"""
	return getPythonPackageVersion('scons')
