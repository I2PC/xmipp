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

from .utils import runJob, versionPackage
from .constants import UNKNOWN_VALUE

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

def getCUDAVersion(dictPackages=None) -> str:
	"""
	### Extracts the NVCC (NVIDIA CUDA Compiler) version.

	#### Returns:
	- (str): CUDA version.
	"""
	# Initializing default version
	nvccVersion = None

	# Extracting version command string
	versionCmdStr = versionPackage('nvcc')

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
	- (str): CMake version.
	"""
	# Initializing default version
	cmakeVersion = None

	# Extracting version command string
	versionCmdStr = versionPackage('cmake')

	# Version number is the last word of the first line of the output text
	if versionCmdStr:
		# Only extract if command output string is not empty
		cmakeVersion = versionCmdStr.split('\n')[0].split()[-1]

	# Return cmake version
	return cmakeVersion

def parsingCompilerVersion(str):
		idx = str.find('\n')
		idx2 = str[:idx].rfind(' ')
		version = str[idx2:idx]
		gxx_version = version.replace(' ', '')
		idx = gxx_version.rfind('.')
		gxx_version = gxx_version[:idx]
		return gxx_version

def gppVersion(dictPackages):
		strVersion = versionPackage(dictPackages['CXX'])
		print(strVersion)
		return parsingCompilerVersion(strVersion)

def gccVersion(dictPackages):
		strVersion = versionPackage(dictPackages['CC'])
		return parsingCompilerVersion(strVersion)

def sconsVersion():
		strVersion = versionPackage('scons')
		idx = strVersion.find('SCons: v')
		sconsV = None
		if idx != -1:
			idx2 = strVersion[idx:].find(', ')
			version = strVersion[idx + len('SCons: v'):idx + idx2].split('.')
			sconsV = '.'.join(version[:3])
		return sconsV
