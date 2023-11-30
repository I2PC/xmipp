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

def osVersion():
		out = runJob('cat /etc/os-release', showCommand=False)
		strTarget = 'PRETTY_NAME="'
		idx = out[1].find(strTarget)
		osV = None
		if idx != -1:
				idx2 = out[1][idx:].find('"\n')
				osV = out[1][len(strTarget):idx2]
		return osV

def architectureVersion():
		architectureV = None
		out = runJob('cat /sys/devices/cpu/caps/pmu_name')
		if out[0] == 0:
				architectureV = out[1]
		return architectureV

def CUDAVersion(dictPackages):
		"""
		Extracts the NVCC (NVIDIA CUDA Compiler) version information from a given string.

		Params:
		- strVersion (str): Input string containing CUDA version details.

		Returns:
		- str: Extracted NVCC version information.
		"""
		strversion = versionPackage(dictPackages['CUDA_HOME'])
		nvccVersion = None
		if strversion.find('release') != -1:
				idx = strversion.find('release ')
				nvccVersion = strversion[idx + len('release '):
																 idx + strversion[idx:].find(',')]
		return nvccVersion

def cmakeVersion():
		# Getting CMake version
		cmakVersion = None
		cmakeV = runJob('cmake --version')
		if cmakeV[0] == 0:
			cmakVersion = cmakeV[1].split('\n')[0].split()[-1]
		return cmakVersion

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

