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

# Xmipp's current version
#####################################
XMIPP_VERSION = '3.23.07.0'					#
XMIPP_VERNAME = 'devel'							#
RELEASE_DATE = '14/07/2023'					#
XMIPP_CORE_VERSION = '3.23.07.0'		#
XMIPP_VIZ_VERSION = '3.23.07.0'			#
XMIPP_PLUGIN_VERSION = '3.23.07.0'	#
#####################################

# Version requirements
#####################################
GCC_MINIMUM = '8.4'									#NEEDED? MAYBE ONLY G++, check with nvcc -- g++
GPP_MINIMUM = GCC_MINIMUM						#
CMAKE_MINIMUM = '3.16'							#
SCONS_MINIMUM = 'Unknown'						#TODO: Find out Scons minimum version
CUDA_MINIMUM = '10.2'								#
#####################################

# Mode list (alphabetical order)
MODE_ADD_MODEL = 'addModel'
MODE_ALL = 'all'
MODE_CHECK_CONFIG = 'checkConfig'
MODE_CLEAN_ALL = 'cleanAll'
MODE_CLEAN_BIN = 'cleanBin'
MODE_CLEAN_DEPRECATED = 'cleanDeprecated'
MODE_COMPILE_AND_INSTALL = 'compileAndInstall'
MODE_CONFIG = 'config'
MODE_GET_MODELS = 'getModels'
MODE_GIT = 'git'
MODE_TEST = 'test'
MODE_VERSION = 'version'
MODES = {
	MODE_ADD_MODEL: 'Takes a DeepLearning model from the modelPath, makes a tgz of it and uploads the .tgz according to the <login>',
	MODE_ALL: 'Default param. Runs config, and compileAndInstall.',
	MODE_CHECK_CONFIG: 'Cheks if the values in the config file are ok.',
	MODE_CLEAN_ALL: 'Removes all compiled binaries and sources, leaves the repository as if freshly cloned (without pulling).',
	MODE_CLEAN_BIN: 'Removes all compiled binaries.',
	MODE_CLEAN_DEPRECATED: 'Removes all deprecated binaries.',
	MODE_COMPILE_AND_INSTALL: 'Compiles and installs Xmipp based on already obtained sources.',
	MODE_CONFIG: 'Generates config file based on system information.',
	MODE_GET_MODELS: 'Download the DeepLearning Models at dir/models (./build/models by default)',
	MODE_GIT: 'Runs the given git action for all source repositories.',
	MODE_TEST: 'Runs a given test.',
	MODE_VERSION: 'Prints important version information.'
}

# Other variables
VALUE_UNKNOWN = 'Unkown'
DEFAULT_JOBS = 8