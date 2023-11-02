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

# Version variables
#	----K-E-E-P----U-P-D-A-T-E-D---- #
#####################################
XMIPP_VERSION = '3.23.07.0'        	#
XMIPP_VERNAME = 'devel'							#
RELEASE_DATE = '14/07/2023'        	#
CMAKE_VERSION_REQUIRED = '3.16'			#
#####################################

# Mode list
MODE_CLEAN_ALL = 'cleanAll'
MODE_CLEAN_BIN = 'cleanBin'
MODE_CLEAN_DEPRECATED = 'cleanDeprecated'
MODE_VERSION = 'version'
MODE_CONFIG = 'config'
MODE_CHECK_CONFIG = 'checkConfig'
MODE_COMPILE_AND_INSTALL = 'compileAndInstall'
MODE_GET_MODELS = 'getModels'
MODE_TEST = 'test'
MODE_ALL = 'all'
MODE_GIT = 'git'
MODE_ADD_MODEL = 'addModel'
MODES = {
	MODE_CLEAN_ALL: 'Removes all compiled binaries and sources, leaves the repository as if freshly cloned (without pulling).',
	MODE_CLEAN_BIN: 'Removes all compiled binaries.',
	MODE_CLEAN_DEPRECATED: 'Removes all deprecated binaries.',
	MODE_VERSION: 'Prints important version information.',
	MODE_CONFIG: 'Generates config file based on system information.',
	MODE_CHECK_CONFIG: 'Cheks if the values in the config file are ok.',
	MODE_COMPILE_AND_INSTALL: 'Compiles and installs Xmipp based on already obtained sources.',
	MODE_GET_MODELS: 'Shows available models.',
	MODE_TEST: 'Runs a given test.',
	MODE_ALL: 'Default param. Runs config, and compileAndInstall.',
	MODE_GIT: 'Runs the given git action for all source repositories.',
	MODE_ADD_MODEL: 'Adds a model specified by the user.'
}

# Other variables
DEFAULT_JOBS = 8