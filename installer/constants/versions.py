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

"""
Submodule containing all version info required for Xmipp's installation process.
"""

# Source names
XMIPP = 'xmipp'
XMIPP_CORE = 'xmippCore'
XMIPP_VIZ = 'xmippViz'
XMIPP_PLUGIN = 'scipion-em-xmipp'

# Xmipp's current versions
#####################################
VERSION_KEY = 'version'							#
VERNAME_KEY = 'vername'							#
XMIPP_VERSIONS = {									#
	XMIPP: {													#
		VERSION_KEY: '3.23.07.0',				#
		VERNAME_KEY: 'v3.23.07-Morpheus'#
	},																#
	XMIPP_CORE: {											#
		VERSION_KEY: '3.23.07.0',				#
		VERNAME_KEY: 'v3.23.07-Morpheus'#
	},																#
	XMIPP_VIZ: {											#
		VERSION_KEY: '3.23.07.0',				#
		VERNAME_KEY: 'v3.23.07-Morpheus'#
	},																#
	XMIPP_PLUGIN: {										#
		VERSION_KEY: '3.23.07.0',				#
		VERNAME_KEY: 'v3.23.07-Morpheus'#
	}																	#
}																		#
DEVEL_BRANCHNAME = 'devel'					#
MASTER_BRANCHNAME = 'master'				#
#####################################

# Version requirements
#####################################
GCC_MINIMUM = '8.4'									#TODO: NEEDED? MAYBE ONLY G++, check with nvcc -- g++
GPP_MINIMUM = GCC_MINIMUM						#
CMAKE_MINIMUM = '3.16'							#
SCONS_MINIMUM = '3.0'								#
CUDA_MINIMUM = '10.2'								#
MPI_MINIMUM = '2.0'									#
PYTHON_MINIMUM = '3.0'							#
NUMPY_MINIMUM = '1.21'							#
MINIMUM_CUDA_VERSION = '10.1'				#
#####################################

# Supported gcc versions
vGCC = [
	'12.3', '12.2', '12.1',
	'11.3', '11.2', '11.1', '11',
	'10.5', '10.4', '10.3', '10.2', '10.1', '10',
	'9.4', '9.3', '9.2', '9.1', '9',
	'8.5', '8.4', '8.3', '8.2', '8.1', '8'
]

# CUDA-gcc compatibility table
CUDA_GCC_COMPATIBILITY = {
	'10.1-10.2': vGCC[vGCC.index('8.5'):],
	'11.0-11.0': vGCC[vGCC.index('9.4'):],
	'11.1-11.3': vGCC[vGCC.index('10.5'):],
	'11.4-11.8': vGCC[vGCC.index('11.3'):],
	'12.0-12.3': vGCC[vGCC.index('12.3'):]
}

# Other variables
UNKNOWN_VALUE = 'Unknown'
