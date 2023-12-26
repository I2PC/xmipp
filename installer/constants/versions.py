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
from .main import XMIPP, XMIPP_CORE, XMIPP_VIZ, XMIPP_PLUGIN


# Xmipp's current versions

#####################################
DEVEL_BRANCHNAME = 'devel'				  #
MASTER_BRANCHNAME = 'master'				#

VERSION_KEY = 'version'							#
VERNAME_KEY = 'vername'							#
XMIPP_VERSIONS = {									#
	XMIPP: {													#
		VERSION_KEY: '3.23.11.0',				#
		VERNAME_KEY: 'v3.23.11-Nereus'  #
	},																#
	XMIPP_CORE: {											#
		VERSION_KEY: '3.23.11.0',				#
		VERNAME_KEY: 'v3.23.11-Nereus'  #
	},																#
	XMIPP_VIZ: {											#
		VERSION_KEY: '3.23.11.0',				#
		VERNAME_KEY: 'v3.23.11-Nereus'  #
	},																#
	XMIPP_PLUGIN: {										#
		VERSION_KEY: '3.23.11.0',				#
		VERNAME_KEY: 'v3.23.11-Nereus'  #
	}																	#
}
#####################################

# Supported gcc versions
vGCC = [
	'12.3', '12.2', '12.1',
	'11.3', '11.2', '11.1', '11',
	'10.5', '10.4', '10.3', '10.2', '10.1', '10',
	'9.4', '9.3', '9.2', '9.1', '9',
	'8.5', '8.4'
]

# Version requirements
#####################################
GCC_MINIMUM = vGCC[-1]							#TODO: NEEDED? MAYBE ONLY G++, check with nvcc -- g++
GPP_MINIMUM = GCC_MINIMUM						#
CMAKE_MINIMUM = '3.16'							#
SCONS_MINIMUM = '4.0'								#
CUDA_MINIMUM = '10.1'								#
MPI_MINIMUM = '3.0'									#
PYTHON_MINIMUM = '3.0'							#
NUMPY_MINIMUM = '1.21'							#
HDF5_MINIMUM = '1.10'								#
FFTW_MINIMUM = '3.0'								#
GIT_MINIMUM = '2.0'									#
RSYNC_MINIMUM  = '3.0'
#####################################

# CUDA-gcc compatibility table
CUDA_GCC_COMPATIBILITY = {
	'10.1-10.2': vGCC[vGCC.index('8.5'):],
	'11.0-11.0': vGCC[vGCC.index('9.4'):],
	'11.1-11.3': vGCC[vGCC.index('10.5'):],
	'11.4-11.8': vGCC[vGCC.index('11.3'):],
	'12.0-12.3': vGCC[vGCC.index('12.3'):]
}

# Config variable names
CC = 'CC'
CXX = 'CXX'
MPI_CC = 'MPI_CC'
MPI_CXX = 'MPI_CXX'
MPI_RUN = 'MPI_RUN'
JAVA = 'JAVA'
MATLAB = 'MATLAB'
OPENCV = 'OPENCV'
CUDA = 'CUDA'
STARPU = 'STARPU'
HDF5 = 'HDF5'
SCONS = 'SCons'
CMAKE = 'CMake'
TIFF = 'TIFF'

# Other variables
UNKNOWN_VALUE = 'Unknown'
