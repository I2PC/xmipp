# ***************************************************************************
# * Authors:		Martín Salinas (martin.salinas@cnb.csic.es)
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

import os
from typing import Optional

# Variable names
__SEND_INSTALLATION_STATISTICS = 'SEND_INSTALLATION_STATISTICS'
CMAKE = 'CMAKE'
CC = 'CMAKE_C_COMPILER'
CXX = 'CMAKE_CXX_COMPILER'
__CC_FLAGS = 'CMAKE_C_FLAGS'
__CXX_FLAGS = 'CMAKE_CXX_FLAGS'
CUDA = 'XMIPP_USE_CUDA'
CUDA_COMPILER = 'CMAKE_CUDA_COMPILER'
MPI = 'XMIPP_USE_MPI'
__PREFIX_PATH = 'CMAKE_PREFIX_PATH'
__MPI_HOME = 'MPI_HOME'
__PYTHON_HOME = 'Python3_ROOT_DIR'
__FFTW_HOME = 'FFTW_ROOT'
__TIFF_HOME = 'TIFF_ROOT'
__HDF5_HOME = 'HDF5_ROOT'
__JPEG_HOME = 'JPEG_ROOT'
__SQLITE_HOME = 'SQLite_ROOT'
__CUDA_CXX = 'CMAKE_CUDA_HOST_COMPILER'
MATLAB = 'XMIPP_USE_MATLAB'
LINK_SCIPION = 'XMIPP_LINK_TO_SCIPION'
__BUILD_TESTING = 'BUILD_TESTING'

# This is not used in cmake
__CONDA_PREFIX = 'CONDA_PREFIX'
__XMIPP_CUDA_BIN = 'XMIPP_CUDA_BIN'
__TUNE_FLAG='-mtune=native'

# Config file variable structure
TOGGLES = 'toggles'
LOCATIONS = 'locations'
COMPILATION_FLAGS = 'flags'
CONFIG_VARIABLES = {
	TOGGLES: [
		__SEND_INSTALLATION_STATISTICS, CUDA, MPI, MATLAB, LINK_SCIPION, __BUILD_TESTING
	],
	LOCATIONS: [
		CMAKE, CC, CXX, __PREFIX_PATH, __MPI_HOME, CUDA_COMPILER,
		__PYTHON_HOME, __FFTW_HOME, __TIFF_HOME, __HDF5_HOME, 
	 	__JPEG_HOME, __SQLITE_HOME, __CUDA_CXX
	],
	COMPILATION_FLAGS: [__CC_FLAGS, __CXX_FLAGS]
}

def __getNvccEnvVariable() -> Optional[str]:
	"""
	### This function returns the CUDA compiler for Scipion's enviroment if exists.

	#### Returns:
	- (str | None): Path to CUDA compiler for Scipion's env.
	"""
	cudaBin = os.environ.get(__XMIPP_CUDA_BIN)
	return cudaBin + '/nvcc' if cudaBin else None

def __getPrefixPath() -> Optional[str]:
	"""
	### This function returns the path for the current Conda enviroment.

	#### Returns:
	- (str | None): Path for current Conda enviroment.
	"""
	return os.environ.get(__CONDA_PREFIX)

ON = 'ON'
OFF = 'OFF'
CONFIG_DEFAULT_VALUES = {
	__SEND_INSTALLATION_STATISTICS: ON,
	CMAKE: None,
	CUDA: ON,
	MPI: ON,
	CC: None,
	CXX: None,
	__CC_FLAGS: __TUNE_FLAG,
	__CXX_FLAGS: __TUNE_FLAG,
	CUDA_COMPILER: __getNvccEnvVariable(),
	__PREFIX_PATH: __getPrefixPath(),
	__MPI_HOME: None,
	__PYTHON_HOME: None,
	__FFTW_HOME: None,
	__TIFF_HOME: None,
	__HDF5_HOME: None,
	__JPEG_HOME: None,
	__SQLITE_HOME: None,
	__CUDA_CXX: None,
	MATLAB: ON,
	LINK_SCIPION: ON,
	__BUILD_TESTING: ON
}

# Do not pass this variables to CMake, only for installer logic
INTERNAL_LOGIC_VARS = [__SEND_INSTALLATION_STATISTICS, CMAKE]
