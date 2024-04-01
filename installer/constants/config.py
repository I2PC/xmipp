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

# Variable names
__SEND_INSTALLATION_STATISTICS = 'SEND_INSTALLATION_STATISTICS'
CMAKE_HOME = 'CMAKE_HOME'
GCC_HOME = 'CMAKE_C_COMPILER'
GXX_HOME = 'CMAKE_CXX_COMPILER'
__CUDA = 'XMIPP_USE_CUDA'
CUDA_HOME = 'CUDAToolkit_ROOT'
__MPI = 'XMIPP_USE_MPI'
__MPI_HOME = 'MPI_HOME'
__PYTHON_HOME = 'Python3_ROOT_DIR'
__FFTW_HOME = 'FFTW_ROOT'
__TIFF_HOME = 'TIFF_ROOT'
__HDF5_HOME = 'HDF5_ROOT'
__JPEG_HOME = 'JPEG_ROOT'
__SQLITE_HOME = 'SQLite_ROOT'
__CUDA_CXX_HOME = 'CMAKE_CUDA_HOST_COMPILER'

# Config file variable structure
TOGGLES = 'toggles'
LOCATIONS = 'locations'
CONFIG_VARIABLES = {
	TOGGLES: [
		__SEND_INSTALLATION_STATISTICS, __CUDA, __MPI
	],
	LOCATIONS: [
		CMAKE_HOME, GCC_HOME, GXX_HOME, __MPI_HOME, CUDA_HOME, __PYTHON_HOME,
		__FFTW_HOME, __TIFF_HOME, __HDF5_HOME, __JPEG_HOME, __SQLITE_HOME, __CUDA_CXX_HOME
	]
}

ON = 'ON'
OFF = 'OFF'
CONFIG_DEFAULT_VALUES = {
	__SEND_INSTALLATION_STATISTICS: ON,
	CMAKE_HOME: None,
	__CUDA: ON,
	__MPI: ON,
	GCC_HOME: None,
	GXX_HOME: None,
	CUDA_HOME: None,
	__MPI_HOME: None,
	__PYTHON_HOME: None,
	__FFTW_HOME: None,
	__TIFF_HOME: None,
	__HDF5_HOME: None,
	__JPEG_HOME: None,
	__SQLITE_HOME: None,
	__CUDA_CXX_HOME: None
}

# Do not pass this variables to CMake, only for installer logic
INTERNAL_LOGIC_VARS = [__SEND_INSTALLATION_STATISTICS, CMAKE_HOME]
