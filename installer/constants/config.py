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
SEND_INSTALLATION_STATISTICS = 'SEND_INSTALLATION_STATISTICS'
GCC_HOME = 'CMAKE_C_COMPILER'
GXX_HOME = 'CMAKE_CXX_COMPILER'
CUDA = 'XMIPP_USE_CUDA'
MPI = 'XMIPP_USE_MPI'
MPI_HOME = 'MPI_HOME'
PYTHON_HOME = 'Python3_ROOT_DIR'
FFTW_HOME = 'FFTW_ROOT'
TIFF_HOME = 'TIFF_ROOT'
HDF5_HOME = 'HDF5_ROOT'
JPEG_HOME = 'JPEG_ROOT'
SQLITE_HOME = 'SQLite_ROOT'

# Config file variable structure
TOGGLES = 'toggles'
LOCATIONS = 'locations'
CONFIG_VARIABLES = {
	TOGGLES: [
		SEND_INSTALLATION_STATISTICS, CUDA, MPI
	],
	LOCATIONS: [
		GCC_HOME, GXX_HOME, MPI_HOME, PYTHON_HOME, FFTW_HOME,
		TIFF_HOME, HDF5_HOME, JPEG_HOME, SQLITE_HOME
	]
}

ON = 'ON'
OFF = 'OFF'
CONFIG_DEFAULT_VALUES = {
	SEND_INSTALLATION_STATISTICS: ON,
	CUDA: ON,
	MPI: ON,
	GCC_HOME: None,
	GXX_HOME: None,
	PYTHON_HOME: None,
	MPI_HOME: None,
	FFTW_HOME: None,
	TIFF_HOME: None,
	HDF5_HOME: None,
	JPEG_HOME: None,
	SQLITE_HOME: None
}

# Do not pass this variables to CMake, only for installer logic
INTERNAL_LOGIC_VARS = [SEND_INSTALLATION_STATISTICS]
