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
		CUDA, MPI
	],
	LOCATIONS: [
		GCC_HOME, GXX_HOME, MPI_HOME, PYTHON_HOME, FFTW_HOME,
		TIFF_HOME, HDF5_HOME, JPEG_HOME, SQLITE_HOME
	]
}

ON = 'ON'
OFF = 'OFF'
CONFIG_DEFAULT_VALUES = {
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

"""
[-D XMIPP_USE_CUDA=ON/OFF] # Por defecto ON
[-D XMIPP_USE_MPI=ON/OFF] # Por defecto ON
[-D CMAKE_C_COMPILER=/path/to/gcc]
[-D CMAKE_CXX_COMPILER=/path/to/g++]
[-D Python3_ROOT_DIR=/path/to/python/root] # No al ejecutable
[-D MPI_HOME=/path/to/mpi/root] 
[-D FFTW_ROOT=/path/to/fftw/root]
[-D TIFF_ROOT=/path/to/tiff/root]
[-D HDF5_ROOT=/path/to/hdf5/root]
[-D JPEG_ROOT=/path/to/jpeg/root]
[-D SQLite_ROOT=/path/to/sqlite/root]
"""
