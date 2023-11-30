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
from .__init__ import cmakeInstallURL
"""
Submodule containing all constants needed for handling errors during Xmipp's installation.
"""

# Error codes
OK = 0
UNKOW_ERROR = 1
SCONS_VERSION_ERROR = 2
SCONS_ERROR = 3
GCC_VERSION_ERROR = 4
CC_NO_EXIST_ERROR = 5
CXX_NO_EXIST_ERROR = 6
CXX_VERSION_ERROR = 7
MPI_VERSION_ERROR = 8
MPI_NOT_FOUND_ERROR = 9
PYTHON_VERSION_ERROR = 10
PYTHON_NOT_FOUND_ERROR = 11
NUMPY_NOT_FOUND_ERROR = 12
JAVA_HOME_PATH_ERROR = 13
MATLAB_ERROR = 14
MATLAB_HOME_ERROR = 15
CUDA_VERSION_ERROR = 16
CUDA_ERROR = 17
HDF5_ERROR = 18
MPI_COMPILLATION_ERROR = 19
MPI_RUNNING_ERROR = 20
JAVAC_DOESNT_WORK_ERROR = 21
JAVA_INCLUDE_ERROR = 22
CMAKE_VERSION_ERROR = 23
CMAKE_ERROR = 24
NETWORK_ERROR = 25
IO_ERROR = 26

# Error messages
ERROR_CODE = {
	UNKOW_ERROR: ['Unkonw error', ''],
	SCONS_VERSION_ERROR: ['scons installation  error', 'We tried to install it on your scipion enviroment but was not posible, please install it manually'],
	SCONS_ERROR: ['scons not found', 'We didnt find the scipion enviroment, please install scons manually on your conda env or in your system'],
	GCC_VERSION_ERROR: ['gcc version not valid', 'The version of gcc is lower than minimum, please review the requirements'],
  CC_NO_EXIST_ERROR: ['CC package does not exist','Please review the CC flag on your xmipp.conf'],
  CXX_NO_EXIST_ERROR: ['CXX package does not exist', 'Please review the CXX flag on your xmipp.conf'],
  CXX_VERSION_ERROR: ['g++ version not valid', 'The version of g++ is lower than minimum, please review the requirements'],
	MPI_VERSION_ERROR: ['mpi version not valid', 'The version of mpi is lower than minimum, please review the requirements'],
	MPI_NOT_FOUND_ERROR: ['mpi package does not exist', 'Please review the MPI_RUN flag on your xmipp.conf'],
  PYTHON_VERSION_ERROR: ['python version not valid', 'The version of python is lower than minimum, please review the requirements'],
  PYTHON_NOT_FOUND_ERROR: ['python not found', 'Please install python on your system'],
  NUMPY_NOT_FOUND_ERROR: ['numpy not found', 'Please install numpy'],
  JAVA_HOME_PATH_ERROR: ['JAVA_HOME path with errors or bad installed', 'bin/jar, bin/javac or include not found but required'],
  MATLAB_ERROR: ['Matlab not found on system', 'Please install matlab or set MATLAB as False on the xmipp.conf file'],
  MATLAB_HOME_ERROR: ['MATLAB_HOME path not found', 'Please review the MATLAB_HOME path or set MATLA as False on the xmipp.conf file'],
	CUDA_VERSION_ERROR: ['CUDA version not compatible with your g++ compiler', 'Please update CUDA or update the compiler or set the CUDA flag on the xmipp.conf to False'],
	CUDA_ERROR: ['CUDA not found', 'Please review the CUDA_HOME flag on your xmipp.conf file'],
  HDF5_ERROR: ['hdf5 libs does not work', 'Please review the LIBDIRFLAGS flag on xmipp.conf'],
  MPI_COMPILLATION_ERROR: ['', ''],
	MPI_RUNNING_ERROR: ['mpirun or mpiexec can not run several process in parallel', 'Please, review the mpi installation, if you are running a virtual machine, please allow several processors not just one'],
	JAVAC_DOESNT_WORK_ERROR: ['JAVAC does not work', 'Check the JAVA_HOME flag on xmipp.conf'],
	JAVA_INCLUDE_ERROR: ['JAVA fails. jni include fails','Check the JNI_CPPPATH, CXX and INCDIRFLAGS'],
	CMAKE_VERSION_ERROR: ['', f'Please update your CMake version by following the instructions at {cmakeInstallURL}\033[0m'],
	CMAKE_ERROR: [f'Please install your CMake version by following the instructions at {cmakeInstallURL}\033[0m'],
	NETWORK_ERROR: ['There was a network error running a command.', ''],
	IO_ERROR: ['Input/output error.', 'This error can be caused by the installer not being able to read/write/create/delete a file. Check your permissions on this directory.']
}
