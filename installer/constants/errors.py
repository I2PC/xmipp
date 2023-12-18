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
HDF5_ERROR = 18
MPI_COMPILLATION_ERROR = 19
MPI_RUNNING_ERROR = 20
JAVAC_DOESNT_WORK_ERROR = 21
JAVA_INCLUDE_ERROR = 22
CMAKE_VERSION_ERROR = 23
CMAKE_ERROR = 24
NETWORK_ERROR = 25
IO_ERROR = 26
HDF5_VERSION_ERROR = 27
TIFF_ERROR = 28
FFTW3_ERROR = 29
TIFF_H_ERROR = 30
FFTW3_H_ERROR = 31
FFTW3_VERSION_ERROR = 32
CLONNING_EXTERNAL_SOURCE_ERROR = 33
CLONNING_XMIPP_SOURCE_ERROR = 34
DOWNLOADING_XMIPP_SOURCE_ERROR = 35
GIT_VERSION_ERROR = 36


# Warning codes
MATLAB_WARNING = 1
MATLAB_HOME_WARNING = 2
CUDA_VERSION_WARNING = 3
CUDA_WARNING = 4
OPENCV_WARNING = 5
OPENCV_CUDA_WARNING = 6
STARPU_INCLUDE_WARNING = 7
STARPU_LIB_WARNING = 8
STARPU_LIBRARY_WARNING = 9
STARPU_RUN_WARNING = 10
STARPU_CUDA_WARNING = 11

# Error messages
#TODO review the messages spelling, maybe mor links to the documentation?
ERROR_CODE = {
	UNKOW_ERROR: ['Unkonw error.', ''],
	SCONS_VERSION_ERROR: ['scons installation  error.', 'We tried to install it on your scipion enviroment but was not posible, please install it manually.'],
	SCONS_ERROR: ['scons not found.', 'We didnt find the scipion enviroment, please install scons manually on your conda env or in your system.'],
	GCC_VERSION_ERROR: ['gcc version not valid.', 'The version of gcc is lower than minimum, please review the requirements.'],
  CC_NO_EXIST_ERROR: ['CC package does not exist.','Please review the CC flag on your xmipp.conf.'],
  CXX_NO_EXIST_ERROR: ['CXX package does not exist.', 'Please review the CXX flag on your xmipp.conf.'],
  CXX_VERSION_ERROR: ['g++ version not valid.', 'The version of g++ is lower than minimum, please review the requirements.'],
	MPI_VERSION_ERROR: ['mpi version not valid.', 'The version of mpi is lower than minimum, please review the requirements.'],
	MPI_NOT_FOUND_ERROR: ['mpi package does not exist.', 'Please review the MPI_RUN flag on your xmipp.conf.'],
  PYTHON_VERSION_ERROR: ['python version not valid.', 'The version of python is lower than minimum, please review the requirements.'],
  PYTHON_NOT_FOUND_ERROR: ['python not found.', 'Please install python on your system.'],
  NUMPY_NOT_FOUND_ERROR: ['numpy not found.', 'Please install numpy.'],
  JAVA_HOME_PATH_ERROR: ['JAVA_HOME path with errors or bad installed.', 'bin/jar, bin/javac or include not found on the JAVA_HOME path but required.'],
  HDF5_ERROR: ['hdf5 libs does not work.', 'Please review the LIBDIRFLAGS flag on xmipp.conf.'],
  MPI_COMPILLATION_ERROR: ['', '.'],
	MPI_RUNNING_ERROR: ['mpirun or mpiexec can not run several process in parallel.', 'Please, review the mpi installation, if you are running a virtual machine, please allow several processors not just one.'],
	JAVAC_DOESNT_WORK_ERROR: ['JAVAC does not work.', 'Check the JAVA_HOME flag on xmipp.conf.'],
	JAVA_INCLUDE_ERROR: ['JAVA fails. jni include fails.','Check the JNI_CPPPATH, CXX and INCDIRFLAGS.'],
	CMAKE_VERSION_ERROR: ['.', f'Please update your CMake version by following the instructions at {cmakeInstallURL}\033[0m'],
	CMAKE_ERROR: [f'Please install your CMake version by following the instructions at {cmakeInstallURL}\033[0m'],
	NETWORK_ERROR: ['There was a network error running a command.', ''],
	IO_ERROR: ['Input/output error.', 'This error can be caused by the installer not being able to read/write/create/delete a file. Check your permissions on this directory.'],
  TIFF_ERROR: ['TIFF library was not found on your system.', 'Please install it or add the path in the flag TIFF_HOME of the xmipp.conf file.'],
  FFTW3_ERROR: ['FFTW3 library was not found on your system.','Please install it or add the path in the flag FFTW3_HOME of the xmipp.conf file.'],
	TIFF_H_ERROR: ['tiffio.h header file was not found.', 'Please install the package.'],
	FFTW3_H_ERROR: ['fftw3.h header file was not found.', 'Please install the package.'],
	FFTW3_VERSION_ERROR: ['fftw version not valid.', 'The version is minor than require, please update it.'],
  CLONNING_EXTERNAL_SOURCE_ERROR: ['Error cloning external repository with git.', 'Please review the internet connection and the git package.'],
	CLONNING_XMIPP_SOURCE_ERROR: ['Error cloning xmipp repository with git.', 'Please review the internet connection and the git package.'],
	DOWNLOADING_XMIPP_SOURCE_ERROR:['Error downloading (wget) xmipp repository.', 'Please review the internet connection.'],
	GIT_VERSION_ERROR: [],


}

# Warning messages
WARNING_CODE = {
	MATLAB_WARNING: ['Matlab not found on system.', 'Please install matlab or set MATLAB as False on the xmipp.conf file.'],
  MATLAB_HOME_WARNING: ['MATLAB_HOME path not found.', 'Please review the MATLAB_HOME path.MATLAB flag set to False on the xmipp.conf file.'],
	CUDA_VERSION_WARNING: ['CUDA version not compatible with your g++ compiler.', 'Please update CUDA or update the compiler. CUDA flag set to False on xmipp.conf.'],
	CUDA_WARNING: ['CUDA not found.', 'Please review the CUDA_HOME flag on your xmipp.conf file.'],
	OPENCV_WARNING: ['OpenCV does not work.',
									 'OPENCV flag was set to False and will not be used inside Xmipp.',
									 'Please review your Opencv installation.'],
  OPENCV_CUDA_WARNING: ['OpenCV CUDA support does not work.',
													'OPENCVSUPPORTSCUDA flag set to False and will not be used inside Xmipp.',
													'Please review your Opencv installation.'],
	STARPU_CUDA_WARNING: ['CUDA must be enabled together with STARPU.', 'Set STARPU flag to False on xmipp.conf'],
	STARPU_INCLUDE_WARNING: ['', 'Set STARPU flag to False on xmipp.conf'],
	STARPU_LIB_WARNING: ['', 'Set STARPU flag to False on xmipp.conf'],
	STARPU_LIBRARY_WARNING: ["STARPU_LIBRARY must be specified (link library name).", 'Set STARPU flag to False on xmipp.conf'],
	STARPU_RUN_WARNING: ["Check STARPU_* settings.", 'Set STARPU flag to False on xmipp.conf'],

}
