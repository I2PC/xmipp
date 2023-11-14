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

"""
Module containing all constants needed for the installation of Xmipp.
"""

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
SCONS_MINIMUM = '3.0'								#
CUDA_MINIMUM = '10.2'								#
MPI_MINIMUM = '2.0'
PYTHON_MINIMUM = '3.0'
NUMPY_MINIMUM = '1.21'
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
	'General': {
		MODE_VERSION: 'Returns the version information. Add \'--short\' to print only the version number.',
		MODE_COMPILE_AND_INSTALL: 'Compiles and installs Xmipp based on already obtained sources.',
		MODE_ALL: 'Default param. Runs config, and compileAndInstall.'
	},
	'Config': {
		MODE_CONFIG: 'Generates config file based on system information.',
		MODE_CHECK_CONFIG: 'Cheks if the values in the config file are ok.'
	},
	'Downloads': {
		MODE_GET_MODELS: 'Download the DeepLearning Models at dir/models (./build/models by default).'
	},
	'Clean': {
		MODE_CLEAN_BIN: 'Removes all compiled binaries.',
		MODE_CLEAN_DEPRECATED: 'Removes all deprecated binaries from src/xmipp/bin.',
		MODE_CLEAN_ALL: 'Removes all compiled binaries and sources, leaves the repository as if freshly cloned (without pulling).'
	},
	'Test': {
		MODE_TEST: 'Runs a given test.'
	},
	'Developers': {
		MODE_GIT: 'Runs the given git action for all source repositories.',
		MODE_ADD_MODEL: 'Takes a DeepLearning model from the modelPath, makes a tgz of it and uploads the .tgz according to the <login>\naddModel login modelPath\nlogin = user@server'
	}
}

# Other variables
VALUE_UNKNOWN = 'Unkown'
DEFAULT_JOBS = 8
COMMON_USAGE_HELP_MESSAGE = 'Run \"./xmipp -h\" for usage help.'

# Files names
CONFIG_FILE = 'xmipp.conf'


# Error Code
ERROR_CODE = {
	0: ['No error', ''],
	1: ['No error', ''],
	2: ['scons not found', 'We tried to install it on your scipion enviroment but was not posible, please install it manually'],
	3: ['scons not found', 'We didnt find the scipion enviroment, please install scons manually on your conda env or in your system'],
	4: ['gcc version not valid', 'The version of gcc is lower than minimum, please review the requirements'],
  5: ['CC package does not exist','Please review the CC flag on your xmipp.conf'],
  6: ['CXX package does not exist', 'Please review the CXX flag on your xmipp.conf'],
  7: ['g++ version not valid', 'The version of g++ is lower than minimum, please review the requirements'],
	8: ['mpi version not valid', 'The version of mpi is lower than minimum, please review the requirements'],
	9: ['mpi package does not exist', 'Please review the MPI_RUN flag on your xmipp.conf'],
  10: ['python version not valid', 'The version of python is lower than minimum, please review the requirements'],
  11: ['python not found', 'Please install python on your system'],
  12: ['numpy not found', 'Please install numpy'],
  13: ['java not found on the system', 'Please install java'],
  14: ['JAVA_HOME path with errors', 'bin/jar, bin/javac or include not found but required'],
  15: ['Matlab not found on system', 'Please install matlab or set MATLA as False on the xmipp.conf file'],
  16: ['MATLAB_HOME path not found', 'Please review the MATLAB_HOME path or set MATLA as False on the xmipp.conf file'],
	17: ['', ''],
	18: ['', ''],
}


'''
CONFIG CONSTANTS:
DEBUG=False

CC=gcc
CXX=g++

INCDIRFLAGS=-I../ -I/home/agarcia/anaconda3/include -I/usr/include/opencv4
LIBDIRFLAGS=-L/home/agarcia/anaconda3/lib

JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

MPI_CC=mpicc
MPI_CXX=mpicxx
MPI_LINKERFORPROGRAMS=mpicxx
MPI_RUN=mpirun

MATLAB=False
MATLAB_DIR=

CUDA=True
CXX_CUDA=/usr/bin/g++-10
CUDA_HOME=

OPENCV=False
OPENCVSUPPORTSCUDA=False

STARPU=False
STARPU_HOME=
STARPU_INCLUDE=/include/starpu/1.3
STARPU_LIB=/lib
STARPU_LIBRARY=libstarpu-1.3

# constants
CCFLAGS=-std=c99
CXXFLAGS=-mtune=native -march=native -flto -std=c++17 -O3
LINKERFORPROGRAMS=g++
LINKFLAGS=-flto

NVCC=/usr/local/cuda/bin/nvcc
NVCC_LINKFLAGS=-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs

#Remove?
MPI_LINKFLAGS=
MPI_CXXFLAGS=

JAR=/usr/lib/jvm/java-11-openjdk-amd64/bin/jar
JAVAC=/usr/lib/jvm/java-11-openjdk-amd64/bin/javac
JAVA_BINDIR=/usr/lib/jvm/java-11-openjdk-amd64/bin
JNI_CPPPATH=/usr/lib/jvm/java-11-openjdk-amd64/include:/usr/lib/jvm/java-11-openjdk-amd64/include/linux


#Not to save
NVCC_CXXFLAGS=--x cu -D_FORCE_INLINES -Xcompiler -fPIC -ccbin /usr/bin/g++-10 -std=c++14 --expt-extended-lambda -gencode=arch=compute_60,code=compute_60 -gencode=arch=compute_61,code=compute_61 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_86,code=compute_86

'''
'''
on the fly values to check

PYTHONINCFLAGS=-I/home/agarcia/anaconda3/include/python3.8 -I/home/agarcia/anaconda3/lib/python3.8/site-packages/numpy/core/include

'''
