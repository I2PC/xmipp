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
MPI_MINIMUM = '2.0'									#
PYTHON_MINIMUM = '3.0'							#
NUMPY_MINIMUM = '1.21'							#
MINIMUM_CUDA_VERSION = '10.1'				#
#####################################
vGCC = ['12.3', '12.2', '12.1',
				'11.3', '11.2', '11.1', '11',
				'10.5', '10.4', '10.3', '10.2', '10.1', '10',
				'9.4', '9.3', '9.2', '9.1', '9',
				'8.5', '8.4', '8.3', '8.2', '8.1', '8']

CUDA_GCC_COMPATIBILITY = {
	'10.1-10.2': vGCC[vGCC.index('8.5'):],
	'11.0-11.0': vGCC[vGCC.index('9.4'):],
	'11.1-11.3': vGCC[vGCC.index('10.5'):],
	'11.4-11.8': vGCC[vGCC.index('11.3'):],
	'12.0-12.3': vGCC[vGCC.index('12.3'):],
}

# Other variables
CXX_FLAGS = ' -mtune=native -march=native -flto -std=c++17 -O3'
VALUE_UNKNOWN = 'Unkown'
DEFAULT_JOBS = 8
COMMON_USAGE_HELP_MESSAGE = 'Run \"./xmipp -h\" for usage help.'
DEFAULT_BUILD_DIR = './build'
DEFAULT_MODELS_DIR = DEFAULT_BUILD_DIR + '/models'
TAB_SIZE = 4
PATH_TO_FIND_HDF5 = ["/usr/lib",
								 "/usr/lib/x86_64-linux-gnu/hdf5/serial",
								 "/usr/lib/x86_64-linux-gnu"]

# Files names
CONFIG_FILE = 'xmipp.conf'

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

# Group list
GROUP_GENERAL = 'General'
GROUP_CONFIG = 'Config'
GROUP_DOWNLOADS = 'Downloads'
GROUP_CLEAN = 'Clean'
GROUP_TEST = 'Test'
GROUP_DEVELOPERS = 'Developers'

# Modes with help message
MODES = {
	GROUP_GENERAL: {
		MODE_VERSION: 'Returns the version information. Add \'--short\' to print only the version number.',
		MODE_COMPILE_AND_INSTALL: 'Compiles and installs Xmipp based on already obtained sources.',
		MODE_ALL: 'Default param. Runs config, and compileAndInstall.'
	},
	GROUP_CONFIG: {
		MODE_CONFIG: 'Generates config file based on system information.',
		MODE_CHECK_CONFIG: 'Cheks if the values in the config file are ok.'
	},
	GROUP_DOWNLOADS: {
		MODE_GET_MODELS: f'Download the DeepLearning Models at dir/models ({DEFAULT_MODELS_DIR} by default).'
	},
	GROUP_CLEAN: {
		MODE_CLEAN_BIN: 'Removes all compiled binaries.',
		MODE_CLEAN_DEPRECATED: 'Removes all deprecated binaries from src/xmipp/bin.',
		MODE_CLEAN_ALL: 'Removes all compiled binaries and sources, leaves the repository as if freshly cloned (without pulling).'
	},
	GROUP_TEST: {
		MODE_TEST: 'Runs a given test.'
	},
	GROUP_DEVELOPERS: {
		MODE_GIT: 'Runs the given git action for all source repositories.',
		MODE_ADD_MODEL: 'Takes a DeepLearning model from the modelPath, makes a tgz of it and uploads the .tgz according to the <login>.'
	}
}

# Arguments of each mode, sorted by group, with their respective help message
MODE_ARGS = {
	MODE_VERSION: {
		'-dir': f"Directory where the xmipp will be installed. Default is \"{DEFAULT_BUILD_DIR}\".",
		'-short': "If set, only version number is shown."
	},
	MODE_COMPILE_AND_INSTALL: {
		'-j': f"Number of jobs. Defaults to {DEFAULT_JOBS}.",
		'-br': "Branch for the source repositories.",
		'-dir': f"Directory where the xmipp will be installed. Default is \"{DEFAULT_BUILD_DIR}\"."
	},
	MODE_ALL: {
		'-j': f"Number of jobs. Defaults to {DEFAULT_JOBS}.",
		'-br': "Branch for the source repositories.",
		'-dir': f"Directory where the xmipp will be installed. Default is \"{DEFAULT_BUILD_DIR}\".",
		'-noAsk': "If set, Xmipp will try to automatically find necessary libraries and compilers."
	},
	MODE_CONFIG: {
		'-noAsk': "If set, Xmipp will try to automatically find necessary libraries and compilers."
	},
	MODE_CHECK_CONFIG: {},
	MODE_GET_MODELS: {
		'-dir': f"Directory where the Deep Learning Models will be downloaded. Default is \"{DEFAULT_MODELS_DIR}\"."
	},
	MODE_CLEAN_BIN: {},
	MODE_CLEAN_DEPRECATED: {},
	MODE_CLEAN_ALL: {},
	MODE_TEST: {
		'testName': "Test to run. If combined with --show, greps the test name from the test list.",
		'-show': "If set, shows the tests available. If combined with a test name, greps that test name within the test list."
	},
	MODE_GIT: {
		'command': "Git command to run on all source repositories."
	},
	MODE_ADD_MODEL: {
		'login': "Login (usr@server) for Nolan machine to upload the model with. Must have write permisions to such machine.",
		'modelPath': "Path to the model to upload to Nolan."
	}
}

# Examples for the help message of each mode
MODE_EXAMPLES = {
	MODE_VERSION: [

	],
	MODE_COMPILE_AND_INSTALL: [

	],
	MODE_ALL: [

	],
	MODE_CONFIG: [

	],
	MODE_CHECK_CONFIG: [],
	MODE_GET_MODELS: [

	],
	MODE_CLEAN_BIN: [],
	MODE_CLEAN_DEPRECATED: [],
	MODE_CLEAN_ALL: [],
	MODE_TEST: [

	],
	MODE_GIT: ['./xmipp git pull'],
	MODE_ADD_MODEL: ['./xmipp addModel myuser@127.0.0.1 /home/myuser/mymodel']
}

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
	17: ['CUDA version not compatible with your g++ compiler', 'Please update CUDA or update the compiler or set the CUDA flag on the xmipp.conf to False'],
	18: ['CUDA not found', 'Please review the CUDA_HOME flag on your xmipp.conf file'],
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