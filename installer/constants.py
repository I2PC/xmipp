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

# Source names
XMIPP = 'xmipp'
XMIPP_CORE = 'xmippCore'
XMIPP_VIZ = 'xmippViz'
XMIPP_PLUGIN = 'scipion-em-xmipp'
CUFFTADVISOR = 'cuFFTAdvisor'
CTPL = 'CTPL'
GTEST = 'googletest'
LIBSVM = 'libsvm'
LIBCIFPP = 'libcifpp'

#XMIPP_VERSION = '3.23.07.0'
#XMIPP_VERNAME = 'Morpheus' 

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

# Source repositories
ORGANIZATION_NAME = 'I2PC'
REPOSITORIES = {
	ORGANIZATION_NAME: 'https://github.com/I2PC/',
	CUFFTADVISOR: 'https://github.com/DStrelak/cuFFTAdvisor.git',
	CTPL: 'https://github.com/vit-vit/CTPL.git',
	GTEST: 'https://github.com/google/googletest',
	LIBSVM: 'https://github.com/cossorzano/libsvm.git',
	LIBCIFPP: 'https://github.com/MartinSalinas98/libcifpp'
}
TAGS_SUBPAGE = 'archive/refs/tags/'

vGCC = [
	'12.3', '12.2', '12.1',
	'11.3', '11.2', '11.1', '11',
	'10.5', '10.4', '10.3', '10.2', '10.1', '10',
	'9.4', '9.3', '9.2', '9.1', '9',
	'8.5', '8.4', '8.3', '8.2', '8.1', '8'
]

CUDA_GCC_COMPATIBILITY = {
	'10.1-10.2': vGCC[vGCC.index('8.5'):],
	'11.0-11.0': vGCC[vGCC.index('9.4'):],
	'11.1-11.3': vGCC[vGCC.index('10.5'):],
	'11.4-11.8': vGCC[vGCC.index('11.3'):],
	'12.0-12.3': vGCC[vGCC.index('12.3'):]
}

# Other variables
CXX_FLAGS = ' -mtune=native -march=native -flto -std=c++17 -O3'
LINK_FLAGS = '-flto'
VALUE_UNKNOWN = 'Unkown'
DEFAULT_JOBS = 8
COMMON_USAGE_HELP_MESSAGE = 'Run \"./xmipp -h\" for usage help.'
DEFAULT_BUILD_DIR = './build'
DEFAULT_MODELS_DIR = DEFAULT_BUILD_DIR + '/models'
TAB_SIZE = 4
INC_PATH = [
	'/usr/local/include/',
	'/usr/include'
]
INC_HDF5_PATH = INC_PATH + [
	"/usr/include/hdf5/serial",
	"/usr/local/include/hdf5/serial"
]
PATH_TO_FIND_HDF5 = [
	"/usr/lib",
	"/usr/lib/x86_64-linux-gnu/hdf5/serial",
	"/usr/lib/x86_64-linux-gnu"
]

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
		'-dir': f"Directory where the xmipp will be installed. Default is \"{DEFAULT_BUILD_DIR}\"."
	},
	MODE_CONFIG: {},
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
		f'./xmipp {MODE_VERSION}',
		f'./xmipp {MODE_VERSION} --short',
		f'./xmipp {MODE_VERSION} -dir /path/to/my/build/dir'
	],
	MODE_COMPILE_AND_INSTALL: [
		f'./xmipp {MODE_COMPILE_AND_INSTALL}',
		f'./xmipp {MODE_COMPILE_AND_INSTALL} -j 20',
		f'./xmipp {MODE_COMPILE_AND_INSTALL} -dir /path/to/my/build/dir',
		f'./xmipp {MODE_COMPILE_AND_INSTALL} -br devel',
		f'./xmipp {MODE_COMPILE_AND_INSTALL} -j 20 dir /path/to/my/build/dir -br devel'
	],
	MODE_ALL: [
		'./xmipp',
		f'./xmipp {MODE_ALL}',
		'./xmipp -j 20',
		'./xmipp -dir /path/to/my/build/dir',
		'./xmipp -br devel',
		f'./xmipp {MODE_ALL} -j 20 dir /path/to/my/build/dir -br devel]'
	],
	MODE_CONFIG: [],
	MODE_CHECK_CONFIG: [],
	MODE_GET_MODELS: [f'./xmipp {MODE_GET_MODELS}', f'./xmipp {MODE_GET_MODELS} -dir /path/to/my/model/directory'],
	MODE_CLEAN_BIN: [],
	MODE_CLEAN_DEPRECATED: [],
	MODE_CLEAN_ALL: [],
	MODE_TEST: [f'./xmipp {MODE_TEST} testName', f'./xmipp {MODE_TEST} --show', f'./xmipp {MODE_TEST} testName --show'],
	MODE_GIT: [f'./xmipp {MODE_GIT} pull', f'./xmipp {MODE_GIT} checkout devel'],
	MODE_ADD_MODEL: [f'./xmipp {MODE_ADD_MODEL} myuser@127.0.0.1 /home/myuser/mymodel']
}

CONFIG_DICT = {'INCDIRFLAGS': '',
								'CC': '',
								'CXX': '',
								'MPI_CC': '',
								'MPI_CXX': '',
								'MPI_RUN': '',
								'JAVA_HOME': '',
								'OPENCV': '',
								'CUDA': '',
								'CUDA_HOME': '',
								'CUDA_CXX': '',
								'STARPU': '',
								'STARPU_HOME': '',
								'STARPU_INCLUDE': '',
								'STARPU_LIB': '',
								'STARPU_LIBRARY': '',
								'MATLAB': '',
								'MATLAB_HOME': '',
								'LIBDIRFLAGS': ''
}

#Error Flags code
OK = 0
UNKOW_ERROR = 1
SCONS_INSTALLATION_ERROR = 2
NO_SCONS_NO_SCIPION_ERROR = 3
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

# Error Code
ERROR_CODE = {
	UNKOW_ERROR: ['Unkonw error', ''],
	SCONS_INSTALLATION_ERROR: ['scons installation  error', 'We tried to install it on your scipion enviroment but was not posible, please install it manually'],
	NO_SCONS_NO_SCIPION_ERROR: ['scons not found', 'We didnt find the scipion enviroment, please install scons manually on your conda env or in your system'],
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
INCDIRFLAGS += python
LIBDIRFLAGS += python

'''
