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
Submodule containing all the general constants needed for Xmipp's installation.
"""

# Dependency names
CUFFTADVISOR = 'cuFFTAdvisor'
CTPL = 'CTPL'
GTEST = 'googletest'
LIBSVM = 'libsvm'
LIBCIFPP = 'libcifpp'

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

# API URL
API_URL = 'http://127.0.0.1:8000/web/attempts/'

# Other variables
CXX_FLAGS = ' -mtune=native -march=native -flto -std=c++17 -O3'
LINK_FLAGS = '-flto'
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

# File names
CONFIG_FILE = 'xmipp.conf'
LOG_FILE = 'compileLOG.txt'
CMD_OUT_LOG_FILE = 'commandOutput.log'
CMD_ERR_LOG_FILE = 'commandError.log'
OUTPUT_POLL_TIME = 0.5

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
