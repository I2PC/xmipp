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
XMIPP = 'xmipp'
XMIPP_CORE = 'xmippCore'
XMIPP_VIZ = 'xmippViz'
XMIPP_PLUGIN = 'scipion-em-xmipp'
CUFFTADVISOR = 'cuFFTAdvisor'
CTPL = 'CTPL'
GTEST = 'googletest'
LIBSVM = 'libsvm'
LIBCIFPP = 'libcifpp'

REPOSITORIES = {
XMIPP: ['https://github.com/I2PC/xmipp.git', ''],
XMIPP_CORE: ['https://github.com/I2PC/xmippCore.git', ''],
XMIPP_VIZ: ['https://github.com/I2PC/xmippViz.git', ''],
XMIPP_PLUGIN: ['https://github.com/I2PC/scipion-em-xmipp.git', ''],
CUFFTADVISOR: ['https://github.com/DStrelak/cuFFTAdvisor.git', 'master'],
CTPL: ['https://github.com/vit-vit/CTPL.git', 'master'],
GTEST: ['https://github.com/google/googletest', 'v1.13.x'],
LIBSVM: ['https://github.com/cossorzano/libsvm.git', 'master'],
LIBCIFPP: ['https://github.com/MartinSalinas98/libcifpp', 'ms_feature_ciflibrary'],
}
# Source repositories
ORGANIZATION_NAME = 'I2PC'
TAGS_SUBPAGE = 'archive/refs/tags/'

# API URL
#API_URL = 'http://127.0.0.1:8000/web/attempts/' #local url
API_URL = 'xmipp.i2pc.margaro.eus/api/attempts/' #temporal url
# documentation xmipp
DOCUMENTATION_URL = 'https://i2pc.github.io/docs/'

# Other variables
CXX_FLAGS = ' -mtune=native -march=native -flto=auto -std=c++17 '
LINKFLAGS = '-flto=auto'
TAB_SIZE = 4
INC_PATH = [
	'/usr/local/include/',
	'/usr/include'
]
INC_HDF5_PATH = INC_PATH + [
	"/usr/include/hdf5/serial",
	"/usr/local/include/hdf5/serial"
]
PATH_TO_FIND = [
	"/usr/lib",
	"/usr/lib/x86_64-linux-gnu/hdf5/serial",
	"/usr/lib/x86_64-linux-gnu",
  "/usr/local/lib",
  "/lib"
]

PATH_TO_FIND_H = ['/usr/include',
									'/usr/local/include']



# File names
CONFIG_FILE = 'xmipp.conf'
OLD_CONFIG_FILE = 'xmipp_old.conf'
XMIPPENV = 'xmippEnv.json'
VERSIONS_FILE = 'xmippVersions.txt'
COMPRESED_FILE = 'report.tar.gz'
LOG_FILE = 'compileLOG.txt'
CMD_OUT_LOG_FILE = 'commandOutput.log'
CMD_ERR_LOG_FILE = 'commandError.log'
OUTPUT_POLL_TIME = 0.5
TAIL_LOG = 1000

INTERNAL_FLAGS = {'CCFLAGS': '',
								  'CXXFLAGS': '',
								  'PYTHONINCFLAGS': '',
								  'LINKERFORPROGRAMS': '',
								  'MPI_LINKERFORPROGRAMS': '',
								  'NVCC_CXXFLAGS': '',
								  'LINKFLAGS_NVCC': '',
									'PYTHON_LIB': '',
								  'JAVA_BINDIR': '',
								  'JAVAC': '',
								  'JAR': '',
									'JNI_CPPPATH': '',
									'LINKFLAGS': '',
									}

CONFIG_DICT = {'INCDIRFLAGS': '',
								'CC': '',
								'CXX': '',
								'MPI_CC': '',
								'MPI_CXX': '',
								'MPI_RUN': '',
								'JAVA_HOME': '',
								'OPENCV': '',
								'OPENCVCUDASUPPORTS': '',
								'CUDA': '',
								'CUDA_HOME': '',
								'CUDACXX': '',
							  'SCONS': '',
								'STARPU': '',
								'STARPU_HOME': '',
								'STARPU_INCLUDE': '',
								'STARPU_LIB': '',
								'STARPU_LIBRARY': '',
								'MATLAB': '',
								'MATLAB_HOME': '',
								'LIBDIRFLAGS': '',
							  'HDF5_HOME': '',
							  'TIFF_SO': '',
							 	'FFTW3_SO': '',
							  'TIFF_H': '',
							 	'FFTW3_H': '',
							  'ANON_DATA_COLLECT': ''
}
