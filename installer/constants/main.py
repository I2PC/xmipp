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

import os

XMIPP = 'xmipp'
XMIPP_CORE = 'xmippCore'
XMIPP_VIZ = 'xmippViz'
XMIPP_PLUGIN = 'scipion-em-xmipp'

UP = "\x1B[1A\r"
REMOVE_LINE = '\033[K'

CUFFTADVISOR = 'cuFFTAdvisor'
CTPL = 'CTPL'
GTEST = 'googletest'
LIBSVM = 'libsvm'
LIBCIFPP = 'libcifpp'

REPOSITORIES = {
	XMIPP: ['https://github.com/I2PC/xmipp', None],
	XMIPP_CORE: ['https://github.com/I2PC/xmippCore', None],
	XMIPP_VIZ: ['https://github.com/I2PC/xmippViz', None],
	XMIPP_PLUGIN: ['https://github.com/I2PC/scipion-em-xmipp', None],
	CUFFTADVISOR: ['https://github.com/DStrelak/cuFFTAdvisor', 'master'],
	CTPL: ['https://github.com/vit-vit/CTPL', 'master'],
	GTEST: ['https://github.com/google/googletest', 'v1.13.x'],
	LIBSVM: ['https://github.com/cossorzano/libsvm', 'master'],
	LIBCIFPP: ['https://github.com/MartinSalinas98/libcifpp', 'ms_feature_ciflibrary'],
}
XMIPP_SOURCES = [XMIPP_CORE, XMIPP_VIZ, XMIPP_PLUGIN]
EXTERNAL_SOURCES = [CUFFTADVISOR, CTPL, GTEST, LIBSVM, LIBCIFPP]
SOURCES_PATH = "src/"
BUILD_PATH = "build/"
INSTALL_PATH = "dist/"
BUILD_TYPE = "Release"
CMAKE_CACHE_PATH = os.path.join(BUILD_PATH, 'CMakeCache.txt')

# Source repositories
ORGANIZATION_NAME = 'I2PC'
TAGS_SUBPAGE = 'archive/refs/tags/'
BRANCHES_SUBPAGE = 'archive/refs/heads/'

# Internal URLs
API_URL = 'xmipp.i2pc.es/api/attempts/'
DOCUMENTATION_URL = 'https://i2pc.github.io/docs/'

# External URLs
CMAKE_INSTALL_DOCS_URL = 'https://i2pc.github.io/docs/Installation/InstallationNotes/index.html#cmake'
MODELS_URL = "http://scipion.cnb.csic.es/downloads/scipion/software/em"
SCIPION_FILES_REMOTE_PATH = "scipionfiles/downloads/scipion/software/em"
SCIPION_TESTS_URLS = "http://scipion.cnb.csic.es/downloads/scipion/data/tests"

# Other variables
TAB_SIZE = 4
SECTION_MESSAGE_LEN = 60

# File names
CONFIG_FILE = 'xmipp.conf'
LOG_FILE = 'compilation.log'
TAIL_LOG_NCHARS = 300
