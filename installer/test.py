# ***************************************************************************
# * Authors:		Alberto Garcia (alberto.garcia@cnb.csic.es)
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
This module contains the necessary functions to run and manage the test.
"""

# General imports
from os import environ, path

# Self imports
from .constants import XMIPP, SCIPION_TESTS_URLS, CONFIG_FILE, XMIPP_USE_CUDA, XMIPP_LINK_TO_SCIPION, ENVIROMENT_ERROR
from .logger import blue, red, logger
from .utils import runJob
from .config import readConfig
from .main import handleRetCode

####################### COMMAND FUNCTIONS #######################


def runTests(testNames):
	"""
	### This function fetches the sources needed for Xmipp to compile.

	#### Params:
	- branch (str): Optional. Branch to clone the sources from.
	"""
	xmippSrc = environ.get('XMIPP_SRC', None)
	#xmippSrc = '/home/agarcia/scipion3/xmipp-bundle/src'
	if xmippSrc and path.isdir(xmippSrc):
		environ['PYTHONPATH'] = ':'.join([
            path.join(environ['XMIPP_SRC'], XMIPP),
            environ.get('PYTHONPATH', '')])
		testsPath = path.join(environ['XMIPP_SRC'], XMIPP, 'tests')
		#testsPath = '/home/agarcia/scipion3/xmipp-bundle/src/xmipp/tests/'
		dataSetPath = path.join(testsPath, 'data')
	
	else:
		handleRetCode(ENVIROMENT_ERROR, predefinedErrorCode=ENVIROMENT_ERROR, sendAPI=False)

	# downloading/updating the dataset
	dataset = 'xmipp_programs'
	if path.isdir(dataSetPath):
		logger(blue("Updating the test files"))
		task = "update"
		showO = False
	else:
		logger(blue("Downloading the test files"))
		task = "download"
		showO = True
	args = "%s %s %s" % ("tests/data", SCIPION_TESTS_URLS, dataset)
	runJob("bin/xmipp_sync_data %s %s" % (task, args), cwd='src/xmipp', showOutput=showO)
	
	configDict = readConfig(CONFIG_FILE) if path.exists(CONFIG_FILE) else {}
	CUDA = configDict.get(XMIPP_USE_CUDA)
	noCudaStr = '--noCuda' if CUDA == 'OFF' else ''
	
	logger(" Tests to do: %s" % ', '.join(testNames))
	
	if configDict.get(XMIPP_LINK_TO_SCIPION) == 'ON':
		pythonExe = 'scipion3 python'
	else:
		pythonExe = 'python3'
		
	pythonExe = 'python3'
	runJob("%s test.py %s %s" % (pythonExe, testNames, noCudaStr), cwd='src/xmipp/tests', showOutput=True, showError=True)

