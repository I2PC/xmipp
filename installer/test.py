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
from .constants import XMIPP, SCIPION_TESTS_URLS, CONFIG_FILE, XMIPP_USE_CUDA, XMIPP_LINK_TO_SCIPION
from .logger import blue, red, logger
from .utils import runJob
from .config import readConfig

####################### COMMAND FUNCTIONS #######################


def runTests(testNames):
	"""
	### This function fetches the sources needed for Xmipp to compile.

	#### Params:
	- branch (str): Optional. Branch to clone the sources from.
	"""
	xmippSrc = environ.get('XMIPP_SRC', None)
	if xmippSrc and path.isdir(xmippSrc):
		environ['PYTHONPATH'] = ':'.join([
            path.join(environ['XMIPP_SRC'], XMIPP),
            environ.get('PYTHONPATH', '')])
		testsPath = path.join(environ['XMIPP_SRC'], XMIPP, 'tests')
	else:
		logger.logError(errorMsg=red('XMIPP_SRC is not in the enviroment.') +
              '\nTo run the tests you need to run: ' +
              blue('source dist/xmipp.bashrc'))
		
		
	dataSetPath = path.join(testsPath, 'data')
	environ["XMIPP_TEST_DATA"] = dataSetPath

	# downloading/updating the dataset
	dataset = 'xmipp_programs'
	if path.isdir(dataSetPath):
		logger(blue("Updating the test files"))
		task = "update"
	else:
		logger(blue("Downloading the test files"))
		task = "download"
	args = "%s %s %s" % ("tests/data", SCIPION_TESTS_URLS, dataset)
	runJob("bin/xmipp_sync_data %s %s" % (task, args), cwd='src/xmipp')
	
	configDict = readConfig(CONFIG_FILE) if path.exists(CONFIG_FILE) else {}
	CUDA = configDict.get(XMIPP_USE_CUDA)
	
	noCudaStr = '--noCuda' if CUDA == 'OFF' else ''
	#logger(" Tests to do: %s" % ', '.join(testNames))
	
	
	if configDict.get(XMIPP_LINK_TO_SCIPION) == 'ON':
		pythonExe = 'scipion3 run'
	else:
		pythonExe = 'python3'

	if not runJob("(cd src/xmipp/tests; %s test.py %s %s)"
	              % (pythonExe, ' '.join(testNames), noCudaStr)):
		logger.logError()
