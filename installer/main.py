# ***************************************************************************
# * Authors:		Martín Salinas (martin.salinas@cnb.csic.es)
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
This module contains the necessary functions to run most installer commands.
"""

# General imports
import os, sys
from typing import Tuple, Dict

# Module imports
from .utils import runJob
from .logger import logger, yellow, green
from .constants import (REPOSITORIES, XMIPP_SOURCES, SOURCES_PATH,
	CONFIG_DEFAULT_VALUES, SOURCE_CLONE_ERROR, INTERNAL_LOGIC_VARS, INTERRUPTED_ERROR)
from .api import sendApiPOST

####################### COMMAND FUNCTIONS #######################
def getSources(branch: str=None):
	"""
	### This function fetches the sources needed for Xmipp to compile.
	
	#### Params:
	- branch (str): Optional. Branch to clone the sources from.
	"""
	# Clone or download internal sources
	for source in XMIPP_SOURCES:
		retCode, output = __cloneSourceRepo(REPOSITORIES[source][0], path=SOURCES_PATH, branch=branch)
		if retCode:
			resultCode = getPredefinedError(realRetCode=retCode, desiredRetCode=SOURCE_CLONE_ERROR)
			message = f"Error getting xmipp sources ({retCode}):\n{output}" if resultCode != retCode else ""
			logger.logError(message, retCode=resultCode, addPortalLink=bool(message))
			exitXmipp(retCode=resultCode)

def getCMakeVarsStr(configDict: Dict) -> str:
	"""
	### This function converts the variables in the config dictionary into a string as CMake args.
	
	#### Params:
	- configDict (dict): Dictionary to obtain the parameters from.
	"""
	cmakeParams = []
	for variable in CONFIG_DEFAULT_VALUES.keys():
		if variable in configDict and variable not in INTERNAL_LOGIC_VARS and configDict[variable] != '':
			cmakeParams.append(f"-D {variable}={configDict[variable]}")
	return ' '.join(cmakeParams)

def exitXmipp(retCode: int=0, configDict: Dict={}):
	"""
	### This function exits Xmipp with the given return code, processing it as a success or an error.
	
	#### Params:
	- retCode (int): Optional. Error code.
	- configDict (dict): Optional. Dictionary containing all config variables. If not empty, an API message is sent.
	"""
	# Send API message
	if configDict:
		sendApiPOST(configDict, retCode=retCode)
	
	# If it was a success, print success message
	if not retCode:
		logger(f"\n{__getSuccessMessage()}", forceConsoleOutput=True)

	# End execution
	sys.exit(retCode)

def getPredefinedError(realRetCode: int=0, desiredRetCode: int=0) -> int:
	"""
	### This function returns the corresponding predefined error for a caller piece of code.
	
	#### Params:
	- realRetCode (int): Optional. Real error code obtained from the process.
	- desiredRetCode (int): Optional. Predefined code corresponding to caller code.
	"""
	return realRetCode if realRetCode == INTERRUPTED_ERROR else desiredRetCode

####################### AUX FUNCTIONS #######################
def __cloneSourceRepo(repo: str, branch: str='', path: str='') -> Tuple[int, str]:
	"""
	### Clones the given source as a repository in the given branch if exists. Defaults to default branch.
	
	#### Params:
	- source (str): Source to clone.
	- branch (branch): Optional. Branch to clone repo from.
	- path (str): Optional. Path to clone the repository into.
	
	#### Returns:
	- (int): 0 if everything worked, or else the return code of the command that failed.
	- (str): Output data from the command if it worked or error if it failed.
	"""
	# If branch is provided, check if exists
	if branch:
		retCode, _ = runJob(f"git ls-remote --heads {repo}.git {branch} | grep -q refs/heads/{branch}")
		branchExists = not retCode
		# If does not exist, show warning
		if not branchExists:
			warningStr = f"Warning: branch \'{branch}\' does not exist for repository with url {repo}.\n"
			warningStr += "Falling back to repository's default branch."
			logger(yellow(warningStr), forceConsoleOutput=True)
			branch = None

	branchStr = f" --branch {branch}" if branch else ''
	# Move to defined path to clone
	currentPath = os.getcwd()
	os.chdir(path)

	# Check if repo already exists. As we do not assume it has been
	# correctly cloned, if exists, delete it and re-clone
	clonedFolder = repo.split("/")[-1]
	if os.path.isdir(clonedFolder):
		runJob(f"rm -rf {clonedFolder}")
	retCode, output = runJob(f"git clone{branchStr} {repo}.git")

	# Go back to previous path
	os.chdir(currentPath)
	return retCode, output

def __getSuccessMessage() -> str:
	"""
	### This function returns the message shown when Xmipp is compiled successfully.
	
	#### Returms:
	- (str): Success message.
	"""
	topBottomBorder = "***********************************************************"
	marginLine = "*                                                         *"
	return '\n'.join([
		topBottomBorder,
		marginLine,
		f"*  {green('Xmipp devel has been successfully installed, enjoy it!')} *",
		marginLine,
		topBottomBorder
	])
