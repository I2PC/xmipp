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
from typing import Tuple, Dict, Optional

# Module imports
from .utils import runJob, getCurrentBranch
from .logger import logger, yellow, green
from .constants import (REPOSITORIES, XMIPP_SOURCES, SOURCES_PATH, MASTER_BRANCHNAME,
	CONFIG_DEFAULT_VALUES, SOURCE_CLONE_ERROR, INTERNAL_LOGIC_VARS, TAG_BRANCH_NAME,
	INTERRUPTED_ERROR, XMIPP_VERSIONS, XMIPP, VERSION_KEY, SECTION_MESSAGE_LEN)
from .api import sendApiPOST

####################### COMMAND FUNCTIONS #######################
def getSources(branch: str=None):
	"""
	### This function fetches the sources needed for Xmipp to compile.
	
	#### Params:
	- branch (str): Optional. Branch to clone the sources from.
	"""
	# Clone or download internal sources
	logger(getSectionMessage("Getting Xmipp sources"), forceConsoleOutput=True)
	for source in XMIPP_SOURCES:
		logger(f"Cloning {source}...", forceConsoleOutput=True)
		retCode, output = __cloneSourceRepo(REPOSITORIES[source][0], path=SOURCES_PATH, branch=branch)
		handleRetCode(retCode, predefinedErrorCode=SOURCE_CLONE_ERROR, message=output)

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
	
	# End execution
	sys.exit(retCode)

def handleRetCode(realRetCode: int, predefinedErrorCode: int=0, configDict: Dict={}, message: str=''):
	"""
	### This function checks the given return code and handles the appropiate actions.

	#### Params:
	- realRetCode (int): Real return code of the called function.
	- predefinedErrorCode (int): Optional. Predefined error code for the caller code block in case of error.
	- configDict (dict): Optional. Dictionary containing all variables in the config file.
	- message (str): Optional. Message that will be displayed if there is an error th
	"""
	if realRetCode:
		resultCode = __getPredefinedError(realRetCode=realRetCode, desiredRetCode=predefinedErrorCode)
		message = message if resultCode != realRetCode else ''
		logger.logError(message, retCode=resultCode, addPortalLink=resultCode != realRetCode)
		exitXmipp(retCode=resultCode, configDict=configDict)
	__logDoneMessage()
	logger("", forceConsoleOutput=True)

def getSuccessMessage() -> str:
	"""
	### This function returns the message shown when Xmipp is compiled successfully.
	
	#### Returms:
	- (str): Success message.
	"""
	# Getting release name
	branchName = getCurrentBranch()
	releaseName = branchName
	if branchName is None or branchName == MASTER_BRANCHNAME:
		releaseName = XMIPP_VERSIONS[XMIPP][VERSION_KEY]

	# Creating message line
	releaseMessage = 'Xmipp {} has been successfully installed, enjoy it!'.format(releaseName)
	releaseMessageWrapper = '*  *'
	messageLine = releaseMessageWrapper[:int(len(releaseMessageWrapper)/2)]
	messageLine += green(f'Xmipp {releaseName} has been successfully installed, enjoy it!')
	messageLine += releaseMessageWrapper[int(len(releaseMessageWrapper)/2):]

	# Creating box around message line
	totalLen = len(releaseMessage) + len(releaseMessageWrapper)
	topBottomBorder = ''.join(['*' for _ in range(totalLen)])
	marginLine = f"*{''.join([' ' for _ in range(totalLen - 2)])}*"

	return '\n'.join([topBottomBorder, marginLine, messageLine, marginLine, topBottomBorder])

def getSectionMessage(text: str) -> str:
	"""
	### This function prints a section message in a specific format.

	#### Params:
	- text (str): Title of the section.

	#### Returns:
	- (str): Formatted section message.
	"""
	# Check if provided text's length has exceeded specified limit
	textLen = len(text)
	remainingLen = SECTION_MESSAGE_LEN - textLen
	if remainingLen < 4:
		return text
	
	# Calculating characters around given text
	nDashes = remainingLen - 2
	nFinalDashes = int(nDashes / 2)
	nInitialDashes = nDashes - nFinalDashes
	finalDashes = ''.join(['-' for _ in range(nFinalDashes)])
	initialDashes = ''.join(['-' for _ in range(nInitialDashes)])
	return f"{initialDashes} {text} {finalDashes}"

####################### AUX FUNCTIONS #######################
def __logDoneMessage():
	"""
	### This function logs a message shown after completing a task.
	"""
	logger(green("Done"), forceConsoleOutput=True, substitute=True)

def __logWorkingMessage():
	"""
	### This function logs a message shown as placeholder for small tasks in progress.
	"""
	logger(yellow("Working..."), forceConsoleOutput=True)

def __cloneSourceRepo(repo: str, branch: str=None, path: str='') -> Tuple[int, str]:
	"""
	### Clones the given source as a repository in the given branch if exists. Defaults to default branch.
	### If the repository already exists, checks out to specified branch (if provided).
	
	#### Params:
	- repo (str): Source to clone.
	- branch (branch): Optional. Branch to clone repo from.
	- path (str): Optional. Path to clone the repository into.
	
	#### Returns:
	- (int): 0 if everything worked, or else the return code of the command that failed.
	- (str): Output data from the command if it worked or error if it failed.
	"""
	retCode = 0
	output = ''
	__logWorkingMessage()
	cloneBranch = __getCloneBranch(repo, branch)

	# If specified branch does not exist, show warning
	if branch and not cloneBranch:
		warningStr = f"Warning: branch \'{branch}\' does not exist for repository with url {repo}.\n"
		warningStr += "Falling back to repository's default branch."
		logger(yellow(warningStr), forceConsoleOutput=True, substitute=True)
		branch = None
		__logWorkingMessage()

	# Move to defined path to clone
	currentPath = os.getcwd()
	os.chdir(path)

	# Check if repo already exists. If so, checkout instead of clone.
	clonedFolder = repo.split("/")[-1]
	if os.path.isdir(clonedFolder):
		if branch:
			os.chdir(clonedFolder)
			retCode, output = runJob(f"git checkout {branch}")
			logger(output)
	else:
		branchStr = f" --branch {branch}" if branch else ''
		retCode, output = runJob(f"git clone{branchStr} {repo}.git")
		logger(output)

	# Go back to previous path
	os.chdir(currentPath)

	return retCode, output

def __getCloneBranch(repo: str, branch: str) -> Optional[str]:
	"""
	### Returns the branch to clone from in the given repository. 
	
	#### Params:
	- repo (str): Repository to clone.
	- branch (branch): Branch to clone repo from.
	
	#### Returns:
	- (str | None): The given branch if it is a valid one, or None to indicate default branch.
	"""
	# If branch exists, use it
	if branch:
		retCode, _ = runJob(f"git ls-remote --heads {repo}.git {branch} | grep -q refs/heads/{branch}")
		if not retCode:
			return branch
	
	# If repository is xmipp source and current branch is a release, clone from corresponding release
	repoName = repo.split("/")[-1]
	if repoName in XMIPP_SOURCES:
		branchName = getCurrentBranch()
		if not branchName or branchName == MASTER_BRANCHNAME or branchName == TAG_BRANCH_NAME:
			return XMIPP_VERSIONS[repoName][VERSION_KEY]
	
	return None

def __getPredefinedError(realRetCode: int=0, desiredRetCode: int=0) -> int:
	"""
	### This function returns the corresponding predefined error for a caller piece of code.
	
	#### Params:
	- realRetCode (int): Optional. Real error code obtained from the process.
	- desiredRetCode (int): Optional. Predefined code corresponding to caller code.
	"""
	return realRetCode if realRetCode == INTERRUPTED_ERROR else desiredRetCode
