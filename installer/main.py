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
from typing import Tuple, Optional

# Module imports
from .utils import runJob, getCurrentBranch, isProductionMode
from .logger import logger, yellow, green, bold
from .constants import (REPOSITORIES, XMIPP_SOURCES, SOURCES_PATH, MASTER_BRANCHNAME,
	SOURCE_CLONE_ERROR, TAG_BRANCH_NAME, INTERRUPTED_ERROR, VERSION_FILE, RELEASE_DATE,
	XMIPP_VERSIONS, XMIPP, VERSION_KEY, SECTION_MESSAGE_LEN, VERNAME_KEY, MODE_GET_SOURCES,
	MODE_CONFIG_BUILD, CONFIG_FILE, XMIPP_PLUGIN)
from .api import sendApiPOST, getOSReleaseName
from .cmake import parseCmakeVersions
from .config import getConfigDate

####################### COMMAND FUNCTIONS #######################
def getSources(branch: str=None, production: bool=False):
	"""
	### This function fetches the sources needed for Xmipp to compile.
	
	#### Params:
	- branch (str): Optional. Branch to clone the sources from.
	"""
	# Clone or download internal sources
	logger(getSectionMessage("Getting Xmipp sources"), forceConsoleOutput=True)
	for source in XMIPP_SOURCES:
		if source == XMIPP_PLUGIN and production:
			pass
		else:
			logger(f"Cloning {source}...", forceConsoleOutput=True)
			retCode, output = __cloneSourceRepo(REPOSITORIES[source][0], path=SOURCES_PATH, branch=branch)
			message = output if retCode else ''
			handleRetCode(retCode, predefinedErrorCode=SOURCE_CLONE_ERROR, message=message, sendAPI=False)

def exitXmipp(retCode: int=0):
	"""
	### This function exits Xmipp with the given return code, processing it as a success or an error.

	#### Params:
	- retCode (int): Optional. Error code.
	"""
	# End execution
	sys.exit(retCode)

def handleRetCode(realRetCode: int, predefinedErrorCode: int=0, message: str='', sendAPI: bool=True):
	"""
	### This function checks the given return code and handles the appropiate actions.

	#### Params:
	- realRetCode (int): Real return code of the called function.
	- predefinedErrorCode (int): Optional. Predefined error code for the caller code block in case of error.
	- message (str): Optional. Message that will be displayed if there is an error th
	- sendAPI (bool): Optional. If True, API message will be sent.
	"""
	if realRetCode:
		resultCode = __getPredefinedError(realRetCode=realRetCode, desiredRetCode=predefinedErrorCode)
		message = message if resultCode != realRetCode else ''
		logger.logError(message, retCode=resultCode, addPortalLink=resultCode != realRetCode)
		
		if sendAPI and resultCode != INTERRUPTED_ERROR:
			sendApiPOST(resultCode)
		exitXmipp(retCode=resultCode)
	else:
		if message:
			logger(message)
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

def getVersionMessage(short: bool=False) -> str:
	"""
	### This function returns the message for the version mode.

	#### Params:
	- short (bool): Optional. If True, only Xmipp's version with name will be returned.

	#### Returns:
	- (str): Message for version mode.
	"""
	if short:
		return XMIPP_VERSIONS[XMIPP][VERNAME_KEY]
	
	# Main info
	rightSectionStart = 25
	versionType = 'release' if isProductionMode() else getCurrentBranch()
	versionMessage = bold(f"Xmipp {XMIPP_VERSIONS[XMIPP][VERSION_KEY]} ({versionType})\n\n")
	versionMessage += f"{__getVersionLineWithSpaces('Release date: ', rightSectionStart)}{RELEASE_DATE}\n"
	versionMessage += f"{__getVersionLineWithSpaces('Compilation date: ', rightSectionStart)}{getConfigDate(CONFIG_FILE)}\n"
	versionMessage += f"{__getVersionLineWithSpaces('System version: ', rightSectionStart)}{getOSReleaseName()}\n"

	# Get sources branch
	for source in XMIPP_SOURCES:
		sourcePath = os.path.join(SOURCES_PATH, source)
		currentCommit = __getCurrentCommit(dir=sourcePath)
		branchName = getCurrentBranch(dir=sourcePath)
		branchName = branchName if branchName != 'HEAD' else __getCommitName(currentCommit, dir=sourcePath)
		sourceText = f"{source} branch: "
		versionMessage += f"{__getVersionLineWithSpaces(sourceText, rightSectionStart)}{branchName} ({currentCommit})\n"
	versionMessage += "\n"

	# Getting versions from version file
	if os.path.exists(VERSION_FILE):
		versionDict = parseCmakeVersions(VERSION_FILE)
		for package in versionDict:
			versionMessage += f"{__getVersionLineWithSpaces(f'{package}: ', rightSectionStart)}{versionDict[package]}\n"
		versionMessage = versionMessage[:-1]
	else:
		warningStr = "This project has not yet been configured, so some detectable dependencies have not been properly detected.\n"
		warningStr += f"Run '{MODE_GET_SOURCES}' and then '{MODE_CONFIG_BUILD}' to be able to show all detectable."
		versionMessage += yellow(warningStr)
	return versionMessage

####################### AUX FUNCTIONS #######################
def __branchExists(repo: str, branch: str) -> bool:
	"""
	### This function checks if the given branch exists.

	#### Params:
	- repo (str): Repository to check the branch from.
	- branch (str): Branch to check.

	#### Returns:
	- (bool): True if the branch exists, False otherwise.
	"""
	retCode, _ = runJob(f"git ls-remote --heads {repo}.git {branch} | grep -q refs/heads/{branch}")
	return not retCode

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
			retCode, output = runJob(f"git checkout {branch}", logOutput=True)
	else:
		branchStr = f" --branch {branch}" if branch else ''
		retCode, output = runJob(f"git clone{branchStr} {repo}.git", logOutput=True)

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
	if branch and __branchExists(repo, branch):
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

def __getCurrentCommit(dir: str="./") -> str:
	"""
	### This function returns the current commit short hash of a given repository:

	#### Params:
	- dir (str): Optional. Directory to repository.

	#### Returns:
	- (str): Current commit short hash, or empty string if it is not a repo or there were errors.
	"""
	retCode, output = runJob("git rev-parse --short HEAD", cwd=dir)
	if retCode or not output:
		return ''
	return output

def __getCommitName(commit: str, dir: str="./") -> str:
	"""
	### This function returns the name of the commit branch. It can be a branch name or a release name.

	#### Params:
	- commit (str): Commit hash
	- dir (str): Optional. Directory to repository.

	#### Returns:
	- (str): Name of the commit branch or release.
	"""
	retCode, output = runJob(f"git name-rev {commit}", cwd=dir)
	if retCode or not output:
		return ''

	# Extract name from commit
	return output.replace(commit, "").replace(" ", "")

def __getVersionLineWithSpaces(text: str, desiredLen: int) -> str:
	"""
	### This function returns the given text with additional spaces so that it has the desired length.

	#### Params:
	- text (str): Text where extra spaces could be added to.
	- desiredLen (int): Target length for the text.

	#### Returns:
	- (str): Given text with as much spaces to the right so that it fits the desired length.
	"""
	# Check if text is already as long or longer than the desired length
	textLen = len(text)
	if textLen >= desiredLen:
		return text
	
	# Crafting spaces and adding them to the right
	spaces = ''.join([' ' for _ in range(desiredLen - textLen)])
	return f"{text}{spaces}"
