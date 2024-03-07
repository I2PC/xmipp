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
from typing import Tuple
from os.path import join

# Module imports
from .utils import runJob, getCurrentBranch
from .logger import logger, yellow
from .constants import (TAGS_SUBPAGE, SCONS_INSTALL_ERROR, XMIPP_VERSIONS, VERNAME_KEY, REPOSITORIES, XMIPP_SOURCES, EXTERNAL_SOURCES,
												CLONNING_XMIPP_SOURCE_ERROR, DOWNLOADING_XMIPP_SOURCE_ERROR)

####################### COMMAND FUNCTIONS #######################
def getSources(branch: str=None):
	"""
	### This function fetches the sources needed for Xmipp to compile.
	
	#### Params:
	- branch (str): Optional. Branch to clone the sources from.
	"""
	# Clone external sources
	retCode, output = __getExternalSources()
	if retCode:
		logger.logError(f"Error cloning external sources:\n{output}", retCode=retCode)
	
	# Clone or download internal sources
	retCode, output = __getXmippSources(branch=branch)
	if retCode:
		logger.logError(f"Error cloning xmipp sources:\n{output}", retCode=retCode)

def installScons(update: bool=False) -> Tuple[int, str]:
	"""
	### Tries to install or update scons in current env.
	### Generates an error if something goes wrong.

	#### Params:
	- update (bool): If True, it will try to update scons instead of installing from scratch.

	#### Returns:
	- (int): Return code of the command.
	- (str): Output data from the command if it worked or error if it failed.
	"""
	updateStr = ' --update' if update else ''
	retCode, output = runJob(f"pip install scons{updateStr}")

	if retCode:
		logger.logError(output, retCode=SCONS_INSTALL_ERROR)
	logger(output)
	
####################### AUX FUNCTIONS #######################
def __getXmippSources(branch: str=None):
	"""
	### Gets all of the xmipp repository sources, either downloads or clones them.

	#### Params:
	- branch (str): Optional. Branch to clone the sources from.
	
	#### Returns:
	- (int): 0 if all commands executed correctly or an error code otherwise.
	- (str): Empty string if all commands executed correctly or an error otherwise.
	"""
	currentBranch = getCurrentBranch()
	pass

def __getExternalSources() -> Tuple[int, str]:
	"""
	### Clones all of the external sources.
	
	#### Returns:
	- (int): 0 if all commands executed correctly or an error code otherwise.
	- (str): Empty string if all commands executed correctly or an error otherwise.
	"""
	for source in EXTERNAL_SOURCES:
		retCode, output = __cloneSourceRepo(REPOSITORIES[source][0], branch=REPOSITORIES[source][1])
		if retCode:
			return retCode, output
	return 0, ''

def __getTagURL(repo: str) -> str:
	"""
	### Returns the full URL to a tag given the repositorie's main URL.
	
	#### Params:
	- repo (str): Repositorie's main URL.
	
	#### Returns:
	- (str): Full URL to the tag.
	"""
	# Getting repository and tag name
	sourceName = repo.split("/")[-1]
	tagName = XMIPP_VERSIONS[sourceName][VERNAME_KEY]
	return join(repo, TAGS_SUBPAGE, f"{tagName}.zip")

def __downloadSourceTag(source: str) -> Tuple[int, str]:
	"""
	### Downloads the given source as a tag.
	
	#### Params:
	- source (str): Source to download.
	
	#### Returns:
	- (int): Return code of the command.
	- (str): Output data from the command if it worked or error if it failed.
	"""
	# Getting source name, full URL, and file name
	sourceName = source.split("/")[-1]
	source = __getTagURL(source)
	fileName = source.split("/")[-1]

	# Getting version name and extracted folder name
	versionName = fileName.replace(".zip", "")
	if versionName.startswith("v"):
		versionName = versionName[1:]
	extractedFolderName = f"{sourceName}-{versionName}"

	# Generating list of commands
	commands = [
		f"wget -O {fileName} {source}",
		f"unzip {fileName}",
		f"mv {extractedFolderName} {sourceName}",
		f"rm -f {fileName}"
	]

	# Running all commands checking output
	for command in commands:
		retCode, output = runJob(command)
		if retCode:
			return retCode, output
	return 0, ''
	
def __cloneSourceRepo(repo: str, branch: str=None) -> Tuple[int, str]:
	"""
	### Clones the given source as a repository in the given branch if exists. Defaults to default branch.
	
	#### Params:
	- source (str): Source to clone.
	- branch (branch): Optional. Branch to clone repo from.
	
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
	return runJob(f"git clone{branchStr} {repo}.git")
