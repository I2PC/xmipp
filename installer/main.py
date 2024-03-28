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
import os
from typing import Tuple, Optional

# Module imports
from .utils import runJob, isTag
from .logger import logger, yellow
from .constants import (TAGS_SUBPAGE, BRANCHES_SUBPAGE, XMIPP_VERSIONS, VERNAME_KEY, REPOSITORIES, XMIPP_SOURCES, EXTERNAL_SOURCES,
												SOURCES_PATH, OK)

####################### COMMAND FUNCTIONS #######################
def getSources(branch: str=None):
	"""
	### This function fetches the sources needed for Xmipp to compile.
	
	#### Params:
	- branch (str): Optional. Branch to clone the sources from.
	"""
	# Check if it is a tag or a branch
	tag = isTag()

	# Clone external sources
	retCode, output = __getExternalSources(path=SOURCES_PATH, tag=tag)
	if retCode:
		logger.logError(f"Error getting external sources ({retCode}):\n{output}", retCode=retCode)
	
	# Clone or download internal sources
	retCode, output = __getXmippSources(branch=branch, path=SOURCES_PATH, tag=tag)
	if retCode:
		logger.logError(f"Error getting xmipp sources ({retCode}):\n{output}", retCode=retCode)

#def getConfig

####################### AUX FUNCTIONS #######################
def __getXmippSources(branch: str='', path: str='', tag: bool=False):
	"""
	### Gets all of the xmipp repository sources, either downloads or clones them.

	#### Params:
	- branch (str): Optional. Branch to clone the sources from.
	- path (str): Optional. Path to clone the sources into.
	- tag (bool): Optional. If True, it is a tag, otherwise, it is a branch.
	
	#### Returns:
	- (int): 0 if all commands executed correctly or an error code otherwise.
	- (str): Empty string if all commands executed correctly or an error otherwise.
	"""
	# Define clone function deppending on if it is a tag or a branch
	cloneFunc = __downloadSourceTag if tag else __cloneSourceRepo

	# Getting sources
	for source in XMIPP_SOURCES:
		retCode, output = cloneFunc(REPOSITORIES[source][0], path=path, branch=branch)
		if retCode:
			break
	return retCode, output
	
def __getExternalSources(path: str='', tag: bool=False) -> Tuple[int, str]:
	"""
	### Clones all of the external sources.

	#### Params:
	- path (str): Optional. Path to clone the sources into.
	- tag (bool): Optional. If True, it is a tag, otherwise, it is a branch.
	
	#### Returns:
	- (int): 0 if all commands executed correctly or an error code otherwise.
	- (str): Empty string if all commands executed correctly or an error otherwise.
	"""
	# Define clone function deppending on if it is a tag or a branch
	cloneFunc = __downloadSourceTag if tag else __cloneSourceRepo

	# Getting sources
	for source in EXTERNAL_SOURCES:
		retCode, output = cloneFunc(REPOSITORIES[source][0], branch=REPOSITORIES[source][1], path=path)
		if retCode:
			break
	return retCode, output

def __getTagURL(repo: str) -> str:
	"""
	### Returns the full URL to a tag given the repositorie's main URL.
	
	#### Params:
	- repo (str): Repository's main URL.
	
	#### Returns:
	- (str): Full URL to the tag.
	"""
	# Getting repository and tag name
	sourceName = repo.split("/")[-1]
	tagName = XMIPP_VERSIONS[sourceName][VERNAME_KEY]
	return os.path.join(repo, TAGS_SUBPAGE, f"{tagName}.zip")

def __getBranchURL(repo: str, branch: str='') -> Optional[str]:
	"""
	### Returns the full URL to a branch given the repositorie's main URL and the branch name. If no branch is received, falls back to default branch.
	
	#### Params:
	- repo (str): Repository's main URL.
	- branch (str): Optional. Branch name.
	
	#### Returns:
	- (str | None): Full URL to the branch or None if something went wrong.
	"""
	if not branch:
		retCode, output = runJob("git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'")
		if not retCode:
			branch = output
	return os.path.join(repo, BRANCHES_SUBPAGE, f"{branch}.zip") if branch else None

def __downloadSourceTag(source: str, branch: str='', path: str='') -> Tuple[int, str]:
	"""
	### Downloads the given source as a tag or branch.
	
	#### Params:
	- source (str): Source to download.
	- branch (str): Optional. If provided, the source to download is a branch from such branch. If False, it is a tag.
	- path (str): Optional. Path to download the file into.
	
	#### Returns:
	- (int): Return code of the command.
	- (str): Output data from the command if it worked or error if it failed.
	"""
	# Getting source name, full URL, and file name
	sourceName = os.path.join(path, source.split("/")[-1])
	source = __getTagURL(source) if not branch else __getBranchURL(source)
	fileName = source.split("/")[-1]
	fullPath = os.path.join(path, fileName)

	# Getting version name and extracted folder name
	versionName = fileName.replace(".zip", "")
	if versionName.startswith("v"):
		versionName = versionName[1:]
	extractedFolderName =f"{sourceName}-{versionName}"

	# Generating list of commands
	commands = [
		f"wget -O {fullPath} {source}",
		f"unzip {fullPath}",
		f"mv {extractedFolderName} {sourceName}",
		f"rm -f {fullPath}"
	]

	# Running all commands checking output
	for command in commands:
		retCode, output = runJob(command)
		if retCode:
			return retCode, output
	return OK, ''
	
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
