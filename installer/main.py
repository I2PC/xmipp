# ***************************************************************************
# * Authors:		Alberto García (alberto.garcia@cnb.csic.es)
# *					Martín Salinas (martin.salinas@cnb.csic.es)
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
from typing import Union, Tuple

# Installer imports
from .constants import XMIPP, XMIPP_CORE, XMIPP_VIZ, XMIPP_PLUGIN, REPOSITORIES, ORGANIZATION_NAME, \
	DEVEL_BRANCHNAME, TAGS_SUBPAGE, VERNAME_KEY, XMIPP_VERSIONS
from .utils import runBackgroundJob, red

####################### COMMAND FUNCTIONS #######################
def getSources(branch: str=None):
	"""
	### This function fetches the sources needed for Xmipp to compile.
	
	#### Params:
	- branch (str): Optional. Branch to clone the sources from.
	"""
	# Enclose multi-word branch names in quotes
	if branch is not None and len(branch.split(' ')) > 1:
		branch = f"\"{branch}\""

	# Detect if Xmipp is in production or in devel mode
	currentBranch = getCurrentBranch()
	
	# Define sources list
	sources = [XMIPP_CORE, XMIPP_VIZ, XMIPP_PLUGIN]
	os.chdir(os.path.expanduser("~/Documents/tmp")) # TODO:REMOVE. TEST PURPOSES

	# For each source, download or clone
	for source in sources:
		# Non-git directories and production branch download from tags, the rest clone
		if currentBranch is None or currentBranch == XMIPP_VERSIONS[XMIPP][VERNAME_KEY]:
			# Download source tag
			status, output = downloadSourceTag(source)
		else:
			# Clone source repository
			status, output = cloneSourceRepo(source, branch=branch)
		
		# If download failed, return error
		if not status:
			print(red(output))
			sys.exit(1) #TODO: CHECK CODE

####################### AUX FUNCTIONS #######################
def getCurrentBranch(dir: str='./') -> Union[str, None]:
	"""
	### This function returns the current branch of the repository of the given directory or None if it is not a repository.
	
	#### Params:
	- dir (str): Optional. Directory of the repository to get current branch from. Default is current directory.
	
	#### Returns:
	(str | None): The name of the branch, or None if given directory is not a repository.
	"""
	# Getting current branch name
	status, output = runBackgroundJob("git rev-parse --abbrev-ref HEAD", cwd=dir)

	# Return branch name or None if command failed
	return output if status else None

def getSourceTargetBranch() -> str:
	"""
	### This function returns the target branch for a given repository.
	
	#### Params:
	- dir (str): Optional. Directory of the repository to get current branch from. Default is current directory.
	
	#### Returns:
	(str): The name of the target branch.
	"""

def downloadSourceTag(source: str) -> Tuple[bool, str]:
	"""
	### This function downloads the given source as a tag.
	
	#### Params:
	- source (str): Source to download.
	
	#### Returns:
	(bool): True if the command worked or False if it did not.
	(str): Output data from the command if it worked or error if it failed.
	"""
	# Download tag
	zipName = XMIPP_VERSIONS[source][VERNAME_KEY]
	status, output = runBackgroundJob(f"wget -O {REPOSITORIES[ORGANIZATION_NAME]}{source}/{TAGS_SUBPAGE}{zipName}.zip")

	# If download failed, return error
	if not status:
		return False, output
	
	# Unzip tag and change folder name to match repository name
	runBackgroundJob(f"unzip {zipName}.zip")

	# Check unzipped folder naming scheme
	folderName = source + '-' + zipName[1:] # Old naming system
	folderName = folderName if os.path.isdir(folderName) else source + '-' + zipName

	# Change folder name to match repository name
	runBackgroundJob(f"mv {folderName} {source} && rm {zipName}.zip")

	# If everything went well, return True. No output text is needed.
	return True, ''

def cloneSourceRepo(repo: str, branch: str=None) -> Tuple[bool, str]:
	"""
	### This function clones the given source as a repository in the given branch.
	
	#### Params:
	- source (str): Source to clone.
	
	#### Returns:
	(bool): True if the command worked or False if it did not.
	(str): Output data from the command if it worked or error if it failed.
	"""
	# If a branch was provided, check if exists in remote repository
	output = ''
	if branch is not None:
		status, output = runBackgroundJob(f"git ls-remote --heads {REPOSITORIES[ORGANIZATION_NAME]}{repo}.git {branch}")
		
		# Check for errors
		if not status:
			return False, output
		
	# If output is empty, it means branch does not exist, default to devel
	if not output:
		branch = DEVEL_BRANCHNAME

	# Clone repository
	return runBackgroundJob(f"git clone --branch {branch} {REPOSITORIES[ORGANIZATION_NAME]}{repo}.git")