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
import os
from typing import Tuple

# Installer imports
from .constants import XMIPP, XMIPP_CORE, XMIPP_VIZ, XMIPP_PLUGIN, REPOSITORIES, ORGANIZATION_NAME, \
	DEVEL_BRANCHNAME, MASTER_BRANCHNAME, TAGS_SUBPAGE, VERNAME_KEY, XMIPP_VERSIONS
from .utils import runJob, getCurrentBranch, showError

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

	# For each source, download or clone
	for source in sources:
		# Non-git directories and production branch (master also counts) download from tags, the rest clone
		if currentBranch is None or currentBranch == XMIPP_VERSIONS[XMIPP][VERNAME_KEY] or currentBranch == MASTER_BRANCHNAME:
			# Download source tag
			status, output = downloadSourceTag(source)
		else:
			# Clone source repository
			status, output = cloneSourceRepo(source, branch=branch)
		
		# If download failed, return error
		if not status:
			showError(output, status=status) #TODO: CHECK CODE

####################### AUX FUNCTIONS #######################
def downloadSourceTag(source: str) -> Tuple[bool, str]:
	"""
	### This function downloads the given source as a tag.
	
	#### Params:
	- source (str): Source to download.
	
	#### Returns:
	(int): Return code of the command.
	(str): Output data from the command if it worked or error if it failed.
	"""
	# If souce already exists, skip
	if os.path.isdir(source):
		return 0, ''

	# Download tag
	zipName = XMIPP_VERSIONS[source][VERNAME_KEY]
	retcode, output = runJob(f"wget -O {REPOSITORIES[ORGANIZATION_NAME]}{source}/{TAGS_SUBPAGE}{zipName}.zip", showOutput=False, showError=False, showCommand=False)

	# If download failed, return error
	if retcode != 0:
		return retcode, output
	
	# Unzip tag and change folder name to match repository name
	runJob(f"unzip {zipName}.zip", showOutput=False, showError=False, showCommand=False)

	# Check unzipped folder naming scheme
	folderName = source + '-' + zipName[1:] # Old naming system
	folderName = folderName if os.path.isdir(folderName) else source + '-' + zipName

	# Change folder name to match repository name
	retcode, output = runJob(f"mv {folderName} {source} && rm {zipName}.zip", showOutput=False, showError=False, showCommand=False)

	# Return last command's code and output.
	return retcode, output

def cloneSourceRepo(repo: str, branch: str=None) -> Tuple[bool, str]:
	"""
	### This function clones the given source as a repository in the given branch.
	
	#### Params:
	- source (str): Source to clone.
	
	#### Returns:
	(int): 0 if everything worked, or else the return code of the command that failed.
	(str): Output data from the command if it worked or error if it failed.
	"""
	# If a branch was provided, check if exists in remote repository
	output = ''
	if branch is not None:
		retcode, output = runJob(f"git ls-remote --heads {REPOSITORIES[ORGANIZATION_NAME]}{repo}.git {branch}", showOutput=False, showError=False, showCommand=False)
		
		# Check for errors
		if retcode != 0:
			return retcode, output
		
	# If output is empty, it means branch does not exist, default to devel
	if not output:
		branch = DEVEL_BRANCHNAME

	# Clone repository
	return runJob(f"git clone --branch {branch} {REPOSITORIES[ORGANIZATION_NAME]}{repo}.git", showOutput=False, showError=False, showCommand=False)
