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
from typing import Tuple

# Module imports
from .utils import runJob, getCurrentBranch
from .logger import logger, yellow
from .constants import SCONS_INSTALL_ERROR, CLONNING_XMIPP_SOURCE_ERROR, DOWNLOADING_XMIPP_SOURCE_ERROR

####################### COMMAND FUNCTIONS #######################
def getSources(branch: str=None):
	"""
	### This function fetches the sources needed for Xmipp to compile.
	
	#### Params:
	- branch (str): Optional. Branch to clone the sources from.
	"""
	currentBranch = getCurrentBranch()
	if currentBranch:

		pass
	# If curent dir is repo, check git
		# if not git or below version, fallback to wget
		# else attempt clone
			# 
	if not currentBranch:
		# Download latest version from github tags with wget
		pass

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
		#TODO: Exit
	logger(output)
	
####################### AUX FUNCTIONS #######################
def downloadSourceTag(source: str) -> Tuple[int, str]:
	"""
	### Downloads the given source as a tag.
	
	#### Params:
	- source (str): Source to download.
	
	#### Returns:
	- (int): Return code of the command.
	- (str): Output data from the command if it worked or error if it failed.
	"""
	# Getting destination file name and downloading
	fileName = source.split("/")[-1]
	return runJob(f"wget -O {fileName} {source}")
	
def cloneSourceRepo(repo: str, branch: str=None) -> Tuple[int, str]:
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
		retCode, _ = runJob(f"git ls-remote --heads {repo} {branch} | grep -q refs/heads/{branch}")
		branchExists = not retCode
		# If does not exist, show warning
		if not branchExists:
			warningStr = f"Warning: branch \'{branch}\' does not exist for repository with url {repo}.\n"
			warningStr += "Falling back to repository's default branch."
			logger(yellow(warningStr), forceConsoleOutput=True)
			branch = None

	branchStr = f" --branch {branch}" if branch else ''
	return runJob(f"git clone{branchStr} {repo}")
