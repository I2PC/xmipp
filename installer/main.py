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
from .utils import runJob
from .logger import logger
from .constants import SCONS_INSTALL_ERROR, CLONNING_XMIPP_SOURCE_ERROR, DOWNLOADING_XMIPP_SOURCE_ERROR

####################### COMMAND FUNCTIONS #######################
def getSources(branch: str=None):
	"""
	### This function fetches the sources needed for Xmipp to compile.
	
	#### Params:
	- branch (str): Optional. Branch to clone the sources from.
	"""
	pass

def installScons():
	"""
	### Tries to install scons in current env.
	### Generates an error if something goes wrong.
	"""
	retCode, output = runJob("pip install scons")

	if retCode:
		logger.logError(output, retCode=SCONS_INSTALL_ERROR)
		#TODO: Exit
	logger(output)
	

####################### AUX FUNCTIONS #######################
def downloadSourceTag(source: str) -> Tuple[bool, str]:
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
	retCode, output = runJob(f"wget -O {fileName} {source}")
	
	if retCode:
		logger.logError(output, retCode=DOWNLOADING_XMIPP_SOURCE_ERROR)
		#TODO: Exit
	logger(output)

def cloneSourceRepo(repo: str, branch: str=None) -> Tuple[bool, str]:
	"""
	### Clones the given source as a repository in the given branch.
	
	#### Params:
	- source (str): Source to clone.
	- branch (branch): Optional. Branch to clone repo from.
	
	#### Returns:
	- (int): 0 if everything worked, or else the return code of the command that failed.
	- (str): Output data from the command if it worked or error if it failed.
	"""
	branchStr = f" --branch {branch} " if branch else ''
	retCode, output = runJob(f"git clone{branchStr}{repo}")

	if retCode:
		logger.logError(output, retCode=CLONNING_XMIPP_SOURCE_ERROR)
		#TODO: Exit
	logger(output)
