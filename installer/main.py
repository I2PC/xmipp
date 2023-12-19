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
from .constants import (XMIPP, XMIPP_CORE, XMIPP_VIZ, XMIPP_PLUGIN, REPOSITORIES,
	ORGANIZATION_NAME, CUFFTADVSOR_ERROR, GOOGLETEST_ERROR,LIBSVM_ERROR, LIBCIFPP_ERROR, \
	DEVEL_BRANCHNAME, MASTER_BRANCHNAME, TAGS_SUBPAGE, VERNAME_KEY, XMIPP_VERSIONS,
  CUFFTADVISOR, CTPL, GTEST, LIBSVM, LIBCIFPP, CLONNING_EXTERNAL_SOURCE_ERROR,
  CLONNING_XMIPP_SOURCE_ERROR, DOWNLOADING_XMIPP_SOURCE_ERROR, GIT_PULL_WARNING)
from .utils import runJob, getCurrentBranch, printError, printMessage, green, printWarning
from .config import readConfig

####################### COMMAND FUNCTIONS #######################
def getSources(branch: str=None):
	"""
	### This function fetches the sources needed for Xmipp to compile.
	
	#### Params:
	- branch (str): Optional. Branch to clone the sources from.
	"""
	printMessage(text='\n- Getting sources...', debug=True)
	# Enclose multi-word branch names in quotes
	if branch is not None and len(branch.split(' ')) > 1:
		branch = f"\"{branch}\""

	# Detect if Xmipp is in production or in devel mode
	currentBranch = getCurrentBranch()
	
	# Define sources list
	external_sources = [CUFFTADVISOR, CTPL, GTEST, LIBSVM, LIBCIFPP]
	sources = [XMIPP_CORE, XMIPP_VIZ, XMIPP_PLUGIN]

	for source in external_sources:
			# Clone source repository
			status, output = cloneSourceRepo(repo=source, branch=REPOSITORIES[source][1])
			if status != 0:
				printError(errorMsg=output, retCode=CLONNING_EXTERNAL_SOURCE_ERROR)

	for source in sources:
		# Non-git directories and production branch (master also counts) download from tags, the rest clone
		if (currentBranch is None or currentBranch == XMIPP_VERSIONS[XMIPP][VERNAME_KEY]
						or currentBranch == MASTER_BRANCHNAME):
			# Download source tag
			status, output = downloadSourceTag(source)
			if status != 0:
					printError(output, retCode=DOWNLOADING_XMIPP_SOURCE_ERROR)
		else:
			# Clone source repository
			status, output = cloneSourceRepo(source, branch=branch)
		# If download failed, return error
		if status != 0:
			printError(output, retCode=CLONNING_XMIPP_SOURCE_ERROR)

def compileExternalSources(jobs):
		printMessage(text='\n- Compiling external sources...', debug=True)
		dictPackage = readConfig()
		if dictPackage['CUDA'] == 'True':
			compile_cuFFTAdvisor()
		compile_googletest()
		compile_libsvm()
		compile_libcifpp(jobs)


def compile_cuFFTAdvisor():
		printMessage('Compiling cuFFTAdvisor...', debug=True)
		advisorDir = "src/cuFFTAdvisor/"
		currDir = os.getcwd()
		libDir = "src/xmipp/lib/"
		if not os.path.exists(libDir):
				os.makedirs(libDir)
		os.chdir(advisorDir)
		retCode, outputStr = runJob("make all")
		if retCode == 0:
				os.chdir(currDir)
				retCode, outputStr = runJob("cp " + advisorDir + "build/libcuFFTAdvisor.so" + " " + libDir)
				if retCode == 0:
						os.chdir(currDir)
						printMessage(text=green('cuFFTAdvisor package compillated'), debug=True)
				else:
						os.chdir(currDir)
						printError(retCode=CUFFTADVSOR_ERROR, errorMsg=outputStr)
		else:
				os.chdir(currDir)
				printError(retCode=CUFFTADVSOR_ERROR, errorMsg=outputStr)

def compile_googletest():
		printMessage(text="Compiling googletest...", debug=True)
		currDir = os.getcwd()
		buildDir = os.path.join("src", "googletest", "build")
		if not os.path.exists(buildDir):
				os.makedirs(buildDir)
		os.chdir(buildDir)
		retCode, outputStr = runJob("cmake ..")
		if retCode == 0:
				retCode, outputStr = runJob("make gtest gtest_main")
				if retCode == 0:
						os.chdir(currDir)
						printMessage(text=green('googletest package compillated'), debug=True)
				else:
						os.chdir(currDir)
						printError(retCode=GOOGLETEST_ERROR, errorMsg=outputStr)
		else:
				os.chdir(currDir)
				printError(retCode=GOOGLETEST_ERROR, errorMsg=outputStr)


def compile_libsvm():
		printMessage(text="Compiling libsvm...", debug=True)
		# if the libsvm repo is updated, remember that the repoFork/Makefile was edited to remove references to libsvm-so.2
		currDir = os.getcwd()
		libsvmDir = os.path.join("src", "libsvm")
		os.chdir(libsvmDir)
		retCode, outputStr = runJob("make lib")
		if retCode == 0:
				libDir = "src/xmipp/lib"
				os.chdir(currDir)
				if not os.path.exists(libDir):
						os.makedirs(libDir)
				retCode, outputStr = runJob("cp " + libsvmDir + "/libsvm.so" + " " + libDir)
				if retCode == 0:
						os.chdir(currDir)
						printMessage(text=green('libsvm package compillated'), debug=True)
				else:
						os.chdir(currDir)
						printError(retCode=LIBSVM_ERROR, errorMsg=outputStr)
		else:
				os.chdir(currDir)
				printError(retCode=LIBSVM_ERROR, errorMsg=outputStr)


def compile_libcifpp(jobs):
		printMessage(text="Compiling libcifpp..", debug=True)
		currDir = os.getcwd()
		# Moving to library directory
		libcifppDir = os.path.join("src", "libcifpp")
		os.chdir(libcifppDir)
		# Installing
		fullDir = os.path.join(currDir, libcifppDir, '')
		retCode, outputStr = runJob("cmake -S . -B build -DCMAKE_INSTALL_PREFIX=" + fullDir +
							" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCIFPP_DOWNLOAD_CCD=OFF -DCIFPP_INSTALL_UPDATE_SCRIPT=OFF")
		if retCode == 0:
				retCode, outputStr = runJob(f"cmake --build build -j {jobs}")
				if retCode == 0:
						retCode, outputStr = runJob("cmake --install build")
						if retCode == 0:
								# Check if libcifpp created up on compiling lib or lib64 directory
								libcifppLibDir = "lib64" if os.path.exists("lib64") else "lib"
								# Copying .so file
								os.chdir(currDir)
								libDir = "src/xmipp/lib"
								if not os.path.exists(libDir):
										os.makedirs(libDir)
								retCode, outputStr = runJob("cp " + os.path.join(libcifppDir, libcifppLibDir,
																							 "libcifpp.so*") + " " + libDir)
								if retCode == 0:
										printMessage(text=green('libcifpp package compillated'), debug=True)
								else:
										printError(retCode=LIBCIFPP_ERROR, errorMsg=outputStr)
						else:
								os.chdir(currDir)
								printError(retCode=LIBCIFPP_ERROR, errorMsg=outputStr)
				else:
						os.chdir(currDir)
						printError(retCode=LIBCIFPP_ERROR, errorMsg=outputStr)
		else:
				os.chdir(currDir)
				printError(retCode=LIBCIFPP_ERROR, errorMsg=outputStr)


def compileSources(jobs):
		sources = [XMIPP_CORE, XMIPP_VIZ, XMIPP]
		dictPackage = readConfig()

		for source in sources:
			printMessage(text='\n- Compiling {}...'.format(source), debug=True)
			retCode, outputStr = runJob("/usr/bin/env python3 -u $(which scons) -j%s" % jobs, "src/%s" % source)
			print(retCode, outputStr)

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
	retcode, output = runJob(f"wget -O {REPOSITORIES[source][0]}/{TAGS_SUBPAGE}{zipName}.zip")

	# If download failed, return error
	if retcode != 0:
		return retcode, output
	
	# Unzip tag and change folder name to match repository name
	runJob(f"unzip {zipName}.zip")

	# Check unzipped folder naming scheme
	folderName = source + '-' + zipName[1:] # Old naming system
	folderName = folderName if os.path.isdir(folderName) else source + '-' + zipName

	# Change folder name to match repository name
	retcode, output = runJob(f"mv {folderName} {source} && rm {zipName}.zip")

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
		retcode, output = runJob(f"git ls-remote --heads {REPOSITORIES[repo][0]} {branch}")
		
		# Check for errors
		if retcode != 0:
			return retcode, output
		
	# If output is empty, it means branch does not exist, default to devel
	if not output:
		branch = DEVEL_BRANCHNAME
	# Clone or checkout repository
	currentPath = os.getcwd()
	srcPath = os.path.join(currentPath, 'src')
	os.chdir(srcPath)
	destinyPath = os.path.join(srcPath,  repo)
	if os.path.exists(destinyPath):
			printMessage(text="The {} repository exists.".format(repo), debug=True)
			os.chdir(destinyPath)
			retcode, output = runJob(f"git pull ")
			if retcode != 0:
					printWarning(text=output, warningCode=GIT_PULL_WARNING)
					retcode = 0
			else:
					printMessage(text=green("{} updated.".format(repo)), debug=True)
	else:
			retcode, output = runJob(f"git clone --branch {branch} {REPOSITORIES[repo][0]}")
			if retcode == 0:
					printMessage(green(text="Clonned repository {}".format(repo)), debug=True)

	os.chdir(currentPath)
	return retcode, output


