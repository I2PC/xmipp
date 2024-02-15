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
import os, shutil
from typing import Tuple

# Installer imports
from .exit import exitXmipp
from .constants import (XMIPP, XMIPP_COMPILE_LINES, XMIPP_CORE_COMPILE_LINES,
												XMIPP_VIZ_COMPILE_LINES, REPOSITORIES,
	ORGANIZATION_NAME, CUFFTADVSOR_ERROR, GOOGLETEST_ERROR,LIBSVM_ERROR, LIBCIFPP_ERROR, \
	DEVEL_BRANCHNAME, MASTER_BRANCHNAME, TAGS_SUBPAGE, HEADER0, HEADER1, HEADER2,
  CUFFTADVISOR, CTPL, GTEST, LIBSVM, LIBCIFPP, CLONNING_EXTERNAL_SOURCE_ERROR,
  CLONNING_XMIPP_SOURCE_ERROR, DOWNLOADING_XMIPP_SOURCE_ERROR, GIT_PULL_WARNING,
	XMIPP_COMPILATION_ERROR,XMIPPCORE_COMPILATION_ERROR,
  XMIPPVIZ_COMPILATION_ERROR, XMIPP_VERSIONS, VERNAME_KEY, DEPRECATE_ERROR,
  CLEANING_SOURCES_WARNING,CONFIG_FILE,CLEANING_BINARIES_WARNING, DONE0, DONE1,DONE2,
  INSTALLATION_ERROR, LINKING2SCIPION, VERSION_KEY, SCIPION_LINK_WARNING,
  CUFFTADVISOR,	CTPL,	GTEST, LIBSVM, LIBCIFPP, XMIPP_CORE,XMIPP_VIZ, XMIPP_PLUGIN)
from .utils import (runJob, getCurrentBranch, printError, printMessage, green,
										printWarning, createDir, getScipionHome, yellow, blue)
from .config.config import readConfig

####################### COMMAND FUNCTIONS #######################
def getSources(branch: str=None, LOG_FILE_path:str=''):
	"""
	### This function fetches the sources needed for Xmipp to compile.
	
	#### Params:
	- branch (str): Optional. Branch to clone the sources from.
	"""
	global logFilePath
	logFilePath = LOG_FILE_path

	# Enclose multi-word branch names in quotes
	if branch is not None and len(branch.split(' ')) > 1:
		branch = f"\"{branch}\""

	# Detect if Xmipp is in production or in devel mode
	currentBranch = getCurrentBranch()
	
	# Define sources list
	external_sources = [CUFFTADVISOR, CTPL, GTEST, LIBSVM, LIBCIFPP]
	sources = [XMIPP_CORE, XMIPP_VIZ, XMIPP_PLUGIN]

	printMessage(text=f'\n{HEADER1} Getting external sources...', debug=True)
	for source in external_sources:
			# Clone source repository
			status, output = cloneSourceRepo(repo=source, branch=REPOSITORIES[source][1])
			if status != 0:
				exitError(retCode=CLONNING_EXTERNAL_SOURCE_ERROR, output=output)
	printMessage(text=green(DONE1), debug=True)


	printMessage(text=f'\n{HEADER1} Getting Xmipp sources...', debug=True)
	for source in sources:
		# Non-git directories and production branch (master also counts) download from tags, the rest clone
		#if (currentBranch is None or currentBranch == XMIPP_VERSIONS[XMIPP][VERNAME_KEY]
		#				or currentBranch == MASTER_BRANCHNAME):
		# Download source tag
		status, output = downloadSourceTag(source)
		if not status:
			exitError(retCode=DOWNLOADING_XMIPP_SOURCE_ERROR, output=output)
		else:
			# Clone source repository
			status, output = cloneSourceRepo(source, branch=branch)
		# If download failed, return error
		if status != 0:
			exitError(retCode=CLONNING_XMIPP_SOURCE_ERROR, output=output)
	printMessage(text=green(DONE1), debug=True)



def compileExternalSources(jobs):
		"""
		Compiles external sources required by Xmipp.

		This function orchestrates the compilation process for external sources necessary for Xmipp.
		It compiles various components such as cuFFTAdvisor, googletest, libsvm, and libcifpp based on
		configurations read from the package. The compilation is performed based on the provided job count.

		Args:
		- jobs (int): The number of jobs/threads to be used for compilation.

		Raises:
		- RuntimeError: If any error occurs during the compilation process of external sources,
		  it raises a RuntimeError with error details.
		"""
		printMessage(text=f'\n{HEADER1} Compiling external sources...', debug=True)
		dictPackage, _ = readConfig()
		if dictPackage['CUDA'] == 'True':
			compile_cuFFTAdvisor()
		compile_googletest()
		compile_libsvm()
		compile_libcifpp(jobs)
		printMessage(text=green(DONE1), debug=True)

def compile_cuFFTAdvisor():
		"""
		Compiles the cuFFTAdvisor library for Xmipp.

		This function compiles the cuFFTAdvisor library required by Xmipp by executing the necessary build commands.
		Upon successful compilation, it copies the resulting library file to the Xmipp library directory.

		Raises:
		- RuntimeError: If any error occurs during the compilation and copying process of cuFFTAdvisor,
		  it raises a RuntimeError with error details.
		"""
		printMessage(f'{HEADER2} Compiling cuFFTAdvisor...', debug=True)
		advisorDir = "src/cuFFTAdvisor/"
		currDir = os.getcwd()
		libDir = "src/xmipp/lib/"
		if not os.path.exists(libDir):
				os.makedirs(libDir)
		os.chdir(advisorDir)
		retCode, outputStr = runJob("make all", printLOG=True, pathLOGFile=currDir)
		if retCode == 0:
				os.chdir(currDir)
				retCode, outputStr = runJob("cp " + advisorDir + "build/libcuFFTAdvisor.so" + " " + libDir)
				if retCode == 0:
						os.chdir(currDir)
				else:
						os.chdir(currDir)
						exitError(retCode=CUFFTADVSOR_ERROR, output=outputStr, pathFile=currDir)
		else:
				os.chdir(currDir)
				exitError(retCode=CUFFTADVSOR_ERROR, output=outputStr, pathFile=currDir)
		printMessage(green(DONE2), debug=True, pathFile=currDir)



def compile_googletest():
		"""
		Compiles the libsvm library for Xmipp.

		This function compiles the libsvm library required by Xmipp by executing the 'make lib' command.
		It copies the resulting library file to the Xmipp library directory upon successful compilation.

		Raises:
		- RuntimeError: If any error occurs during the compilation and copying process of libsvm,
		  it raises a RuntimeError with error details.
		"""

		printMessage(text=f'{HEADER2} Compiling googletest...', debug=True)
		currDir = os.getcwd()
		buildDir = os.path.join("src", "googletest", "build")
		if not os.path.exists(buildDir):
				os.makedirs(buildDir)
		os.chdir(buildDir)
		retCode, outputStr = runJob("cmake ..", printLOG=True, pathLOGFile=currDir)
		if retCode == 0:
				retCode, outputStr = runJob("make gtest gtest_main", printLOG=True, pathLOGFile=currDir)
				if retCode == 0:
						os.chdir(currDir)
						printMessage(text=green(DONE2), debug=True, pathFile=currDir)
				else:
						os.chdir(currDir)
						exitError(retCode=GOOGLETEST_ERROR, output=outputStr, pathFile=currDir)
		else:
				os.chdir(currDir)
				exitError(retCode=GOOGLETEST_ERROR, output=outputStr, pathFile=currDir)


def compile_libsvm():
		printMessage(text=f'{HEADER2} Compiling libsvm...', debug=True)
		# if the libsvm repo is updated, remember that the repoFork/Makefile was edited to remove references to libsvm-so.2
		currDir = os.getcwd()
		libsvmDir = os.path.join("src", "libsvm")
		os.chdir(libsvmDir)
		retCode, outputStr = runJob("make lib", printLOG=True, pathLOGFile=currDir)
		if retCode == 0:
				libDir = "src/xmipp/lib"
				os.chdir(currDir)
				if not os.path.exists(libDir):
						os.makedirs(libDir)
				retCode, outputStr = runJob("cp " + libsvmDir + "/libsvm.so" + " " + libDir, printLOG=True, pathLOGFile=currDir)
				if retCode == 0:
						os.chdir(currDir)
						printMessage(text=green(DONE2), debug=True, pathFile=currDir)
				else:
						os.chdir(currDir)
						exitError(retCode=LIBSVM_ERROR, output=outputStr, pathFile=currDir)
		else:
				os.chdir(currDir)
				exitError(retCode=LIBSVM_ERROR, output=outputStr, pathFile=currDir)


def compile_libcifpp(jobs):
		"""
		Compiles the libcifpp library for Xmipp.

		This function compiles the libcifpp library required by Xmipp using CMake. It sets up the build environment,
		compiles the library, and installs it within the Xmipp directory.

		Args:
		- jobs (int): The number of jobs/threads to be used for compilation.

		Raises:
		- RuntimeError: If any error occurs during the compilation and installation process of libcifpp,
		  it raises a RuntimeError with error details.
		"""
		printMessage(text=f'{HEADER2} Compiling libcifpp..', debug=True)
		currDir = os.getcwd()
		# Moving to library directory
		libcifppDir = os.path.join("src", "libcifpp")
		os.chdir(libcifppDir)
		# Installing
		fullDir = os.path.join(currDir, libcifppDir, '')
		retCode, outputStr = runJob("cmake -S . -B build -DCMAKE_INSTALL_PREFIX=" + fullDir +
							" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCIFPP_DOWNLOAD_CCD=OFF -DCIFPP_INSTALL_UPDATE_SCRIPT=OFF"
																, printLOG=True, pathLOGFile=currDir)
		if retCode == 0:
				retCode, outputStr = runJob(f"cmake --build build -j {jobs}", printLOG=True, pathLOGFile=currDir)
				if retCode == 0:
						retCode, outputStr = runJob("cmake --install build", printLOG=True, pathLOGFile=currDir)
						if retCode == 0:
								# Check if libcifpp created up on compiling lib or lib64 directory
								libcifppLibDir = "lib64" if os.path.exists("lib64") else "lib"
								# Copying .so file
								os.chdir(currDir)
								libDir = "src/xmipp/lib"
								if not os.path.exists(libDir):
										os.makedirs(libDir)
								retCode, outputStr = runJob("cp " + os.path.join(libcifppDir, libcifppLibDir,
																							 "libcifpp.so*") + " " + libDir, printLOG=True, pathLOGFile=currDir)
								if retCode == 0:
										printMessage(text=green(DONE2), debug=True, pathFile=currDir)
								else:
										exitError(retCode=LIBCIFPP_ERROR, output=outputStr, pathFile=currDir)

						else:
								os.chdir(currDir)
								exitError(retCode=LIBCIFPP_ERROR, output=outputStr, pathFile=currDir)

				else:
						os.chdir(currDir)
						exitError(retCode=LIBCIFPP_ERROR, output=outputStr, pathFile=currDir)

		else:
				os.chdir(currDir)
				exitError(retCode=LIBCIFPP_ERROR, output=outputStr, pathFile=currDir)


def compileSources(jobs, sconsPath:str):
		"""
		Compiles Xmipp source code.

		This function compiles the Xmipp core, Xmipp, and Xmipp Viz from their respective source directories.
		It utilizes the SCons build system with specified job parallelism to compile the source code.

		Args:
		- jobs (int): The number of jobs/threads to be used for compilation.

		Raises:
		- RuntimeError: If any error occurs during the compilation process for Xmipp components,
		  it raises an appropriate RuntimeError with error details.
		"""
		sources = [[XMIPP_CORE, XMIPPCORE_COMPILATION_ERROR, XMIPP_CORE_COMPILE_LINES],
						   [XMIPP, XMIPP_COMPILATION_ERROR, XMIPP_COMPILE_LINES],
							 [XMIPP_VIZ, XMIPPVIZ_COMPILATION_ERROR, XMIPP_VIZ_COMPILE_LINES]]
		dictPackage, _ = readConfig()
		for source in sources:
				compileXmippRun(source=source[0], sourceError=source[1], compileLines=source[2], sconsPath=sconsPath, jobs=jobs)


def compileXmippRun(source:str, sourceError:str, compileLines:list, sconsPath:str, jobs:int):
		printMessage(text=f'\n{HEADER1} Compiling {source}...', debug=True)
		retCode, outputStr = runJob(
								f"/usr/bin/env python3 -u {sconsPath} -j{jobs}",
								f"src/{source}",
								streaming=True, showOutput=False, showError=True,
								linesCompileBar=compileLines)
		if retCode != 0:
				exitError(retCode=sourceError, output=outputStr)
		printMessage(text=green('{}'.format(DONE1)), debug=True)


def compileAndInstall(args):
	# Get sources
	printMessage('\n---------------------------------------', debug=True)
	printMessage(text=f'\n{HEADER0} Get sources {HEADER0}', debug=True)
	getSources(branch=args.branch)
	# Compile external dependencies
	printMessage('\n---------------------------------------', debug=True)
	printMessage(text=f'\n{HEADER0} External compilations {HEADER0}', debug=True)
	compileExternalSources(jobs=args.jobs)
	printMessage('\n---------------------------------------', debug=True)
	# Compile Xmipp
	printMessage(text=f'\n{HEADER0} Xmipp compilation {HEADER0}', debug=True)
	dictPackages, _ = readConfig()
	compileSources(jobs=args.jobs, sconsPath=dictPackages['SCONS'])
	printMessage('\n---------------------------------------', debug=True)
	#Install
	printMessage(text=f'\n{HEADER0} Installation {HEADER0}', debug=True)
	install(directory=args.directory)


def install(directory):
		"""
		Installs Xmipp components to the specified directory.

		This function orchestrates the installation process of various Xmipp components to the given directory.
		It copies libraries, scripts, bindings, resources, and configuration files from the source directories
		to the specified installation directory.

		Args:
		- directory (str): The target directory where Xmipp components will be installed.

		Raises:
		- RuntimeError: If any error occurs during the installation process, a RuntimeError is raised.
		"""
		currentBranch = getCurrentBranch()
		if XMIPP_VERSIONS[XMIPP] == currentBranch:
				verbose = False
		else:
				verbose = True
		cleanDeprecated()
		cpCmd = "rsync -LptgoD "
		createDir(directory)
		createDir(directory + "/lib")
		retCode, outputStr = runJob(cpCmd + " src/*/lib/lib* " + directory + "/lib/")
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)
		if os.path.exists(directory + "/bin"):
				shutil.rmtree(directory + "/bin")
		if not os.path.exists(directory + "/bin"):
				os.makedirs(directory + "/bin")
		dirBin = os.path.join(os.getcwd(), "src/xmipp/bin/")
		filenames = [f for f in os.listdir(dirBin)]
		for f in filenames:
				if os.path.islink(os.path.join(dirBin, f)):
						retCode, outputStr = runJob('ln -s ' + os.path.join(dirBin, f) + ' ' + os.path.join(directory, 'bin', f))
						if retCode != 0:
								exitError(retCode=INSTALLATION_ERROR, output=outputStr)

				else:
						retCode, outputStr = runJob(cpCmd + os.path.join(dirBin, f) + ' ' + os.path.join(directory, 'bin', f))
						if retCode != 0:
								exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		destPathPyModule = os.path.expanduser(
				os.path.abspath(os.path.join(directory, "pylib", "xmippPyModules")))
		createDir(destPathPyModule)
		initFn = destPathPyModule + "/__init__.py"
		if not os.path.isfile(initFn):
				with open(initFn, 'w') as f:
						pass  # just to create a init file to be able to import it as module

		dirBin = os.path.join(os.getcwd(), 'src/xmipp/libraries/py_xmipp')
		folderNames = [x for x in os.walk(dirBin)]
		for folder in folderNames[1:]:
				folderName = os.path.basename(folder[0])
				for file in folder[2]:
						createDir(os.path.join(destPathPyModule, folderName))
						retCode, outputStr = runJob("ln -sf " + os.path.join(folder[0], file) + ' ' + os.path.join(destPathPyModule, folderName, file))
						if retCode != 0:
								exitError(retCode=INSTALLATION_ERROR, output=outputStr)


		createDir(directory + "/bindings")
		createDir(directory + "/bindings/matlab")
		retCode, outputStr = runJob(cpCmd + " src/xmipp/bindings/matlab/*.m* " + directory + "/bindings/matlab/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		createDir(directory + "/bindings/python")
		retCode, outputStr = runJob(
				cpCmd + " src/xmipp/bindings/python/xmipp_base.py " + directory + "/bindings/python/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)
		retCode, outputStr = runJob(cpCmd + " src/xmipp/bindings/python/xmipp.py " + directory + "/bindings/python/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " src/xmipp/bindings/python/xmipp_conda_envs.py " + directory + "/bindings/python/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " -r src/xmipp/bindings/python/envs_DLTK/ " + directory + "/bindings/python/envs_DLTK", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " src/xmipp/lib/xmippLib.so " + directory + "/bindings/python/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " src/xmipp/lib/_swig_frm.so " + directory + "/bindings/python/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		createDir(directory + "/bindings/python/sh_alignment")
		retCode, outputStr = runJob(cpCmd + " -r src/xmipp/external/sh_alignment/python/* " + directory + "/bindings/python/sh_alignment/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " src/xmipp/external/sh_alignment/swig_frm.py " + directory + "/bindings/python/sh_alignment/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		createDir(directory + "/resources")
		retCode, outputStr = runJob(cpCmd + " -r src/*/resources/* " + directory + "/resources/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		createDir(directory + "/bindings/java")
		retCode, outputStr = runJob(cpCmd + " -Lr src/xmippViz/java/lib " + directory + "/bindings/java/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " -Lr src/xmippViz/java/build " + directory + "/bindings/java/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " -Lr src/xmippViz/external/imagej " + directory + "/bindings/java/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " src/xmippViz/bindings/python/xmippViz.py " + directory + "/bindings/python/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " xmippEnv.json " + directory + "/xmippEnv.json", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		printMessage(text='Xmipp installed on {}'.format(os.path.join(os.getcwd(), directory.replace('./', ''))), debug=True)

		# Scipion connection
		linkToScipion(directory, verbose)

		runJob("touch %s/v%s" % (directory, XMIPP_VERSIONS[XMIPP][VERSION_KEY]), showCommand=verbose)  # version token
		fhBash = open(directory + "/xmipp.bashrc", "w")
		fhFish = open(directory + "/xmipp.fish", "w")
		fhBash.write("# This script is valid for bash and zsh\n\n")
		fhFish.write("# This script is valid for fish\n\n")

		XMIPP_HOME = os.path.realpath(directory)
		fhBash.write("export XMIPP_HOME=%s\n" % XMIPP_HOME)
		fhFish.write("set -x XMIPP_HOME %s\n" % XMIPP_HOME)

		XMIPP_SRC = os.path.realpath("src")
		fhBash.write("export XMIPP_SRC=%s\n" % XMIPP_SRC)
		fhFish.write("set -x XMIPP_SRC %s\n" % XMIPP_SRC)

		virtEnvDir = os.environ.get('VIRTUAL_ENV', '')  # if virtualEnv is used
		virtEnvLib = os.path.join(virtEnvDir, 'lib') if virtEnvDir else ''
		condaDir = os.environ.get('CONDA_PREFIX', '')  # if conda is used
		condaLib = os.path.join(condaDir, 'lib') if condaDir else ''
		fhBash.write("export PATH=%s/bin:$PATH\n" % XMIPP_HOME)
		fhBash.write(
				"export LD_LIBRARY_PATH=%s/lib:%s/bindings/python:%s:%s:$LD_LIBRARY_PATH\n"
				% (XMIPP_HOME, XMIPP_HOME, virtEnvLib, condaLib))
		fhBash.write(
				"export PYTHONPATH=%s/bindings/python:%s/pylib:$PYTHONPATH\n" % (
				XMIPP_HOME, XMIPP_HOME))
		fhFish.write("set -px PATH %s/bin\n" % XMIPP_HOME)
		fhFish.write("set -px LD_LIBRARY_PATH %s/lib %s/bindings/python %s %s\n"
								 % (XMIPP_HOME, XMIPP_HOME, virtEnvLib, condaLib))
		fhFish.write(
				"set -px PYTHONPATH %s/bindings %s/pylib\n" % (XMIPP_HOME, XMIPP_HOME))

		fhBash.write('\n')
		fhBash.write("alias x='xmipp'\n")
		fhBash.write("alias xsj='xmipp_showj'\n")
		fhBash.write("alias xio='xmipp_image_operate'\n")
		fhBash.write("alias xis='xmipp_image_statistics'\n")
		fhBash.write("alias xih='xmipp_image_header'\n")
		fhBash.write("alias xmu='xmipp_metadata_utilities'\n")
		fhFish.write('\n')
		fhFish.write("alias x 'xmipp'\n")
		fhFish.write("alias xsj 'xmipp_showj'\n")
		fhFish.write("alias xio 'xmipp_image_operate'\n")
		fhFish.write("alias xis 'xmipp_image_statistics'\n")
		fhFish.write("alias xih 'xmipp_image_header'\n")
		fhFish.write("alias xmu 'xmipp_metadata_utilities'\n")

		fhBash.close()
		fhFish.close()

def install(directory):
		"""
		Installs Xmipp components to the specified directory.

		This function orchestrates the installation process of various Xmipp components to the given directory.
		It copies libraries, scripts, bindings, resources, and configuration files from the source directories
		to the specified installation directory.

		Args:
		- directory (str): The target directory where Xmipp components will be installed.

		Raises:
		- RuntimeError: If any error occurs during the installation process, a RuntimeError is raised.
		"""

		currentBranch = getCurrentBranch()
		if XMIPP_VERSIONS[XMIPP] == currentBranch:
				verbose = False
		else:
				verbose = True
		cleanDeprecated()
		printMessage(text=f'{HEADER1} Linking Xmipp...', debug=True)

		cpCmd = "rsync -LptgoD "
		createDir(directory)
		createDir(directory + "/lib")
		retCode, outputStr = runJob(cpCmd + " src/*/lib/lib* " + directory + "/lib/")
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)
		if os.path.exists(directory + "/bin"):
				shutil.rmtree(directory + "/bin")
		if not os.path.exists(directory + "/bin"):
				os.makedirs(directory + "/bin")
		dirBin = os.path.join(os.getcwd(), "src/xmipp/bin/")
		filenames = [f for f in os.listdir(dirBin)]
		for f in filenames:
				if os.path.islink(os.path.join(dirBin, f)):
						retCode, outputStr = runJob('ln -s ' + os.path.join(dirBin, f) + ' ' + os.path.join(directory, 'bin', f))
						if retCode != 0:
								exitError(retCode=INSTALLATION_ERROR, output=outputStr)

				else:
						retCode, outputStr = runJob(cpCmd + os.path.join(dirBin, f) + ' ' + os.path.join(directory, 'bin', f))
						if retCode != 0:
								exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		destPathPyModule = os.path.expanduser(
				os.path.abspath(os.path.join(directory, "pylib", "xmippPyModules")))
		createDir(destPathPyModule)
		initFn = destPathPyModule + "/__init__.py"
		if not os.path.isfile(initFn):
				with open(initFn, 'w') as f:
						pass  # just to create a init file to be able to import it as module

		dirBin = os.path.join(os.getcwd(), 'src/xmipp/libraries/py_xmipp')
		folderNames = [x for x in os.walk(dirBin)]
		for folder in folderNames[1:]:
				folderName = os.path.basename(folder[0])
				for file in folder[2]:
						createDir(os.path.join(destPathPyModule, folderName))
						retCode, outputStr = runJob("ln -sf " + os.path.join(folder[0], file) + ' ' + os.path.join(destPathPyModule, folderName, file))
						if retCode != 0:
								exitError(retCode=INSTALLATION_ERROR, output=outputStr)


		createDir(directory + "/bindings")
		createDir(directory + "/bindings/matlab")
		retCode, outputStr = runJob(cpCmd + " src/xmipp/bindings/matlab/*.m* " + directory + "/bindings/matlab/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		createDir(directory + "/bindings/python")
		retCode, outputStr = runJob(
				cpCmd + " src/xmipp/bindings/python/xmipp_base.py " + directory + "/bindings/python/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " src/xmipp/bindings/python/xmipp.py " + directory + "/bindings/python/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " src/xmipp/bindings/python/xmipp_conda_envs.py " + directory + "/bindings/python/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " -r src/xmipp/bindings/python/envs_DLTK/ " + directory + "/bindings/python/envs_DLTK", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " src/xmipp/lib/xmippLib.so " + directory + "/bindings/python/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " src/xmipp/lib/_swig_frm.so " + directory + "/bindings/python/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		createDir(directory + "/bindings/python/sh_alignment")
		retCode, outputStr = runJob(cpCmd + " -r src/xmipp/external/sh_alignment/python/* " + directory + "/bindings/python/sh_alignment/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " src/xmipp/external/sh_alignment/swig_frm.py " + directory + "/bindings/python/sh_alignment/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		createDir(directory + "/resources")
		retCode, outputStr = runJob(cpCmd + " -r src/*/resources/* " + directory + "/resources/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		createDir(directory + "/bindings/java")
		retCode, outputStr = runJob(cpCmd + " -Lr src/xmippViz/java/lib " + directory + "/bindings/java/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " -Lr src/xmippViz/java/build " + directory + "/bindings/java/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " -Lr src/xmippViz/external/imagej " + directory + "/bindings/java/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " src/xmippViz/bindings/python/xmippViz.py " + directory + "/bindings/python/", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		retCode, outputStr = runJob(cpCmd + " xmippEnv.json " + directory + "/xmippEnv.json", showCommand=verbose)
		if retCode != 0:
				exitError(retCode=INSTALLATION_ERROR, output=outputStr)

		printMessage(text='Xmipp installed on {}'.format(os.path.join(os.getcwd(), directory.replace('./', ''))), debug=True)

		printMessage(text=green(f'\n{DONE1}'), debug=True)

		# Scipion connection
		printMessage(f"{HEADER1} Linking to Scipion...",debug=True)
		linkToScipion(directory, verbose)
		runJob("touch %s/v%s" % (directory, XMIPP_VERSIONS[XMIPP][VERSION_KEY]), showCommand=verbose)  # version token

		printMessage(f"{HEADER1} Creating the xmipp.bashrc file...",debug=True)
		fhBash = open(directory + "/xmipp.bashrc", "w")
		fhFish = open(directory + "/xmipp.fish", "w")
		fhBash.write("# This script is valid for bash and zsh\n\n")
		fhFish.write("# This script is valid for fish\n\n")

		XMIPP_HOME = os.path.realpath(directory)
		fhBash.write("export XMIPP_HOME=%s\n" % XMIPP_HOME)
		fhFish.write("set -x XMIPP_HOME %s\n" % XMIPP_HOME)

		XMIPP_SRC = os.path.realpath("src")
		fhBash.write("export XMIPP_SRC=%s\n" % XMIPP_SRC)
		fhFish.write("set -x XMIPP_SRC %s\n" % XMIPP_SRC)

		virtEnvDir = os.environ.get('VIRTUAL_ENV', '')  # if virtualEnv is used
		virtEnvLib = os.path.join(virtEnvDir, 'lib') if virtEnvDir else ''
		condaDir = os.environ.get('CONDA_PREFIX', '')  # if conda is used
		condaLib = os.path.join(condaDir, 'lib') if condaDir else ''
		fhBash.write("export PATH=%s/bin:$PATH\n" % XMIPP_HOME)
		fhBash.write(
				"export LD_LIBRARY_PATH=%s/lib:%s/bindings/python:%s:%s:$LD_LIBRARY_PATH\n"
				% (XMIPP_HOME, XMIPP_HOME, virtEnvLib, condaLib))
		fhBash.write(
				"export PYTHONPATH=%s/bindings/python:%s/pylib:$PYTHONPATH\n" % (
				XMIPP_HOME, XMIPP_HOME))
		fhFish.write("set -px PATH %s/bin\n" % XMIPP_HOME)
		fhFish.write("set -px LD_LIBRARY_PATH %s/lib %s/bindings/python %s %s\n"
								 % (XMIPP_HOME, XMIPP_HOME, virtEnvLib, condaLib))
		fhFish.write(
				"set -px PYTHONPATH %s/bindings %s/pylib\n" % (XMIPP_HOME, XMIPP_HOME))

		fhBash.write('\n')
		fhBash.write("alias x='xmipp'\n")
		fhBash.write("alias xsj='xmipp_showj'\n")
		fhBash.write("alias xio='xmipp_image_operate'\n")
		fhBash.write("alias xis='xmipp_image_statistics'\n")
		fhBash.write("alias xih='xmipp_image_header'\n")
		fhBash.write("alias xmu='xmipp_metadata_utilities'\n")
		fhFish.write('\n')
		fhFish.write("alias x 'xmipp'\n")
		fhFish.write("alias xsj 'xmipp_showj'\n")
		fhFish.write("alias xio 'xmipp_image_operate'\n")
		fhFish.write("alias xis 'xmipp_image_statistics'\n")
		fhFish.write("alias xih 'xmipp_image_header'\n")
		fhFish.write("alias xmu 'xmipp_metadata_utilities'\n")

		fhBash.close()
		fhFish.close()
		printMessage(green(DONE1), debug=True)

def cleanDeprecated():
		"""
		Cleans deprecated Xmipp programs and scripts.

		This function searches for deprecated Xmipp programs and scripts within the source directories
		and removes them from the 'src/xmipp/bin/' directory. Deprecated programs are identified by their
		absence from the current program listings and are removed to maintain an updated codebase.

		Note:
		The function relies on specific directory structures and filenames within the 'src/xmipp/'
		directory. It identifies deprecated programs based on their absence in the current program listings.

		Raises:
		- RuntimeError: If an error occurs during the removal process, a RuntimeError is raised.
		"""
		printMessage(text=f'\n{HEADER1} Cleaning deprecated programs...', debug=True)
		listCurrentPrograms = []
		retCode, outputStr = runJob('find src/xmipp/bin/*')
		files = outputStr.split('\n')
		listFilesXmipp = [os.path.basename(x).replace('xmipp_', '') for x in files]
		listCurrentFolders = [x[0] for x in os.walk('src/xmipp/applications/programs/')]
		for folder in listCurrentFolders[1:]:
				for file in os.listdir(folder):
						if '.cpp' in file:
								listCurrentPrograms.append(os.path.basename(folder))
		listCurrentScripts = [os.path.basename(x[0]) for x in os.walk('src/xmipp/applications/scripts/')]
		listCurrentPrograms = listCurrentPrograms + listCurrentScripts[1:]
		list2RemoveXmipp = [x for x in listFilesXmipp if (x not in listCurrentPrograms and 'test_' not in x)]
		for x in list2RemoveXmipp:
				retCode, outputStr = runJob('rm src/xmipp/bin/xmipp_{}'.format(x))
		if retCode != 0:
				exitError(retCode=DEPRECATE_ERROR, output=outputStr)

		if len(list2RemoveXmipp) > 0:
			printMessage(text=green('Deprecated programs removed'), debug=True)
		printMessage(green(DONE1), debug=True)

def cleanSources():
		DEPENDENCIES = [CUFFTADVISOR,	CTPL,	GTEST, LIBSVM, LIBCIFPP, XMIPP_CORE, XMIPP_VIZ, XMIPP_PLUGIN]
		for dep in DEPENDENCIES:
				runJob("rm -rf src/%s" % dep)
		retCode, outputStr = runJob("rm -rf src/xmipp/bin")
		if retCode != 0:
				printWarning(text=outputStr, warningCode=CLEANING_SOURCES_WARNING)
		retCode, outputStr = runJob("rm -rf src/xmipp/lib")
		if retCode != 0:
				printWarning(text=outputStr, warningCode=CLEANING_SOURCES_WARNING)
		retCode, outputStr = runJob("rm -rf src/xmipp/.sconsign.dblite")
		if retCode != 0:
				printWarning(text=outputStr, warningCode=CLEANING_SOURCES_WARNING)
		retCode, outputStr = runJob("git stash")  # to get exactly like in repo
		if retCode != 0:
				printWarning(text=outputStr, warningCode=CLEANING_SOURCES_WARNING)


def cleanBin():
		printMessage('Deleting binaries files...', debug=True)
		for ext in ['so', 'os', 'o']:
				runJob('find src/* -name "*.%s" -exec rm -rf {} \;' % ext)
		retCode, outputStr = runJob('find . -iname "*.pyc" -delete')
		if retCode != 0:
				printWarning(text=outputStr, warningCode=CLEANING_BINARIES_WARNING)
		retCode, outputStr = runJob("rm -rf %s build" % CONFIG_FILE)
		if retCode != 0:
				printWarning(text=outputStr, warningCode=CLEANING_BINARIES_WARNING)
		retCode, outputStr = runJob('find . -iname "*.dblite" -delete')
		if retCode != 0:
				printWarning(text=outputStr, warningCode=CLEANING_BINARIES_WARNING)
		cleanEmptyFolders()
		printMessage(green('Done'), debug=True)



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

	currentPath = os.getcwd()
	srcPath = os.path.join(currentPath, 'src')
	os.chdir(srcPath)
	# Download tag
	zipName = XMIPP_VERSIONS[source][VERNAME_KEY]
	baseName = REPOSITORIES[source][0].replace(f'{source}.git',f'{source}')
	url = f'{baseName}/{TAGS_SUBPAGE}{zipName}.zip'
	print(url)
	retcode, output = runJob(f"wget {url} ")

	# If download failed, return error
	if retcode != 0:
		os.chdir(currentPath)
		return False, output
	# Unzip tag and change folder name to match repository name
	retcode, output = runJob(f"yes | unzip {zipName}.zip", showCommand=True)
	if retcode != 0:
		os.chdir(currentPath)
		return False, output
	# Check unzipped folder naming scheme
	folderName = source + '-' + zipName[1:] # Old naming system
	folderName = folderName if os.path.isdir(folderName) else source + '-' + zipName
	# Change folder name to match repository name
	retcode, output = runJob(f"mv {folderName} {source} && rm {zipName}.zip")
	if retcode != 0:
		os.chdir(currentPath)
		return False, output
	# Return last command's code and output.
	os.chdir(currentPath)
	return True, ''

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
		printMessage(text="The {} repository exists.".format(repo), debug=True, pathFile=currentPath)
		os.chdir(destinyPath)
		retcode, output = runJob(f"git pull ")
		if retcode != 0 and retcode != 1:
			printWarning(text=output, warningCode=GIT_PULL_WARNING, pathFile=currentPath)
			retcode = 0
		else:
			printMessage(text=green("{} updated.".format(repo)), debug=True, pathFile=currentPath)
	else:
		retcode, output = runJob(f"git clone --branch {branch} {REPOSITORIES[repo][0]}")
		if retcode == 0:
			printMessage(green(text="Clonned repository {}".format(repo)), debug=True, pathFile=currentPath)
	os.chdir(currentPath)
	if retcode == 1: retcode = 0
	return retcode, output

def linkToScipion(directory:str, verbose:bool=False):
		"""
		:param directory:
		:param verbose:
		:return:

		Creates symbolic links to integrate Xmipp with Scipion.

		This function creates symbolic links to integrate Xmipp with Scipion by linking specific libraries
		and bindings required for their interaction.

		Args:
		- directory (str): The directory containing the necessary files for linking.
		- verbose (bool, optional): Controls the verbosity of the function (default: False).

		Returns:
		None

		Raises:
		- RuntimeError: If linking encounters errors, it raises a RuntimeError with details.
		- Warning: If the expected Xmipp directory is not found, it issues a warning.

		Note:
		This function assumes the presence of certain directories and files within the environment.
		Specifically, it expects the directory structure to include paths required for the linking process.
		"""

		scipionSoftware = os.environ.get('SCIPION_SOFTWARE', os.path.join(getScipionHome(), 'software'))
		scipionLibs = os.path.join(scipionSoftware, 'lib')
		scipionBindings = os.path.join(scipionSoftware, 'bindings')
		scipionSoftwareEM = os.path.join(scipionSoftware, 'em')
		xmippHomeLink = os.path.join(scipionSoftwareEM, 'xmipp')
		currentDir = os.getcwd()
		dirnameAbs = os.path.join(currentDir, directory)
		if os.path.isdir(scipionLibs) and os.path.isdir(scipionBindings):
			printMessage('scipionSoftware: {}'.format(scipionSoftware), debug=True)
		if os.path.isdir(xmippHomeLink):
				retCode, outputStr = runJob("rm %s" %xmippHomeLink, showCommand=verbose)
				if retCode != 0:
					exitError(retCode=LINKING2SCIPION, output=outputStr)
				retCode, outputStr = runJob("ln -srf %s %s" % (dirnameAbs, xmippHomeLink), showCommand=verbose)
				if retCode != 0:
						exitError(retCode=LINKING2SCIPION, output=outputStr)
				xmippLink = os.readlink(xmippHomeLink)
				coreLib = os.path.join(xmippLink, "lib", "libXmippCore.so")
				xmippLib = os.path.join(xmippLink, "lib", "libXmipp.so")
				SVMLib = os.path.join(xmippLink, "lib", "libsvm.so")
				CIFPPLib = os.path.join(xmippLink, "lib", "libcifpp.so*")
				bindings = os.path.join(xmippLink, "bindings", "python")
				runLinkScipion("ln -srf %s %s" % (coreLib, scipionLibs), cwd=scipionSoftwareEM, verbose=verbose)
				runLinkScipion("ln -srf %s %s" % (SVMLib, scipionLibs), cwd=scipionSoftwareEM, verbose=verbose)
				runLinkScipion("ln -srf %s %s" % (CIFPPLib, scipionLibs), cwd=scipionSoftwareEM, verbose=verbose)
				runLinkScipion("ln -srf %s %s" % (xmippLib, scipionLibs), cwd=scipionSoftwareEM, verbose=verbose)
				runLinkScipion("ln -srf %s %s" % (bindings, scipionBindings), cwd=scipionSoftwareEM, verbose=verbose)
				printMessage(text=blue(str("Xmipp linked to Scipion on " + xmippHomeLink) + (' ' * 150)), debug=True)
				printMessage(green(DONE1), debug=True)

		else:
				printWarning(text='', warningCode=SCIPION_LINK_WARNING)

def runLinkScipion(cmd:str, cwd:str, verbose:bool):
		retCode, outputStr = runJob(cwd=cwd, cmd=cmd, showCommand=verbose)
		if retCode != 0:
				exitError(retCode=LINKING2SCIPION, output=outputStr)

def cleanEmptyFolders():
		log = []
		path = "src/xmipp/applications/programs/"
		retCode, outputStr = runJob("find {} -type d -empty".format(path))
		if retCode != 0:
				printWarning(text=outputStr, warningCode=CLEANING_BINARIES_WARNING)
		for folder in log:
				if os.path.isdir(folder):
						retCode, outputStr = runJob("rm -rf {}".format(folder))
						if retCode != 0:
								printWarning(text=outputStr, warningCode=CLEANING_BINARIES_WARNING)

def exitError(output:str='', retCode:int=0, pathFile:str=''):
		printError(errorMsg=output, retCode=retCode, pathFile=pathFile)
		dictPackages, _ = readConfig()
		exitXmipp(retCode=retCode, dictPackages=dictPackages)