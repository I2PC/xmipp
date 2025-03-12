# ***************************************************************************
# * Authors:		Alberto García (alberto.garcia@cnb.csic.es)
# *							Martín Salinas (martin.salinas@cnb.csic.es)
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
Module containing useful functions used by the installation process.
"""

# General imports
import os, multiprocessing
from typing import List, Tuple, Callable, Any, Optional
from subprocess import Popen, PIPE
from threading import Thread
from io import BufferedReader

# Installer imports
from .constants import XMIPP, VERNAME_KEY, XMIPP_VERSIONS, INTERRUPTED_ERROR
from .constants.versions import LATEST_RELEASE_NAME
from .logger import blue, red, logger

####################### RUN FUNCTIONS #######################
def runJob(cmd: str, cwd: str='./', showOutput: bool=False, showError: bool=False,
					 showCommand: bool=False, substitute: bool=False, logOutput: bool=False) -> Tuple[int, str]:
	"""
	### This function runs the given command.

	#### Params:
	- cmd (str): Command to run.
	- cwd (str): Optional. Path to run the command from. Default is current directory.
	- showOutput (bool): Optional. If True, output is printed.
	- showError (bool): Optional. If True, errors are printed.
	- showCommand (bool): Optional. If True, command is printed in blue.
	- substitute (bool): Optional. If True, output will replace previous line.
	- logOutput (bool): Optional. If True, output will be stored in the log.

	#### Returns:
	- (int): Return code.
	- (str): Output of the command, regardless of if it is an error or regular output.
	"""
	# Printing command if specified
	__logToSelection(blue(cmd), sendToLog=logOutput, sendToTerminal=showCommand, substitute=substitute)

	# Running command
	process = Popen(cmd, cwd=cwd, env=os.environ, stdout=PIPE, stderr=PIPE, shell=True)
	try:
		process.wait()
	except KeyboardInterrupt:
		return INTERRUPTED_ERROR, ""
	
	# Defining output string
	retCode = process.returncode
	output, err = process.communicate()
	outputStr = output.decode() if not retCode and output else err.decode()
	outputStr = outputStr[:-1] if outputStr.endswith('\n') else outputStr

	# Printing output if specified
	if not retCode:
		__logToSelection(f"{outputStr}", sendToLog=logOutput, sendToTerminal=showOutput, substitute=substitute)

	# Printing errors if specified
	if retCode and showError:
		if logOutput:
			logger.logError(outputStr)
		else:
			print(red(outputStr))

	# Returing return code
	return retCode, outputStr

def runInsistentJob(cmd: str, cwd: str='./', showOutput: bool=False, showError: bool=False, showCommand: bool=False, nRetries: int=5) -> Tuple[int, str]:
	"""
	### This function runs the given network command and retries it the number given of times until one of the succeeds or it fails for all the retries.

	#### Params:
	- cmd (str): Command to run.
	- cwd (str): Optional. Path to run the command from. Default is current directory.
	- showOutput (bool): Optional. If True, output is printed.
	- showError (bool): Optional. If True, errors are printed.
	- showCommand (bool): Optional. If True, command is printed in blue.
	- nRetries (int): Optional. Maximum number of retries for the command.

	#### Returns:
	- (int): Return code.
	- (str): Output of the command, regardless of if it is an error or regular output.
	"""
	# Running command up to nRetries times (improves resistance to small network errors)
	for _ in range(nRetries):
		retCode, output = runJob(cmd, cwd=cwd)
		# Break loop if success was achieved
		if retCode == 0:
			break
	
	# Enforce message showing deppending on value
	if showCommand:
		print(blue(cmd))
	if showOutput:
		print('{}\n'.format(output))
	if showError:
		print(red(output))
	
	# Returning output and return code
	return retCode, output

def runParallelJobs(funcs: List[Tuple[Callable, Tuple[Any]]], nJobs: int=multiprocessing.cpu_count()) -> List:
	"""
	### This function runs the given command list in parallel.

	#### Params:
	- funcs (list(tuple(callable, tuple(any)))): Functions to run with parameters, if there are any.

	#### Returns:
	- (list): List containing the return of each function.
	"""
	# Creating a pool of n concurrent jobs
	with multiprocessing.Pool(nJobs) as p:
		# Run each function and obtain results
		results = p.starmap(__runLambda, funcs)
	
	# Return obtained result list
	return results

def runStreamingJob(cmd: str, cwd: str='./', showOutput: bool=False, showError: bool=False, substitute: bool=False) -> int:
	"""
	### This function runs the given command and shows its output as it is being generated.

	#### Params:
	- cmd (str): Command to run.
	- cwd (str): Optional. Path to run the command from. Default is current directory.
	- showOutput (bool): Optional. If True, output is printed.
	- showError (bool): Optional. If True, errors are printed.
	- substitute (bool): Optional. If True, output will replace previous line.

	#### Returns:
	- (int): Return code.
	"""
	# Create a Popen instance and error stack
	logger(cmd)
	process = Popen(cmd, cwd=cwd, stdout=PIPE, stderr=PIPE, shell=True)
	
	# Create and start threads for handling stdout and stderr
	threadOut = Thread(target=__handleOutput, args=(process.stdout, showOutput, substitute))
	threadErr = Thread(target=__handleOutput, args=(process.stderr, showError, substitute, True))
	threadOut.start()
	threadErr.start()

	# Wait for execution, handling keyboard interruptions
	try:
		process.wait()
		threadOut.join()
		threadErr.join()
	except (KeyboardInterrupt):
		process.returncode = INTERRUPTED_ERROR
	
	return process.returncode

####################### GIT FUNCTIONS #######################
def getCurrentBranch(dir: str='./') -> str:
	"""
	### This function returns the current branch of the repository of the given directory or empty string if it is not a repository or a recognizable tag.
	
	#### Params:
	- dir (str): Optional. Directory of the repository to get current branch from. Default is current directory.
	
	#### Returns:
	- (str): The name of the branch, 'HEAD' if a tag, or empty string if given directory is not a repository or a recognizable tag.
	"""
	# Getting current branch name
	retcode, branchName = runJob("git rev-parse --abbrev-ref HEAD", cwd=dir)

	# If there was an error, we are in no branch
	return branchName if not retcode else ''
	
def isProductionMode(dir: str='./') -> bool:
	"""
	### This function returns True if the current Xmipp repository is in production mode.
	
	#### Params:
	- dir (str): Optional. Directory of the repository where the check will happen. Default is current directory.
	
	#### Returns:
	- (bool): True if the repository is in production mode. False otherwise.
	"""
	currentBranch = getCurrentBranch(dir=dir)
	return currentBranch is None or currentBranch == XMIPP_VERSIONS[XMIPP][VERNAME_KEY]

def isTag(dir: str='./') -> bool:
	"""
	### This function returns True if the current Xmipp repository is in a tag.

	#### Params:
	- dir (str): Optional. Directory of the repository where the check will happen. Default is current directory.
	
	#### Returns:
	- (bool): True if the repository is a tag. False otherwise.
	"""
	currentBranch = getCurrentBranch(dir=dir)
	return not currentBranch or currentBranch == "HEAD"


def getCurrentName():
	"""
	### This function returns the current branch of the repository of the given directory or the name of the tag.

	#### Params:
	- dir (str): Optional. Directory of the repository where the check will happen. Default is current directory.

	#### Returns:
	- (str): The name of the branch or the tag.
	"""
	if isTag():
		return LATEST_RELEASE_NAME
	else:
		return getCurrentBranch()



def isBranchUpToDate(dir: str='./') -> bool:
	"""
	### This function returns True if the current branch is up to date, or False otherwise or if some error happened.
	
	#### Params:
	- dir (str): Optional. Directory of the repository to get current branch from. Default is current directory.
	
	#### Returns:
	- (bool): True if the current branch is up to date, or False otherwise or if some error happened.
	"""
	# Getting current branch
	currentBranch = getCurrentBranch(dir=dir)

	# Check if previous command succeeded
	if currentBranch is None:
		return False
	
	# Update branch
	retCode = runInsistentJob("git fetch")[0]

	# Check if command succeeded
	if retCode != 0:
		return False

	# Get latest local commit
	localCommit = runJob(f"git rev-parse {currentBranch}")[1]

	# Get latest remote commit
	retCode, remoteCommit = runInsistentJob(f"git rev-parse origin/{currentBranch}")

	# Check if command succeeded
	if retCode != 0:
		return False
	
	# Return commit comparison
	return localCommit == remoteCommit

####################### VERSION FUNCTIONS #######################
def getPackageVersionCmd(packageName: str) -> Optional[str]:
	"""
	### Retrieves the version of a package or program by executing '[packageName] --version' command.

	Params:
	- packageName (str): Name of the package or program.

	Returns:
	- (str | None): Version information of the package or None if not found or errors happened.
	"""
	# Running command
	retCode, output = runJob(f'{packageName} --version')

	# Check result if there were no errors
	return output if retCode == 0 else None

####################### AUX FUNCTIONS (INTERNAL USE ONLY) #######################
def __handleOutput(stream: BufferedReader, show: bool=False, substitute: bool=False, err: bool=False):
	"""
	### This function receives a process output stream and logs its lines.

	#### Params:
	- stream (BufferedReader): Function to run.
	- show (bool): Optional. If True, output will also be printed through terminal.
	- substitute (bool): Optional. If True, output will replace previous line. Only used when show is True.
	- err (bool): Optional. If True, the stream contains an error. Otherwise, it is regular output.
	"""
	# If print through terminal is enabled with substitution, add a first line break
	if show and substitute:
		print("")

	# Print all lines in the process output
	for line in iter(stream.readline, b''):
		line = line.decode().replace("\n", "")
		if err:
			line = red(line)
		logger(line, forceConsoleOutput=show, substitute=substitute)

def __runLambda(function: Callable, args: Tuple[Any]=()):
	"""
	### This function is used to run other functions (intented for use inside a worker pool, so it can be picked).

	#### Params:
	- function (callable): Function to run.
	- args (tuple(any)): Optional. Function arguments.

	#### Returns:
	- (Any): Return value/(s) of the called function.
	"""
	return function(*args)

def __logToSelection(message: str, sendToLog: bool=True, sendToTerminal: bool=False, substitute: bool=False):
	"""
	### This function logs the given message into the selected logging platform.

	#### Params:
	- message (str): Message to log.
	- sendToLog (bool): Optional. If True, message is sent to the logger (into file).
	- sendToTerminal (bool): Optional. If True, message is sent to terminal.
	- substitute (bool): Optional. If True, message will replace last terminal printed message. Only used when all other variables are True.
	"""
	if sendToLog:
		logger(message, forceConsoleOutput=sendToTerminal, substitute=substitute)
	else:
		if sendToTerminal:
			print(message)
