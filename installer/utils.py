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
import sys, os, time, multiprocessing
from typing import List, Tuple, Union, Callable, Any
from io import FileIO
from subprocess import Popen, PIPE

# Installer imports
from .constants import (MODES, TAB_SIZE, XMIPP, VERNAME_KEY, LOG_FILE, IO_ERROR, ERROR_CODE,
	CMD_OUT_LOG_FILE, CMD_ERR_LOG_FILE, OUTPUT_POLL_TIME, XMIPP_VERSIONS, SCONS_INSTALL_ERROR)

####################### RUN FUNCTIONS #######################
def runJob(cmd: str, cwd: str='./', showOutput: bool=False, showError: bool=False, showCommand: bool=False, streaming: bool=False) -> Tuple[int, str]:
	"""
	### This function runs the given command.

	#### Params:
	- cmd (str): Command to run.
	- cwd (str): Optional. Path to run the command from. Default is current directory.
	- showOutput (bool): Optional. If True, output is printed.
	- showError (bool): Optional. If True, errors are printed.
	- showCommand (bool): Optional. If True, command is printed in blue.
	- streaming (bool): Optional. If True, output is shown in real time as it is being produced.

	#### Returns:
	- (int): Return code.
	- (str): Output of the command, regardless of if it is an error or regular output.
	"""
	# Printing command if specified
	if showCommand == True:
		print(blue(cmd))

	# Running command
	if streaming:
		retCode, outputStr = runStreamingJob(cmd, cwd=cwd, showOutput=showOutput, showError=showError)
	else:
		process = Popen(cmd, cwd=cwd, env=os.environ, stdout=PIPE, stderr=PIPE, shell=True)
		
		# Defining output string
		retCode = process.returncode
		output, err = process.communicate()
		outputStr = output.decode() if retCode else err.decode()

	# Printing output if specified
	if not streaming and showOutput:
		print('{}\n'.format(outputStr))

	# Printing errors if specified
	if not streaming and err and showError:
		print(red(outputStr))

	# Returing return code
	outputStr = outputStr[:-1] if outputStr.endswith('\n') else outputStr
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
		results = p.starmap(runLambda, funcs)
	
	# Return obtained result list
	return results

####################### PRINT FUNCTIONS #######################
def getFormattingTabs(text: str) -> str:
	"""
	### This method returns the given text, formatted to expand tabs into a fixed tab size.

	### Params:
	- text (str): The text to be formatted.

	### Returns:
	- (str): Formatted text.
	"""
	return text.expandtabs(TAB_SIZE)

def printError(errorMsg: str, retCode: int=1):
	"""
	### This function prints an error message and exits with the given return code.

	#### Params:
	- errorMsg (str): Error message to show.
	- retCode (int): Optional. Return code to end the exection with.
	"""
	# Print the error message in red color
	print(red(errorMsg))
	sys.exit(retCode) # TODO: Try API POST. Remove responsibility?

def printMessage(text: str, debug: bool=False):
	"""
	### This method prints the given text into the log file, and, if debug mode is active, also through terminal.

	### Params:
	- text (str): The text to be printed.
	- debug (bool): Indicates if debug mode is active.
	"""
	# If debug mode is active, print through terminal
	if debug:
		print(text, flush=True)
	
	# Open the file to add text
	try:
		with open(LOG_FILE, mode="a") as file:
			file.write(f"{text}\n")
	# If there was an error during the process, show error and exit
	except OSError:
		printError(f"Could not open log file to add info.\n{ERROR_CODE[IO_ERROR]}", retCode=IO_ERROR)

####################### EXECUTION MODE FUNCTIONS #######################
def getModeGroups() -> List[str]:
	"""
	### Returns all the group names of all the available execution modes.
	
	#### Returns:
	- (List[str]): List of all mode groups.
	"""
	return list(MODES.keys())

def getAllModes() -> List[str]:
	"""
	### Returns all the available execution modes.
	
	#### Returns:
	- (List[str]): List of all available modes.
	"""
	# Defining empty list to store modes
	modes = []

	# For each mode group, obtain mode names
	for modeGroup in getModeGroups():
		for mode in list(MODES[modeGroup].keys()):
			# Add mode to list
			modes.append(mode)
	
	# Return full mode list
	return modes

####################### COLORS #######################
def green(text: str) -> str:
	"""
	### This function returns the given text formatted in green color.

	#### Params:
	- text (str): Text to format.

	#### Returns:
	- (str): Text formatted in green color.
	"""
	return f"\033[92m{text}\033[0m"

def yellow(text: str) -> str:
	"""
	### This function returns the given text formatted in yellow color.

	#### Params:
	- text (str): Text to format.

	#### Returns:
	- (str): Text formatted in yellow color.
	"""
	return f"\033[93m{text}\033[0m"

def red(text: str) -> str:
	"""
	### This function returns the given text formatted in red color.

	#### Params:
	- text (str): Text to format.

	#### Returns:
	- (str): Text formatted in red color.
	"""
	return f"\033[91m{text}\033[0m"

def blue(text: str) -> str:
	"""
	### This function returns the given text formatted in blue color.

	#### Params:
	- text (str): Text to format.

	#### Returns:
	- (str): Text formatted in blue color.
	"""
	return f"\033[34m{text}\033[0m"

def bold(text: str) -> str:
	"""
	### This function returns the given text formatted in bold.

	#### Params:
	- text (str): Text to format.

	#### Returns:
	- (str): Text formatted in bold.
	"""
	return f"\033[1m{text}\033[0m"

####################### GIT FUNCTIONS #######################
def getCurrentBranch(dir: str='./') -> Union[str, None]:
	"""
	### This function returns the current branch of the repository of the given directory or None if it is not a repository.
	
	#### Params:
	- dir (str): Optional. Directory of the repository to get current branch from. Default is current directory.
	
	#### Returns:
	- (str | None): The name of the branch, or None if given directory is not a repository.
	"""
	# Getting current branch name
	retcode, output = runJob("git rev-parse --abbrev-ref HEAD", cwd=dir)

	# Return branch name or None if command failed
	return output if retcode == 0 else None

def isProductionMode() -> bool:
	"""
	### This function returns True if the current Xmipp repository is in production mode.
	
	#### Returns:
	- (bool): True if the repository is in production mode. False otherwise.
	"""
	currentBranch = getCurrentBranch()
	return currentBranch is None or currentBranch == XMIPP_VERSIONS[XMIPP][VERNAME_KEY]

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

####################### ENV FUNCTIONS #######################
def getCurrentEnvName() -> str:
	"""
	### This function returns the name of the active enviroment.

	#### Returns:
	- (str): Name of the active enviroment, or an empty string if there is no active enviroment or if errors happened.
	"""
	# Getting conda prefix path
	retCode, envPath = runJob("echo $CONDA_PREFIX")

	# If command failed, we assume we are not in an enviroment
	# If enviroment's path is empty, we are also in no env
	if retCode != 0 or not envPath:
		return ''

	# Return enviroment name, which is the last directory within the
	# path contained in $CONDA_PREFIX
	return envPath.split('/')[-1]

def isScipionEnv() -> bool:
	"""
	### This function returns True if the current active enviroment is Scipion's enviroment, or False otherwise or if some error happened.

	#### Returns:
	- (bool): True if the current active enviroment is Scipion's enviroment, or False otherwise or if some error happened.
	"""
	# Getting enviroment name
	envPath = getCurrentEnvName()

	# Enviroment's path needs to be a valid path containing binaries's
	# directory with scipion executable inside
	return envPath and os.path.exists(os.path.join(envPath, 'bin', 'scipion'))

####################### VERSION FUNCTIONS #######################
def versionToNumber(strVersion: str) -> float:
	"""
	### This function converts the version string into a version number that can be numerically compared.
	#### Supports any length of version numbers, but designed for three, in format X.Y.Z (mayor.minor.micro).

	#### Params:
	- strVersion (str): String containing the version numbers.

	#### Returns:
	- (float): Number representing the value of the version numbers combined.
	"""
	# Defining the most significant version number value
	mayorMultiplier = 100

	# Getting version numbers separated by dots
	listVersion = strVersion.split('.')

	# Getting the numeric version for each element
	numberVersion = 0
	for i in range(len(listVersion)):
		try:
			# Multiply each next number by the mayor multiplier divided by 10 in each iteration
			# That way, mayor * 100, minor * 10, micro * 1, next * 0.1, ...
			numberVersion += int(listVersion[i]) * (mayorMultiplier / (10 ** i))
		except Exception:
			# If there is some error, exit the loop
			break
	
	# Returning result number
	return numberVersion

def getPackageVersionCmd(packageName: str) -> Union[str, None]:
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

def getPackageVersionCmd(packageName: str) -> Union[str, None]:
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

def getPythonPackageVersion(packageName: str) -> Union[str, None]:
	"""
	### Retrieves the version of a Python package.

	Params:
	- packageName (str): Name of the Python package.

	Returns:
	- (str | None): Version string of the Python package or None if not found or errors happened.
	"""
	# Running command
	retCode, output = runJob(f'pip show {packageName}')

	# Extract variable if there were no errors
	if retCode == 0 and output:
		# Split output into lines and select the one which starts with 'Version:'
		output = output.splitlines()

		for line in output:
			if line.startswith('Version'):
				# If that line was found, return last word of such line
				return line.split()[-1]

####################### OTHER FUNCTIONS #######################
def installScons() -> bool:
	"""
	### This function attempts to install Scons in the current enviroment.
	"""
	# Attempt installing/upgrading Scons
	retCode = runJob('pip install --upgrade scons', streaming=True)[0]

	# Obtain enviroment's name for log's message
	envName = getCurrentEnvName()

	# If command failed, show error message and exit
	if retCode != 0:
		instructionStr = "Please, install it manually."
		envNameStr = f'Scons could not be installed in enviroment "{envName}".' if envName else f'Scons does not install automatically system wide by default.'
		printError(f'{envNameStr} {instructionStr}', retCode=SCONS_INSTALL_ERROR)

		return False
	
	# If succeeded, log message
	printMessage(f'Succesfully installed or updated Scons on {envName} enviroment.')
	return True

####################### AUX FUNCTIONS (INTERNAL USE ONLY) #######################
def runStreamingJob(cmd: str, cwd: str='./', showOutput: bool=False, showError: bool=False):
	"""
	### This function runs the given command and shows its output as it is being generated.
	#### Params:
	- cmd (str): Command to run.
	- cwd (str): Optional. Path to run the command from. Default is current directory.
	- showOutput (bool): Optional. If True, output is printed.
	- showError (bool): Optional. If True, errors are printed.
	#### Returns:
	- (int): Return code.
	- (str): Output of the command, regardless of if it is an error or regular output.
	"""
	# Creating writer and reader buffers in same tmp file
	error = False
	try:
		with open(CMD_OUT_LOG_FILE, "wb") as writerOut, open(CMD_OUT_LOG_FILE, "rb", 0) as readerOut,\
			open(CMD_ERR_LOG_FILE, "wb") as writerErr, open(CMD_ERR_LOG_FILE, "rb", 0) as readerErr:
			# Configure stdout and stderr deppending on param values
			stdout = writerOut if showOutput else PIPE
			stderr = writerErr if showError else PIPE

			# Run command and write output
			process = Popen(cmd, cwd=cwd, stdout=stdout, stderr=stderr, shell=True)
			outputStr = writeProcessOutput(process, readerOut, readerErr, showOutput=showOutput, showError=showError)
	except (KeyboardInterrupt, OSError) as e:
		error = True
		errorText = str(e)

	# Remove tmp files
	runJob(f"rm -f {CMD_OUT_LOG_FILE} {CMD_ERR_LOG_FILE}", cwd=cwd)

	# If there were errors, show them instead of returning
	if error:
		printError(errorText)

	# Return result
	return process.returncode, outputStr

def writeProcessOutput(process: Popen, readerOut: FileIO=None, readerErr: FileIO=None, showOutput: bool=False, showError: bool=False):
	"""
	### This function captures the output and errors of the given process as it runs.
	#### Params:
	- process (Popen): Running process.
	- readerOut (FileIO): Output reader.
	- readerErr (FileIO): Error reader.
	- showOutput (bool): Optional. If True, output is printed.
	- showError (bool): Optional. If True, errors are printed.
	#### Returns:
	- (str): Output of the command, regardless of if it is an error or regular output.
	"""
	# While process is still running, write output
	outputStr = ""
	while True:
		# Get process running status and print output
		isProcessFinished = process.poll() is not None
		outputStr += writeReaderLine(readerOut, show=showOutput)
		outputStr += writeReaderLine(readerErr, show=showError, err=True)

		# If process has finished, exit loop
		if isProcessFinished:
			break

		# Sleep before continuing to next iteration
		time.sleep(OUTPUT_POLL_TIME)

	return outputStr

def writeReaderLine(reader: FileIO, show: bool=False, err: bool=False):
	"""
	### This function captures the output and errors of the given process as it runs.

	#### Params:
	- reader (FileIO): Process reader.
	- show (bool): Optional. If True, reader text is printed.
	- err (bool): Optional. If True, reader's output is treated as an error.

	#### Returns:
	- (str): Output of the reader.
	"""
	# Getting raw line
	line = reader.read().decode()

	# If line is not empty, print it
	if line:
		# The line to print has to remove the last '\n'
		printedLine = line[:-1] if line.endswith('\n') else line
		printMessage(red(printedLine) if err else printedLine, debug=show)

	# Return line
	return red(line) if err else line

def writeReaderLine(reader: FileIO, show: bool=False, err: bool=False):
	"""
	### This function captures the output and errors of the given process as it runs.
	#### Params:
	- reader (FileIO): Process reader.
	- show (bool): Optional. If True, reader text is printed.
	- err (bool): Optional. If True, reader's output is treated as an error.
	#### Returns:
	- (str): Output of the reader.
	"""
	# Getting raw line
	line = reader.read().decode()

	# If line is not empty, print it
	if line:
		# The line to print has to remove the last '\n'
		printedLine = line[:-1] if line.endswith('\n') else line
		printMessage(red(printedLine) if err else printedLine, debug=show)

	# Return line
	return red(line) if err else line

def runLambda(function: Callable, args: Tuple[Any]=()):
	"""
	### This function is used to run other functions (intented for use inside a worker pool, so it can be picked).

	#### Params:
	- function (callable): Function to run.
	- args (tuple(any)): Optional. Function arguments.

	#### Returns:
	- (Any): Return value/(s) of the called function.
	"""
	return function(*args)
