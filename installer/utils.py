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
import pkg_resources, sys, glob, distutils.spawn, os, io, time
from typing import List, Tuple, Union
from sysconfig import get_paths
from subprocess import Popen, PIPE
from io import FileIO

# Installer imports
from .constants import SCONS_MINIMUM, MODES, CUDA_GCC_COMPATIBILITY, vGCC,\
	TAB_SIZE, XMIPP_VERSIONS, XMIPP, VERNAME_KEY, LOG_FILE, IO_ERROR, ERROR_CODE,\
	CMD_OUT_LOG_FILE, CMD_ERR_LOG_FILE, OUTPUT_POLL_TIME, SCONS_VERSION_ERROR
from .versions import getPackageVersionCmd

####################### GENERAL FUNCTIONS #######################
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
		output, err = process.communicate()
		retCode = process.returncode
		outputStr = output.decode() if not retCode else err.decode()

	# Printing output if specified
	if not streaming and showOutput:
		print('{}\n'.format(outputStr))

	# Printing errors if specified
	if not streaming and err and showError:
		print(red(outputStr))

	# Returing return code
	outputStr = outputStr[:-1] if outputStr.endswith('\n') else outputStr
	return retCode, outputStr

def runNetworkJob(cmd: str, cwd: str='./', showOutput: bool=False, showError: bool=False, showCommand: bool=False, nRetries: int=5) -> Tuple[int, str]:
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
	printMessage(red(errorMsg), debug=True)
	sys.exit(retCode)

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
	retCode = runNetworkJob("git fetch")[0]

	# Check if command succeeded
	if retCode != 0:
		return False

	# Get latest local commit
	localCommit = runJob(f"git rev-parse {currentBranch}")[1]

	# Get latest remote commit
	retCode, remoteCommit = runNetworkJob(f"git rev-parse origin/{currentBranch}")

	# Check if command succeeded
	if retCode != 0:
		return False
	
	# Return commit comparison
	return localCommit == remoteCommit

####################### ENV FUNCTIONS #######################
def getCurrentEnvName() -> str:
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

	# If enviroment's path is empty, we are in no env or an error happened
	if not envPath:
		return False
	
	# Returning result. Enviroment's path needs to be a valid path 
	# containing binaries's directory with scipion executable inside
	return os.path.exists(os.path.join(envPath, 'bin', 'scipion'))

####################### VERSION FUNCTIONS #######################
# UTILS
def findFileInDirList(fnH, dirlist):
	"""
	Finds a file in a list of directories.

	Params:
	- fnH (str): File name or pattern to search for.
	- dirlist (str or list of str): Single directory or list of directories to search in.

	Returns:
	- str: Directory containing the file if found, otherwise an empty string.
	"""
	if isinstance(dirlist, str):
			dirlist = [dirlist]

	for dir in dirlist:
			validDirs = glob.glob(os.path.join(dir, fnH))
			if len(validDirs) > 0:
					return os.path.dirname(validDirs[0])
	return ''

def whereIsPackage(packageName):
		"""
		Finds the directory of a specific package or program in the system.

		Params:
		- packageName (str): Name of the package or program.

		Returns:
		- str or None: Directory containing the package or program, or None if not found.
		"""
		programPath = distutils.spawn.find_executable(packageName)
		if programPath:
				programPath = os.path.realpath(programPath)
				return os.path.dirname(programPath)
		else:
				return None

def existPackage(packageName):
		"""Return True if packageName exist, else False"""
		path = pathPackage(packageName)
		if path and getPackageVersionCmd(path) is not None:
				return True
		return False

def pathPackage(packageName):
		"""
		Finds the path of a specific package in the system.

		Params:
		- packageName (str): Name of the package.

		Returns:
		- str: Path to the package.
		"""
		return runJob('which {}'.format(packageName), showError=True)[1]

def getINCDIRFLAG():
		return ' -I ' + os.path.join(get_paths()['data'].replace(' ', ''),  'include')

def versionToNumber(strVersion: str) -> float:
	"""
	### This function converts the version string into a version number that can be numerically compared.
	#### Supports any length of version numbers, but designed for three, in format X.Y.Z (mayor.minor.micro).

	#### Params:
	strVersion (str): String containing the version numbers.

	#### Returns:
	(float): Number representing the value of the version numbers combined.
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

def sconsVersion():
		# TODO: Revisar: no se "puede" devolver un número indeterminado de argumentos, tienes que devolver siempre el mismo numero, aunque a veces uno no sirva
		# No se entiende muy bien qué hace (y creo que se puede optimizar)
		"""
		Checks if the installed version of SCons meets a minimum requirement.

		Returns:
		- bool or tuple: If the installed SCons version meets the requirement, returns True.
										 If not, and it's possible to install SCons in a Scipion environment, it attempts to do so.
										 Returns a tuple (error_code, False) if installation fails.
		"""
		try:
			textVersion = pkg_resources.get_distribution("scons").version
			if versionToNumber(textVersion) >= versionToNumber(SCONS_MINIMUM):
				return True
		except Exception:
			pass
		if isScipionEnv():
			status, output = runJob('pip install scons', showOutput=True, showCommand=True)
			if status == 0:
				return True
			else:
				printError(output, retCode=2)
		else:
			print(blue('Scons package not found, please install it  with \"pip install scons\".'))
			return False

def getCompatibleGCC(nvccVersion):
		"""
		Retrieves compatible versions of GCC based on a given NVCC (NVIDIA CUDA Compiler) version.

		Params:
		- nvccVersion (str): Version of NVCC.

		Returns:
		- tuple: A tuple containing compatible GCC versions and a boolean indicating compatibility.
		"""
		# https://gist.github.com/ax3l/9489132
		for key, value in CUDA_GCC_COMPATIBILITY.items():
				versionList = key.split('-')
				if versionToNumber(nvccVersion) >= versionToNumber(versionList[0]) and \
								versionToNumber(nvccVersion) <= versionToNumber(versionList[1]):
						return value, True
		return vGCC, False

def CXXVersion(string):
		"""
		Extracts the version of a C++ compiler from a given string.

		Params:
		- string (str): Input string containing compiler version information.

		Returns:
		- str: Extracted C++ compiler version.
		"""
		idx = string.find('\n')
		idx2 = string[:idx].rfind(' ')
		version = string[idx2:idx]
		gxx_version = version.replace(' ', '')
		idx = gxx_version.rfind('.')
		gxx_version = gxx_version[:idx]
		return gxx_version

def MPIVersion(string):
		"""
		Extracts the MPI version information from a given string.

		Params:
		- string (str): Input string containing MPI version details.

		Returns:
		- str: Extracted MPI version information.
		"""
		idx = string.find('\n')
		idx2 = string[:idx].rfind(' ')
		return string[idx2:idx].replace(' ', '')

def findFileInDirList(fnH, dirlist):
		"""
    Searches for a specific file within a list of directories.

    Params:
    - fnH (str): Name of the file to be found.
    - dirlist (str or list): List of directories to search in.

    Returns:
    - str: Directory containing the specified file, or an empty string if not found.
		"""
		if isinstance(dirlist, str):
				dirlist = [dirlist]

		for dir in dirlist:
				validDirs = glob.glob(os.path.join(dir, fnH))
				if len(validDirs) > 0:
						return os.path.dirname(validDirs[0])
		return ''

def checkLib(gxx, libFlag):
		"""
		Checks if a specific library can be linked by a given compiler.

		Params:
		- gxx (str): Compiler command.
		- libFlag (str): Flag representing the library.

		Returns:
		- bool: True if the library can be linked, False otherwise.
		"""
		# TODO: Revisar: funciona como queremos?
		status = runJob('echo "int main(){}" > xmipp_check_lib.cpp ; ' + gxx + ' ' + libFlag + ' xmipp_check_lib.cpp', showError=True)[0]
		os.remove('xmipp_check_lib.cpp')
		os.remove('a.out') if os.path.isfile('a.out') else None
		return status == 0

def get_Hdf5_name(libdirflags):
		"""
		Identifies the HDF5 library name based on the given library directory flags.

		Params:
		- libdirflags (str): Flags specifying library directories.

		Returns:
		- str: Name of the HDF5 library ('hdf5', 'hdf5_serial', or 'hdf5' as default).
		"""
		libdirs = libdirflags.split("-L")
		for dir in libdirs:
				if os.path.exists(os.path.join(dir.strip(), "libhdf5.so")):
						return "hdf5"
				elif os.path.exists(os.path.join(dir.strip(), "libhdf5_serial.so")):
						return "hdf5_serial"
		return "hdf5"

def installScons():
	"""
	### This function attempts to install Scons.
	"""
	# Attempt installing/upgrading Scons
	retCode = runJob('pip install --upgrade scons', streaming=True)[0]

	# Obtain enviroment's name for log's message
	envName = getCurrentEnvName()

	# If command failed, show error message and exit
	if retCode != 0:
		printError(f'Scons could not be installed in enviroment "{envName}". Please, install it manually.', retCode=SCONS_VERSION_ERROR)

	# If succeeded, log message
	printMessage(f'Succesfully installed or updated Scons on {envName} enviroment.')
		
####################### AUX FUNCTIONS (INTERNAL USE ONLY) #######################
def runStreamingJob(cmd: str, cwd: str='./', showOutput: bool=False, showError: bool=False) -> Tuple[int, str]:
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
		with io.open(CMD_OUT_LOG_FILE, "wb") as writerOut, io.open(CMD_OUT_LOG_FILE, "rb", 0) as readerOut,\
			io.open(CMD_ERR_LOG_FILE, "wb") as writerErr, io.open(CMD_ERR_LOG_FILE, "rb", 0) as readerErr:
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

def writeProcessOutput(process: Popen, readerOut: FileIO, readerErr: FileIO, showOutput: bool=False, showError: bool=False) -> str:
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

def writeReaderLine(reader: FileIO, show: bool=False, err: bool=False) -> str:
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
