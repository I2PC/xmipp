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
import subprocess, pkg_resources, sys, glob, distutils.spawn, os
from typing import List, Tuple, Union
from sysconfig import get_paths

# Installer imports
from .constants import SCONS_MINIMUM, MODES, CUDA_GCC_COMPATIBILITY, vGCC,\
	TAB_SIZE, XMIPP_VERSIONS, XMIPP, VERNAME_KEY, LOG_FILE, IO_ERROR, ERROR_CODE

####################### GENERAL FUNCTIONS #######################
def showError(errorMsg: str, retCode: int=1):
	"""
	### This function prints an error message and exits with the given return code.

	#### Params:
	- errorMsg (str): Error message to show.
	- retCode (int): Optional. Return code to end the exection with.
	"""
	# Print the error message in red color
	print(red(errorMsg))
	sys.exit(retCode)

def runJob(cmd: str, cwd: str='./', showOutput: bool=False, showError: bool=False, showCommand: bool=False) -> Tuple[int, str]:
	"""
	### This function runs the given command.

	#### Params:
	- cmd (str): Command to run.
	- cwd (str): Optional. Path to run the command from. Default is current directory.
	- showOutput (bool): Optional. If True, output is printed.
	- showError (bool): Optional. If True, errors are printed.
	- showCommand (bool): Optional. If True, command is printed in blue.

	#### Returns:
	- (int): Return code.
	- (str): Output of the command, regardless of if it is an error or regular output.
	"""
	# Running command
	process = subprocess.Popen(cmd, cwd=cwd, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
	output, err = process.communicate()

	# Defining output string
	outputStr = output.decode("utf-8") if process.returncode == 0 else err.decode("utf-8")

	# Printing command if specified
	if showCommand == True:
		print(blue(cmd))

	# Printing output if specified
	if showOutput == True:
		print('{}\n'.format(outputStr))

	if err:
		# Printing errors if specified
		if showError == True:
			print(red(outputStr))

	# Returing return code
	outputStr = outputStr[:-1] if outputStr.endswith('\n') else outputStr
	return process.returncode, outputStr

def runNetworkJob(cmd: str, cwd: str='/.', showOutput: bool=False, showError: bool=False, showCommand: bool=False, nRetries: int=5) -> Tuple[int, str]:
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
	# Running command up to nRetries times
	for _ in range(nRetries):
		retCode, output = runJob(cmd, cwd=cwd)
		# Break loop if success was achieved
		if retCode:
			break
	
	# Enforce message showing deppending on value
	if showCommand:
		print(blue(cmd))
	if showOutput:
		print('{}\n'.format(output))
	if showError:
		print(red(output))
	
	# Returning output and return code
	return retCode, showOutput

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

def printMessage(text: str, debug: bool=False):
	"""
	### This method prints the given text into the log file, and, if debug mode is active, also through terminal.

	### Params:
	- text (str): The text to be printed.
	- debug (bool): Indicates if debug mode is active.
	"""
	# Create log file if it does not exist
	if not os.path.exists(LOG_FILE):
		status, output = runJob(f"touch {LOG_FILE}")

		# Check if file was created successfully
		if not status:
			showError(f"{ERROR_CODE[IO_ERROR]}\nLog file could not be created. Check your directory permissions.\n{output}", retCode=IO_ERROR)
	
	# If debug mode is active, print through terminal
	if debug:
		print(text)
		sys.stdout.flush()
	
	# Open the file to add text
	try:
		with open(LOG_FILE, mode="+a") as file:
			file.write(f"{text}\n")
	# If there was an error during the process, show error and exit
	except Exception:
		showError(f"Could not open log file to add info.\n{ERROR_CODE[IO_ERROR]}", retCode=IO_ERROR)

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

def versionPackage(package):
	"""
	Retrieves the version of a package or program by executing '[package] --version' command.

	Params:
	- package (str): Name of the package or program.

	Returns:
	- str: Version information of the package or an empty string if not found.
	"""
	cmd = '{} --version'.format(package)
	status, output = runJob(cmd, showError=True)
	if status != 0 or output.find('not found') != -1:
		return ''
	return output

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
		if path != '' and versionPackage(path) != '':
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


def existPath(path):
		"""Return True if path exist, else False"""
		pass

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
		if isScipionVersion():
			status, output = runJob('pip install scons', showOutput=True, showCommand=True)
			if status == 0:
				return True
			else:
				showError(output, retCode=2)
		else:
			print(blue('Scons package not found, please install it  with \"pip install scons\".'))
			return False





def isScipionVersion():
		"""
		Checks if the current environment is a Scipion version.

		Returns:
		- bool: True if the environment is Scipion, False otherwise.
		"""
		status, output = runJob('echo $CONDA_PREFIX', showError=True)
		if status == 0 and output.find('scipion3') != -1:
			return True
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
				if float(nvccVersion) >= float(versionList[0]) and \
								float(nvccVersion) <= float(versionList[1]):
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

def isScipionEnviroment():
		status = runJob('conda env list | grep "scipion3 "')
		if status[0] == 0:
				return True
		else:
				False

def installScons():
		if isScipionEnviroment():
				status = runJob('conda activate scipion3')
				if status[0] == 0:
						status = runJob('pip install --upgrade scons')
						if status[0] == 0 and status[1].find('Successfully installed scons') != -1:
								print('Succesfully installed or updated Scons on scipion3 enviroment')
								return True
						else:
								return False, 'conda could not be installed on scipion3 enviroment with pip'
				else:
						return False, 'scipion3 enviroment could not be activated'
		else:
				return False, 'scipion3 enviroment not found'