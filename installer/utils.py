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
import subprocess, pkg_resources, sys
from os import environ, remove, path
from typing import Union, List, Tuple

# Installer imports
from .constants import SCONS_MINIMUM, MODES, CUDA_GCC_COMPATIBILITY, vGCC, TAB_SIZE
import glob
import distutils.spawn
from os import path
from sysconfig import get_paths


####################### COLORS #######################
def green(text: str) -> str:
	"""
	### This function returns the given text formatted in green color.

	#### Params:
	text (str): Text to format.

	#### Returns:
	(str): Text formatted in green color.
	"""
	return "\033[92m" + text + "\033[0m"

def yellow(text: str) -> str:
	"""
	### This function returns the given text formatted in yellow color.

	#### Params:
	text (str): Text to format.

	#### Returns:
	(str): Text formatted in yellow color.
	"""
	return "\033[93m" + text + "\033[0m"

def red(text: str) -> str:
	"""
	### This function returns the given text formatted in red color.

	#### Params:
	text (str): Text to format.

	#### Returns:
	(str): Text formatted in red color.
	"""
	return "\033[91m" + text + "\033[0m"

def blue(text: str) -> str:
	"""
	### This function returns the given text formatted in blue color.

	#### Params:
	text (str): Text to format.

	#### Returns:
	(str): Text formatted in blue color.
	"""
	return "\033[34m" + text + "\033[0m"

def bold(text: str) -> str:
	"""
	### This function returns the given text formatted in bold.

	#### Params:
	text (str): Text to format.

	#### Returns:
	(str): Text formatted in bold.
	"""
	return "\033[1m" + text + "\033[0m"

####################### GENERAL FUNCTIONS #######################
def getFormattingTabs(text: str) -> str:
	"""
	### This method returns the given text, formatted to expand tabs into a fixed tab size.

	### Params:
	- text (str): The text to be formatted.

	### Returns:
	(str): Formatted text.
	"""
	return text.expandtabs(TAB_SIZE)

def showError(errorMsg: str, retCode: int=1):
	"""
	### This function prints an error message and exits with the given return code.

	#### Params:
	errorMsg (str): Error message to show.
	retCode (int): Optional. Return code to end the exection with.
	"""
	# Print the error message in red color
	print(red(errorMsg))
	sys.exit(retCode)

def runJob(cmd: str, cwd: str='./', showOutput: bool=True, logOut: list=None, logErr: list=None, showError: bool=True, showCommand: bool=True) -> Union[bool, None]:
	"""
	### This function runs the given command.

	#### Params:
	cmd (str): Command to run.
	cwd (str): Optional. Path to run the command from. Default is current directory.
	showOutput (bool): Optional. If True, output is printed.
	logOut (list): Optional. List to store the output into.
	logErr (list): Optional. List to store the errors into.
	showError (bool): Optional. If True, errors are printed.
	showCommand (bool): Optional. If True, command is printed in blue.

	#### Returns:
	(bool): True if there were no errors. If there were errors, False if error string is not empty, None otherwise.
	"""
	# Running command
	p = subprocess.Popen(cmd, cwd=cwd, env=environ, stdout=subprocess.PIPE,
												stderr=subprocess.PIPE, shell=True)
	output, err = p.communicate()

	# Printing command if specified
	if showCommand == True:
		print(blue(cmd))

	# Printing output if specified
	if showOutput == True:
		print('{}\n'.format(output.decode("utf-8")))
	# Storing output in list if specified
	if logOut is not None:
		logOut.append(output.decode("utf-8"))

	if err:
		# Printing errors if specified
		if showError == True:
			print(red(err.decode("utf-8")))
		# Storing errors in list if specified
		if logErr is not None:
			logErr.append(err.decode("utf-8"))
		# If error string is not empty, return False
		if err.decode("utf-8") != '':
			return False
	else:
		# If there were no errors, return True
		return True

def runBackgroundJob(cmd: str, cwd: str='./') -> Tuple[bool, str]:
	"""
	### This function runs the given command in the background and returns the results.

	#### Params:
	cmd (str): Command to run.
	cwd (str): Optional. Path to run the command from. Default is current directory.

	#### Returns:
	(bool): True if the command worked or False if it did not.
	(str): Output data from the command if it worked or error if it failed.
	"""
	# Running command and capturing output
	process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
	output, error = process.communicate()

	# Returning output
	if process.returncode == 0:
		return True, output.decode('utf-8').strip()
	else:
		return False, error.decode('utf-8')

####################### EXECUTION MODE FUNCTIONS #######################
def getModeGroups():
	"""
	### Returns all the group names of all the available execution modes.
	
	#### Returns:
	(List[str]): List of all mode groups.
	"""
	return list(MODES.keys())

def getAllModes() -> List[str]:
	"""
	### Returns all the available execution modes.
	
	#### Returns:
	(List[str]): List of all available modes.
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
				validDirs = glob.glob(path.join(dir, fnH))
				if len(validDirs) > 0:
						return path.dirname(validDirs[0])
		return ''


def versionPackage(package):
		"""
		Retrieves the version of a package or program by executing '[package] --version' command.

		Params:
		- package (str): Name of the package or program.

		Returns:
		- str: Version information of the package or an empty string if not found.
		"""
		str = []
		cmd = '{} --version'.format(package)
		if runJob(cmd, showOutput=False, logOut=str, showCommand=False):
				for line in str:
						if line.find('not found') != -1:
								return ''
		return str[0]


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
				programPath = path.realpath(programPath)
				return path.dirname(programPath)
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
		path = []
		runJob('which {}'.format(packageName), showCommand=False,
					 showOutput=False, logOut=path)
		path = path[0].replace('\n', '')
		return path


def existPath(path):
		"""Return True if path exist, else False"""
		pass

def getINCDIRFLAG():
		return ' -I ' + path.join(get_paths()['data'].replace(' ', ''),  'include')



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
			outlog = []
			errlog = []
			if runJob('pip install scons', logOut=outlog, logErr=errlog, showError=False, showCommand=True):
				return True
			else:
				print(red(errlog[0]))
				return 2, False
		else:
			print(blue('Scipion enviroment not found, please install manually scons library'))
			return False


def CUDAVersion(strVersion):
		nvcc_version = ''
		if strVersion.find('release') != -1:
				idx = strVersion.find('release ')
				nvcc_version = strVersion[idx + len('release '):
																		idx + strVersion[idx:].find(',')]
		return nvcc_version



def isScipionVersion():
		"""
		Checks if the current environment is a Scipion version.

		Returns:
		- bool: True if the environment is Scipion, False otherwise.
		"""
		condaEnv = []
		if runJob('echo $CONDA_PREFIX', logOut=condaEnv, showError=True):
			if condaEnv[0].find('scipion3') != -1:
				return True
			else:
				return False
		else:
			return False


def getCompatibleGCC(nvcc_version):
		"""
		Retrieves compatible versions of GCC based on a given NVCC (NVIDIA CUDA Compiler) version.

		Params:
		- nvcc_version (str): Version of NVCC.

		Returns:
		- tuple: A tuple containing compatible GCC versions and a boolean indicating compatibility.
		"""
		# https://gist.github.com/ax3l/9489132
		for key, value in CUDA_GCC_COMPATIBILITY.items():
				list = key.split('-')
				if float(nvcc_version) >= float(list[0]) and \
								float(nvcc_version) <= float(list[1]):
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
		idx = string.find('\n')
		idx2 = string[:idx].rfind(' ')
		return string[idx2:idx].replace(' ', '')

def findFileInDirList(fnH, dirlist):
    """ :returns the dir where found or an empty string if not found.
        dirs can contain *, then first found is returned.
    """
    if isinstance(dirlist, str):
        dirlist = [dirlist]

    for dir in dirlist:
        validDirs = glob.glob(path.join(dir, fnH))
        if len(validDirs) > 0:
            return path.dirname(validDirs[0])
    return ''

def checkLib(gxx, libFlag):
    """ Returns True if lib is found. """
    logErr = []
    logOut = []
    result = runJob('echo "int main(){}" > xmipp_check_lib.cpp ; ' +
            gxx + ' ' + libFlag + ' xmipp_check_lib.cpp',
            showOutput=False, showCommand=False, logOut=logOut, logErr=logErr)
    remove('xmipp_check_lib.cpp')
    remove('a.out') if path.isfile('a.out') else None
    return result

def get_Hdf5_name(libdirflags):
		libdirs = libdirflags.split("-L")
		for dir in libdirs:
				if path.exists(path.join(dir.strip(), "libhdf5.so")):
						return "hdf5"
				elif path.exists(path.join(dir.strip(), "libhdf5_serial.so")):
						return "hdf5_serial"
		return "hdf5"