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
from os import environ
from typing import Union, List

# Installer imports
from .constants import SCONS_MINIMUM, MODES, CUDA_GCC_COMPATIBILITY, vGCC
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


def versionPackage(package):
		"""Return the version of the package if found, else return False"""
		str = []
		cmd = '{} --version'.format(package)
		if runJob(cmd, showOutput=False, logOut=str, showCommand=False):
				for line in str:
						if line.find('not found') != -1:
								return ''
		return str[0]


def whereIsPackage(packageName):
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
		path = []
		runJob('which {}'.format(packageName), showCommand=False,
					 showOutput=False, logOut=path)
		path = path[0].replace('\n', '')
		return path


def existPath(path):
		"""Return True if path exist, else False"""
		pass

def getINCDIRFLAG():
		get_paths()
		return " -I%s" % "%s/include" % get_paths()['data']

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
		return 3, False

def isScipionVersion():
	condaEnv = []
	if runJob('echo $CONDA_PREFIX', logOut=condaEnv, showError=True):
		if condaEnv[0].find('scipion3') != -1:
			return True
		else:
			return False
	else:
		return False


def get_compatible_GCC(nvcc_version):
		# https://gist.github.com/ax3l/9489132
		for key, value in CUDA_GCC_COMPATIBILITY.items():
				list = key.split('-')
				if float(nvcc_version) >= float(list[0]) and \
								float(nvcc_version) <= float(list[1]):
						return value, True
		return vGCC, False

