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
import sys, glob, distutils.spawn, json, os, io, time, subprocess, shutil, multiprocessing
import re
from typing import List, Tuple, Union, Callable, Any
from sysconfig import get_paths

# Installer imports
from .constants import (MODES, CUDA_GCC_COMPATIBILITY, vGCC,\
	TAB_SIZE, XMIPP, VERNAME_KEY, LOG_FILE, IO_ERROR, ERROR_CODE,\
	CMD_OUT_LOG_FILE, CMD_ERR_LOG_FILE, OUTPUT_POLL_TIME,
  XMIPP_VERSIONS, MODE_GET_MODELS, WARNING_CODE, XMIPPENV, urlModels, remotePath,
  DOCUMENTATION_URL, urlTest, SCONS_INSTALLATION_WARINING, DONE0, DONE1, HEADER0, HEADER1, HEADER2)

####################### RUN FUNCTIONS #######################
def runJob(cmd: str, cwd: str='./', showOutput: bool=False, showError: bool=False,
					 showCommand: bool=False, streaming: bool=False, printLOG:bool=False,
					 pathLOGFile:str='') -> Tuple[int, str]:
	"""
	### This function runs the given command.

	#### Params:
	- cmd (str): Command to run.
	- cwd (str): Optional. Path to run the command from. Default is current directory.
	- showOutput (bool): Optional. If True, output is printed.
	- showError (bool): Optional. If True, errors are printed.
	- showCommand (bool): Optional. If True, command is printed in blue.
	- streaming (bool): Optional. If True, output is shown in real time as it is being produced.
	- printLOG (bool): Optiona. If True, output on no streaming is printed on log

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
		process = subprocess.Popen(cmd, cwd=cwd, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

		# Defining output string
		output, err = process.communicate()
		retCode = process.returncode
		outputStr = output.decode() if not retCode else err.decode()

		# Printing output if specified
		if showOutput:
			print('{}\n'.format(outputStr))

		# Printing errors if specified
		if err and showError:
			print(red(outputStr))

		if printLOG:
			printMessage(text=outputStr, debug=False, printLOG_FILE=True, pathFile=pathLOGFile)

	# Returning return code
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

def printError(errorMsg: str, retCode: int=1, pathFile:str=''):
	"""
	### This function prints an error message.

	#### Params:
	- errorMsg (str): Error message to show.
	- retCode (int): Optional. Return code to end the exection with.
	"""
	# Print the error message in red color
	errorStr = (f'!! ERROR {retCode}: {errorMsg}\n{ERROR_CODE[retCode][0]}\n'
							f'{ERROR_CODE[retCode][1]}'
							f'\nMore details on the Xmipp documentation portal: {DOCUMENTATION_URL}')
	printMessage(red(errorStr), debug=True, pathFile=pathFile)

def printMessage(text: str, debug: bool=False, pathFile:str='', printLOG_FILE:bool=True):
	"""
	### This function prints the given text into the log file, and, if debug mode is active, also through terminal.

	### Params:
	- text (str): The text to be printed.
	- debug (bool): Indicates if debug mode is active.
	"""
	# If debug mode is active, print through terminal
	if debug:
		print(text, flush=True)
	# Open the file to add text
	try:
		if not pathFile:
				pathFile = LOG_FILE
		else:
				pathFile = os.path.join(pathFile, LOG_FILE)
		if printLOG_FILE:
				with open(pathFile, mode="a") as file:
					text = remove_color_codes(text)
					file.write(f"{text}\n")
					file.flush()
	# If there was an error during the process, show error and exit
	except OSError:
		printError(f"Could not open log file to add info.\n{ERROR_CODE[IO_ERROR]}", retCode=IO_ERROR)

def printWarning(text: str, warningCode: int, debug: bool=True, pathFile:str=''):
	"""
	### This function logs the given text as a warning.

	### Params:
	- text (str): The text to be printed.
	- warningCode (int): Code of the controlled warning.
	- debug (bool): Indicates if debug mode is active.
	"""
	printMessage(yellow(f'! Warning code {warningCode}: {WARNING_CODE[warningCode][0]}\n{WARNING_CODE[warningCode][1]}\n'),
							 debug=debug, pathFile=pathFile)

def remove_color_codes(coloredText):
		"""
		Removes ANSI color codes from a given text.
		Args:
				text_with_colors (str): The input text containing ANSI color codes.

		Returns:
				str: The text with color codes removed.
		"""
		patron = re.compile(r'\x1b\[[0-9;]*[mK]')
		texto_sin_colores = patron.sub('', coloredText)
		return texto_sin_colores

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

def addDeepLearninModel(login, modelPath='', update=None):
		""" Takes the folder name modelName from models dir and
				makes a .tgz, uploads the .tgz to xmipp server.
		"""
		modelPath = modelPath.rstrip("/")
		if not os.path.isdir(modelPath):
				print("<modelsPath> is not a directory. Please, check the path. \n"
							"The name of the model will be the name of that folder.\n")

		modelName = os.path.basename(modelPath)
		modelsDir = os.path.dirname(modelPath)
		tgzFn = "xmipp_model_%s.tgz" % modelName
		localFn = os.path.join(modelsDir, tgzFn)

		printMessage("Creating the '%s' model." % tgzFn, debug=True)
		runJob("tar czf %s %s" % (tgzFn, modelName), cwd=modelsDir)

		printMessage("Warning: Uploading, please BE CAREFUL! This can be dangerous.")
		printMessage('You are going to be connected to "%s" to write in folder '
					'"%s".' % (login, remotePath))
		if input("Continue? YES/no\n").lower() == 'no':
				sys.exit()

		printMessage("Trying to upload the model using '%s' as login" % login)
		args = "%s %s %s %s" % (
		login, os.path.abspath(localFn), remotePath, update)
		if runJob("src/xmipp/bin/xmipp_sync_data upload %s" % args):
				printMessage("'%s' model successfully uploaded! Removing the local .tgz"
							% modelName)
				runJob("rm %s" % localFn)

def downloadDeepLearningModels(dest:str='build'):
    if not os.path.exists('build/bin/xmipp_sync_data'):
        printMessage(red('Xmipp has not been installed. Please, first install Xmipp '))
        return False
    if dest == 'build':
        modelsPath = 'models'
    else:
        modelsPath = dest
    dataSet = "DLmodels"

    # downloading/updating the DLmodels
    if os.path.isdir(os.path.join(dest, modelsPath)):
        printMessage(f"{HEADER0} Updating the Deep Learning models...", debug=True)
        task = "update"
    else:
        printMessage("-- Downloading Deep Learning models...", debug=True)
        task = "download"
    global pDLdownload
    retCode, outputStr = runJob("bin/xmipp_sync_data %s %s %s %s"
                         % (task, modelsPath, urlModels, dataSet),
                         cwd='build', streaming=True, showOutput=True)
    if retCode != 0:
        printMessage(red('Unable to download models. Try again with ./xmipp {}\n{}'.format(MODE_GET_MODELS, outputStr)), debug=True)
    else:
        printMessage(green('Models downloaded in the path: {}'.format(modelsPath)), debug=True)
        printMessage(green(DONE0), debug=True)

def runTests(testName:str='', show:bool=False, allPrograms:bool=False,
						 allFuncs:bool=False, CUDA: bool=True):
    str2Test = ''
    if testName:
        str2Test += testName
    if show:
        str2Test += ' -show'
    if allPrograms:
        str2Test += ' -allPrograms'
    if allFuncs:
        str2Test += ' -allFuncs'
    xmippSrc = os.environ.get('XMIPP_SRC', None)
    if xmippSrc and os.path.isdir(xmippSrc):
        os.environ['PYTHONPATH'] = ':'.join([
            os.path.join(os.environ['XMIPP_SRC'], XMIPP),
            os.environ.get('PYTHONPATH', '')])
        testsPath = os.path.join(os.environ['XMIPP_SRC'], XMIPP, 'tests')
    else:
        retCode, outputStr = runJob('source build/xmipp.bashrc')
        if retCode != 0:
            printMessage(red('XMIPP_SRC is not in the enviroment.') +
								 '\nTo run the tests you need to run: ' +
								 blue('source build/xmipp.bashrc'), debug=True,
								 printLOG_FILE=False)
            return

    dataSetPath = os.path.join(testsPath, 'data')
    os.environ["XMIPP_TEST_DATA"] = dataSetPath

    # downloading/updating the dataset
    dataset = 'xmipp_programs'
    if os.path.isdir(dataSetPath):
        printMessage("\n- Updating the test files...", debug=True, printLOG_FILE=False)
        task = "update"
        showOutput=False
    else:
        printMessage("\n- Downloading the test files...", debug=True, printLOG_FILE=False)
        task = "download"
        showOutput=True
    args = "%s %s %s" % ("tests/data", urlTest, dataset)
    retCode, outputStr = runJob("bin/xmipp_sync_data %s %s" % (task, args),
												cwd='src/xmipp', showOutput=showOutput)
    if retCode != 0:
        printMessage(red('Error downloading test files.\n{}'.format(outputStr)), printLOG_FILE=False)
    else:
        printMessage(text=green('Done'), debug=True, printLOG_FILE=False)

    noCudaStr = '-noCuda' if not CUDA else ''
    if testName or allPrograms:
        printMessage('\n-----------------------------', debug=True, printLOG_FILE=False)
        printMessage("Tests to do: %s" % (str2Test), debug=True, printLOG_FILE=False)
    retCode, outputStr = runJob("%s test.py %s %s" % ('python3', str2Test, noCudaStr),
					cwd=testsPath,  streaming=False, showError=True, showOutput=True	)
    if retCode != 0:
        printMessage(red('Error runnig test.\n{}'.format(outputStr)), printLOG_FILE=False)


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

def updateEnviron(pathenviron:str='', path2Add:str=''):
		"""
		 This function updates the environment variable by adding a new path to it if it's not already present.

		 Params:
		 pathenviron (str): Name of the environment variable to be updated.
		 path2Add (str): Path to be added to the environment variable.

		 Returns:
		 None
		 """
		path_collected = os.environ.get(pathenviron, '')
		if path2Add not in path_collected.split(':'):
				path_collected += ':' + path2Add
		os.environ[pathenviron] = path_collected

def updateXmippEnv(pos='begin', realPath=True, **kwargs):
		""" Add/update a variable in self.env dictionary
				pos = {'begin', 'end', 'replace'}
		"""
		env = readXmippEnv()
		for key, value in kwargs.items():
				isString = isinstance(value, str)
				if isString and realPath:
						value = os.path.realpath(value)
				if key in env:
						if env[key].find(str(value)) == -1:
								if pos == 'begin' and isString:
										env[key] = value + os.pathsep + env[key]
								elif pos == 'end' and isString:
										env[key] = env[key] + os.pathsep + value
								elif pos == 'replace':
										env[key] = str(value)
				else:
						env[key] = str(value)

		writeXmippEnv(env)

def readXmippEnv():
		try:
			with open(XMIPPENV, 'r') as f:
					data = json.load(f)
			return data
		except FileNotFoundError:
				return {}

def writeXmippEnv(env):
		with open(XMIPPENV, 'w') as f:
				json.dump(env, f, indent=4)


####################### VERSION FUNCTIONS #######################

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

def getPackageVersionCmdReturn(packageName: str) -> Union[str, None]:
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
	return output, retCode

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
def printHappyEnd():
		branch = branchName()
		if branch == 'master':
				branch = ''
		else:
				branch = 'devel'
		strXmipp = 'Xmipp {}/{} has been installed, enjoy it!'.format(
				XMIPP_VERSIONS['xmipp']['vername'], branch)
		lenStr = len(strXmipp)
		border = '*' * (lenStr + 4)
		spaceStr = ' ' * (lenStr + 2)
		print('\n')
		print(border)
		print('*' + spaceStr + '*')
		print('* ', end='')
		print(green(strXmipp), end='')
		print(' *')
		print('*' + spaceStr + '*')
		print(border)
		printMessage(text=strXmipp, debug=False)
		printMessage('More about Xmipp: {}'.format(DOCUMENTATION_URL), debug=True)

def branchName():
		retCode, outputStr = runJob('git status')
		if retCode == 0:
			if outputStr.find('On branch') != -1:
				branch = outputStr[outputStr.find('On branch') + len('On branch'):outputStr.find('\n')]
				return branch

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
		path = shutil.which(packageName)
		if path and getPackageVersionCmd(path) is not None:
				return True
		return False

def getINCDIRFLAG():
		return ' -I ' + os.path.join(get_paths()['data'].replace(' ', ''),  'include')

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

def get_Hdf5_name(libdirflags):
		"""
		Identifies the HDF5 library name based on the given library directory flags.

		Params:
		- libdirflags (str): Flags specifying library directories.

		Returns:
		- str: Name of the HDF5 library ('hdf5', 'hdf5_serial', or 'hdf5' as default).
		"""
		libdirs = ['/usr/lib',
		'/usr/lib64']

		#libdirs = libdirflags.split("-L")
		for dir in libdirs:
				if os.path.exists(os.path.join(dir.strip(), "libhdf5.so")):
						return "hdf5"
				elif os.path.exists(os.path.join(dir.strip(), "libhdf5_serial.so")):
						return "hdf5_serial"
		return "hdf5"

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
		printWarning(f'{envNameStr} {instructionStr}', warningCode=SCONS_INSTALLATION_WARINING)
		return False
	
	# If succeeded, log message
	printMessage(f'Succesfully installed or updated Scons on {envName} enviroment.', debug=True)
	return True


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
			stdout = writerOut if showOutput else subprocess.PIPE
			stderr = writerErr if showError else subprocess.PIPE

			# Run command and write output
			process = subprocess.Popen(cmd, cwd=cwd, stdout=stdout, stderr=stderr, shell=True)
			outputStr = writeProcessOutput(process, readerOut, readerErr, showOutput=showOutput, showError=showError)
	except (KeyboardInterrupt, OSError) as e:
		error = True
		errorText = str(e)

	# If there were errors, show them instead of returning
	if error:
		printError(errorText)

	# Return result
	return process.returncode, outputStr

def writeProcessOutput(process: subprocess.Popen, readerOut: io.FileIO, readerErr: io.FileIO, showOutput: bool=False, showError: bool=False) -> str:
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
		outputStr += writeReaderLine(readerOut, show=False)
		outputStr += writeReaderLine(readerErr, show=showError, err=True)

		# If process has finished, exit loop
		if isProcessFinished:
			break

		printMessage(outputStr)
		# Sleep before continuing to next iteration
		time.sleep(OUTPUT_POLL_TIME)

	return outputStr

def writeReaderLine(reader: io.FileIO, show: bool=False, err: bool=False) -> str:
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
		if err:
				if printedLine.find('warning:') != -1 \
								or printedLine.find('Note:') != -1\
								or printedLine.find('Warning:') != -1:
					printMessage(yellow(f'{printedLine}'), debug=True)
				elif printedLine.find('serial compilation of 2 LTRANS jobs') != -1:
						pass
				else:
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

def createDir(path):
	"""
	Create a directory if it doesn't exist.

	Args:
	- path (str): The path of the directory to be created.
	"""
	if not os.path.exists(path):
		os.makedirs(path)

def getScipionHome():
	""" Returns SCIPION_HOME, the directory for scipion3 or EMPTY str. """
	return os.environ.get("SCIPION_HOME", whereis("scipion3")) or ''

def whereis(program, findReal=False, env=None):
	"""
	Find the directory path of a specified program.

	Args:
	- program (str): The name of the program to search for.
	- findReal (bool, optional): If True, returns the real path of the program (default: False).
	- env (str, optional): The environment variable to use for the search (default: None).

	Returns:
	- str or None: Returns the directory path where the program is located. If not found, returns None.
	"""
	programPath = distutils.spawn.find_executable(program, path=env)
	if programPath:
		if findReal:
			programPath = os.path.realpath(programPath)
		return os.path.dirname(programPath)
	else:
		return None