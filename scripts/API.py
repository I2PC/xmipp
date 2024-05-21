# ***************************************************************************
# * Authors:		Alberto García (alberto.garcia@cnb.csic.es)
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
Module containing all functions needed for the metric's API request.
"""

# General imports
import re, hashlib, http.client, json, ssl
from typing import Dict, Optional, Tuple

# Self imports
from .utils import runJob #TODO replace with from .utils import runJob

API_URL = 'xmipp.i2pc.es/api/attempts/'

def sendApiPOST(retCode: int = 0, XMIPP_VERSION:str = ''):
	"""
	### Sends a POST request to Xmipp's metrics's API.

	#### Params:
	- retCode (int): Optional. Return code for the API request.
	"""
	try:
		# Getting JSON data for curl command
		bodyParams = __getJSON(retCode=retCode, XMIPP_VERSION=XMIPP_VERSION)

		# Send API POST request if there were no errors
		if bodyParams is not None:
			# Define the parameters for the POST request
			params = json.dumps(bodyParams)
			# Set up the headers
			headers = {"Content-type": "application/json"}

			# Establish a connection
			url = API_URL.split("/", maxsplit=1)
			path = f"/{url[1]}"
			url = url[0]
			conn = http.client.HTTPSConnection(url,context=ssl._create_unverified_context())  # Unverified context because url does not have an ssl certificate

			# Send the POST request
			conn.request("POST", path, params, headers)
			response = conn.getresponse()
			data = response.read()
			print(response.headers.get('Location'))
			# Imprimir la respuesta
			print("Status:", response.status)
			print("Response:", data)
			# Close the connection
			conn.close()
	except Exception as e:
		pass


####################### UTILS FUNCTIONS #######################
def getOSReleaseName() -> str:
	"""
	### This function returns the name of the current system OS release.

	#### Returns:
	- (str): OS release name.
	"""
	# Initializing default release name
	releaseName = 'Unknown'

	# Text around release name
	textBefore = 'PRETTY_NAME="'
	textAfter = '"\n'
	log = []
	# Obtaining os release name
	retCode = runJob('cat /etc/os-release', show_output=False, show_command=False, log=log)
	name = '\n'.join(log)
	# Look for release name if command did not fail
	if retCode == True:
		# Find release name's line in command output
		targetStart = name.find(textBefore)
		if targetStart != 1:
			# Search end of release name's line
			nameEnd = name[targetStart:].find(textAfter)

			# Calculate release name's start index
			nameStart = targetStart + len(textBefore)

			if nameEnd != -1 and nameStart != nameEnd:
				# If everything was correctly found and string is
				# not empty, extract release name
				releaseName = name[nameStart:nameEnd]

	# Return release name
	return releaseName


def __getJSON(retCode: int = 0, XMIPP_VERSION: str = '') -> Optional[Dict]:
	"""
	### Creates a JSON with the necessary data for the API POST message.

	#### Params:
	- retCode (int): Optional. Return code for the API request.
	- XMIPP_VERSION (str): version of release

	#### Return:
	- (dict | None): JSON with the required info or None if user id could not be produced.
	"""
	try:
		# Getting user id and checking if it exists
		userId = __getUserId()
		if userId is None:
			return

		# Obtaining variables in parallel
		CUDA_version = ''
		GCC_version = ''
		GPP_version = ''
		configFile = '../xmipp.conf'#TODO change path
		compileFile = '../compileLOG.txt'
		with open(configFile, 'r') as file:
			lines = file.readlines()
		for l in lines:
			log = []
			if l.find('CC')!= -1 and l.find('CCFLAGS')== -1 and l.find('MPI_CC')== -1\
				and l.find('NVCC') == -1 and l.find('NVCC_CXXFLAGS')== -1 and l.find('NVCC_LINKFLAGS')== -1:
				compiler = l.split('=')[-1]
				compiler = compiler.replace('\n', '')
				runJob('{} --version'.format(compiler), show_output=False, show_command=False, log=log)
				GCC_version = log[0].split(' ')[-1]
			if l.find('CXX')!= -1 and l.find('CXXFLAGS')== -1 and l.find('CXX_CUDA')== -1\
				and l.find('MPI_CXX') == -1 and l.find('MPI_CXXFLAGS')== -1 and l.find('NVCC_CXXFLAGS')== -1:
				compiler = l.split('=')[-1]
				compiler = compiler.replace('\n', '')
				runJob('{} --version'.format(compiler), show_output=False, show_command=False, log=log)
				GPP_version = log[0].split(' ')[-1]
			if l.find('NVCC')!= -1 and l.find('NVCC_CXXFLAGS')== -1 and l.find('NVCC_LINKFLAGS')== -1:
				compiler = l.split('=')[-1]
				compiler = compiler.replace('\n', '')
				runJob('{} --version'.format(compiler), show_output=False, show_command=False, log=log)
				CUDA_version = log[-2][log[-2].find('release')+ 8 :log[-2].find('release') + 12]


		with open(compileFile, 'r') as file:
			lines = file.readlines()
			logTail = lines[-100:]

		currentBranch = getCurrentBranch()

		# If branch is master or there is none, get release name
		branchName = XMIPP_VERSION if not currentBranch or currentBranch == 'master' else currentBranch

		# Introducing data into a dictionary
		return {
			"user": {
				"userId": userId
			},
			"version": {
				"os": getOSReleaseName(),
				"architecture": __getArchitectureName(),
				"cuda": CUDA_version,
				"cmake": None,
				"gcc": GCC_version,
				"gpp": GPP_version,
				"scons": None
			},
			"xmipp": {
				"branch": branchName,
				"updated": isBranchUpToDate()
			},
			"returnCode": retCode,
			"logTail": logTail if retCode else None
			# Only needs log tail if something went wrong
		}
		"""
		return {
				"user": {
					"userId": userId
				},
				"version": {
					"os": getOSReleaseName(),
					"architecture": __getArchitectureName(),
					"cuda": CUDA_version,
					"cmake": None,
					"gcc": GCC_version,
					"gpp": GPP_version,
					"mpi": None,
					"python": None,
					"sqlite": None,
					"java": None,
					"hdf5": None,
					"jpeg": None
				},
				"xmipp": {
					"branch": branchName,
					"updated": isBranchUpToDate()
				},
				"returnCode": retCode,
				"logTail": logTail if retCode else None
				# Only needs log tail if something went wrong
			}
	""" #TODO replace for this json format
	except Exception as e:
		pass
def __getMACAddress() -> Optional[str]:
	"""
	### This function returns a physical MAC address for this machine. It prioritizes ethernet over wireless.

	#### Returns:
	- (str | None): MAC address, or None if there were any errors.
	"""
	# Run command to get network interfaces info
	log = []
	status = runJob("ip addr", show_output=False, log=log, show_command=False, in_parallel=False)

	# If command failed, return None to avoid sending POST request
	if status == False:
		return

	# Regular expression to match the MAC address and interface names
	macRegex = r"link/ether ([0-9a-f:]{17})"
	interfaceRegex = r"^\d+: (enp|wlp|eth)\w+"

	# Split the output into lines
	lines = log

	# Iterate over the lines looking for MAC address
	macAddress = None
	for line in lines:
		# If this line contains an interface name
		if re.match(interfaceRegex, line):
			# Extract the interface name
			interfaceName = re.match(interfaceRegex, line).group(1)

			# If the interface name starts with 'enp', 'wlp', or 'eth
			if interfaceName.startswith(('enp', 'wlp', 'eth')):
				# Extract the MAC address from the next line and exit
				macAddress = re.search(macRegex,
				                       lines[lines.index(line) + 1]).group(1)
				break

	return macAddress


def __getUserId() -> Optional[str]:
	"""
	### This function returns the unique user id for this machine.

	#### Returns:
	- (str | None): User id, or None if there were any errors.
	"""
	# Obtaining user's MAC address
	macAddress = __getMACAddress()

	# If no physical MAC address was found, user id cannot be created
	if macAddress is None or not macAddress:
		return

	# Create a new SHA-256 hash object
	sha256 = hashlib.sha256()

	# Update the hash object with the bytes of the MAC address
	sha256.update(macAddress.encode())

	# Return hexadecimal representation of the hash
	return sha256.hexdigest()


def __getArchitectureName() -> str:
	"""
	### This function returns the name of the system's architecture name.

	#### Returns:
	- (str): Architecture name.
	"""
	# Initializing to unknown value
	archName = 'Unknow'
	log = []
	# Obtaining architecture name
	retCode = runJob('cat /sys/devices/cpu/caps/pmu_name', show_output=False, show_command=False, log=log)
	if len(log) > 0:
		architecture = log[0]
	# If command worked and returned info, extract it
	if retCode == True and architecture:
		archName = architecture

	# Returing architecture name
	return archName


def getCurrentBranch(dir: str = './') -> str:
	"""
	### This function returns the current branch of the repository of the given directory or empty string if it is not a repository or a recognizable tag.

	#### Params:
	- dir (str): Optional. Directory of the repository to get current branch from. Default is current directory.

	#### Returns:
	- (str): The name of the branch, 'HEAD' if a tag, or empty string if given directory is not a repository or a recognizable tag.
	"""
	log = []
	# Getting current branch name
	retcode = runJob("git rev-parse --abbrev-ref HEAD", show_output=False, show_command=False, cwd=dir, log=log)
	if len(log) > 0:
		branchName = log[0]
	else:
		return ''
	# If there was an error, we are in no branch
	return branchName if not retcode else ''

def isBranchUpToDate(dir: str = './') -> bool:
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
	retCode, remoteCommit = runInsistentJob(
		f"git rev-parse origin/{currentBranch}")

	# Check if command succeeded
	if retCode != 0:
		return False

	# Return commit comparison
	return localCommit == remoteCommit


def runInsistentJob(cmd: str, cwd: str = './', showOutput: bool = False,
                    showError: bool = False, showCommand: bool = False,
                    nRetries: int = 5) -> Tuple[int, str]:
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
		output=[]
		retCode = runJob(cmd, cwd=cwd, log=output, show_output=False,show_command=False)
		# Break loop if success was achieved
		if retCode == 0:
			break
	if len(output) > 0:
		output = output[0]
	# Enforce message showing deppending on value
	if showCommand:
		print(cmd)
	if showOutput:
		print('{}\n'.format(output))
	if showError:
		print(output)

	# Returning output and return code
	return retCode, output


if __name__ == '__main__':
	sendApiPOST(retCode=0)
