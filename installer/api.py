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
Module containing all functions needed for the metric's API request.
"""

# General imports
import json, re, hashlib
from typing import Dict, Union

# Self imports
from .versions import getOSReleaseName, getArchitectureName, getCUDAVersion,\
	getCmakeVersion, getGPPVersion, getGCCVersion, getSconsVersion
from .utils import runJob, runNetworkJob, getCurrentBranch, isBranchUpToDate
from .constants import API_URL, LOG_FILE

def sendApiPost(dictPackage: Dict, retCode: int=0):
	"""
	### Sends a POST request to Xmipp's metrics's API.
	
	#### Params:
	- dictPackage (Dict): Dictionary containing all discovered or config variables.
	- retCode (int): Optional. Return code for the API request.
	"""
	# Getting JSON data for curl command
	jsonStr = getJSONString(dictPackage, retCode=retCode)

	# Send API POST request if there were no errors
	if jsonStr is not None:
		runNetworkJob(getCurlStr(API_URL, jsonStr))
	
####################### UTILS FUNCTIONS #######################
def getJSONString(dictPackage: Dict, retCode: int=0) -> Union[str, None]:
	"""
	### Creates a JSON string with the necessary data for the APU POST message.
	
	#### Params:
	- dictPackage (Dict): Dictionary containing all discovered or config variables.
	- retCode (int): Optional. Return code for the API request.
	
	#### Return:
	- (str|None): JSON string with the required info or None if user id could not be produced.
	"""
	# Getting user id and checking if it exists
	userId = getUserId()
	if userId is None:
		return

	# Introducing data into a dictionary
	jsonDict: Dict = {
		"user": {
			"userId": userId
		},
		"version": {
			"os": getOSReleaseName(),
			"architecture": getArchitectureName(),
			"cuda": getCUDAVersion(dictPackage),
			"cmake": getCmakeVersion(),
			"gcc": getGCCVersion(dictPackage),
			"gpp": getGPPVersion(dictPackage),
			"scons": getSconsVersion()
		},
		"xmipp": {
			"branch": getCurrentBranch(),
			"updated": isBranchUpToDate()
		},
		"returnCode": retCode,
		"logTail": getLogTail()
	}

	# Return JSON object with all info
	return json.dumps(jsonDict)

def getCurlStr(url: str, jsonStr: str) -> str:
	"""
	### Creates a curl command string to send a POST message to metrics's API.
	
	#### Params:
	- url (str): Ulr to send the POST message to.
	- jsonStr (str): JSON in a string format containing all the data for the curl command.
	
	#### Return:
	- (str): Curl command string.
	"""
	# Creating and returning command string
	cmd = "curl --header \"Content-Type: application/json\" -X POST"
	cmd += f" --data \'{jsonStr}\' --request POST {url}"
	return cmd

def getMACAddress() -> Union[str, None]:
	"""
	### This function returns a physical MAC address for this machine. It prioritizes ethernet over wireless.
	
	#### Returns:
	- (str|None): MAC address, or None if there were any errors.
	"""
	# Run command to get network interfaces info
	status, output = runJob("ip addr")

	# If command failed, return None to avoid sending POST request
	if status != 0:
		return
	
	# Regular expression to match the MAC address and interface names
	macRegex = r"link/ether ([0-9a-f:]{17})"
	interfaceRegex = r"^\d+: (enp|wlp)\w+"

	# Split the output into lines
	lines = output.split('\n')

	# Iterate over the lines looking for MAC address
	macAddress = None
	for line in lines:
		# If this line contains an interface name
		if re.match(interfaceRegex, line):
			# Extract the interface name
			interfaceName = re.match(interfaceRegex, line).group(1)
			
			# If the interface name starts with 'enp' or 'wlp'
			if interfaceName.startswith(('enp', 'wlp')):
				# Extract the MAC address from the next line and exit
				macAddress = re.search(macRegex, lines[lines.index(line) + 1]).group(1)
				break
	
	return macAddress

def getUserId() -> Union[str, None]:
	"""
	### This function returns the unique user id for this machine.
	
	#### Returns:
	- (str|None): User id, or None if there were any errors.
	"""
	# Obtaining user's MAC address
	macAddress = getMACAddress()

	# If no physical MAC address was found, user id cannot be created
	if macAddress is None or not macAddress:
		return
	
	# Create a new SHA-256 hash object
	sha256 = hashlib.sha256()
	
	# Update the hash object with the bytes of the MAC address
	sha256.update(macAddress.encode())
	
	# Return hexadecimal representation of the hash
	return sha256.hexdigest()

def getLogTail() -> Union[str, None]:
	"""
	### This function returns the last lines of the installation log.
	
	#### Returns:
	- (str|None): Installation log's last lines, or None if there were any errors.
	"""
	# Obtaining log tail
	retCode, output = runJob(f"tail {LOG_FILE}")

	# Return content if it went right
	return output if retCode == 0 else None
