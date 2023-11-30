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

# General imports
import json, re, hashlib
from typing import Dict, Union

# Self imports
from .versionsCollector import osVersion, architectureVersion, CUDAVersion,\
	cmakeVersion, gppVersion, gccVersion, sconsVersion
from .utils import runJob, showError, getCurrentBranch
from .constants import NETWORK_ERROR, API_URL

def sendApiPost(dictPackage: Dict):
	"""
	### Sends a POST request to Xmipp's metrics's API.
	
	#### Params:
	- dictPackage (Dict): Dictionary containing all discovered or config variables.
	"""
	# If there is no user id, don't send request
	if getUserId() is None:
		return
	
	# Get curl command string
	curlCmd = getCurlStr(API_URL, dictPackage)

	# Send API POST message. Retry up to N times (improves resistance to small network errors)
	for _ in range(5):
		status, output = runJob(curlCmd)
		# Break loop if success was achieved
		if status:
			break
	
	# Show error if it failed
	# TODO: THIS IS FOR TESTING, IGNORE ERROR IN PRODUCTION VERSION
	showError(output, retCode=NETWORK_ERROR)

####################### UTILS FUNCTIONS #######################
def getJSONString(dictPackage: Dict, retCode: int=0) -> str:
	"""
	### Creates a JSON string with the necessary data for the APU POST message.
	
	#### Params:
	- dictPackage (Dict): Dictionary containing all discovered or config variables.
	
	#### Return:
	- (str): JSON string with the required info.
	"""
	# Introducing data into a dictionary
	jsonDict: Dict = {
		"user": {
			"userId": getUserId()
		},
		"version": {
			"os": osVersion(),
			"architecture": architectureVersion(),
			"cuda": CUDAVersion(dictPackage),
			"cmake": cmakeVersion(),
			"gcc": gccVersion(dictPackage),
			"gpp": gppVersion(dictPackage),
			"scons": sconsVersion()
		},
		"xmipp": {
			"branch": getCurrentBranch(),
			"updated": True
		},
		"returnCode": retCode,
		"logTail": "muchas lines"
	}

	# Return JSON object with all info
	return json.dumps(jsonDict)

def getCurlStr(url: str, dictPackage: Dict) -> str:
	"""
	### Creates a curl command string to send a POST message to metrics's API.
	
	#### Params:
	- url (str): Ulr to send the POST message to.
	- dictPackage (Dict): Dictionary containing all discovered or config variables.
	
	#### Return:
	- (str): Curl command string.
	"""
	# Creating and returning command string
	cmd = "curl --header \"Content-Type: application/json\" -X POST"
	cmd += f" --data \'{getJSONString(dictPackage)}\' --request POST {url}"
	return cmd

def getMACAddress() -> Union[str, None]:
	"""
	### This function returns a physical MAC address for this machine. It prioritizes ethernet over wireless.
	
	#### Returns:
	- (str): MAC address, or None if there were any errors.
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
	- (str): User id, or None if there were any errors.
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
