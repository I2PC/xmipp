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
import re, hashlib, http.client, json, ssl
from typing import Dict, Optional

# Self imports
from .config import (getOSReleaseName, getArchitectureName, getCUDAVersion,
	getCmakeVersion, getGXXVersion, getGCCVersion, getCC, getCXX, getNVCC, getCMake)
from .utils import runJob, getCurrentBranch, isBranchUpToDate, runParallelJobs
from .cmake_cache import parseCmakeCache
from .constants import (API_URL, LOG_FILE, TAIL_LOG_NCHARS,
	XMIPP_VERSIONS, XMIPP, VERSION_KEY, MASTER_BRANCHNAME)

def sendApiPOST(configDict:Dict, retCode: int=0):
	"""
	### Sends a POST request to Xmipp's metrics's API.
	
	#### Params:
	- configDict (Dict): Dictionary containing all discovered or config variables.
	- retCode (int): Optional. Return code for the API request.
	"""
	# Getting JSON data for curl command
	bodyParams = __getJSON(configDict, retCode=retCode)

	# Send API POST request if there were no errors
	if bodyParams is not None:
		# Define the parameters for the POST request
		params = json.dumps(bodyParams)
		params = params.replace("null", "\"null\"") # TEMPORARY, REMOVE WHEN BACK END ALLOWS REAL NULLS

		# Set up the headers
		headers = {"Content-type": "application/json"}

		# Establish a connection
		url = API_URL.split("/", maxsplit=1)
		path = f"/{url[1]}"
		url = url[0]
		conn = http.client.HTTPSConnection(url, context=ssl._create_unverified_context()) # Unverified context because url does not have an ssl certificate

		# Send the POST request
		conn.request("POST", path, params, headers)

		# Close the connection
		conn.close()
	
####################### UTILS FUNCTIONS #######################
def __getJSON(configDict: Dict, retCode: int=0) -> Optional[Dict]:
	"""
	### Creates a JSON with the necessary data for the API POST message.
	
	#### Params:
	- configDict (Dict): Dictionary containing all discovered or config variables.
	- retCode (int): Optional. Return code for the API request.
	
	#### Return:
	- (dict | None): JSON with the required info or None if user id could not be produced.
	"""
	# Getting user id and checking if it exists
	userId = __getUserId()
	if userId is None:
		return
	
	# Obtaining variables in parallel
	jsonData = runParallelJobs([
		(getOSReleaseName, ()),
		(getArchitectureName, ()),
		(getCUDAVersion, (getNVCC(configDict),)),
		(getCmakeVersion, (getCMake(configDict),)),
		(getGCCVersion, (getCC(configDict),)),
		(getGXXVersion, (getCXX(configDict),)),
		(getCurrentBranch, ()),
		(isBranchUpToDate, ()),
		(__getLogTail, ())
	])

	# If branch is master or there is none, get release name
	branchName = XMIPP_VERSIONS[XMIPP][VERSION_KEY] if not jsonData[6] or jsonData[6] == MASTER_BRANCHNAME else jsonData[6]

	# Introducing data into a dictionary
	return {
		"user": {
			"userId": userId
		},
		"version": {
			"os": jsonData[0],
			"architecture": jsonData[1],
			"cuda": jsonData[2],
			"cmake": jsonData[3],
			"gcc": jsonData[4],
			"gpp": jsonData[5],
			"scons": None
		},
		"xmipp": {
			"branch": branchName,
			"updated": jsonData[7]
		},
		"returnCode": retCode,
		"logTail": jsonData[8] if retCode else None # Only needs log tail if something went wrong
	}

def __getMACAddress() -> Optional[str]:
	"""
	### This function returns a physical MAC address for this machine. It prioritizes ethernet over wireless.
	
	#### Returns:
	- (str | None): MAC address, or None if there were any errors.
	"""
	# Run command to get network interfaces info
	status, output = runJob("ip addr")

	# If command failed, return None to avoid sending POST request
	if status != 0:
		return
	
	# Regular expression to match the MAC address and interface names
	macRegex = r"link/ether ([0-9a-f:]{17})"
	interfaceRegex = r"^\d+: (enp|wlp|eth)\w+"

	# Split the output into lines
	lines = output.split('\n')

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
				macAddress = re.search(macRegex, lines[lines.index(line) + 1]).group(1)
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

def __getLogTail() -> Optional[str]:
	"""
	### This function returns the last lines of the installation log.
	
	#### Returns:
	- (str | None): Installation log's last lines, or None if there were any errors.
	"""
	# Obtaining log tail
	retCode, output = runJob(f"tail -n {TAIL_LOG_NCHARS} {LOG_FILE}")

	# Return content if it went right
	return output if retCode == 0 else None
