# ***************************************************************************
# * Authors:		Alberto García (alberto.garcia@cnb.csic.es)
# *					Martín Salinas (martin.salinas@cnb.csic.es)
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
import json
from typing import Dict

# Self imports
from .versionsCollector import osVersion, architectureVersion, CUDAVersion,\
	cmakeVersion, gppVersion, gccVersion, sconsVersion
from .utils import runJob, showError
from .constants import NETWORK_ERROR, API_URL

def sendApiPost(dictPackage: Dict):
	"""
	### Sends a POST request to Xmipp's metrics's API.
	
	#### Params:
	- dictPackage (Dict): Dictionary containing all discovered or config variables.
	"""
	# Send API POST message. Retry up to N times (improves resistance to small network errors)
	for _ in range(5):
		status, output = runJob(getCurlStr(API_URL, dictPackage))
		# Break loop if success was achieved
		if status:
			break
	
	# Show error if it failed
	# TODO: THIS IS FOR TESTING, IGNORE ERROR IN PRODUCTION VERSION
	showError(output, retCode=NETWORK_ERROR)

def getJSONString(dictPackage: Dict) -> str:
	"""
	### Creates a JSON string with the necessary data for the APU POST message.
	
	#### Params:
	- dictPackage (Dict): Dictionary containing all discovered or config variables.
	
	#### Return:
	- (str): JSON string with the required info.
	"""
	# Introducing data into a dictionary
	jsonDict: Dict= {
		"user": {
			"userId": "hashMachine5" #TODO: get hash
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
			"branch": "agm_API",
			"updated": True
		},
		"returnCode": "0 con espacio",
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
