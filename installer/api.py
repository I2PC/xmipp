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

import json

from .versionsCollector import osVersion, architectureVersion, CUDAVersion,\
	cmakeVersion, gppVersion, gccVersion, sconsVersion

def postAPI(dictPackage):
	osV = osVersion()
	architectureV = architectureVersion()
	CUDAV = CUDAVersion(dictPackage)
	cmakeV = cmakeVersion()
	gppV = gppVersion(dictPackage)
	gccV = gccVersion(dictPackage)
	sconsV = sconsVersion()
	print(osV)

def getJSONString(dictPackage) -> str:
	"""
	### Creates a JSON string with the necessary data for the APU POST message.
	
	#### Params:
	- dictPackage (Namespace): Command line arguments parsed by argparse library.
	
	#### Return:
	- (str): JSON string with the required info
	"""
	# Introducing data into a dictionary
	jsonDict = {
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

"""
--data '{
       "user": {
         "userId": "hashMachine5"
       },
       "version": {
         "os": "Centor",
         "cuda": "NoSequeeseso",
         "cmake": "3.5.6",
         "gcc": "4.perocentos",
         "gpp": "gepusplas",
         "scons": "4.3.3"
       },
       "xmipp": {
         "branch": "agm_API",
         "updated": true
       },
       "returnCode": "0 con espacio",
       "logTail": "muchas lines"
     }'      http://127.0.0.1:8000/web/attempts/
"""
