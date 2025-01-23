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
import re, hashlib, http.client, json
from typing import Dict, Optional
from urllib.parse import urlparse
import os

# Self imports
from .cmake import parseCmakeVersions
from .utils import runJob, getCurrentName, isBranchUpToDate, runParallelJobs
from .constants import (API_URL, LOG_FILE, TAIL_LOG_NCHARS, UNKNOWN_VALUE,
    XMIPP_VERSIONS, XMIPP, VERSION_KEY, MASTER_BRANCHNAME, VERSION_FILE, CMAKE_PYTHON,
    CMAKE_CUDA, CMAKE_MPI, CMAKE_HDF5, CMAKE_JPEG, CMAKE_SQLITE, CMAKE_JAVA,
    CMAKE_CMAKE, CMAKE_GCC, CMAKE_GPP)
from .logger import logger


def sendApiPOST(retCode: int=0):
    """
    ### Sends a POST request to Xmipp's metrics's API.

    #### Params:
    - retCode (int): Optional. Return code for the API request.
    """
    # Getting JSON data for curl command
    bodyParams = __getJSON(retCode=retCode)
    with open(os.path.join(os.getcwd(), 'datosRequest.txt'), 'w') as file:
        file.write(f'bodyParams: {bodyParams}')
    # Send API POST request if there were no errors
    #if bodyParams is not None:
    # Define the parameters for the POST request
    params = json.dumps(bodyParams)
    # Set up the headers
    headers = {"Content-type": "application/json"}
    parsedUrl = urlparse(API_URL)
    # Establish a connection
    conn = http.client.HTTPSConnection(parsedUrl.hostname, parsedUrl.port, timeout=5)
    try:
        # Send the POST request
        conn.request("POST", parsedUrl.path, body=params, headers=headers)
    except Exception:
        pass
    finally:
        # Close the connection
        conn.close()


####################### UTILS FUNCTIONS #######################
def getOSReleaseName() -> str:
    """
    ### This function returns the name of the current system OS release.

    #### Returns:
    - (str): OS release name.
    """
    # Initializing default release name
    releaseName = UNKNOWN_VALUE

    # Text around release name
    textBefore = 'PRETTY_NAME="'
    textAfter = '"\n'

    # Obtaining os release name
    retCode, name = runJob('cat /etc/os-release')

    # Look for release name if command did not fail
    if retCode == 0:
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

def __getJSON(retCode: int=0) -> Optional[Dict]:
    """
    ### Creates a JSON with the necessary data for the API POST message.

    #### Params:
    - retCode (int): Optional. Return code for the API request.

    #### Return:
    - (dict | None): JSON with the required info or None if user id could not be produced.
    """
    # Getting user id and checking if it exists
    userId = __getUserId()
    if userId is None:
        userId = 'Anonymous'
        #return

    # Obtaining variables in parallel
    data = parseCmakeVersions(VERSION_FILE)
    jsonData = runParallelJobs([
        (getOSReleaseName, ()),
        (__getCPUFlags, ()),
        (getCurrentName, ()),
        (isBranchUpToDate, ()),
        (__getLogTail, ())
    ])

    # If branch is master or there is none, get release name
    branchName = XMIPP_VERSIONS[XMIPP][VERSION_KEY] if not jsonData[2] or jsonData[2] == MASTER_BRANCHNAME else jsonData[2]
    installedByScipion = bool(os.getenv("SCIPION_SOFTWARE"))

    # Introducing data into a dictionary
    return {
        "user": {
            "userId": userId
        },
        "version": {
            "os": jsonData[0],
            "architecture": jsonData[1],
            "cuda": data.get(CMAKE_CUDA),
            "cmake": data.get(CMAKE_CMAKE),
            "gcc": data.get(CMAKE_GCC),
            "gpp": data.get(CMAKE_GPP),
            "mpi": data.get(CMAKE_MPI),
            "python": data.get(CMAKE_PYTHON),
            "sqlite": data.get(CMAKE_SQLITE),
            "java": data.get(CMAKE_JAVA),
            "hdf5": data.get(CMAKE_HDF5),
            "jpeg": data.get(CMAKE_JPEG)
        },
        "xmipp": {
            "branch": branchName,
            "updated": jsonData[3],
            "installedByScipion": installedByScipion
        },
        "returnCode": retCode,
        "logTail": jsonData[4] if retCode else None # Only needs log tail if something went wrong
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

            # If the interface name starts with 'enp', 'ens', 'wlp', or 'eth'
            if interfaceName.startswith(('enp', 'wlp', 'eth', 'ens')):
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

def __getCPUFlags() -> str:
    """
    ### This function returns a string with the flags provided by lscpu.
    """
    returnCode, outputStr = runJob('lscpu | grep Flags')
    if returnCode == 0:
        flagsCPU = outputStr.replace('Flags:', '').strip()
        return flagsCPU
    return UNKNOWN_VALUE
