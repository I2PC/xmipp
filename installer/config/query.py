# ***************************************************************************
# * Authors:		Alberto García (alberto.garcia@cnb.csic.es)
# *							Martín Salinas (martin.salinas@cnb.csic.es)
# *							Oier Lauzirika Zarrabeitia (olauzirika@cnb.csic.es)
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

from typing import Dict, Any, Callable
import os
import shutil

from ..constants import GCC_HOME, GXX_HOME, CUDA_HOME, CMAKE_HOME

def __getFunctionSkeleton(packages: Dict[str, Any], 
                          key: str,
                          finder: Callable):
  result = packages.get(key)
  if not result:
    result = finder()
    packages[key] = result
  return result
 
def findCC() -> str:
  result = os.environ.get('CC')
  if not result:
    result = shutil.which('gcc')
  return result
    
def getCC(packages: Dict[str, Any]) -> Dict:
  """
  ### Retrieves information about the CC (GCC) package and updates the dictionary accordingly.

  #### Params:
  - packages (dict): Dictionary containing package information.

  #### Returns:
  - (dict): Param 'packages' with the 'GCC_HOME' key updated based on the availability of 'gcc'.
  """
  return __getFunctionSkeleton(packages, key=GCC_HOME, finder=findCC)
	
def findCXX() -> str:
  result = os.environ.get('CXX')
  if not result:
    result = shutil.which('g++')
  return result
    
def getCXX(packages: Dict[str, Any]) -> Dict:
  """
  ### Retrieves information about the g++ package and updates the dictionary accordingly.

  #### Params:
  - packages (dict): Dictionary containing package information.

  #### Returns:
  - (dict): Param 'packages' with the 'GXX_HOME' key updated based on the availability of 'g++'.
  """
  return __getFunctionSkeleton(packages, key=GXX_HOME, finder=findCXX)

def findNVCC() -> str:
  unixDefaultNVCCPath = '/usr/local/cuda/bin/nvcc'
  if os.path.exists(unixDefaultNVCCPath):
    return unixDefaultNVCCPath
  else:
    return shutil.which('nvcc')

def getNVCC(packages: Dict[str, Any]) -> Dict:
  """
  ### Retrieves information about the NVCC package and updates the dictionary accordingly.

  #### Params:
  - packages (dict): Dictionary containing package information.

  #### Returns:
  - (dict): Param 'packages' with the 'CUDA_HOME' key updated based on the availability of 'nvcc'.
  """
  baseMatch = __getFunctionSkeleton(packages, key=CUDA_HOME, finder=findNVCC)
  # If base match is cuda root dir, add path to nvcc. Otherwise return as is
  return os.path.join(baseMatch, 'bin', 'nvcc') if 'cuda' in baseMatch.split('/')[-1] else baseMatch

def findCMake() -> str:
  return shutil.which('cmake')

def getCMake(packages: Dict[str, Any]) -> Dict:
  """
  ### Retrieves information about the CMake package and updates the dictionary accordingly.

  #### Params:
  - packages (dict): Dictionary containing package information.

  #### Returns:
  - (dict): Param 'packages' with the 'CMAKE_HOME' key updated based on the availability of 'cmake'.
  """
  return __getFunctionSkeleton(packages, key=CMAKE_HOME, finder=findCMake)
