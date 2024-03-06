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

from ..constants import CC, CXX, SCONS

def _getFunctionSkeleton(packages: Dict[str, Any], 
                         key: str,
                         finder: Callable[[]]):
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
    
def getCC(packages: Dict[str, Any]):
  """
  Retrieves information about the CC (GCC) package and updates the dictionary accordingly.

  Params:
  - packages (dict): Dictionary containing package information.

  Modifies:
  - dictPackages: Updates the 'CC' key based on the availability of 'gcc'.
  """
  return _getFunctionSkeleton(packages, key=CC, finder=findCC)
	
def findCXX() -> str:
  result = os.environ.get('CXX')
  if not result:
    result = shutil.which('g++')
  return result
    
def getCXX(packages: Dict[str, Any]):
  """
  Retrieves information about the g++ package and updates the dictionary accordingly.

  Params:
  - packages (dict): Dictionary containing package information.

  Modifies:
  - dictPackages: Updates the 'CXX' key based on the availability of 'g++'.
  """
  return _getFunctionSkeleton(packages, key=CXX, finder=findCXX)

def findJava():
  """
  Retrieves information about the Java package.
  """
  # Find javac program
  javaBinPath = whereIsPackage('javac')
  if not javaBinPath:
      javaBinPath = findFileInDirs('javac', ['/usr/lib/jvm/java-*/bin'])
    
  javaHome = None
  if javaBinPath:
      javaHome = os.path.join(os.path.split(javaBinPath)[:-1])

  return javaHome
    
def getJavaHome(packages: Dict[str, Any]):
  """
  Retrieves information about the java package and updates the dictionary accordingly.

  Params:
  - packages (dict): Dictionary containing package information.

  Modifies:
  - dictPackages: Updates the 'JAVA_HOME' key based on the availability of java.
  """
  return _getFunctionSkeleton(packages, key='JAVA_HOME', finder=findJava)

def findScons():
  """
  Retrieves information about the Scons package.
  """
  return shutil.which('scons')

def getScons(packages: Dict[str, Any]):
  return _getFunctionSkeleton(packages, key=SCONS, finder=findScons)