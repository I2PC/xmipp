# ***************************************************************************
# * Authors:		Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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

from typing import Dict, Any, List
from .constants import CMAKE, DEFAULT_CMAKE, INTERNAL_LOGIC_VARS
from .utils import runJob
import shutil

def getCMake(config: Dict[str, Any]) -> str:
	"""
	### Retrieves information about the CMake package and updates the dictionary accordingly.

	#### Params:
	- packages (dict): Dictionary containing package information.

	#### Returns:
	- (dict): Param 'packages' with the 'CMAKE' key updated based on the availability of 'cmake'.
	"""
	cmake = config.get(CMAKE)
	return cmake if cmake else shutil.which(DEFAULT_CMAKE)

def getCMakeVars(config: Dict) -> List[str]:
	"""
	### This function converts the variables in the config dictionary into a list as CMake args.
	
	#### Params:
	- configDict (dict): Dictionary to obtain the parameters from.
	"""
	result = []
	for (key, value) in config.items():
		if key not in INTERNAL_LOGIC_VARS and bool(value):
			result.append(f"-D{key}={value}")
	return result

def getCMakeVarsStr(config: Dict) -> str:
	"""
	### This function converts the variables in the config dictionary into a string as CMake args.
	
	#### Params:
	- configDict (dict): Dictionary to obtain the parameters from.
	"""
	return ' '.join(getCMakeVars(config))

def checkPackage(package: str, config: Dict[str, Any]) -> bool:
  cmake = getCMake(config)
  args = []
  args.append(f'-DNAME={package}')
  args.append(f'-DCOMPILER_ID=GNU')
  args.append(f'-DLANGUAGE=C')
  args.append(f'-DMODE=EXIST')
  args += getCMakeVars(config)
  
  cmd = cmake + ' ' + ' '.join(args)
  ret, _ = runJob(cmd, logOutput=False)
  return ret == 0
  