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

from typing import Dict, Tuple, Optional
from datetime import datetime
from ..constants import CONFIG_VARIABLES, CONFIG_DEFAULT_VALUES, TOGGLES, LOCATIONS

ASSIGNMENT_SEPARATOR = '='
COMMENT_ESCAPE = '#'

def __parseConfigLine(lineNumber: int, line: str) -> Optional[Tuple[str, str]]:
  """
	### Reads the given line from the config file and returns the key-value pair as a tuple.

	#### Params:
  - lineNumber (int): Line nunber inside the config file.
	- line (str): Line to parse.
	
	#### Returns:
	- (tuple(str, str)): Tuple containing the read key-value pair.
	"""
  # Skip if comments
  line = line.split(COMMENT_ESCAPE, maxsplit=2)[0].strip()
  
  # Check if empty line
  if not line:
    return None
  
  # Try to parse the line
  tokens = line.split(ASSIGNMENT_SEPARATOR, maxsplit=1)
  if len(tokens) != 2:
    raise RuntimeError(f'Unable to parse line {lineNumber+1}: {line}')
  
  key = tokens[0].strip()
  value = tokens[1].strip()
  return key, value
  
def __makeConfigLine(key: str, value: str) -> str:
  """
	### Composes a config file line given a key-value pair to write.

	#### Params:
  - key (int): Name of the variable.
	- value (str): Value of the variable.
	
	#### Returns:
	- (str): String containing the appropiately formatted key-value pair.
	"""
  value = '' if value is None else value
  return key + ASSIGNMENT_SEPARATOR + value

def readConfig(path: str) -> Dict[str, str]:
  """
	### Reads the config file and returns a dictionary with all the parsed variables.

	#### Params:
	- path (str): Path to the config file.
	
	#### Returns:
	- (dict): Dictionary containing all the variables found in the config file.
	"""
  result = dict()
  
  with open(path, 'r') as configFile:
    for i, line in enumerate(configFile):
      keyval = __parseConfigLine(i, line)
      if keyval is not None:
        key, value = keyval
        result[key] = value
  
  return result

def writeConfig(path: str):
  """
	### Writes a template config file with empty variables.

	#### Params:
	- path (str): Path to the config file.
	"""
  lines = []
  with open(path, 'w') as configFile:
    lines.append("##### TOGGLE SECTION #####\n")
    lines.append("# Activate or deactivate this features using values ON/OFF\n")
    for toggle in CONFIG_VARIABLES[TOGGLES]:
      lines.append(__makeConfigLine(toggle, CONFIG_DEFAULT_VALUES[toggle]) + '\n')

    lines.append("\n##### PACKAGE HOME SECTION #####\n")
    lines.append("# Use this variables to use custom installation paths for the required packages.\n")
    lines.append("# If left empty, CMake will search for those packages within your system.\n")
    for location in CONFIG_VARIABLES[LOCATIONS]:
      lines.append(__makeConfigLine(location, CONFIG_DEFAULT_VALUES[location]) + '\n')
    configFile.writelines(lines)
