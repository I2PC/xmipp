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

import os, re
from typing import Dict, Tuple, Optional
from datetime import datetime
from copy import copy
from .logger import logger, yellow
from .constants import (CONFIG_VARIABLES, CONFIG_DEFAULT_VALUES, TOGGLES,
  LOCATIONS, COMPILATION_FLAGS, ON, OFF, CONFIG_FILE)

__ASSIGNMENT_SEPARATOR = '='
__COMMENT_ESCAPE = '#'
__LAST_MODIFIED_TEXT = "Config file automatically generated on"

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
  line = line.split(__COMMENT_ESCAPE, maxsplit=2)[0].strip()
  
  # Check if empty line
  if not line:
    return None
  
  # Try to parse the line
  tokens = line.split(__ASSIGNMENT_SEPARATOR, maxsplit=1)
  if len(tokens) != 2:
    raise RuntimeError(f'Unable to parse line {lineNumber+1}: {line}')
  
  key = tokens[0].strip()
  value = tokens[1].strip()
  return key, value
  
def __makeConfigLine(key: str, value: str, defaultValue: str) -> str:
  """
	### Composes a config file line given a key-value pair to write.

	#### Params:
  - key (int): Name of the variable.
	- value (str): Value of the variable found in the config file.
  - defaultValue (str): Default value of the variable.
	
	#### Returns:
	- (str): String containing the appropiately formatted key-value pair.
	"""
  defaultValue = '' if defaultValue is None else defaultValue
  value = defaultValue if value is None else value
  return key + __ASSIGNMENT_SEPARATOR + value

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
      try:
        keyval = __parseConfigLine(i, line)
      except RuntimeError as rte:
        warningStr = f"WARNING: There was an error parsing {CONFIG_FILE} file: {rte}\n"
        warningStr += "Contents of config file won't be read, default values will be used instead.\n"
        warningStr += "You can create a new file template from scratch running './xmipp config -o'."
        logger(yellow(warningStr), forceConsoleOutput=True)
        return CONFIG_DEFAULT_VALUES
      if keyval is not None:
        key, value = keyval
        result[key] = value
  
  return result

def writeConfig(path: str, configDict: Dict=None):
  """
	### Writes a template config file with given variables, leaving the rest with default values.

	#### Params:
	- path (str): Path to the config file.
  - configDict (dict): Optional. Dictionary containig already existing variables.
	"""
  variables = copy(configDict) if configDict else {}
  lines = []
  with open(path, 'w') as configFile:
    lines.append("##### TOGGLE SECTION #####\n")
    lines.append(f"# Activate or deactivate this features using values {ON}/{OFF}\n")
    for toggle in CONFIG_VARIABLES[TOGGLES]:
      lines.append(__makeConfigLine(toggle, variables.get(toggle), CONFIG_DEFAULT_VALUES[toggle]) + '\n')
      variables.pop(toggle, None)

    lines.append("\n##### PACKAGE HOME SECTION #####\n")
    lines.append("# Use this variables to use custom installation paths for the required packages.\n")
    lines.append("# If left empty, CMake will search for those packages within your system.\n")
    for location in CONFIG_VARIABLES[LOCATIONS]:
      lines.append(__makeConfigLine(location, variables.get(location), CONFIG_DEFAULT_VALUES[location]) + '\n')
      variables.pop(location, None)
    
    lines.append("\n##### COMPILATION FLAGS #####\n")
    lines.append("# We recommend not modifying this variables unless you know what you are doing.\n")
    for flag in CONFIG_VARIABLES[COMPILATION_FLAGS]:
      lines.append(__makeConfigLine(flag, variables.get(flag), CONFIG_DEFAULT_VALUES[flag]) + '\n')
      variables.pop(flag, None)
    
    # If there are extra unkown flags, add them in a different section
    if variables:
      lines.append("\n##### UNKNOWN VARIABLES #####\n")
      lines.append("# This variables were not expected, but are kept here in case they might be needed.\n")
      for variable in variables.keys():
        lines.append(__makeConfigLine(variable, variables[variable], '') + '\n')

    lines.append(f"\n# {__LAST_MODIFIED_TEXT} {datetime.today()}\n")
    configFile.writelines(lines)

def getConfigDate(path: str) -> str:
  """
  ### This function obtains from the config file the date of its last modification.

  #### Params:
  - path (str): Path to the config file.

  #### Returns:
  - (str): Date formatted in dd/mm/yyyy.
  """
  if not os.path.exists(path):
    return ''
  
  # Extract line with date, and date from such line
  dateStr = ''
  with open(path, 'r') as configFile:
    for line in configFile:
        if __LAST_MODIFIED_TEXT in line:
          match = re.search(r'\d{4}-\d{2}-\d{2}', line)
          if match:
            dateStr = match.group()
  
  if dateStr:
    dateStr = datetime.strptime(dateStr, '%Y-%m-%d').strftime('%d/%m/%Y')

  return dateStr
