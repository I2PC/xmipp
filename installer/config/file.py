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

ASSIGNMENT_SEPARATOR = '='
COMMENT_ESCAPE = '#'

def _parseConfigLine(i: int, line: str) -> Optional[Tuple[str, str]]:
  # Skip if comments
  line = line.split(COMMENT_ESCAPE, maxsplit=2)[0]
  
  # Remove spaces
  line = line.strip()
  
  # Check if empty line
  if not line:
    return None
  
  # Try to parse the line
  tokens = line.split(ASSIGNMENT_SEPARATOR, maxsplit=2)
  if len(tokens) != 2:
    raise RuntimeError(f'Unable to parse line {i+1}: {line}')
  
  key = tokens[0].strip()
  value = tokens[1].strip()
  return key, value
  
def _makeConfigLine(key: str, value: str) -> str:
  return key + ASSIGNMENT_SEPARATOR + value

def readConfig(path: str) -> Dict[str, str]:
  result = dict()
  
  with open(path, 'r') as file:
    for i, line in enumerate(file):
      keyval = _parseConfigLine(i, line)
      if keyval is not None:
        key, value = keyval
        result[key] = value
  
  return result

def writeConfig(config: Dict[str, str], path: str):
  with open(path, 'w') as file:
    print(
      COMMENT_ESCAPE,
      'Configuration file automatically generated at', 
      datetime.now(),
      file=file
    )
    
    for key, value in config.items():
      print(_makeConfigLine(key, value), file=file)
