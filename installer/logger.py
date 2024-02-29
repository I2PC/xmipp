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

"""
Provides a global logger
"""

import re

from .constants import LOG_FILE

####################### TEXT MODE #######################
def green(text: str) -> str:
	"""
	### This function returns the given text formatted in green color.

	#### Params:
	- text (str): Text to format.

	#### Returns:
	- (str): Text formatted in green color.
	"""
	return f"\033[92m{text}\033[0m"

def yellow(text: str) -> str:
	"""
	### This function returns the given text formatted in yellow color.

	#### Params:
	- text (str): Text to format.

	#### Returns:
	- (str): Text formatted in yellow color.
	"""
	return f"\033[93m{text}\033[0m"

def red(text: str) -> str:
	"""
	### This function returns the given text formatted in red color.

	#### Params:
	- text (str): Text to format.

	#### Returns:
	- (str): Text formatted in red color.
	"""
	return f"\033[91m{text}\033[0m"

def blue(text: str) -> str:
	"""
	### This function returns the given text formatted in blue color.

	#### Params:
	- text (str): Text to format.

	#### Returns:
	- (str): Text formatted in blue color.
	"""
	return f"\033[34m{text}\033[0m"

def bold(text: str) -> str:
	"""
	### This function returns the given text formatted in bold.

	#### Params:
	- text (str): Text to format.

	#### Returns:
	- (str): Text formatted in bold.
	"""
	return f"\033[1m{text}\033[0m"

def remove_text_formatting(text: str) -> str:
	"""
	### This function returns the given text without fancy formatting

	#### Params:
	- text (str): Text to remove format

	#### Returns:
	- (str): Text without format
	"""
	ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
	return ansi_escape.sub('', text)

class Logger:
	"""
	### Logger class for keeping track of installation messages
	"""
 
	def __init__(self, logPath: str, outputToConsole: bool = False):
		"""
		### Constructor
		
		#### Params:
		- logPath (str): Path to the log file.
		- ouputToConsoloe (str): Print messages to console 
		"""
		self.logFile = open(logPath, 'w')
		self.outputToConsole = outputToConsole
	
	def setConsoleOutput(self, outputToConsole: bool):
		"""
		### Modifies console output beaviour
		
		#### Params:
		- ouputToConsoloe (str): Enable printing messages to console 
		"""
		self.outputToConsole = outputToConsole
 
	def __call__(self, text: str):
		"""
		### Log a message
		
		#### Params:
		- text (str): Message to be logged. Supports fancy formatting
		"""
		print(remove_text_formatting(text), file=self.logFile, flush=True)
		if self.printToConsole:
			print(text, flush=True)
	

"""
### Global logger
"""
logger = Logger(LOG_FILE)
