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

# General imports
import re

# Installer imports
from .constants import LOG_FILE, ERROR_CODE, DOCUMENTATION_URL, UP, REMOVE_LINE

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

def removeTextFormatting(text: str) -> str:
	"""
	### This function returns the given text without fancy formatting

	#### Params:
	- text (str): Text to remove format

	#### Returns:
	- (str): Text without format
	"""
	ansiEscape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
	return ansiEscape.sub('', text)

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
 
	def __call__(self, text: str, forceConsoleOutput: bool = False, substitute: bool = False):
		"""
		### Log a message
		
		#### Params:
		- text (str): Message to be logged. Supports fancy formatting
		- forceConsoleOutput (bool): Optional. If True, text is also printed through terminal.
		- substitute (bool): Optional. If True, previous line is substituted with new text. Only used when forceConsoleOutput = True.
		"""
		print(removeTextFormatting(text), file=self.logFile, flush=True)
		if self.outputToConsole or forceConsoleOutput:
			text = text if not substitute else f"{UP}{REMOVE_LINE}{text}"
			print(text, flush=True)
	 
	def logError(self, errorMsg: str, retCode: int=1, addPortalLink: bool=True):
		"""
		### This function prints an error message.

		#### Params:
		- errorMsg (str): Error message to show.
		- retCode (int): Optional. Return code to end the exection with.
		- addPortalLink (bool): If True, a message linking the documentation portal is shown.
		"""
		errorStr = errorMsg + '\n\n'
		errorStr += f'Error {retCode}: {ERROR_CODE[retCode][0]}'
		errorStr += f"\n{ERROR_CODE[retCode][1]} " if ERROR_CODE[retCode][1] else ''
		if addPortalLink:
			errorStr += f'\nMore details on the Xmipp documentation portal: {DOCUMENTATION_URL}'

		self.__call__(red(errorStr), forceConsoleOutput=True)

"""
### Global logger
"""
logger = Logger(LOG_FILE)
