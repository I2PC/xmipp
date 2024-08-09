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
Provides a global logger.
"""

# General imports
import shutil, math

# Installer imports
from .constants import (ERROR_CODE, DOCUMENTATION_URL, UP, REMOVE_LINE,
	BOLD, BLUE, RED, GREEN, YELLOW, END_FORMAT, FORMATTING_CHARACTERS)

####################### TEXT MODE #######################
def green(text: str) -> str:
	"""
	### This function returns the given text formatted in green color.

	#### Params:
	- text (str): Text to format.

	#### Returns:
	- (str): Text formatted in green color.
	"""
	return f"{GREEN}{text}{END_FORMAT}"

def yellow(text: str) -> str:
	"""
	### This function returns the given text formatted in yellow color.

	#### Params:
	- text (str): Text to format.

	#### Returns:
	- (str): Text formatted in yellow color.
	"""
	return f"{YELLOW}{text}{END_FORMAT}"

def red(text: str) -> str:
	"""
	### This function returns the given text formatted in red color.

	#### Params:
	- text (str): Text to format.

	#### Returns:
	- (str): Text formatted in red color.
	"""
	return f"{RED}{text}{END_FORMAT}"

def blue(text: str) -> str:
	"""
	### This function returns the given text formatted in blue color.

	#### Params:
	- text (str): Text to format.

	#### Returns:
	- (str): Text formatted in blue color.
	"""
	return f"{BLUE}{text}{END_FORMAT}"

def bold(text: str) -> str:
	"""
	### This function returns the given text formatted in bold.

	#### Params:
	- text (str): Text to format.

	#### Returns:
	- (str): Text formatted in bold.
	"""
	return f"{BOLD}{text}{END_FORMAT}"

def removeNonPrintable(text: str) -> str:
	"""
	### This function returns the given text without non printable characters.

	#### Params:
	- text (str): Text to remove format.

	#### Returns:
	- (str): Text without format.
	"""
	for formattingChar in FORMATTING_CHARACTERS:
		text = text.replace(formattingChar, "")
	return text

class Logger:
	"""
	### Logger class for keeping track of installation messages.
	"""
 
	def __init__(self, outputToConsole: bool = False):
		"""
		### Constructor.
		
		#### Params:
		- ouputToConsoloe (bool): Print messages to console.
		"""
		self.__logFile = None
		self.__outputToConsole = outputToConsole
		self.__lenLastPrintedElem = 0
		self.__allowSubstitution = True
	
	def startLogFile(self, logPath: str):
		"""
		### Initiates the log file.

		#### Params:
		- logPath (str): Path to the log file.
		"""
		self.__logFile = open(logPath, 'w')

	def setConsoleOutput(self, outputToConsole: bool):
		"""
		### Modifies console output behaviour.
		
		#### Params:
		- ouputToConsoloe (str): Enable printing messages to console.
		"""
		self.__outputToConsole = outputToConsole
 
	def setAllowSubstitution(self, allowSubstitution: bool):
		"""
		### Modifies console output behaviour, allowing or disallowing substitutions.
		
		#### Params:
		- allowSubstitution (bool): If False, console outputs won't be substituted.
		"""
		self.__allowSubstitution = allowSubstitution


	def __call__(self, text: str, forceConsoleOutput: bool = False, substitute: bool = False):
		"""
		### Log a message.
		
		#### Params:
		- text (str): Message to be logged. Supports fancy formatting.
		- forceConsoleOutput (bool): Optional. If True, text is also printed through terminal.
		- substitute (bool): Optional. If True, previous line is substituted with new text. Only used when forceConsoleOutput = True.
		"""
		if self.__logFile is not None:
			print(removeNonPrintable(text), file=self.__logFile, flush=True)
			
		if self.__outputToConsole or forceConsoleOutput:
			# Calculate number of lines to substitute if substitution was requested
			substitutionStr = ''.join([f'{UP}{REMOVE_LINE}' for _ in range(self.__getNLastLines())])
			text = f"{substitutionStr}{text}" if self.__allowSubstitution and substitute else text
			print(text, flush=True)
			# Store length of printed string for next substitution calculation
			self.__lenLastPrintedElem = len(removeNonPrintable(text))
	 
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

	def __getNLastLines(self) -> int:
		"""
		### This function returns the number of lines of the terminal the last print occupied.

		#### Returns:
		- (int): Number of lines of the last print. 
		"""
		return math.ceil(self.__lenLastPrintedElem / shutil.get_terminal_size().columns)

"""
### Global logger.
"""
logger = Logger()
