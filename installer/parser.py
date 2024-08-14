# ***************************************************************************
# * Authors:		Martín Salinas (martin.salinas@cnb.csic.es)
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
This module contains a class that extends the capabilities of standard argparser.
"""

# General imports
import argparse, shutil
from typing import List, Tuple

# Installer imports
from .constants import (MODES, MODE_ARGS, TAB_SIZE, MODE_EXAMPLES,
	MODE_ALL, PARAMS, SHORT_VERSION, LONG_VERSION, DESCRIPTION)
from .logger import yellow, red

# File specific constants
SECTION_N_DASH = 45
SECTION_SPACE_MODE_HELP = 2
SECTION_HELP_START = TAB_SIZE + SECTION_N_DASH + SECTION_SPACE_MODE_HELP
LINE_SIZE_LOWER_LIMIT = int(SECTION_HELP_START * 1.5)

####################### AUX FUNCTIONS #######################
def __getLineSize() -> int:
	"""
	### This function returns the maximum size for a line.

	### Returns:
	- (int): Maximum line size.
	"""
	# Getting column size in characters
	size = shutil.get_terminal_size().columns

	# Return size with lower limit
	return LINE_SIZE_LOWER_LIMIT if size < LINE_SIZE_LOWER_LIMIT else size

def __fitWordsInLine(words: List[str], sizeLimit: int) -> Tuple[str, List[str]]:
	"""
	### This function returns a tuple containig a line with the words from the given list that could fit given the size limit, and the list with the remaining words.

	### Params:
	- words (List[str]): List of words to try to fit into a line.
	- sizeLimit (int): Size limit for the text.

	### Returns:
	- (str): Line with the words that were able to fit in it.
	- (List[str]): List containing the words that could not fit in the line.
	"""
	# Initializing line and creating copy of word list
	# The copy is made because original list cannot be edited mid iteration
	line = ''
	remainingWords = words

	# Check if each word fits in the line
	for word in words:
		# If the line is not empty, len includes extra space
		if line:
			if len(line + ' ' + word) > sizeLimit:
				return line, remainingWords
			else:
				line += ' ' + word
				remainingWords = remainingWords[1:]
		else:
			# If the first word already exceeds the size limit,
			# it means it is a huge word, but we need to print it
			# anyways and move on to the next line
			if len(word) >= sizeLimit:
				return word, remainingWords[1:]
			else:
				# If word fits, add to line, and remove it from word list
				line = word
				remainingWords = remainingWords[1:]
	
	# If we exited the loop, it means all words were introduced in the line
	return line, []

def __multiLineHelpText(text: str, sizeLimit: int, leftFill: str) -> str:
	"""
	### This function returns the given text, formatted in several lines so that it does not exceed the given character limit.

	### Params:
	- text (str): The text to be formatted.
	- sizeLimit (int): Size limit for the text.
	- leftFill (str): String to add at the left of each new line.

	### Returns:
	- (str): Formatted text.
	"""
	if len(text) <= sizeLimit:
		# If its size is within the limits, return as is
		formattedText = text
	else:
		# If size exceeds limits, split into lines
		# We need to calculate each word size to not split the string in the
		# middle of a word
		textWords = text.split(' ')

		# Initializing list to store lines
		lines = []

		# While there are still words outside of a line, parse them into one.
		while textWords:
			# Getting new line and removing fitted words in such line
			line, textWords = __fitWordsInLine(textWords, sizeLimit)

			# Add line to list
			if line:
				# If it's not the first line, add the left fill
				line = leftFill + line if lines else line
				lines.append(line)

		# Join lines into a single string
		formattedText = '\n'.join(lines)
	
	# Return resulting text
	return formattedText

def getFormattingTabs(text: str) -> str:
	"""
	### This method returns the given text, formatted to expand tabs into a fixed tab size.

	### Params:
	- text (str): The text to be formatted.

	### Returns:
	- (str): Formatted text.
	"""
	return text.expandtabs(TAB_SIZE)

def helpSeparator() -> str:
	"""
	### This function returns the line that separates sections inside the help message.

	### Returns:
	- (str): Line that separates sections inside the help message.
	"""
	dashes = ['-' for _ in range(SECTION_N_DASH)]
	return getFormattingTabs(f"\t{''.join(dashes)}\n")

def textWithLimits(previousText: str, text: str) -> str:
	"""
	### This function returns the given text, formatted so that it does not exceed the character limit by line for the param help section.

	### Params:
	- previousText (str): Text inserted before the one to be returned.
	- text (str): The text to be formatted.

	### Returns:
	- (str): Formatted text.
	"""
	# Obtain previous text length
	previousLength = len(getFormattingTabs(previousText))

	# Check if such length exceeds the space reserved for modes and params
	if previousLength >= SECTION_HELP_START:
		# If so, it means that section space for modes and params 
		# is too low and should be set to a higher number, but for now we need to print anyways, 
		# so we reduce space from the one reserved for mode help
		remainingSpace = __getLineSize() - previousLength

		# Add minimum fill in space possible
		fillInSpace = ' '
	else:
		# If such section is within the expected size range, calculate remaining size
		# based on the expected help section beginning
		remainingSpace = __getLineSize() - SECTION_HELP_START

		# Add fill in space
		fillInSpace = ''.join([' ' for _ in range(SECTION_HELP_START - previousLength)])
	
	# Format string so it does not exceed size limit
	formattedHelp = __multiLineHelpText(text, remainingSpace, ''.join([' ' for _ in range(SECTION_HELP_START)]))

	return previousText + fillInSpace + formattedHelp + '\n'

def getParamFirstName(paramKey: str) -> str:
	"""
	### This function returns the first name of the given param key. Short name has priority over long name.

	### Params:
	- paramKey (str): Key to identify the param.

	### Returns:
	- (str): Formatted text.
	"""
	return PARAMS[paramKey].get(SHORT_VERSION, PARAMS[paramKey].get(LONG_VERSION, ''))

####################### HELP FUNCTIONS #######################
def getModeHelp(mode: str, general: bool=True) -> str:
	"""
	### This method returns the help message of a given mode.

	### Params:
	- mode (str): Mode to get help text for.
	- general (bool). Optional. If True, only the general help message is displayed.

	### Returns:
	- (str): Help of the mode (empty if mode not found).
	"""
	# Find mode group containing current mode
	for group in list(MODES.keys()):
		if mode in list(MODES[group].keys()):
			messageList = MODES[group][mode]
			if general:
				return messageList[0]
			else:
				return '\n'.join(messageList)
	
	# If it was not found, return empty string
	return ''

def getParamNames(paramKey: str) -> List[str]:
	"""
	### This method returns the list of possible names a given param has.

	### Params:
	- paramKey (str): Key to find the param.

	### Returns:
	- (List[str]): List of all the names of the given param.
	"""
	nameList = [PARAMS[paramKey].get(SHORT_VERSION, ''), PARAMS[paramKey].get(LONG_VERSION, '')]
	return [name for name in nameList if name]

####################### PARSER CLASS #######################
class ErrorHandlerArgumentParser(argparse.ArgumentParser):
	"""
	This class overrides the error function of the standard argument parser
	to display better error messages.
	"""
	####################### OVERRIDED PUBLIC FUNCTIONS #######################
	def error(self, message):
		"""
		### This method prints through stderr the error message and exits with specific return code.
		
		#### Params:
		- message (str): Error message.
		"""
		# Getting mode and usage help from text
		textList = self.prog.split(' ')
		mode = textList[-1]

		# If text list only contains one item, mode is generic and
		# we need to get the help text
		if len(textList) > 1:
			textList = ' '.join(textList[:-1])
			extraLineBreak = '\n'
		else:
			textList = self.format_help()
			extraLineBreak = ''

		# Exiting with message
		errorMessage = red(f"{mode}: error: {message}\n")
		self.exit(1, getFormattingTabs(f"{textList}{extraLineBreak}{errorMessage}"))

class GeneralHelpFormatter(argparse.HelpFormatter):
	"""
	This class overrides the default help formatter to display a custom help message.
	"""
	def __getModeArgsStr(self, mode: str) -> str:
		"""
		### This method returns the args text for a given mode.

		### Params:
		- mode (str): Mode to get args text for.

		### Returns:
		- (str): Args text for given mode.
		"""
		# Getting argument list for the mode  
		argList = MODE_ARGS[mode]

		# Formatting every element
		paramNames = []
		for param in argList:
			paramName = getParamFirstName(param)
			if paramName:
				paramNames.append(f'[{paramName}]')

		# Returning all formatted param names as a string
		return ' '.join(paramNames)

	def __getModeArgsAndHelpStr(self, previousText: str, mode: str) -> str:
		"""
		### This method returns the args and help text for a given mode.

		### Params:
		- previousText (str): Text inserted before the one to be returned.
		- mode (str): Mode to get help text for.

		### Returns:
		- (str): Args and help text for given mode.
		"""
		# Initializing help string to format
		modeHelpStr = ''

		# Find mode group containing current mode
		modeHelpStr = getModeHelp(mode)

		# Return formatted text formed by the previous text, 
		# the args for the mode, and its help text
		return textWithLimits(previousText + self.__getModeArgsStr(mode), modeHelpStr)

	def format_help(self):
		"""
		### This method prints the help message of the argument parser.
		"""
		# Base message
		helpMessage = "Run Xmipp's installer script\n\nUsage: xmipp [options]\n"

		# Add every section
		for section in list(MODES.keys()):
			# Adding section separator and section name
			helpMessage += helpSeparator() + f"\t# {section} #\n\n"

			# Adding help text for every mode in each section
			for mode in list(MODES[section].keys()):
				helpMessage += self.__getModeArgsAndHelpStr(f"\t{mode} ", mode)

		# Adding epilog and returning to print
		epilog = "Example 1: ./xmipp\n"
		epilog += "Example 2: ./xmipp compileAndInstall -j 4\n"
		helpMessage += '\n' + epilog

		# Adding note about mode specific help
		noteMessage = "Note: You can also view a specific help message for each mode with \"./xmipp [mode] -h\".\n"
		noteMessage += f"Example: ./xmipp {MODE_ALL} -h\n"
		helpMessage += yellow(noteMessage)
		return getFormattingTabs(helpMessage)

class ModeHelpFormatter(argparse.HelpFormatter):
	"""
	This class overrides the default help formatter to display a custom help message deppending on the mode selected.
	"""
	def __argsContainOptional(self, argNames: List[str]) -> bool:
		"""
		### This method returns True if the param name list contains at least one optional param.

		### Params:
		- argNames (List[str]): List containing the param names.

		### Returns:
		- (bool): True if there is at least one optional param. False otherwise.
		"""
		# For every param name, check if starts with '-'
		for name in argNames:
			if name.startswith('-'):
				return True
		
		# If execution gets here, there were no optional params
		return False

	def format_help(self):
		"""
		### This method prints the help message of the argument parser.
		"""
		# Getting the selected mode from the parent help message
		# Message received is the format_help of the main parser's
		# formatter, adding the mode at the end
		mode = self._prog.split(' ')[-1]

		# Initialize the help message
		helpMessage = getModeHelp(mode, general=False) + '\n\n'

		# Get mode args
		args = MODE_ARGS[mode]

		# Add extra messages deppending on if there are args
		optionsStr = ''
		separator = ''
		if len(args) > 0:
			argNames = [getParamFirstName(argName) for argName in args]
			if self.__argsContainOptional(argNames):
				helpMessage += yellow("Note: only params starting with '-' are optional. The rest are required.\n")
			optionsStr = ' [options]'
			separator = helpSeparator() + '\t# Options #\n\n'
		helpMessage += f'Usage: xmipp {mode}{optionsStr}\n{separator}'

		# Adding arg info
		for arg in args:
			helpMessage += textWithLimits('\t' + ', '.join(getParamNames(arg)), PARAMS[arg][DESCRIPTION])

		# Adding a few examples
		examples = MODE_EXAMPLES[mode]
		for i in range(len(examples)):
			numberStr = '' if len(examples) == 1 else f' {i+1}'	
			helpMessage += f"\nExample{numberStr}: {examples[i]}"
		
		# If any test were added, add extra line break
		if len(examples) > 0:
			helpMessage += '\n'

		return getFormattingTabs(helpMessage)
	