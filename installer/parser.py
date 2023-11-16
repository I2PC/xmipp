# ***************************************************************************
# * Authors:		Alberto García (alberto.garcia@cnb.csic.es)
# *							Martín Salinas (martin.salinas@cnb.csic.es)
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
from .constants import MODES, MODE_ARGS, TAB_SIZE, MODE_EXAMPLES
from .utils import getFormattingTabs, yellow, red

# File specific constants
N_TABS_MODE_TEXT = 6
TABS_MODE_TEXT = ''.join(['\t' for _ in range(N_TABS_MODE_TEXT)])
SECTION_N_DASH = 40
SECTION_SPACE_MODE_HELP = 2
SECTION_HELP_START = TAB_SIZE + SECTION_N_DASH + SECTION_SPACE_MODE_HELP
LINE_SIZE_LOWER_LIMIT = int(SECTION_HELP_START * 1.5)

####################### AUX FUNCTIONS #######################
def getLineSize() -> int:
	"""
	### This function returns the maximum size for a line.

	### Returns:
	(int): Maximum line size.
	"""
	# Getting column size in characters
	size = shutil.get_terminal_size().columns

	# Return size with lower limit
	return LINE_SIZE_LOWER_LIMIT if size < LINE_SIZE_LOWER_LIMIT else size

def helpSeparator() -> str:
	"""
	### This function returns the line that separates sections inside the help message.

	### Returns:
	(str): Line that separates sections inside the help message.
	"""
	dashes = ['-' for _ in range(SECTION_N_DASH)]
	return getFormattingTabs(f"\t{''.join(dashes)}\n")

def fitWordsInLine(words: List[str], sizeLimit: int) -> Tuple[str, List[str]]:
	"""
	### This function returns a tuple containig a line with the words from the given list that could fit given the size limit, and the list with the remaining words.

	### Params:
	- words (List[str]): List of words to try to fit into a line.
	- sizeLimit (int): Size limit for the text.

	### Returns:
	(str): Line with the words that were able to fit in it.
	(List[str]): List containing the words that could not fit in the line.
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

def multiLineHelpText(text: str, sizeLimit: int, leftFill: str) -> str:
	"""
	### This function returns the given text, formatted in several lines so that it does not exceed the given character limit.

	### Params:
	- text (str): The text to be formatted.
	- sizeLimit (int): Size limit for the text.
	- leftFill (str): String to add at the left of each new line.

	### Returns:
	(str): Formatted text.
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
			line, textWords = fitWordsInLine(textWords, sizeLimit)

			# Add line to list
			if line:
				# If it's not the first line, add the left fill
				line = leftFill + line if lines else line
				lines.append(line)

		# Join lines into a single string
		formattedText = '\n'.join(lines)
	
	# Return resulting text
	return formattedText

def textWithLimits(previousText: str, text: str) -> str:
	"""
	### This function returns the given text, formatted so that it does not exceed the character limit by line for the param help section.

	### Params:
	- previousText (str): Text inserted before the one to be returned.
	- text (str): The text to be formatted.

	### Returns:
	(str): Formatted text.
	"""
	# Obtain previous text length
	previousLength = len(getFormattingTabs(previousText))

	# Check if such length exceeds the space reserved for modes and params
	if previousLength >= SECTION_HELP_START:
		# If so, it means that section space for modes and params 
		# is too low and should be set to a higher number, but for now we need to print anyways, 
		# so we reduce space from the one reserved for mode help
		remainingSpace = getLineSize() - previousLength

		# Add minimum fill in space possible
		fillInSpace = ' '
	else:
		# If such section is within the expected size range, calculate remaining size
		# based on the expected help section beginning
		remainingSpace = getLineSize() - SECTION_HELP_START

		# Add fill in space
		fillInSpace = ''.join([' ' for _ in range(SECTION_HELP_START - previousLength)])
	
	# Format string so it does not exceed size limit
	formattedHelp = multiLineHelpText(text, remainingSpace, ''.join([' ' for _ in range(SECTION_HELP_START)]))

	return previousText + fillInSpace + formattedHelp + '\n'

####################### HELP FUNCTIONS #######################
def argsContainOptional(argNames: List[str]) -> bool:
	"""
	### This method returns True if the param name list contains at least one optional param.

	### Params:
	- argNames (List[str]): List containing the param names.

	### Returns:
	(bool): True if there is at least one optional param. False otherwise.
	"""
	# For every param name, check if starts with '-'
	for name in argNames:
		if name.startswith('-'):
			return True
	
	# If execution gets here, there were no optional params
	return False

def getModeHelp(mode: str) -> str:
	"""
	### This method returns the help message of a given mode.

	### Params:
	- mode (str): Mode to get help text for.

	### Returns:
	(str): Help of the mode (empty if mode not found).
	"""
	# Find mode group containing current mode
	for group in list(MODES.keys()):
		if mode in list(MODES[group].keys()):
			return MODES[group][mode]
	
	# If it was not found, return empty string
	return ''

def getModeArgsStr(mode: str) -> str:
	"""
	### This method returns the args text for a given mode.

	### Params:
	- mode (str): Mode to get args text for.

	### Returns:
	(str): Args text for given mode.
	"""
	# Getting argument dictionary for the mode  
	argDict = MODE_ARGS[mode]

	# Formatting every element
	paramNames = [f'[{paramName}]' for paramName in list(argDict.keys())]

	# Returning all formatted param names as a string
	return ' '.join(paramNames)

def getModeArgsAndHelpStr(previousText: str, mode: str) -> str:
	"""
	### This method returns the args and help text for a given mode.

	### Params:
	- previousText (str): Text inserted before the one to be returned.
	- mode (str): Mode to get help text for.

	### Returns:
	(str): Args and help text for given mode.
	"""
	# Initializing help string to format
	modeHelpStr = ''

	# Find mode group containing current mode
	modeHelpStr = getModeHelp(mode)

	# Return formatted text formed by the previous text, 
	# the args for the mode, and its help text
	return textWithLimits(previousText + getModeArgsStr(mode), modeHelpStr)

####################### PARSER CLASS #######################
class ComplexArgumentParser(argparse.ArgumentParser):
	"""
	This class extends the capabilities of the standard argument parser to be able
	to handle complex argument dependencies.
	"""
	def __init__(self, *args, mainParamName=None, **kwargs):
		"""
		### This constructor adds the ability to keep track of argument enforcement conditions.

		#### Params:
		- *args: Positional arguments passed to the parent class method.
		- mainParamName (str): Name of the main param.
		- **kwargs: Keyword arguments passed to the parent class method.
		"""
		super().__init__(*args, **kwargs)
		self.mainParamName = mainParamName

	####################### AUX PRIVATE FUNCTIONS #######################
	def _updateRequiredParam(self, paramName: str):
		"""
		### This method updates the given param to make it a requirement if it wasn't already.
		
		#### Params:
		- paramName (str): Name of the parameter.
		"""
		# Searching for the action we need
		for action in self._actions:
			if action.dest == paramName:
				# If nargs was '?' (1 or 0), set to None (1)
				if action.nargs == '?':
					action.nargs = None
				# If nargs was '*' (any number), set to '+' (at least 1)
				elif action.nargs == '*':
					action.nargs = '+'
				action.default = None
				action.required = True
				break
		
	def _updateMainParamIfPositionalArg(self, paramName: str):
		"""
		### This method updates the main param if it receives a different positional param.
		### This is done to ensure value integrity for both params.
		
		#### Params:
		- paramName (str): Name of the parameter.
		"""
		# Checking if argument is positional (optionals start with '-')
		if self.mainParamName is not None and not paramName.startswith('-'):
			# Update mode param so it cannot be blank now.
			# Otherwise, it will aquire the default value and the value
			# supposed to be for mode will end up in the positional param, as
			# that one cannot be blank and mode can
			self._updateRequiredParam(self.mainParamName)

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
		self.exit(2, f"{textList}{extraLineBreak}{errorMessage}")

	def parse_args(self, *args, **kwargs) -> argparse.Namespace:
		"""
		### This method parses the introduced args, making the main one a requirement if it is needed.
		
		#### Params:
		- *args: Positional arguments passed to the parent class method.
		- **kwargs: Keyword arguments passed to the parent class method.
		
		#### Returns:
		- (Namespace): The Namespace object containing the parsed arguments.
		"""
		# Parsing known args
		try:
			knownArgs = self.parse_known_args(*args, **kwargs)[0]
		except SystemExit:
			self._updateRequiredParam(self.mainParamName)
			# Parse the args again with the updated settings
			knownArgs = super().parse_args(*args, **kwargs)

		#print(knownArgs)
		#return knownArgs

	#def parse_args(self, *args, **kwargs) -> argparse.Namespace:
	#	"""
	#	### This method parses the introduced args, only enforcing the ones that fulfill their condition.
	#	
	#	#### Params:
	#	- *args: Positional arguments passed to the parent class method.
	#	- **kwargs: Keyword arguments passed to the parent class method.
	#	
	#	#### Returns:
	#	- (Namespace): The Namespace object containing the parsed arguments.
	#	"""
	#	# Obtaining conditional args dicitionary's number of elements
	#	nParams = len(self.conditionalArgs)
#
	#	# Iterate until dictionary is empty or max number of iterations has been reached (max = nParams)
	#	# Max iterations is number of params because, worst case, only one new param fulfills its condition
	#	# for every iteration, in case every conditional param deppends on another conditional param except for
	#	# one of them (at least one needs to deppend on a fixed param).
	#	for _ in range(nParams):
	#		# If dictionary is empty, stop iterating
	#		if not self.conditionalArgs:
	#			break
#
	#		# Parsing known args
	#		knownArgs = self.parse_known_args(*args, **kwargs)[0]
#
	#		# Obtaining all params that meet their condition
	#		metParamNames = self._getArgsWithMetCondition(knownArgs)
#
	#		# Adding all the params meeting their conditions and removing them from the dictionary
	#		for paramName in metParamNames:
	#			argList = self.conditionalArgs[paramName]
#
	#			# If argument is positional, make mode param a requirement
	#			self._updateMainParamIfPositionalArg(argList['args'][0])
#
	#			# Adding extra param
	#			self.add_argument(*argList['args'], **argList['kwargs'])
#
	#			# Removing param from dictionary
	#			self.conditionalArgs.pop(paramName)
#
	#	# Parse args
	#	return super().parse_args(*args, **kwargs)

class GeneralHelpFormatter(argparse.HelpFormatter):
	"""
	This class overrides the default help formatter to display a custom help message.
	"""
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
				helpMessage += getModeArgsAndHelpStr(f"\t{mode} ", mode)

		# Adding epilog and returning to print
		epilog = "Example 1: ./xmipp\n"
		epilog += "Example 2: ./xmipp compileAndInstall -j 4\n"
		helpMessage += '\n' + epilog
		return getFormattingTabs(helpMessage)

class ModeHelpFormatter(argparse.HelpFormatter):
	"""
	This class overrides the default help formatter to display a custom help message deppending on the mode selected.
	"""
	def format_help(self):
		"""
		### This method prints the help message of the argument parser.
		"""
		# Getting the selected mode from the parent help message
		# Message received is the format_help of the main parser's
		# formatter, adding the mode at the end
		mode = self._prog.split(' ')[-1]

		# Initialize the help message
		helpMessage = getModeHelp(mode) + '\n\n'

		# Get mode args
		args = list(MODE_ARGS[mode].keys())

		# Add extra messages deppending on if there are args
		optionsStr = ''
		separator = ''
		if len(args) > 0:
			if argsContainOptional(args):
				helpMessage += yellow("Note: only params starting with '-' are optional. The rest are required.\n")
			optionsStr = ' [options]'
			separator = helpSeparator() + '\t# Options #\n\n'
		helpMessage += f'Usage: xmipp {mode}{optionsStr}\n{separator}'

		# Adding arg info
		for arg in args:
			helpMessage += textWithLimits('\t' + arg, MODE_ARGS[mode][arg])

		# Adding a few examples
		examples = MODE_EXAMPLES[mode]
		for i in range(len(examples)):
			numberStr = '' if len(examples) == 1 else f' {i}'	
			helpMessage += f"\nExample{numberStr}: {examples[i]}"
		
		# If any test were added, add extra line break
		if len(examples) > 0:
			helpMessage += '\n'

		return getFormattingTabs(helpMessage)
	