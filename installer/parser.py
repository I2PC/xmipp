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
import argparse
from typing import List, Tuple

# Installer imports
from .constants import MODES, MODE_ARGS, TAB_SIZE
from .utils import getFormattingTabs

# File specific constants
N_TABS_MODE_TEXT = 6
TABS_MODE_TEXT = ''.join(['\t' for _ in range(N_TABS_MODE_TEXT)])
SECTION_N_DASH = 40
SECTION_SPACE_MODE_HELP = 2
SECTION_HELP_START = TAB_SIZE + SECTION_N_DASH + SECTION_SPACE_MODE_HELP
LINE_SIZE_LIMIT = SECTION_HELP_START * 2.25

####################### AUX FUNCTIONS #######################
def helpSeparator() -> str:
	"""
	### This method returns the line that separates sections inside the help message.

	### Returns:
	(str): Line that separates sections inside the help message.
	"""
	dashes = ['-' for _ in range(SECTION_N_DASH)]
	return getFormattingTabs(f"\t{''.join(dashes)}\n")

def fitWordsInLine(words: List[str], sizeLimit: int) -> Tuple[str, List[str]]:
	"""
	### This method returns a tuple containig a line with the words from the given list that could fit given the size limit, and the list with the remaining words.

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
	### This method returns the given text, formatted in several lines so that it does not exceed the given character limit.

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
	### This method returns the given text, formatted so that it does not exceed the character limit by line for the param help section.

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
		remainingSpace = LINE_SIZE_LIMIT - previousLength

		# Add minimum fill in space possible
		fillInSpace = ' '
	else:
		# If such section is within the expected size range, calculate remaining size
		# based on the expected help section beginning
		remainingSpace = LINE_SIZE_LIMIT - SECTION_HELP_START

		# Add fill in space
		fillInSpace = ''.join([' ' for _ in range(SECTION_HELP_START - previousLength)])
	
	# Format string so it does not exceed size limit
	formattedHelp = multiLineHelpText(text, remainingSpace, ''.join([' ' for _ in range(SECTION_HELP_START)]))

	return previousText + fillInSpace + formattedHelp + '\n'

####################### HELP FUNCTIONS #######################
def getModeArgs(mode: str) -> str:
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

def getModeArgsAndHelp(previousText: str, mode: str) -> str:
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
	for group in list(MODES.keys()):
		if mode in list(MODES[group].keys()):
			modeHelpStr = MODES[group][mode]
			break

	# Return formatted text formed by the previous text, 
	# the args for the mode, and its help text
	return textWithLimits(previousText + getModeArgs(mode), modeHelpStr)

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
		self.conditionalArgs = {}
		self.mainParamName = mainParamName

	####################### AUX PRIVATE FUNCTIONS #######################
	def _getArgsWithMetCondition(self, knownArgs: argparse.Namespace) -> List[str]:
		"""
		### This method returns a list containing all the conditional param names that meet their condition.
		
		#### Params:
		- knownArgs (Namespace): Namespace object composed of the already parsed and recognized arguments.

		#### Returns:
		(List[str]): List containing all the param names for all the params whose condition is fulfilled.
		"""
		# Initialize empty list to store param names
		metParamNames = []

		# Only makes sense processing if there are any known args
		if not knownArgs:
			return metParamNames

		# Storing all param variables into local variables to allow eval comparison
		for variable, value in knownArgs._get_kwargs():
			locals()[variable] = value

		# Iterate conditional params getting every param with a fulfilled condition
		for paramName, argList in list(self.conditionalArgs.items()):
			try:
				# Checking if condition is met. Try-except is needed for params that might
				# deppend on another conditional param. When evaluating, they will try to 
				# compare with variables not yet defined.
				if eval(argList['condition']):
					# If param's condition is met, add to list
					metParamNames.append(paramName)
			except NameError:
				continue

		return metParamNames

	def _updateRequiredParam(self, paramName: str):
		"""
		### This method updates the given param to make it a requirement if it wasn't already.
		
		#### Params:
		- paramName (str): Name of the parameter.
		"""
		for action in self._actions:
			if action.dest == paramName:
				action.nargs = None
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
	def add_argument(self, *args, condition: str=None, **kwargs):
		"""
		### This method adds the given parameter to the argument list, while
		### keeping track of its enforcement condition.
		
		#### Params:
		- *args: Positional rguments passed to the parent class method.
		- condition (str): The enforcement condition for the argument.
		- **kwargs: Keyword arguments passed to the parent class method.
		"""
		# Call the original add_argument method if no condition is provided
		if condition is None:
			super().add_argument(*args, **kwargs)
		else:
			# Store the condition for this argument
			argName = args[0]
			self.conditionalArgs[argName] = {'condition': condition, 'args': args, 'kwargs': kwargs}

	def format_help(self):
		"""
		### This method prints the help message of the argument parser.
		"""
		# Base message
		helpMessage = self.description + '\n\nUsage: xmipp [options]\n'

		# Add every section
		for section in list(MODES.keys()):
			# Adding section separator and section name
			helpMessage += helpSeparator() + f"\t# {section} #\n\n"

			# Adding help text for every mode in each section
			for mode in list(MODES[section].keys()):
				helpMessage += getModeArgsAndHelp(f"\t{mode} ", mode)

		# Adding epilog and printing
		helpMessage += '\n' + self.epilog
		print(getFormattingTabs(helpMessage))

	def parse_args(self, *args, **kwargs) -> argparse.Namespace:
		"""
		### This method parses the introduced args, only enforcing the ones that fulfill their condition.
		
		#### Params:
		- *args: Positional arguments passed to the parent class method.
		- **kwargs: Keyword arguments passed to the parent class method.
		
		#### Returns:
		- (Namespace): The Namespace object containing the parsed arguments.
		"""
		# Obtaining conditional args dicitionary's number of elements
		nParams = len(self.conditionalArgs)

		# Iterate until dictionary is empty or max number of iterations has been reached (max = nParams)
		# Max iterations is number of params because, worst case, only one new param fulfills its condition
		# for every iteration, in case every conditional param deppends on another conditional param except for
		# one of them (at least one needs to deppend on a fixed param).
		for _ in range(nParams):
			# If dictionary is empty, stop iterating
			if not self.conditionalArgs:
				break

			# Parsing known args
			knownArgs = self.parse_known_args(*args, **kwargs)[0]

			# Obtaining all params that meet their condition
			metParamNames = self._getArgsWithMetCondition(knownArgs)

			# Adding all the params meeting their conditions and removing them from the dictionary
			for paramName in metParamNames:
				argList = self.conditionalArgs[paramName]

				# If argument is positional, make mode param a requirement
				self._updateMainParamIfPositionalArg(argList['args'][0])

				# Adding extra param
				self.add_argument(*argList['args'], **argList['kwargs'])

				# Removing param from dictionary
				self.conditionalArgs.pop(paramName)

		# Parse args
		return super().parse_args(*args, **kwargs)
