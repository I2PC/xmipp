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
from typing import List

# Installer imports
from .constants import MODES, MODE_ARGS
from .utils import getFormattingTabs

# File specific constants
N_TABS_MODE_TEXT = 6
TABS_MODE_TEXT = ''.join(['\t' for _ in range(N_TABS_MODE_TEXT)])

####################### AUX FUNCTIONS #######################
def helpSeparator() -> str:
	"""
	### This method returns the line that separates sections inside the help message.

	### Returns:
	(str): Line that separates sections inside the help message.
	"""
	return "\t----------------------------------------\n"

def textWithLimits(previousText: str, text: str) -> str:
	"""
	### This method returns the given text, formatted so that it does not exceed the character limit by line for the param help section.

	### Params:
	- previousText (str): Text inserted before the one to be returned.
	- text (str): The text to be formatted.

	### Returns:
	(str): Formatted text.
	"""
	#TODO: Implement
	return previousText + '\n'

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
			helpMessage += helpSeparator() + f"\t{section}\n\n"

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

"""
"Usage: xmipp [options]\n"
"   ----------------------------\n"
"   version [dir=build]         Returns the version information. Add '--short' to print only the version number.\n"
"   compile [N]                 Compile with N processors (8 by default)\n"
"   install [dir]               Install at dir (./build by default)\n"
"   compileAndInstall [N] [dir] Compile with N processors (8 by default) and install in the dir directory ('build' by\n"
"                               default)\n"
"   all [op1=opt1 op2=opt2...]  (Default) Retrieve [br=branch], configure, check, compile [N=8], install [dir=build]\n"
"   ----------------------------\n"
"   config [noAsk]              Configure compilation variables. If 'noAsk' is passed, it will try to automatically \n"
"                               found some libraries and compilers. \n"
"                               for compiling using system libraries\n"
"   check_config                Check that the configuration is correct\n"
"   ----------------------------\n"
"   get_dependencies            Retrieve dependencies from github\n"
"   get_devel_sources [branch]  Retrieve development sources from github for a given branch (devel branch by default)\n"
"   get_models [dir]            Download the Deep Learning Models at dir/models (./build/models by default).\n"
"   ----------------------------\n"
"   cleanBin                    Clean all already compiled files (build, .so,.os,.o in src/* and " + Config.FILE_NAME + ")\n"
"   cleanDeprecated             Clean all deprecated executables from src/xmipp/bin).\n"
"   cleanAll                    Delete all (sources and build directories)\n"
"   ----------------------------\n"
"   test [--show] testName:     Run tests to check Xmipp programs (without args, it shows a detailed help).\n"
"                               if --show is activated without testName all are shown, \n"
"                               instead a grep of testName is done \n"
"   ----------------------------\n"
"   For developers:\n"
"   create_devel_paths          Create bashrc files for devel\n"
"   git [command]               Git command to all 4 repositories\n"
"   gitConfig                   Change the git config from https to git\n"
"   addModel login modelPath    Takes a deepLearning model from the 'modelPath', makes a tgz of it and \n"
"                               uploads the .tgz according to the <login>. \n"
"                               Note the login (usr@server) must have write permisions to Nolan machine.\n"
"   tar <mode> [v=ver] [br=br]  Create a bundle of the xmipp (without arguments shows a detailed help)\n"
"                               <mode> can be 'Sources', 'BinDebian' or 'BinCentos', when Sources \n"
"                               put a branch (default: master).'\n"
"                               <ver> usually X.YY.MM (add debug to package this local script and \n"
"                               the local scripts/tar.py) \n"
"""