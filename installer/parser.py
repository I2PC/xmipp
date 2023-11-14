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
from argparse import Namespace
from typing import Dict

# Installer imports
from .utils import yellow, red

####################### AUX FUNCTIONS #######################
def printHelpSeparator():
	"""
	### This method prints the line that separates sections inside the help message.
	"""
	print("\t----------------------------\n")

####################### PARSER CLASS #######################
class ComplexArgumentParser(argparse.ArgumentParser):
	"""
	This class extends the capabilities of the standard argument parser to be able
	to handle complex argument dependencies.
	"""
	def __init__(self, *args, **kwargs):
		"""
		### This constructor adds the ability to keep track of argument enforcement conditions.

		#### Params:
		- *args, **kwargs: Positional arguments passed to the parent class method.
		- **kwargs: Keyword arguments passed to the parent class method.
		"""
		super().__init__(*args, **kwargs)
		self.conditionalArgs = {}

	####################### AUX FUNCTIONS #######################
	def _updateModeIfPositionalArg(self, paramName: str):
		"""
		### This method updates the mode param if it receives a different positional param.
		### This is done to ensure value integrity for both params.
		
		#### Params:
		- argList (str): Name of the parameter.
		"""
		# Checking if argument is positional (optionals start with '-')
		if not paramName.startswith('-'):
			# Update mode param so it cannot be blank now.
			# Otherwise, it will aquire the default value and the value
			# supposed to be for mode will end up in the positional param, as
			# that one cannot be blank and mode can
			for action in self._actions:
				if action.dest == 'mode':
					action.nargs = None
					action.default = None
					break

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
		customMessage = """HELP MESSAGE WIP."""

		#for dest, condition in self.argumentConditions.items():
		#	# Fetch the argument
		#	argument = self._option_string_actions[dest]

		#	# Format the argument as a string
		#	argumentStr = f"{argument.option_strings} {argument.metavar or argument.dest.upper()}"

		#	# Add the argument and its condition to the help message
		#	customMessage += f"\n{argumentStr}: (only if {condition}) {argument.help}"

		print(customMessage)

	def parse_args(self, *args, **kwargs) -> Namespace:
		"""
		### This method parses the introduced args, only enforcing the ones that fulfill their condition.
		
		#### Params:
		- *args: Positional arguments passed to the parent class method.
		- **kwargs: Keyword arguments passed to the parent class method.
		
		#### Returns:
		- (Namespace): The Namespace object containing the parsed arguments.
		"""
		# Keep iterating until conditional args dictionary is empty
		while self.conditionalArgs:
			# Parsing known args
			knownArgs = self.parse_known_args(*args, **kwargs)[0]

			# Storing all param variables into local variables to allow eval comparison
			if knownArgs:
				for variable, value in knownArgs._get_kwargs():
					locals()[variable] = value

			# Checking if current conditional arg must be added as an extra param due to the values of known args
			paramName = list(self.conditionalArgs.keys())[0]
			argList = self.conditionalArgs[paramName]
			try:
				if eval(argList['condition']):
					# If argument is positional, make mode param a requirement
					self._updateModeIfPositionalArg(argList['args'][0])

					# Adding extra param
					self.add_argument(*argList['args'], **argList['kwargs'])

				# Removing conditional param from dictionary and reparsing
				# This is done to support conditions between conditional args
				# For example, conditional arg1 might deppend on fixed arg, but conditional arg2 might
				# deppend on value of conditional arg1, so, to be able to check arg2's condition, arg1 must have been parsed before 
				self.conditionalArgs.pop(paramName)
			except NameError as err:
				warningStr = yellow("Did you define a conditional variable that deppends on a different conditional variable?\n"
												"If so, the one that does not have such dependency must be defined before the one deppending on it.\n"
												"Example: We have 3 args: fixed, cond1, cond2.\n"
												"\tcond1 deppends on fixed, and cond2 deppends on cond1. Therefore, cond1 needs to be defined before cond2.")
				raise NameError(f"{red('Error with condition ' + argList['condition'] + ': ' + str(err))}\n{warningStr}")

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