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
Submodule containing all constants needed for the argument parsing part of Xmipp's installation.
"""

# Other variables
COMMON_USAGE_HELP_MESSAGE = 'Run \"./xmipp -h\" for usage help.'
DEFAULT_BUILD_DIR = './build'
DEFAULT_MODELS_DIR = DEFAULT_BUILD_DIR + '/models'

# Mode list (alphabetical order)
MODE_ADD_MODEL = 'addModel'
MODE_ALL = 'all'
MODE_CHECK_CONFIG = 'checkConfig'
MODE_CLEAN_ALL = 'cleanAll'
MODE_CLEAN_BIN = 'cleanBin'
MODE_CLEAN_DEPRECATED = 'cleanDeprecated'
MODE_COMPILE_AND_INSTALL = 'compileAndInstall'
MODE_CONFIG = 'config'
MODE_GET_MODELS = 'getModels'
MODE_GIT = 'git'
MODE_TEST = 'test'
MODE_VERSION = 'version'

# Group list
GROUP_GENERAL = 'General'
GROUP_CONFIG = 'Config'
GROUP_DOWNLOADS = 'Downloads'
GROUP_CLEAN = 'Clean'
GROUP_TEST = 'Test'
GROUP_DEVELOPERS = 'Developers'

# Modes with help message
MODES = {
	GROUP_GENERAL: {
		MODE_VERSION: 'Returns the version information. Add \'--short\' to print only the version number.',
		MODE_COMPILE_AND_INSTALL: 'Compiles and installs Xmipp based on already obtained sources.',
		MODE_ALL: 'Default param. Runs config, and compileAndInstall.'
	},
	GROUP_CONFIG: {
		MODE_CONFIG: 'Generates config file based on system information.',
		MODE_CHECK_CONFIG: 'Cheks if the values in the config file are ok.'
	},
	GROUP_DOWNLOADS: {
		MODE_GET_MODELS: f'Download the DeepLearning Models at dir/models ({DEFAULT_MODELS_DIR} by default).'
	},
	GROUP_CLEAN: {
		MODE_CLEAN_BIN: 'Removes all compiled binaries.',
		MODE_CLEAN_DEPRECATED: 'Removes all deprecated binaries from src/xmipp/bin.',
		MODE_CLEAN_ALL: 'Removes all compiled binaries and sources, leaves the repository as if freshly cloned (without pulling).'
	},
	GROUP_TEST: {
		MODE_TEST: 'Runs a given test.'
	},
	GROUP_DEVELOPERS: {
		MODE_GIT: 'Runs the given git action for all source repositories.',
		MODE_ADD_MODEL: 'Takes a DeepLearning model from the modelPath, makes a tgz of it and uploads the .tgz according to the <login>.'
	}
}

# Arguments of each mode, sorted by group, with their respective help message
MODE_ARGS = {
	MODE_VERSION: {
		'-dir': f"Directory where the xmipp will be installed. Default is \"{DEFAULT_BUILD_DIR}\".",
		'-short': "If set, only version number is shown."
	},
	MODE_COMPILE_AND_INSTALL: {
		'-j': f"Number of jobs. Defaults to all available.",
		'-br': "Branch for the source repositories.",
		'-dir': f"Directory where the xmipp will be installed. Default is \"{DEFAULT_BUILD_DIR}\"."
	},
	MODE_ALL: {
		'-j': f"Number of jobs. Defaults to all available.",
		'-br': "Branch for the source repositories.",
		'-dir': f"Directory where the xmipp will be installed. Default is \"{DEFAULT_BUILD_DIR}\"."
	},
	MODE_CONFIG: {},
	MODE_CHECK_CONFIG: {},
	MODE_GET_MODELS: {
		'-dir': f"Directory where the Deep Learning Models will be downloaded. Default is \"{DEFAULT_MODELS_DIR}\"."
	},
	MODE_CLEAN_BIN: {},
	MODE_CLEAN_DEPRECATED: {},
	MODE_CLEAN_ALL: {},
	MODE_TEST: {
		'testName': "Test to run. If combined with --show, greps the test name from the test list.",
		'-show': "If set, shows the tests available. If combined with a test name, greps that test name within the test list."
	},
	MODE_GIT: {
		'command': "Git command to run on all source repositories."
	},
	MODE_ADD_MODEL: {
		'login': "Login (usr@server) for Nolan machine to upload the model with. Must have write permisions to such machine.",
		'modelPath': "Path to the model to upload to Nolan."
	}
}

# Examples for the help message of each mode
MODE_EXAMPLES = {
	MODE_VERSION: [
		f'./xmipp {MODE_VERSION}',
		f'./xmipp {MODE_VERSION} --short',
		f'./xmipp {MODE_VERSION} -dir /path/to/my/build/dir'
	],
	MODE_COMPILE_AND_INSTALL: [
		f'./xmipp {MODE_COMPILE_AND_INSTALL}',
		f'./xmipp {MODE_COMPILE_AND_INSTALL} -j 20',
		f'./xmipp {MODE_COMPILE_AND_INSTALL} -dir /path/to/my/build/dir',
		f'./xmipp {MODE_COMPILE_AND_INSTALL} -br devel',
		f'./xmipp {MODE_COMPILE_AND_INSTALL} -j 20 dir /path/to/my/build/dir -br devel'
	],
	MODE_ALL: [
		'./xmipp',
		f'./xmipp {MODE_ALL}',
		'./xmipp -j 20',
		'./xmipp -dir /path/to/my/build/dir',
		'./xmipp -br devel',
		f'./xmipp {MODE_ALL} -j 20 dir /path/to/my/build/dir -br devel]'
	],
	MODE_CONFIG: [],
	MODE_CHECK_CONFIG: [],
	MODE_GET_MODELS: [f'./xmipp {MODE_GET_MODELS}', f'./xmipp {MODE_GET_MODELS} -dir /path/to/my/model/directory'],
	MODE_CLEAN_BIN: [],
	MODE_CLEAN_DEPRECATED: [],
	MODE_CLEAN_ALL: [],
	MODE_TEST: [f'./xmipp {MODE_TEST} testName', f'./xmipp {MODE_TEST} --show', f'./xmipp {MODE_TEST} testName --show'],
	MODE_GIT: [f'./xmipp {MODE_GIT} pull', f'./xmipp {MODE_GIT} checkout devel'],
	MODE_ADD_MODEL: [f'./xmipp {MODE_ADD_MODEL} myuser@127.0.0.1 /home/myuser/mymodel']
}