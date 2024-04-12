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
DEFAULT_MODELS_DIR = 'build'

# Mode list (alphabetical order)
MODE_ADD_MODEL = 'addModel'
MODE_ALL = 'all'
MODE_CLEAN_ALL = 'cleanAll'
MODE_CLEAN_BIN = 'cleanBin'
MODE_COMPILE_AND_INSTALL = 'compileAndInstall'
MODE_BUILD_CONFIG = 'configBuild'
MODE_CONFIG = 'config'
MODE_GET_MODELS = 'getModels'
MODE_GIT = 'git'
MODE_TEST = 'test'
MODE_VERSION = 'version'

# Modes with help message
MODES = {
	'General': {
		MODE_VERSION: ['Returns the version information. Add \'--short\' to print only the version number.'],
		MODE_COMPILE_AND_INSTALL: ['Compiles and installs Xmipp based on already obtained sources.'],
		MODE_ALL: [f'Default param. Runs {MODE_CONFIG}, {MODE_BUILD_CONFIG}, and {MODE_COMPILE_AND_INSTALL}.'],
		MODE_BUILD_CONFIG: ['Configures the project with CMake.']
	},
	'Config': {
		MODE_CONFIG: ['Generates a config file template with default values.'],
	},
	'Downloads': {
		MODE_GET_MODELS: [f'Download the DeepLearning Models at dir/models ({DEFAULT_MODELS_DIR} by default).']
	},
	'Clean': {
		MODE_CLEAN_BIN: ['Removes all compiled binaries.'],
		MODE_CLEAN_ALL: ['Removes all compiled binaries and sources, leaves the repository as if freshly cloned (without pulling).']
	},
	'Test': {
		MODE_TEST: ['Runs a given test.']
	},
	'Developers': {
		MODE_GIT: ['Runs the given git action for all source repositories.'],
		MODE_ADD_MODEL: [
			"Takes a DeepLearning model from the modelPath, makes a tgz of it and uploads the .tgz according to the <login>.",
			"This mode is used to upload a model folder to the Scipion/Xmipp server.",
			"Usually the model folder contains big files used to fed deep learning procedures"
			"with pretrained data. All the models stored in the server will be downloads"
			"using the 'get_models' mode or during the compilation/installation time"
			"or with scipion3 installb deepLearningToolkit modelsPath must be the absolute path.",
			"",
			"Usage: -> ./xmipp addModel <usr@server> <modelsPath> [--update]",
			"Steps:	0. modelName = basename(modelsPath) <- Please, check the folders name!",
			"        1. Packing in 'xmipp_model_modelName.tgz'",
			"        2. Check if that model already exists (use --update to override an existing model)",
			"        3. Upload the model to the server.",
			"        4. Update the MANIFEST file.",
			"",
			"The model name will be the folder name in <modelsPath>",
			"Must have write permisions to such machine."
		]
	}
}

# Definition of all params found in the
SHORT_VERSION = 'short'
LONG_VERSION = 'long'
DESCRIPTION = 'description'
# Possible param list
PARAM_XMIPP_DIRECTORY = 'xmipp-directory'
PARAM_SHORT = 'short'
PARAM_JOBS = 'jobs'
PARAM_BRANCH = 'branch'
PARAM_MODELS_DIRECTORY = 'models-directory'
PARAM_TEST_NAME = 'test-name'
PARAM_SHOW_TESTS = 'show-tests'
PARAM_GIT_COMMAND = 'git-command'
PARAM_LOGIN = 'login'
PARAM_MODEL_PATH = 'model-path'
PARAM_UPDATE = 'update'
PARAM_OVERWRITE = 'overwrite'
PARAMS = {
	PARAM_XMIPP_DIRECTORY: {
		SHORT_VERSION: "-d",
		LONG_VERSION: "--directory",
		DESCRIPTION: f"Directory where the xmipp will be installed. Default is \"{DEFAULT_BUILD_DIR}\"."
	},
	PARAM_SHORT: {
		LONG_VERSION: "--short",
		DESCRIPTION: "If set, only version number is shown."
	},
	PARAM_JOBS: {
		SHORT_VERSION: "-j",
		LONG_VERSION: "--jobs",
		DESCRIPTION: "Number of jobs. Defaults to all available."
	},
	PARAM_BRANCH: {
		SHORT_VERSION: "-b",
		LONG_VERSION: "--branch",
		DESCRIPTION: "Branch for the source repositories."
	},
	PARAM_MODELS_DIRECTORY: {
		SHORT_VERSION: "-d",
		LONG_VERSION: "--directory",
		DESCRIPTION: f"Directory where the xmipp will be installed. Default is \"{DEFAULT_BUILD_DIR}\"."
	},
	PARAM_TEST_NAME: {
		SHORT_VERSION: "testName",
		DESCRIPTION: "Run certain test. If combined with --show, greps the test name from the test list."
	},
	PARAM_SHOW_TESTS: {
		LONG_VERSION: "--show",
		DESCRIPTION: "Shows the tests available and how to invoke those."
	},
	PARAM_GIT_COMMAND: {
		SHORT_VERSION: "command",
		DESCRIPTION: "Git command to run on all source repositories."
	},
	PARAM_LOGIN: {
		SHORT_VERSION: "login",
		DESCRIPTION: "Login (usr@server) for Nolan machine to upload the model with. Must have write permisions to such machine."
	},
	PARAM_MODEL_PATH: {
		SHORT_VERSION: "modelPath",
		DESCRIPTION: "Path to the model to upload to Nolan."
	},
	PARAM_UPDATE: {
		LONG_VERSION: "--update",
		DESCRIPTION: "Flag to update an existing model"
	},
	PARAM_OVERWRITE: {
		SHORT_VERSION: "-o",
		LONG_VERSION: "--overwrite",
		DESCRIPTION: "If set, current config file will be overwritten with a new one."
	}
}

# Arguments of each mode, sorted by group
MODE_ARGS = {
	MODE_VERSION: [PARAM_XMIPP_DIRECTORY, PARAM_SHORT],
	MODE_COMPILE_AND_INSTALL: [PARAM_JOBS, PARAM_BRANCH, PARAM_XMIPP_DIRECTORY],
	MODE_ALL: [PARAM_JOBS, PARAM_BRANCH, PARAM_XMIPP_DIRECTORY],
	MODE_BUILD_CONFIG: [],
	MODE_CONFIG: [],
	MODE_GET_MODELS: [PARAM_MODELS_DIRECTORY],
	MODE_CLEAN_BIN: [],
	MODE_CLEAN_ALL: [],
	MODE_TEST: [PARAM_TEST_NAME, PARAM_SHOW_TESTS],
	MODE_GIT: [PARAM_GIT_COMMAND],
	MODE_ADD_MODEL: [PARAM_LOGIN, PARAM_MODEL_PATH, PARAM_UPDATE]
}

# Examples for the help message of each mode
MODE_EXAMPLES = {
	MODE_VERSION: [
		f'./xmipp {MODE_VERSION}',
		f'./xmipp {MODE_VERSION} {PARAMS[PARAM_SHORT][LONG_VERSION]}',
		f'./xmipp {MODE_VERSION} {PARAMS[PARAM_XMIPP_DIRECTORY][SHORT_VERSION]} /path/to/my/build/dir'
	],
	MODE_COMPILE_AND_INSTALL: [
		f'./xmipp {MODE_COMPILE_AND_INSTALL}',
		f'./xmipp {MODE_COMPILE_AND_INSTALL} {PARAMS[PARAM_JOBS][SHORT_VERSION]} 20',
		f'./xmipp {MODE_COMPILE_AND_INSTALL} {PARAMS[PARAM_XMIPP_DIRECTORY][SHORT_VERSION]} /path/to/my/build/dir',
		f'./xmipp {MODE_COMPILE_AND_INSTALL} {PARAMS[PARAM_BRANCH][SHORT_VERSION]} devel',
		f'./xmipp {MODE_COMPILE_AND_INSTALL} {PARAMS[PARAM_JOBS][SHORT_VERSION]} '
		f'20 {PARAMS[PARAM_XMIPP_DIRECTORY][SHORT_VERSION]} /path/to/my/build/dir {PARAMS[PARAM_BRANCH][SHORT_VERSION]} devel'
	],
	MODE_ALL: [
		'./xmipp',
		f'./xmipp {MODE_ALL}',
		f'./xmipp {PARAMS[PARAM_JOBS][SHORT_VERSION]} 20',
		f'./xmipp {PARAMS[PARAM_XMIPP_DIRECTORY][SHORT_VERSION]} /path/to/my/build/dir',
		f'./xmipp {PARAMS[PARAM_BRANCH][SHORT_VERSION]} devel',
		f'./xmipp {MODE_ALL} {PARAMS[PARAM_JOBS][SHORT_VERSION]} 20 '
		f'{PARAMS[PARAM_XMIPP_DIRECTORY][SHORT_VERSION]} /path/to/my/build/dir {PARAMS[PARAM_BRANCH][SHORT_VERSION]} devel'
	],
	MODE_BUILD_CONFIG: [],
	MODE_CONFIG: [
		f'./xmipp {MODE_CONFIG} {PARAMS[PARAM_OVERWRITE][SHORT_VERSION]}'
	],
	MODE_GET_MODELS: [
		f'./xmipp {MODE_GET_MODELS}',
		f'./xmipp {MODE_GET_MODELS} {PARAMS[PARAM_MODELS_DIRECTORY][SHORT_VERSION]} /path/to/my/model/directory'
	],
	MODE_CLEAN_BIN: [],
	MODE_CLEAN_ALL: [],
	MODE_TEST: [
		f'./xmipp {MODE_TEST} xmipp_sample_test',
		f'./xmipp {MODE_TEST} {PARAMS[PARAM_SHORT][LONG_VERSION]}',
	],
	MODE_GIT: [
		f'./xmipp {MODE_GIT} pull',
		f'./xmipp {MODE_GIT} checkout devel'
	],
	MODE_ADD_MODEL: [
		f'./xmipp {MODE_ADD_MODEL} myuser@127.0.0.1 /home/myuser/mymodel'
	]
}
