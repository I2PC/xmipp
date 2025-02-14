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
Submodule containing all constants needed for handling errors during Xmipp's installation.
"""

from .main import CMAKE_INSTALL_DOCS_URL, LOG_FILE, CONFIG_FILE

# Error codes
INTERRUPTED_ERROR = -1
OK = 0
UNKOW_ERROR = 1
SOURCE_CLONE_ERROR = 2
CMAKE_ERROR = 3
CMAKE_CONFIGURE_ERROR = 4
CMAKE_COMPILE_ERROR = 5
CMAKE_INSTALL_ERROR = 6
IO_ERROR = 7
ENVIROMENT_ERROR = 8

# Error messages
__CHECK_LOG_MESSAGE = f'Check the inside file \'{LOG_FILE}\'.'
ERROR_CODE = {
	INTERRUPTED_ERROR: ['Process was interrupted by the user.', ''],
	UNKOW_ERROR: ['', ''],
	SOURCE_CLONE_ERROR: ['Error cloning xmipp repository with git.', 'Please review the internet connection and the git package.'],
	CMAKE_ERROR: ['There was an error with CMake.', f'Please install it by following the instructions at {CMAKE_INSTALL_DOCS_URL}'],
	CMAKE_CONFIGURE_ERROR: ['Error configuring with CMake.', __CHECK_LOG_MESSAGE],
	CMAKE_COMPILE_ERROR: ['Error compiling with CMake.', __CHECK_LOG_MESSAGE],
	CMAKE_INSTALL_ERROR: ['Error installing with CMake.', __CHECK_LOG_MESSAGE],
	IO_ERROR: ['Input/output error.', 'This error can be caused by the installer not being able to read/write/create/delete a file. Check your permissions on this directory.'],
ENVIROMENT_ERROR: ['XMIPP_SRC is not in the enviroment. To run the tests you need to run: source dist/xmipp.bashrc', '']

}
