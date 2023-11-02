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

# Version variables
#	----K-E-E-P----U-P-D-A-T-E-D---- #
#####################################
XMIPP_VERSION = '3.23.07.0'        	#
XMIPP_VERNAME = 'devel'							#
RELEASE_DATE = '14/07/2023'        	#
CMAKE_VERSION_REQUIRED = '3.16'			#
#####################################

# Mode list
MODES = {
	'cleanAll': 'Removes all compiled binaries and sources, leaves the repository as if freshly cloned (without pulling).',
	'cleanBin': 'Removes all compiled binaries.',
	'cleanDeprecated': 'Removes all deprecated binaries.',
	'version': 'Prints '
} 

# Other variables
DEFAULT_JOBS = 8