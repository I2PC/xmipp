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

"""
Submodule containing all version info required for Xmipp's installation process.
"""

from .main import XMIPP, XMIPP_CORE, XMIPP_VIZ, XMIPP_PLUGIN

# Xmipp's current versions
__LATEST_RELEASE_NUMBER = '3.XX.YY.0'
__LATEST_RELEASE_NAME = 'v3.XX.YY-TBD'
#####################################
DEVEL_BRANCHNAME = 'devel'					
MASTER_BRANCHNAME = 'master'				
																		
VERSION_KEY = 'version'							
VERNAME_KEY = 'vername'							
XMIPP_VERSIONS = {									
	XMIPP: {													
		VERSION_KEY: __LATEST_RELEASE_NUMBER,				
		VERNAME_KEY: __LATEST_RELEASE_NAME  
	},																
	XMIPP_CORE: {											
		VERSION_KEY: __LATEST_RELEASE_NUMBER,				
		VERNAME_KEY: __LATEST_RELEASE_NAME  
	},																
	XMIPP_VIZ: {											
		VERSION_KEY: __LATEST_RELEASE_NUMBER,				
		VERNAME_KEY: __LATEST_RELEASE_NAME  
	},																
	XMIPP_PLUGIN: {										
		VERSION_KEY: __LATEST_RELEASE_NUMBER,				
		VERNAME_KEY: __LATEST_RELEASE_NAME  
	}																	
}																		
#####################################

# Supported gcc versions
vGCC = [
	'12.3', '12.2', '12.1',
	'11.3', '11.2', '11.1', '11',
	'10.5', '10.4', '10.3', '10.2', '10.1', '10',
	'9.4', '9.3', '9.2', '9.1', '9',
	'8.5', '8.4'
]

# Version requirements
CMAKE_MINIMUM = '3.16' # 2019

# CUDA-gcc compatibility table
# https://gist.github.com/ax3l/9489132
CUDA_GCC_COMPATIBILITY = {
	'10.1-10.2': vGCC[vGCC.index('8.5'):],
	'11.0-11.0': vGCC[vGCC.index('9.4'):],
	'11.1-11.3': vGCC[vGCC.index('10.5'):],
	'11.4-11.8': vGCC[vGCC.index('11.3'):],
	'12.0-12.3': vGCC[vGCC.index('12.3'):]
}

# Other variables
UNKNOWN_VALUE = 'Unknown'
