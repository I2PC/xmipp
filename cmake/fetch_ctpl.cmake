#***************************************************************************
# Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# 02111-1307  USA
#
#  All comments concerning this program package may be sent to the
#  e-mail address 'xmipp@cnb.csic.es'
# ***************************************************************************

include(ExternalProject)

function(fetch_ctpl)
	set(SOURCE_DIR ${CMAKE_BINARY_DIR}/ctpl)
	ExternalProject_Add(
		download_ctpl
		GIT_REPOSITORY https://github.com/vit-vit/CTPL.git
		GIT_TAG master
		SOURCE_DIR ${SOURCE_DIR}
		UPDATE_COMMAND ""
		CONFIGURE_COMMAND ""
		BUILD_COMMAND ""
		INSTALL_COMMAND ""
		BUILD_IN_SOURCE TRUE
	)

	add_library(ctpl INTERFACE IMPORTED)
	target_include_directories(ctpl INTERFACE ${SOURCE_DIR})
	add_dependencies(ctpl download_ctpl)
endfunction()
