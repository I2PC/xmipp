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

function(fetch_cuFFTAdvisor)
	set(SOURCE_DIR ${CMAKE_BINARY_DIR}/cuFFTAdvisor)

	ExternalProject_Add(
		build_cuFFTAdvisor
		GIT_REPOSITORY https://github.com/HiPerCoRe/cuFFTAdvisor.git
		GIT_TAG master
		SOURCE_DIR ${SOURCE_DIR}
		UPDATE_COMMAND ""
		CONFIGURE_COMMAND ""
		BUILD_COMMAND make -C <SOURCE_DIR> libcuFFTAdvisor.so
		INSTALL_COMMAND ""
		BUILD_IN_SOURCE TRUE
	)

	add_library(cuFFTAdvisor SHARED IMPORTED)
	set_target_properties(cuFFTAdvisor PROPERTIES IMPORTED_LOCATION ${SOURCE_DIR}/build/libcuFFTAdvisor.so)
	target_include_directories(cuFFTAdvisor INTERFACE ${SOURCE_DIR})
	add_dependencies(cuFFTAdvisor build_cuFFTAdvisor)
	install(
		FILES ${SOURCE_DIR}/build/libcuFFTAdvisor.so
 		DESTINATION ${CMAKE_INSTALL_LIBDIR}
	)
endfunction()
