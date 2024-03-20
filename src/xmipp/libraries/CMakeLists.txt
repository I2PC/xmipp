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

cmake_minimum_required(VERSION 3.16)

file(
	GLOB_RECURSE 
	SOURCES 
		${CMAKE_CURRENT_SOURCE_DIR}/classification/*.cpp 
		${CMAKE_CURRENT_SOURCE_DIR}/data/*.cpp 
		${CMAKE_CURRENT_SOURCE_DIR}/dimred/*.cpp 
		${CMAKE_CURRENT_SOURCE_DIR}/interface/*.cpp 
		${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/*.cpp 
		${CMAKE_CURRENT_SOURCE_DIR}/tomo/*.cpp 
)
file(
	GLOB_RECURSE 
	HEADERS 
		${CMAKE_CURRENT_SOURCE_DIR}/classification/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/data/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/dimred/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/interface/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/reconstruction/*.h
		${CMAKE_CURRENT_SOURCE_DIR}/tomo/*.h
)

add_library(xmippLib SHARED ${SOURCES})
target_include_directories(
	xmippLib
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_precompile_headers(
	xmippLib
	PRIVATE
		${HEADERS}
)
target_link_libraries(
	xmippLib
	PUBLIC
		xmippCore
)
