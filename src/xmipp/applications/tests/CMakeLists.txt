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
	TEST_SOURCES 
		${CMAKE_CURRENT_SOURCE_DIR}/function_tests/*.cpp 
)


foreach(TEST_SOURCE ${TEST_SOURCES})
	get_filename_component(PROGRAM_NAME ${TEST_SOURCE} NAME_WE)
	set(PROGRAM_NAME "xmipp_${PROGRAM_NAME}")

	set(IS_CUDA_PROGRAM ${PROGRAM_NAME} MATCHES "cuda") 
	set(IS_MPI_PROGRAM ${PROGRAM_NAME} MATCHES "mpi") 
	
	if(${IS_CUDA_PROGRAM} AND NOT ${XMIPP_USE_CUDA})
		message("Skipping ${PROGRAM_NAME}")
		continue()
	endif()

	if(${IS_MPI_PROGRAM} AND NOT ${XMIPP_USE_MPI})
		message("Skipping ${PROGRAM_NAME}")
		continue()
	endif()

	add_executable(
		${PROGRAM_NAME}
		${TEST_SOURCE}
	)
	target_link_libraries(
		${PROGRAM_NAME}
		Xmipp
		GTest::gtest_main
	)

	if(${IS_MPI_PROGRAM})
		target_link_libraries(
			${PROGRAM_NAME}
			XmippParallel
		)
	endif()
	
	if(${IS_CUDA_PROGRAM})
		target_link_libraries(
			${PROGRAM_NAME}
			XmippInterfaceCuda
		)
	endif()
	
	if(${IS_MPI_PROGRAM} AND ${IS_CUDA_PROGRAM})
		target_link_libraries(
			${PROGRAM_NAME}
			XmippParallelCuda
		)
	endif()

	install(
		TARGETS ${PROGRAM_NAME}
		RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
	)
endforeach()
