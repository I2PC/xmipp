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
	GLOB 
	PROGRAM_DIRECTORIES 
	RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*
)

foreach(PROGRAM_DIRECTORY ${PROGRAM_DIRECTORIES})
	if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${PROGRAM_DIRECTORY})
		set(PROGRAM_NAME xmipp_${PROGRAM_DIRECTORY})

		set(IS_MPI_PROGRAM ${PROGRAM_NAME} MATCHES "mpi") 
		set(IS_CUDA_PROGRAM ${PROGRAM_NAME} MATCHES "cuda")
		set(IS_CUDA11_PROGRAM ${PROGRAM_NAME} MATCHES "cuda11") 
		if(${IS_CUDA11_PROGRAM})
			string(REPLACE "cuda11" "cuda" PROGRAM_NAME ${PROGRAM_NAME})
		endif()

		if(${IS_CUDA_PROGRAM})
			if(${XMIPP_USE_CUDA})
				if(${CUDAToolkit_VERSION} VERSION_LESS 11 AND ${IS_CUDA11_PROGRAM})
					message("Skipping ${PROGRAM_NAME} as CUDA 11 is not supported")
					continue()
				endif()
			else()
				message("Skipping ${PROGRAM_NAME} as CUDA is not enabled")
				continue()
			endif()
		endif()

		if(${IS_MPI_PROGRAM} AND NOT ${XMIPP_USE_MPI})
			message("Skipping ${PROGRAM_NAME} as MPI is not enabled")
			continue()
		endif()

		file(GLOB_RECURSE PROGRAM_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${PROGRAM_DIRECTORY}/*.cpp)	

		if (NOT PROGRAM_SOURCES)
			message(WARNING "${PROGRAM_DIRECTORY} directory is empty. Skipping program.")
			continue()
		endif()

		add_executable(
			${PROGRAM_NAME}
			${PROGRAM_SOURCES}
		)
		target_link_libraries(
			${PROGRAM_NAME}
			Xmipp
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

	endif()
endforeach()
