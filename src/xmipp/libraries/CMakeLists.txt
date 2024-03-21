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

find_package(MPI COMPONENTS C CXX)
find_package(CUDAToolkit 10.2)

# Find source files
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
file(
	GLOB_RECURSE 
	CUDA_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/reconstruction_cuda/*.cu
		${CMAKE_CURRENT_SOURCE_DIR}/reconstruction_cuda/*.cpp
)
file(
	GLOB_RECURSE 
	PARALLEL_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/parallel/*.cpp
)
file(
	GLOB_RECURSE 
	PARALLEL_CUDA_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/parallel_adapt_cuda/*.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/reconstruction_adapt_cuda/*.cpp
)


add_library(xmippLib SHARED ${SOURCES})
target_include_directories(
	xmippLib
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
#target_precompile_headers(
#	xmippLib
#	PRIVATE
#		${HEADERS}
#)
target_link_libraries(
	xmippLib
	PUBLIC
		xmippCore
		xmippExternal
		cifpp
		ctpl
)

# Register CUDA sources
if(CUDAToolkit_FOUND)
	target_sources(
		xmippLib
		PRIVATE ${CUDA_SOURCES}
	)
	set_target_properties( 
		xmippLib
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
	)
	target_link_libraries(
		xmippLib
		PUBLIC
			CUDA::cudart
			CUDA::cufft
	)
endif()

# Register MPI sources
if(MPI_FOUND)
	target_sources(
		xmippLib
		PRIVATE ${PARALLEL_SOURCES}
	)
	target_link_libraries(
		xmippLib
		PUBLIC
			MPI::MPI_C
	)
endif()

# Register combined MPI and CUDA sources
if(MPI_FOUND AND CUDAToolkit_FOUND)
	target_sources(
		xmippLib
		PRIVATE ${PARALLEL_CUDA_SOURCES}
	)
endif()
