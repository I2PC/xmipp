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

# Define the project
project(
  xmipp
	VERSION 3.23.11
	LANGUAGES C CXX
)

include(CheckLanguage)
include(CTest)

# Set compile options
option(XMIPP_USE_CUDA "Compile CUDA programs" ON)
option(XMIPP_USE_MPI "Compile MPI programs" ON)
option(XMIPP_USE_MATLAB "Compile MATLAB binding" ON)

# Set C++ 17
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Find CUDA and MPI
if(${XMIPP_USE_CUDA})
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --x cu")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fPIC")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
    
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
  else()
    set(XMIPP_USE_CUDA OFF)
    message("CUDA was requested but nvcc was not found. Compiling without CUDA")
  endif()
endif()

if(${XMIPP_USE_MPI})
	find_package(MPI COMPONENTS C CXX)
  if (NOT MPI_FOUND)
    set(XMIPP_USE_MPI OFF)
    message("MPI was requested but MPI was not found. Compiling without MPI")
  endif()
endif()

if(${XMIPP_USE_MATLAB})
	find_package(Matlab)

  if (NOT Matlab_FOUND)
    set(XMIPP_USE_MATLAB OFF)
    message("Matlab was requested but Matlab was not found. Compiling without Matlab")
  endif()
endif()

add_subdirectory(src)
