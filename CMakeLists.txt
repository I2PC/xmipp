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

# Import functions
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(fetch_cifpp)
include(fetch_ctpl)
include(fetch_cuFFTAdvisor)
include(fetch_googletest)
include(fetch_libsvm)
include(write_bashrc)
include(link_to_scipion)

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
option(XMIPP_LINK_TO_SCIPION "Link to scipion3" ON)
option(XMIPP_SAVE_VERSIONS "Save versions.txt with dependency versions" ON)

# Set C++ 17
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Find Python
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
string(APPEND XMIPP_VERSIONS "Python3=${Python3_VERSION}\n")

# Find CUDA
if(${XMIPP_USE_CUDA})
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit 10.2 REQUIRED)
	  string(APPEND XMIPP_VERSIONS "CUDA=${CUDAToolkit_VERSION}\n")

    set(CMAKE_CUDA_STANDARD 14)

    string(APPEND CMAKE_CUDA_FLAGS " --x cu")
    string(APPEND CMAKE_CUDA_FLAGS " -D_FORCE_INLINES")
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler")
    string(APPEND CMAKE_CUDA_FLAGS " -fPIC")
    string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda")

    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
      set(CMAKE_CUDA_ARCHITECTURES 60 75 86)
    else()
      string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=sm_60")
      string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_75,code=sm_75")
      string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_86,code=sm_86")
    endif()
    
  else()
    set(XMIPP_USE_CUDA OFF)
    message("CUDA was requested but nvcc was not found. Compiling without CUDA")
  endif()
endif()

# Find MPI
if(${XMIPP_USE_MPI})
	find_package(MPI COMPONENTS CXX)
  if (NOT MPI_FOUND)
    set(XMIPP_USE_MPI OFF)
    message("MPI was requested but MPI was not found. Compiling without MPI")
  else()
	  string(APPEND XMIPP_VERSIONS "MPI=${MPI_CXX_VERSION}\n")
  endif()
endif()

# Find MATLAB
if(${XMIPP_USE_MATLAB})
	find_package(Matlab)

  if (NOT Matlab_FOUND)
    set(XMIPP_USE_MATLAB OFF)
    message("Matlab was requested but Matlab was not found. Compiling without Matlab")
  endif()
endif()

# Fetch dependencies
fetch_cifpp()
fetch_ctpl()
fetch_googletest()
fetch_libsvm()
if(XMIPP_USE_CUDA)
  fetch_cuFFTAdvisor()
endif()

# Import subdirectories
add_subdirectory(src/xmippCore)
add_subdirectory(src/xmipp)
add_subdirectory(src/xmippViz)

# Save a version file for debug
if(XMIPP_SAVE_VERSIONS)
  string(APPEND XMIPP_VERSIONS "CMake=${CMAKE_VERSION}\n")
  string(APPEND XMIPP_VERSIONS "CC=${CMAKE_C_COMPILER_ID}-${CMAKE_C_COMPILER_VERSION}\n")
  string(APPEND XMIPP_VERSIONS "CXX=${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}\n")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/versions.txt" "${XMIPP_VERSIONS}")
endif()

# Configure bashrc file for installation
write_bashrc(${CMAKE_CURRENT_BINARY_DIR}/xmipp.bashrc ${CMAKE_INSTALL_PREFIX})
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/xmipp.bashrc
  DESTINATION ./
)

# Link to scipion
if(XMIPP_LINK_TO_SCIPION)
  if (NOT SCIPION_SOFTWARE)
    set(SCIPION_SOFTWARE "$ENV{SCIPION_SOFTWARE}")
  endif()

  if (SCIPION_SOFTWARE)
    message("Scipion software directory found at ${SCIPION_SOFTWARE}")
    link_to_scipion(${CMAKE_INSTALL_PREFIX} ${SCIPION_SOFTWARE})
  else()
    message("Linking to scipion was requested, but SCIPION_SOFTWARE is not set")
  endif()
endif()
