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

# Set compile options
option(XMIPP_USE_CUDA "Compile CUDA programs" ON)
option(XMIPP_USE_MPI "Compile MPI programs" ON)

# Set C++ 17
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Find CUDA and MPI
if(${XMIPP_USE_CUDA})
  enable_language(CUDA)
  find_package(CUDAToolkit 10.2 REQUIRED)

  set(CMAKE_CUDA_FLAGS "--x cu -D_FORCE_INLINES -Xcompiler -fPIC -std=c++14 --expt-extended-lambda --expt-relaxed-constexpr")
  
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
endif()

if(${XMIPP_USE_MPI})
	find_package(MPI COMPONENTS C CXX REQUIRED)
endif()

add_subdirectory(src)
