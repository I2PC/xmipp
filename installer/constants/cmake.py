# ***************************************************************************
# * Authors:		Martín Salinas (martin.salinas@cnb.csic.es)
# * 						Oier Lauzirika Zarrabeitia (martin.salinas@cnb.csic.es)
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
# ***************************************************************************/

from .config import CUDA, MPI, MATLAB, LINK_SCIPION, GCC_HOME, GXX_HOME, CUDA_COMPILER

# CMake cache file variables to look for
XMIPP_USE_CUDA=CUDA
XMIPP_USE_MPI=MPI
XMIPP_USE_MATLAB=MATLAB
XMIPP_LINK_TO_SCIPION=LINK_SCIPION
CMAKE_BUILD_TYPE='CMAKE_BUILD_TYPE'
CMAKE_C_COMPILER=GCC_HOME
CMAKE_CXX_COMPILER=GXX_HOME
CMAKE_CUDA_COMPILER=CUDA_COMPILER
