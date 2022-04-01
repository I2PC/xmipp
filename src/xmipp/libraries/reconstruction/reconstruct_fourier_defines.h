/***************************************************************************
 *
 * Authors:     David Strelak (davidstrelak@gmail.com)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#ifndef XMIPP_LIBRARIES_DATA_RECONSTRUCT_FOURIER_DEFINES_H_
#define XMIPP_LIBRARIES_DATA_RECONSTRUCT_FOURIER_DEFINES_H_

typedef float MATRIX[3][3];
constexpr int BLOB_TABLE_SIZE_SQRT= 10000;
constexpr float ACCURACY= 0.001;

#define PASCAL

#ifdef KEPLER
// GPU specific
constexpr int  BLOCK_DIM=  16;
constexpr int  SHARED_BLOB_TABLE=  0;
constexpr int  SHARED_IMG=  0;
constexpr int  PRECOMPUTE_BLOB_VAL=  0;
constexpr int  TILE=  8;
constexpr int  GRID_DIM_Z=  1;
#endif

#ifdef MAXWELL
// GPU specific
constexpr int  BLOCK_DIM=  8;
constexpr int  SHARED_BLOB_TABLE=  0;
constexpr int  SHARED_IMG=  0;
constexpr int  PRECOMPUTE_BLOB_VAL=  1;
constexpr int  TILE=  4;
constexpr int  GRID_DIM_Z=  8;
#endif

#ifdef PASCAL
// GPU specific
constexpr int BLOCK_DIM=  16;
constexpr int SHARED_BLOB_TABLE=  1;
constexpr int SHARED_IMG=  0;
constexpr int PRECOMPUTE_BLOB_VAL=  1;
constexpr int TILE=  2;
constexpr int GRID_DIM_Z=  1;
#endif

#endif /* XMIPP_LIBRARIES_DATA_RECONSTRUCT_FOURIER_DEFINES_H_ */
