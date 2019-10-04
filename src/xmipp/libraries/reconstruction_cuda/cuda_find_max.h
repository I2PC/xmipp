/***************************************************************************
 *
 * Authors:    David Strelak (davidstrelak@gmail.com)
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

#ifndef LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_GPU_FINDMAX_H_
#define LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_GPU_FINDMAX_H_

#include <cassert>
#include "reconstruction_cuda/gpu.h"
#include "data/dimensions.h"

template <typename T, bool dataOnGPU>
void sFindMax1D(const GPU &gpu,
        const Dimensions &dims,
        const T * __restrict__ data,
        T * __restrict__ positions,
        T * __restrict__ values);

template <typename T, bool dataOnGPU>
void sFindMax2DNear(const GPU &gpu,
        const Dimensions &dims,
        const T * __restrict__ data,
        T * __restrict__ positions,
        T * __restrict__ values,
        size_t maxDist);

#endif /* LIBRARIES_RECONSTRUCTION_ADAPT_CUDA_GPU_FINDMAX_H_ */
