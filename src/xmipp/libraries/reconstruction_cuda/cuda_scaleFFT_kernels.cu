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

#pragma once

#include "reconstruction_cuda/cuda_basic_math.h"

/**
 * Kernel performing scaling of the 2D FFT images, with possible normalization,
 * filtering and centering
 * @param in input data
 * @param out output data
 * @param noOfImages images to process
 * @param inX input X dim
 * @param inY input Y dim
 * @param outX output X dim (must be less or equal to input)
 * @param outY output Y dim (must be less or equal to input)
 * @param filter to apply. Only if 'applyFilter' is true
 * @param normFactor normalization factor to use (x *= normFactor).
 * Only if 'normalize' is true
 */
template<typename T, typename U, bool applyFilter, bool normalize, bool center, bool ignoreLowFreq>
__global__
void scaleFFT2DKernel(const T* __restrict__ in, T* __restrict__ out,
        int noOfImages, size_t inX, size_t inY, size_t outX, size_t outY,
    const U* __restrict__ filter, U normFactor) {
    // assign pixel to thread
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;

    if (idx >= outX || idy >= outY ) return;
    size_t fIndex = idy*outX + idx; // index within single image
    U filterCoef = filter[fIndex];
    U centerCoef = 1-2*((idx+idy)&1); // center FT, input must be even
    int yhalf = outY/2;

    size_t origY = (idy <= yhalf) ? idy : (inY - (outY-idy)); // take top N/2+1 and bottom N/2 lines
    for (int n = 0; n < noOfImages; n++) {
        size_t iIndex = n*inX*inY + origY*inX + idx; // index within consecutive images
        size_t oIndex = n*outX*outY + fIndex; // index within consecutive images
        out[oIndex] = in[iIndex];
        if (applyFilter) {
            out[oIndex] *= filterCoef;
        }
        if (ignoreLowFreq) {
            if (0 == idx || 0 == idy) {
                out[oIndex] = {0, 0}; // ignore low frequency, this should increase precision a bit
            }
        }
        if (normalize) {
            out[oIndex] *= normFactor;
        }
        if (center) {
            out[oIndex] *= centerCoef;
        }
    }
}