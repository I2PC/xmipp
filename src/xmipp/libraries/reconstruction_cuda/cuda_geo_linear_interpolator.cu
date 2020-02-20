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
#include "cuda_basic_math.h"

template<typename T>
__global__
void interpolateKernel(const T * __restrict__ in, T * __restrict__ out, float * __restrict matrices,
        int xDim, int yDim) {
    // assign pixel to thread
    unsigned inX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned inY = blockIdx.y * blockDim.y + threadIdx.y;
    if (inX >= xDim) return;
    if (inY >= yDim) return;
    T x = (int)inX - (xDim / 2);
    T y = (int)inY - (yDim / 2);
    unsigned signal = blockIdx.z;

    const T *src = in + (signal * xDim * yDim);
    T *dest = out + (signal * xDim * yDim);
    const float *t = matrices + (signal * 9);

    T outX = x * t[0] + y * t[1] + t[2] + (xDim / (T)2);
    T outY = x * t[3] + y * t[4] + t[5] + (yDim / (T)2);

    T val = 0;
    if ((outX >= 0) && (outX < ((T)xDim - 1))
        && (outY >= 0) && (outY < ((T)yDim - 1))) {
        val = biLerp(src, xDim, yDim, outX, outY);
    }

    unsigned offset = inY * xDim + inX;
    dest[offset] = val;
}

template<typename T>
__global__
void sumKernel(const T * __restrict__ in, T * __restrict__ out,
        const T * __restrict__ weights,
        unsigned xDim, unsigned yDim, unsigned nDim, T normFactor) {
    // assign pixel to thread
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= xDim) return;
    if (y >= yDim) return;

    unsigned pos = (y * xDim) + x;
    double v = in[pos] * weights[0]; // double on purpose, to improve the precision
    for (unsigned n = 1; n < nDim; ++n) {
        unsigned offset = n * xDim * yDim;
        v += in[offset + pos] * weights[n];
    }
    out[pos] = v / normFactor;
}
