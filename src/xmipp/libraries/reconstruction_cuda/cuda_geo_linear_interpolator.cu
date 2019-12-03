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
    // assign element to thread, we have at thead for each column (+ some extra)
    // single block processes single signal
    unsigned inX = blockIdx.x * blockDim.x + threadIdx.x;
    if (inX >= xDim) return;
    int x = (int)inX - (xDim / 2);
    unsigned signal = blockIdx.y;

    const T *src = in + (signal * xDim * yDim);
    T *dest = out + (signal * xDim * yDim);
    float *t = matrices + (signal * 9);
    for (int inY = 0; inY < yDim; ++inY) {
        int y = inY - (yDim / 2);

        T outX = x * t[0] + y * t[1] + t[2] + (xDim / 2);
        T outY = x * t[3] + y * t[4] + t[5] + (yDim / 2);

        T val = 0;
        if ((outX >= 0) && (outX < xDim)
            && (outY >= 0) && (outY < yDim)) {
            val = biLerp(src, xDim, yDim, outX, outY);
        }

        unsigned offset = inY * xDim + inX;
        dest[offset] = val;
    }
}
