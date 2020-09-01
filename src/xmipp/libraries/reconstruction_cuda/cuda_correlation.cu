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

#ifndef LIBRARIES_RECONSTRUCTION_CUDA_CUDA_CORRELATION_CU_
#define LIBRARIES_RECONSTRUCTION_CUDA_CUDA_CORRELATION_CU_

#include "cuda_compatibility.h"

/**
 * Function computes sum and sum squared for each 2D signal
 * U should be vector-2 version of T (i.e. U=float2 for T=float)
 * @param in input 2D signals
 * @param xDim number of columns
 * @param yDim number of rows
 * @param nDim number of signals
 * @param res for each signal, two values are stored: [sum, sum squared].
 *        Must be empty (zero-initialized)
 */
template<typename T, typename U>
__global__
void computeSumSumSqr2D(const T * __restrict__ in,
    unsigned xDim, unsigned yDim, unsigned nDim,
    U * __restrict__ res) {
    // input is 2D signal: map thread to column
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned maxThread = xDim * nDim;
    if (idx >= maxThread) return;

    unsigned x = idx % xDim; // sample position == column
    unsigned n = idx / xDim; // signal index

    bool isSameSignalInWarp = false;
    unsigned warpId = idx / warpSize; // warp has 32 threads
    unsigned idOfFirstThreadInWarp = warpId * warpSize;
    unsigned idOfLastThreadInWarp = idOfFirstThreadInWarp + warpSize - 1;
    unsigned nForFirst = idOfFirstThreadInWarp / xDim;
    unsigned nForLast = idOfLastThreadInWarp / xDim;
    if ((idOfLastThreadInWarp < maxThread) && (nForFirst == nForLast)) {
        isSameSignalInWarp = true;
    }

    // compute statistics
    T sum = 0;
    T sum2 = 0;
    unsigned signalOffset = (n * xDim * yDim);
    // each tread will sum 'its' column
    for (unsigned y = 0; y < yDim; ++y) {
        int offset =  signalOffset + (y * xDim) + x;
        T val = in[offset];
        sum += val;
        sum2 += val * val;
    }
    if (isSameSignalInWarp) {
        // all threads in warp (columns) are processing the same signal
        // perform additional (column-wise) sum
        // intrawarp sum
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down(sum, offset);
            sum2 += __shfl_down(sum2, offset);
        }
        if (idx != idOfFirstThreadInWarp) {
            // only first thread in the warp will write to the output
            return;
        }
    }
    // update the result
    atomicAdd(&res[n].x, sum);
    atomicAdd(&res[n].y, sum2);
}


/**
 * Function computes correlation index (inner product, dot product)
 * of one signal and many others, and at the same time provides statistical info
 * about 'other signals'.
 * U should be vector-3 version of T (i.e. U=float3 for T=float)
 * @param ref first signal in spatial domain, not normalized
 * @param others other signals in spatial domain, not normalized
 * @param xDim of the signal
 * @param yDim of the signal
 * @param nDim no of other signals
 * @param res for each signal, three values are stored: [correlation index, sum of the signal, sum squared of the signal].
 *        Must be empty (zero-initialized)
 */
template<typename T, typename U>
__global__
void computeCorrIndexStat2DOneToN(
        const T* __restrict__ ref,
        T* __restrict__ others,
        int xDim, int yDim, int nDim,
        U* __restrict__ res) {
    // input is 2D signal: map thread to column
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned maxThread = xDim * nDim;
    if (idx >= maxThread) return;

    unsigned x = idx % xDim; // sample position == column
    unsigned n = idx / xDim; // signal index

    bool isSameSignalInWarp = false;
    unsigned warpId = idx / warpSize; // warp has 32 threads
    unsigned idOfFirstThreadInWarp = warpId * warpSize;
    unsigned idOfLastThreadInWarp = idOfFirstThreadInWarp + warpSize - 1;
    unsigned nForFirst = idOfFirstThreadInWarp / xDim;
    unsigned nForLast = idOfLastThreadInWarp / xDim;
    if ((idOfLastThreadInWarp < maxThread) && (nForFirst == nForLast)) {
        isSameSignalInWarp = true;
    }

    // compute statistics
    T corr = 0;
    T sum = 0;
    T sum2 = 0;
    unsigned signalOffset = (n * xDim * yDim);
    // each tread will process 'its' column
    for (unsigned y = 0; y < yDim; ++y) {
        int offsetR = (y * xDim) + x;
        int offsetO = signalOffset + (y * xDim) + x;
        T valR = ref[offsetR];
        T valO = others[offsetO];
        corr += valR * valO;
        sum += valO;
        sum2 += valO * valO;
    }
    if (isSameSignalInWarp) {
        // all threads in warp (columns) are processing the same signal
        // perform additional (column-wise) sum
        // intrawarp sum
        for (int offset = 16; offset > 0; offset /= 2) {
            corr += __shfl_down(corr, offset);
            sum += __shfl_down(sum, offset);
            sum2 += __shfl_down(sum2, offset);
        }
        if (idx != idOfFirstThreadInWarp) {
            // only first thread in the warp will write to the output
            return;
        }
    }
    // update the result
    atomicAdd(&res[n].x, corr);
    atomicAdd(&res[n].y, sum);
    atomicAdd(&res[n].z, sum2);
}


#endif /* LIBRARIES_RECONSTRUCTION_CUDA_CUDA_CORRELATION_CU_ */
