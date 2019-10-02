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

#ifndef LIBRARIES_RECONSTRUCTION_CUDA_CUDA_FIND_MAX_CU_
#define LIBRARIES_RECONSTRUCTION_CUDA_CUDA_FIND_MAX_CU_

#include <float.h> // FLT_MAX
#include <type_traits>

template<typename T, typename T2>
 __device__
void update(T2 &orig, const T * __restrict data, unsigned index) {
    T tmp = data[index];
    if (tmp > orig.x) {
        orig.x = tmp;
        orig.y = (T)index;
    }
}

template<typename T>
__device__
T update(T &orig, T &cand) {
    if (cand.x > orig.x) {
        orig.x = cand.x;
        orig.y = cand.y;
    }
    return orig;
}

template <typename T, typename T2, unsigned blockSize>
__device__
void findMax(
        const T * __restrict__ in,
        T * __restrict__ outPos,
        T * __restrict__ outVal,
        unsigned samples)
{
    // one block process one signal
    // map each thread to some sample of the signal
    unsigned int tid = threadIdx.x;
    unsigned int signal = blockIdx.x;

    T2 ldata;
    ldata.x = -FLT_MAX;
    ldata.y = -1;

    // load data from global memory
    const T *data = in + (signal * samples);
    for(unsigned i = tid; i < samples; i += blockSize) {
        update(ldata, data, i);
    }
    // we have read all data, one of the thread knows the result
    extern __shared__ char smem[];
    T2 *sdata = reinterpret_cast<T2*>(smem);
    sdata[tid] = ldata;
    __syncthreads(); // wait till all threads store their data
    // reduce
#pragma unroll
    for (unsigned counter = blockSize / 2; counter >= 32; counter /= 2) {
        if (tid < counter) {
            sdata[tid] = update(ldata, sdata[tid + counter]);
        }
        __syncthreads();
    }
    // manually unwrap last warp for better performance
    // many of these blocks will be optimized out by compiler based on template
    if ((blockSize >= 32) && (tid < 16)) {
        sdata[tid] = update(ldata, sdata[tid + 16]);
    }
    __syncthreads();
    if ((blockSize >= 16) && (tid < 8)) {
        sdata[tid] = update(ldata, sdata[tid + 8]);
    }
    __syncthreads();
    if ((blockSize >= 8) && (tid < 4)) {
        sdata[tid] = update(ldata, sdata[tid + 4]);
    }
    __syncthreads();
    if ((blockSize >= 4) && (tid < 2)) {
        sdata[tid] = update(ldata, sdata[tid + 2]);
    }
    __syncthreads();
    if ((blockSize >= 2) && (tid < 1)) {
        sdata[tid] = update(ldata, sdata[tid + 1]);
    }
    __syncthreads();

    // last thread now holds the result
    if (tid == 0) {
        outVal[signal] = ldata.x;
        outPos[signal] = ldata.y;
    }
}

template <typename T, unsigned blockSize>
__global__
void findMax(const T * __restrict__ in,
        T * __restrict__ outPos,
        T * __restrict__ outVal,
        unsigned samples)
{
    if (std::is_same<T, float> ::value) {
        findMax<float, float2, blockSize> (
                (float*)in,
                (float*)outPos,
                (float*)outVal,
                samples);
    } else if (std::is_same<T, double> ::value) {
        findMax<double, double2, blockSize>(
                (double*)in,
                (double*)outPos,
                (double*)outVal,
                samples);
    }
}

#endif /* LIBRARIES_RECONSTRUCTION_CUDA_CUDA_FIND_MAX_CU_ */
