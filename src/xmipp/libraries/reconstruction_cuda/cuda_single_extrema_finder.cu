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

#include <type_traits>

template<typename T, typename T2, typename C>
 __device__
void update(const C &comp, T2 &orig, const T * __restrict data, unsigned index) {
    T tmp = data[index];
    if (comp(tmp, orig.x)) {
        orig.x = tmp;
        orig.y = (T)index;
    }
}

template<typename T, typename C>
__device__
T update(const C &comp, T &orig, T &cand) {
    if (comp(cand.x, orig.x)) {
        orig.x = cand.x;
        orig.y = cand.y;
    }
    return orig;
}

template<typename T, unsigned blockSize, typename C>
__device__
void findUniversalInSharedMem(const C &comp, T &ldata) {
    unsigned int tid = threadIdx.x;
    // we have read all data, one of the thread knows the result
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);
    sdata[tid] = ldata;
    __syncthreads(); // wait till all threads store their data
    // reduce
#pragma unroll
    for (unsigned counter = blockSize / 2; counter >= 32; counter /= 2) {
        if (tid < counter) {
            sdata[tid] = update(comp, ldata, sdata[tid + counter]);
        }
        __syncthreads();
    }
    // manually unwrap last warp for better performance
    // many of these blocks will be optimized out by compiler based on template
    if ((blockSize >= 32) && (tid < 16)) {
        sdata[tid] = update(comp, ldata, sdata[tid + 16]);
    }
    __syncthreads();
    if ((blockSize >= 16) && (tid < 8)) {
        sdata[tid] = update(comp, ldata, sdata[tid + 8]);
    }
    __syncthreads();
    if ((blockSize >= 8) && (tid < 4)) {
        sdata[tid] = update(comp, ldata, sdata[tid + 4]);
    }
    __syncthreads();
    if ((blockSize >= 4) && (tid < 2)) {
        sdata[tid] = update(comp, ldata, sdata[tid + 2]);
    }
    __syncthreads();
    if ((blockSize >= 2) && (tid < 1)) {
        sdata[tid] = update(comp, ldata, sdata[tid + 1]);
    }
    __syncthreads();
}

template <typename T, typename T2, unsigned blockSize, typename C>
__device__
void findMax1D(
        const C &comp,
        T startVal,
        const T * __restrict__ in,
        float * __restrict__ outPos,
        T * __restrict__ outVal,
        unsigned samples)
{
    // one block process one signal
    // map each thread to some sample of the signal
    unsigned int tid = threadIdx.x;
    unsigned int signal = blockIdx.x;

    T2 ldata;
    ldata.x = startVal;
    ldata.y = -1;

    // load data from global memory
    const T *data = in + (signal * samples);
    for(unsigned i = tid; i < samples; i += blockSize) {
        update(comp, ldata, data, i);
    }
    __syncthreads(); // wait till all threads are ready
    findUniversalInSharedMem<T2, blockSize>(comp, ldata);

    // last thread now holds the result
    if (tid == 0) {
        if (nullptr != outVal) {
            outVal[signal] = ldata.x;
        }
        if (nullptr != outPos) {
            outPos[signal] = ldata.y;
        }
    }
}

template <typename T, typename T2, unsigned blockSize, typename C>
__device__
void findUniversal2DNearCenter(
        const C &comp,
        T startVal,
        const T * __restrict__ in,
        float * __restrict__ outPos,
        T * __restrict__ outVal,
        unsigned xSize,
        unsigned ySize,
        unsigned maxDist) // we don't check the boundaries, must be checked by caller
{
    // one block process one signal
    // map each thread to some sample of the signal
    unsigned int tid = threadIdx.x;
    unsigned int signal = blockIdx.x;
    unsigned xHalf = xSize / 2;
    unsigned yHalf = ySize / 2;
    unsigned maxDistSq = maxDist * maxDist;
    unsigned yMin = yHalf - maxDist;
    unsigned yMax = yHalf + maxDist + 1;
    unsigned xMin = xHalf - maxDist + tid;
    unsigned xMax = xHalf + maxDist + 1;

    T2 ldata;
    ldata.x = startVal;
    ldata.y = -1;

    // load data from global memory
    const T *data = in + (signal * xSize * ySize); // beginning of the signal
    for(unsigned y = yMin; y < yMax; ++y) {
        int logicY = (int)y - yHalf;
        unsigned ySq = logicY * logicY;
        for(unsigned x = xMin; x < xMax; x += blockSize) {
            int logicX = (int)x - xHalf;
            // continue if the Euclidean distance is too far
            if ((ySq + (logicX * logicX)) > maxDistSq) {
                continue;
            }
            // we are within the proper radius
            unsigned offset = (y * xSize) + x;
            update(comp, ldata, data, offset);
        }
    }
    __syncthreads(); // wait till all threads are ready
    findUniversalInSharedMem<T2, blockSize>(comp, ldata);

    // last thread now holds the result
    if (tid == 0) {
        if (nullptr != outVal) {
            outVal[signal] = ldata.x;
        }
        if (nullptr != outPos) {
            outPos[signal] = ldata.y;
        }
    }
}

template <typename T, unsigned blockSize, typename C>
__global__
void findUniversal(
        const C &comp,
        T startVal,
        const T * __restrict__ in,
        float * __restrict__ outPos,
        T * __restrict__ outVal,
        unsigned samples)
{
    if (std::is_same<T, float> ::value) {
        findMax1D<float, float2, blockSize> (
                comp,
                startVal,
                (float*)in,
                outPos,
                (float*)outVal,
                samples);
    } else if (std::is_same<T, double> ::value) {
        findMax1D<double, double2, blockSize>(
                comp,
                startVal,
                (double*)in,
                outPos,
                (double*)outVal,
                samples);
    }
}

template <typename T, unsigned blockSize, typename C>
__global__
void findUniversal2DNearCenter(
        const C &comp,
        T startVal,
        const T * __restrict__ in,
        float * __restrict__ outPos,
        T * __restrict__ outVal,
        unsigned xSize,
        unsigned ySize,
        unsigned maxDist)
{
    if (std::is_same<T, float> ::value) {
        findUniversal2DNearCenter<float, float2, blockSize> (
                comp,
                startVal,
                (float*)in,
                outPos,
                (float*)outVal,
                xSize, ySize, maxDist);
    } else if (std::is_same<T, double> ::value) {
        findUniversal2DNearCenter<double, double2, blockSize>(
                comp,
                startVal,
                (double*)in,
                outPos,
                (double*)outVal,
                xSize, ySize, maxDist);
    }
}

#endif /* LIBRARIES_RECONSTRUCTION_CUDA_CUDA_FIND_MAX_CU_ */
