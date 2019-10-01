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

#include "reconstruction_cuda/cuda_find_max.h"
#include "reconstruction_cuda/cuda_find_max.cu"
#include "reconstruction_cuda/cuda_asserts.h"

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

template <typename T, bool dataOnGPU>
void sFindMax(const T * __restrict__ data,
        T * __restrict__ positions,
        T * __restrict__ values,
        unsigned samples, unsigned signals)
{
    assert(samples > 0);
    assert(signals > 0);
    // we need at least one warp, max 512 threads
    size_t maxThreads = 512;
    size_t threads = (samples < maxThreads) ? nextPow2(samples) : maxThreads;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(signals, 1, 1);

    T *d_data;
    T *d_positions;
    T *d_values;

    if (dataOnGPU) {
        d_data = (T*)data;
        d_positions = positions;
        d_values = values;

    } else {
        auto bytesAux = signals * sizeof(T);
        auto bytesData = samples * signals * sizeof(T);

        gpuErrchk(cudaMalloc(&d_data, bytesData));
        gpuErrchk(cudaMalloc(&d_positions, bytesAux));
        gpuErrchk(cudaMalloc(&d_values, bytesAux));

        gpuErrchk(cudaMemcpy(d_data, data, bytesData, cudaMemcpyHostToDevice));
    }

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = 2 * threads * sizeof(T);
    switch (threads)
    {
        case 512:
            findMax<T, 512><<< dimGrid, dimBlock, smemSize >>>(d_data, d_positions, d_values, samples);
            break;

        case 256:
            findMax<T, 256><<< dimGrid, dimBlock, smemSize >>>(d_data, d_positions, d_values, samples);
            break;

        case 128:
            findMax<T, 128><<< dimGrid, dimBlock, smemSize >>>(d_data, d_positions, d_values, samples);
            break;

        case 64:
            findMax<T,  64><<< dimGrid, dimBlock, smemSize >>>(d_data, d_positions, d_values, samples);
            break;

        case 32:
            findMax<T,  32><<< dimGrid, dimBlock, smemSize >>>(d_data, d_positions, d_values, samples);
            break;

        case 16:
            findMax<T,  16><<< dimGrid, dimBlock, smemSize >>>(d_data, d_positions, d_values, samples);
            break;

        case  8:
            findMax<T,   8><<< dimGrid, dimBlock, smemSize >>>(d_data, d_positions, d_values, samples);
            break;

        case  4:
            findMax<T,   4><<< dimGrid, dimBlock, smemSize >>>(d_data, d_positions, d_values, samples);
            break;

        case  2:
            findMax<T,   2><<< dimGrid, dimBlock, smemSize >>>(d_data, d_positions, d_values, samples);
            break;

        case  1:
            findMax<T,   1><<< dimGrid, dimBlock, smemSize >>>(d_data, d_positions, d_values, samples);
            break;
    }
    if ( ! dataOnGPU) {
        auto bytesAux = signals * sizeof(T);
        gpuErrchk(cudaMemcpy(positions, d_positions, bytesAux, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(values, d_values, bytesAux, cudaMemcpyDeviceToHost));

        gpuErrchk(cudaFree(d_data));
        gpuErrchk(cudaFree(d_positions));
        gpuErrchk(cudaFree(d_values));
    }
}

template
void sFindMax<float, true>(const float * __restrict__ data,
        float * __restrict__ positions,
        float * __restrict__ values,
        unsigned samples, unsigned signals);

template
void sFindMax<float, false>(const float * __restrict__ data,
        float * __restrict__ positions,
        float * __restrict__ values,
        unsigned samples, unsigned signals);

template
void sFindMax<double, true>(const double * __restrict__ data,
        double * __restrict__ positions,
        double * __restrict__ values,
        unsigned samples, unsigned signals);
template
void sFindMax<double, false>(const double * __restrict__ data,
        double * __restrict__ positions,
        double * __restrict__ values,
        unsigned samples, unsigned signals);
