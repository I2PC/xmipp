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

#include "reconstruction_cuda/cuda_find_extrema.h"
#include "reconstruction_cuda/cuda_find_max.cu"
#include "reconstruction_cuda/cuda_asserts.h"

namespace ExtremaFinder {

template<typename T>
void CudaExtremaFinder<T>::setDefault() {
    // FIXME DS implement
}

template<typename T>
void CudaExtremaFinder<T>::release() {
    // FIXME DS implement
}

template<typename T>
void CudaExtremaFinder<T>::check() {
    // FIXME DS implement
}

template<typename T>
void CudaExtremaFinder<T>::initMax(bool reuse) {
    // FIXME DS implement
}

template<typename T>
void CudaExtremaFinder<T>::findMax(T *data) {
    // FIXME DS implement
}

template<typename T>
void CudaExtremaFinder<T>::initMaxAroundCenter(bool reuse) {
    // FIXME DS implement
}

template<typename T>
void CudaExtremaFinder<T>::findMaxAroundCenter(T *data) {
    // FIXME DS implement
}

template<typename T>
size_t CudaExtremaFinder<T>::ceilPow2(size_t x)
{
    if (x <= 1) return 1;
    int power = 2;
    x--;
    while (x >>= 1) power <<= 1;
    return power;
}

template<typename T>
void CudaExtremaFinder<T>::sFindMax(const GPU &gpu,
    const Dimensions &dims,
    const T * __restrict__ d_data,
    T * __restrict__ d_positions,
    T * __restrict__ d_values) {
    // check input
    assert(dims.sizeSingle() > 0);
    assert(dims.n() > 0);
    assert(nullptr != d_data);
    assert(nullptr != d_positions);
    assert(nullptr != d_values);
    assert(dims.size() <= std::numeric_limits<unsigned>::max()); // indexing overflow in the kernel

    // create threads / blocks
    size_t maxThreads = 512;
    size_t threads = (dims.sizeSingle() < maxThreads) ? ceilPow2(dims.sizeSingle()) : maxThreads;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(dims.n(), 1, 1);
    auto stream = *(cudaStream_t*)gpu.stream();

    // for each thread, we need two variables in shared memory
    size_t smemSize = 2 * threads * sizeof(T);
    switch (threads) {
        case 512:
            return findMax1D<T, 512><<< dimGrid, dimBlock, smemSize, stream>>>(
                    d_data, d_positions, d_values, dims.sizeSingle());
        case 256:
            return findMax1D<T, 256><<< dimGrid, dimBlock, smemSize, stream>>>(
                    d_data, d_positions, d_values, dims.sizeSingle());
        case 128:
            return findMax1D<T, 128><<< dimGrid, dimBlock, smemSize, stream>>>(
                    d_data, d_positions, d_values, dims.sizeSingle());
        case 64:
            return findMax1D<T, 64><<< dimGrid, dimBlock, smemSize, stream>>>(
                    d_data, d_positions, d_values, dims.sizeSingle());
        case 32:
            return findMax1D<T, 32><<< dimGrid, dimBlock, smemSize, stream>>>(
                    d_data, d_positions, d_values, dims.sizeSingle());
        case 16:
            return findMax1D<T, 16><<< dimGrid, dimBlock, smemSize, stream>>>(
                    d_data, d_positions, d_values, dims.sizeSingle());
        case 8:
            return findMax1D<T, 8><<< dimGrid, dimBlock, smemSize, stream>>>(
                    d_data, d_positions, d_values, dims.sizeSingle());
        case 4:
            return findMax1D<T, 4><<< dimGrid, dimBlock, smemSize, stream>>>(
                    d_data, d_positions, d_values, dims.sizeSingle());
        case 2:
            return findMax1D<T, 2><<< dimGrid, dimBlock, smemSize, stream>>>(
                    d_data, d_positions, d_values, dims.sizeSingle());
        case 1:
            return findMax1D<T, 1><<< dimGrid, dimBlock, smemSize, stream>>>(
                    d_data, d_positions, d_values, dims.sizeSingle());
        default: REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Unsupported number of threads");
    }
}

template<typename T>
void CudaExtremaFinder<T>::sFindMax2DNearCenter(
        const GPU &gpu,
        const Dimensions &dims,
        const T * d_data,
        T * d_positions,
        T * d_values,
        size_t maxDist) {
    // check input
    assert(dims.is2D());
    assert( ! dims.isPadded());
    assert(dims.sizeSingle() > 0);
    assert(dims.n() > 0);
    assert(nullptr != d_data);
    assert(nullptr != d_positions);
    assert(nullptr != d_values);
    assert(0 < maxDist);
    int xHalf = dims.x() / 2;
    int yHalf = dims.y() / 2;
    assert((2 * xHalf) > maxDist);
    assert((2 * yHalf) > maxDist);

    // prepare threads / blocks
    size_t maxThreads = 512;
    // threads should process a single row of the signal
    size_t threads = (dims.x() < maxThreads) ? ceilPow2(dims.x()) : maxThreads;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(dims.n(), 1, 1);
    auto stream = *(cudaStream_t*)gpu.stream();

    // for each thread, we need two variables in shared memory
    int smemSize = 2 * threads * sizeof(T);
    switch (threads) {
        case 512:
            return findMax2DNearCenter<T, 512><<< dimGrid, dimBlock, smemSize, stream>>>(
                d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 256:
            return findMax2DNearCenter<T, 256><<< dimGrid, dimBlock, smemSize, stream>>>(
                d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 128:
            return findMax2DNearCenter<T, 128><<< dimGrid, dimBlock, smemSize, stream>>>(
                d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 64:
            return findMax2DNearCenter<T, 64><<< dimGrid, dimBlock, smemSize, stream>>>(
                d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 32:
            return findMax2DNearCenter<T, 32><<< dimGrid, dimBlock, smemSize, stream>>>(
                d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 16:
            return findMax2DNearCenter<T, 16><<< dimGrid, dimBlock, smemSize, stream>>>(
                d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 8:
            return findMax2DNearCenter<T, 8><<< dimGrid, dimBlock, smemSize, stream>>>(
                d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 4:
            return findMax2DNearCenter<T, 4><<< dimGrid, dimBlock, smemSize, stream>>>(
                d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 2:
            return findMax2DNearCenter<T, 2><<< dimGrid, dimBlock, smemSize, stream>>>(
                d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 1:
            return findMax2DNearCenter<T, 1><<< dimGrid, dimBlock, smemSize, stream>>>(
                d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        default: REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Unsupported number of threads");
    }
}

// explicit instantiation
template class CudaExtremaFinder<float>;
template class CudaExtremaFinder<double>;

} /* namespace ExtremaFinder */

//#include "reconstruction_cuda/cuda_find_max.cu"
//#include "reconstruction_cuda/cuda_asserts.h"
//

//
//template <typename T, bool dataOnGPU>
//void sFindMax1D(const GPU &gpu,
//        const Dimensions &dims,
//        const T * __restrict__ data,
//        T * __restrict__ positions,
//        T * __restrict__ values)
//{
//    assert(dims.sizeSingle() > 0);
//    assert(dims.n() > 0);
//    // we need at most 512 threads (physical limitation)
//    size_t maxThreads = 512;
//    size_t threads = (dims.sizeSingle() < maxThreads) ? nextPow2(dims.sizeSingle()) : maxThreads;
//    dim3 dimBlock(threads, 1, 1);
//    dim3 dimGrid(dims.n(), 1, 1);
//
//    T *d_data;
//    T *d_positions;
//    T *d_values;
//
//    auto stream = *(cudaStream_t*)gpu.stream();
//
//    if (dataOnGPU) {
//        d_data = (T*)data;
//        d_positions = positions;
//        d_values = values;
//
//    } else {
//        auto bytesAux = dims.n() * sizeof(T);
//        auto bytesData = dims.size() * sizeof(T);
//
//        gpuErrchk(cudaMalloc(&d_data, bytesData));
//        gpuErrchk(cudaMalloc(&d_positions, bytesAux));
//        gpuErrchk(cudaMalloc(&d_values, bytesAux));
//
//        gpuErrchk(cudaMemcpyAsync(d_data, data, bytesData, cudaMemcpyHostToDevice, stream));
//    }
//
//    // when there is only one warp per block, we need to allocate two warps
//    // worth of shared memory so that we don't index shared memory out of bounds
//    int smemSize = 2 * threads * sizeof(T);
//    switch (threads)
//    {
//        case 512:
//            findMax1D<T, 512><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.sizeSingle());
//            break;
//
//        case 256:
//            findMax1D<T, 256><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.sizeSingle());
//            break;
//
//        case 128:
//            findMax1D<T, 128><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.sizeSingle());
//            break;
//
//        case 64:
//            findMax1D<T,  64><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.sizeSingle());
//            break;
//
//        case 32:
//            findMax1D<T,  32><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.sizeSingle());
//            break;
//
//        case 16:
//            findMax1D<T,  16><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.sizeSingle());
//            break;
//
//        case  8:
//            findMax1D<T,   8><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.sizeSingle());
//            break;
//
//        case  4:
//            findMax1D<T,   4><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.sizeSingle());
//            break;
//
//        case  2:
//            findMax1D<T,   2><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.sizeSingle());
//            break;
//
//        case  1:
//            findMax1D<T,   1><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.sizeSingle());
//            break;
//    }
//    if ( ! dataOnGPU) {
//        auto bytesAux = dims.n() * sizeof(T);
//        gpuErrchk(cudaMemcpyAsync(positions, d_positions, bytesAux, cudaMemcpyDeviceToHost, stream));
//        gpuErrchk(cudaMemcpyAsync(values, d_values, bytesAux, cudaMemcpyDeviceToHost, stream));
//
//        gpuErrchk(cudaFree(d_data));
//        gpuErrchk(cudaFree(d_positions));
//        gpuErrchk(cudaFree(d_values));
//    }
//}
//
//template <typename T, bool dataOnGPU>
//void sFindMax2DNear(const GPU &gpu,
//        const Dimensions &dims,
//        const T * __restrict__ data,
//        T * __restrict__ positions,
//        T * __restrict__ values,
//        size_t maxDist) {
//    assert(dims.sizeSingle() > 0);
//    assert(dims.n() > 0);
//    assert(nullptr != data);
//    assert(nullptr != positions);
//    assert(nullptr != values);
//
//    assert(0 < dims.x());
//    assert(0 < dims.y());
//    assert(1 == dims.zPadded());
//    assert(0 < dims.n());
//    assert(0 < maxDist);
//
//    int xHalf = dims.x() / 2;
//    int yHalf = dims.y() / 2;
//
//    assert((2 * xHalf) > maxDist);
//    assert((2 * yHalf) > maxDist);
//
//    // we need at most 512 threads (physical limitation)
//    size_t maxThreads = 512;
//    // threads should process a single row of the signal
//    size_t threads = (dims.x() < maxThreads) ? nextPow2(dims.x()) : maxThreads;
//    dim3 dimBlock(threads, 1, 1);
//    dim3 dimGrid(dims.n(), 1, 1);
//
//    T *d_data;
//    T *d_positions;
//    T *d_values;
//
//    auto stream = *(cudaStream_t*)gpu.stream();
//
//    if (dataOnGPU) {
//        d_data = (T*)data;
//        d_positions = positions;
//        d_values = values;
//
//    } else {
//        auto bytesAux = dims.n() * sizeof(T);
//        auto bytesData = dims.size() * sizeof(T);
//
//        gpuErrchk(cudaMalloc(&d_data, bytesData));
//        gpuErrchk(cudaMalloc(&d_positions, bytesAux));
//        gpuErrchk(cudaMalloc(&d_values, bytesAux));
//
//        gpuErrchk(cudaMemcpyAsync(d_data, data, bytesData, cudaMemcpyHostToDevice, stream));
//    }
//
//    // when there is only one warp per block, we need to allocate two warps
//    // worth of shared memory so that we don't index shared memory out of bounds
//    int smemSize = 2 * threads * sizeof(T);
//    switch (threads)
//    {
//        case 512:
//            findMax2DNearCenter<T, 512><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
//            break;
//
//        case 256:
//            findMax2DNearCenter<T, 256><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
//            break;
//
//        case 128:
//            findMax2DNearCenter<T, 128><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
//            break;
//
//        case 64:
//            findMax2DNearCenter<T,  64><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
//            break;
//
//        case 32:
//            findMax2DNearCenter<T,  32><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
//            break;
//
//        case 16:
//            findMax2DNearCenter<T,  16><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
//            break;
//
//        case  8:
//            findMax2DNearCenter<T,   8><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
//            break;
//
//        case  4:
//            findMax2DNearCenter<T,   4><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
//            break;
//
//        case  2:
//            findMax2DNearCenter<T,   2><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
//            break;
//
//        case  1:
//            findMax2DNearCenter<T,   1><<< dimGrid, dimBlock, smemSize, stream>>>(d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
//            break;
//    }
//    if ( ! dataOnGPU) {
//        auto bytesAux = dims.n() * sizeof(T);
//        gpuErrchk(cudaMemcpyAsync(positions, d_positions, bytesAux, cudaMemcpyDeviceToHost, stream));
//        gpuErrchk(cudaMemcpyAsync(values, d_values, bytesAux, cudaMemcpyDeviceToHost, stream));
//
//        gpuErrchk(cudaFree(d_data));
//        gpuErrchk(cudaFree(d_positions));
//        gpuErrchk(cudaFree(d_values));
//    }
//
//}
//// sFindMax1D
//template
//void sFindMax1D<float, true>(const GPU &gpu,
//        const Dimensions &dims,
//        const float * __restrict__ data,
//        float * __restrict__ positions,
//        float * __restrict__ values);
//
//template
//void sFindMax1D<float, false>(const GPU &gpu,
//        const Dimensions &dims,
//        const float * __restrict__ data,
//        float * __restrict__ positions,
//        float * __restrict__ values);
//
//template
//void sFindMax1D<double, true>(const GPU &gpu,
//        const Dimensions &dims,
//        const double * __restrict__ data,
//        double * __restrict__ positions,
//        double * __restrict__ values);
//template
//void sFindMax1D<double, false>(const GPU &gpu,
//        const Dimensions &dims,
//        const double * __restrict__ data,
//        double * __restrict__ positions,
//        double * __restrict__ values);
//
//
//// sFindMax2DNear
//template
//void  sFindMax2DNear<float, true>(const GPU &gpu,
//        const Dimensions &dims,
//        const float * __restrict__ data,
//        float * __restrict__ positions,
//        float * __restrict__ values,
//        size_t maxDist);
//
//template
//void  sFindMax2DNear<float, false>(const GPU &gpu,
//        const Dimensions &dims,
//        const float * __restrict__ data,
//        float * __restrict__ positions,
//        float * __restrict__ values,
//        size_t maxDist);
//
//template
//void  sFindMax2DNear<double, true>(const GPU &gpu,
//        const Dimensions &dims,
//        const double * __restrict__ data,
//        double * __restrict__ positions,
//        double * __restrict__ values,
//        size_t maxDist);
//template
//void  sFindMax2DNear<double, false>(const GPU &gpu,
//        const Dimensions &dims,
//        const double * __restrict__ data,
//        double * __restrict__ positions,
//        double * __restrict__ values,
//        size_t maxDist);
