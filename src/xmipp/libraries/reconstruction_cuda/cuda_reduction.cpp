#pragma once

#include <algorithm>
#include <stdexcept>
#include "reconstruction_cuda/cuda_reduction.h"
#include "reconstruction_cuda/cuda_reduction.cu"

template<typename T>
void GpuReduction<T>::initializeReductionBuffers(size_t n) {
    if (d_reductionBuffer1 != nullptr) cudaFree(d_reductionBuffer1);
    if (d_reductionBuffer2 != nullptr) cudaFree(d_reductionBuffer2);

    // TODO checks and better computation of the size
    cudaMalloc(&d_reductionBuffer1, n * sizeof(T));
    cudaMalloc(&d_reductionBuffer2, n * sizeof(T));
}

template<typename T>
GpuReduction<T>::GpuReduction()
{
    stream = new cudaStream_t;
    cudaStreamCreate(reinterpret_cast<cudaStream_t*>(stream));
}

template<typename T>
GpuReduction<T>::~GpuReduction() 
{
    reset();
    //FIXME: wait for stream to finish computations
    cudaStreamDestroy(*reinterpret_cast<cudaStream_t*>(stream));
    delete reinterpret_cast<cudaStream_t*>(stream);
}

template<typename T>
void GpuReduction<T>::reset() 
{
    if (d_reductionBuffer1) cudaFree(d_reductionBuffer1);
    if (d_reductionBuffer2) cudaFree(d_reductionBuffer2);
    d_reductionBuffer1 = nullptr;
    d_reductionBuffer2 = nullptr;
    lastReducedSize = 0;
}

template<typename T>
void GpuReduction<T>::reduction(const T* d_input, T* d_output, size_t n,
        size_t blockSize, size_t gridSize)
{
    cudaStream_t* cudaStream = reinterpret_cast<cudaStream_t*>(stream);
    size_t sharedMemory = 0;
    switch (blockSize) {
        case 512:
            reduction_kernel<T, 512>
                <<<gridSize, blockSize, sharedMemory, *cudaStream>>>(d_input, d_output, n);
            break;
        case 256:
            reduction_kernel<T, 256>
                <<<gridSize, blockSize, sharedMemory, *cudaStream>>>(d_input, d_output, n);
            break;
        case 128:
            reduction_kernel<T, 128>
                <<<gridSize, blockSize, sharedMemory, *cudaStream>>>(d_input, d_output, n);
            break;
        case 64:
            reduction_kernel<T, 64>
                <<<gridSize, blockSize, sharedMemory, *cudaStream>>>(d_input, d_output, n);
            break;
        case 32:
            reduction_kernel<T, 32>
                <<<gridSize, blockSize, sharedMemory, *cudaStream>>>(d_input, d_output, n);
            break;
        default:
            throw std::invalid_argument("Unsupported blockSize");
    }
}

template<typename T>
void GpuReduction<T>::reduceDeviceArrayAsync(const T* d_inData, size_t n, T* h_output) 
{
    if (n > lastReducedSize) {
        initializeReductionBuffers(n);
        lastReducedSize = n;
    }

    cudaStream_t* cudaStream = reinterpret_cast<cudaStream_t*>(stream);

    T* d_input = d_reductionBuffer1;
    T* d_output = d_reductionBuffer2;

    int blockSize = 128;//TODO blockSize choosing
    int gridSize = (n + blockSize - 1) / blockSize;
    reduction(d_inData, d_output, n, blockSize, gridSize);
    n = gridSize;
    //cudaStreamSynchronize(*cudaStream);

    while (n > 1) { 
        std::swap(d_input, d_output);
        gridSize = (n + blockSize - 1) / blockSize;
        reduction(d_input, d_output, n, blockSize, gridSize);
        n = gridSize;
        //cudaStreamSynchronize(*cudaStream);
    }

    //check
    if (cudaMemcpyAsync(h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost, *cudaStream) != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
}

template<typename T>
T GpuReduction<T>::reduceDeviceArray(const T* d_inData, size_t n) 
{
    if (n > lastReducedSize) {
        initializeReductionBuffers(n);
        lastReducedSize = n;
    }

    cudaStream_t* cudaStream = reinterpret_cast<cudaStream_t*>(stream);

    T* d_input = d_reductionBuffer1;
    T* d_output = d_reductionBuffer2;

    int blockSize = 128;//TODO blockSize choosing
    int gridSize = (n + blockSize - 1) / blockSize;
    reduction(d_inData, d_output, n, blockSize, gridSize);
    n = gridSize;
    cudaStreamSynchronize(*cudaStream);

    while (n > 1) { 
        std::swap(d_input, d_output);
        gridSize = (n + blockSize - 1) / blockSize;
        reduction(d_input, d_output, n, blockSize, gridSize);
        n = gridSize;
        cudaStreamSynchronize(*cudaStream);
    }

    T result{};
    //check
    cudaMemcpy(&result, d_output, sizeof(T), cudaMemcpyDeviceToHost);


    return result;
}

// Explicit instantiation
template class GpuReduction<float>;
template class GpuReduction<double>;
