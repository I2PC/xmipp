#pragma once

//#include <algorithm>
//#include <stdexcept>
//#include "cuda_reduction.cu"

template <typename T>
class GpuReduction 
{
public:
    ~GpuReduction();
    T reduceHostArray(const T* h_inData, size_t n);
    T reduceDeviceArray(const T* d_inData, size_t n);
    void reset();
private:
    void initializeReductionBuffers(size_t n);
    void reduction(const T* d_input, T* d_output, size_t n,
            size_t blockSize, size_t gridSize);

    T* d_reductionBuffer1 = nullptr;
    T* d_reductionBuffer2 = nullptr;
    size_t lastReducedSize = 0;
};

/*
template<typename T>
void GpuReduction<T>::initializeReductionBuffers(size_t n) {
    if (d_reductionBuffer1 != nullptr) cudaFree(d_reductionBuffer1);
    if (d_reductionBuffer2 != nullptr) cudaFree(d_reductionBuffer2);

    // TODO checks and better computation of the size
    cudaMalloc(&d_reductionBuffer1, n * sizeof(T));
    cudaMalloc(&d_reductionBuffer2, n * sizeof(T));
}

// Template methods definitions
template<typename T>
GpuReduction<T>::~GpuReduction() 
{
    reset();
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
    switch (blockSize) {
        case 512:
            reduction_kernel<T, 512>
                <<<gridSize, blockSize>>>(d_input, d_output, n);
            break;
        case 256:
            reduction_kernel<T, 256>
                <<<gridSize, blockSize>>>(d_input, d_output, n);
            break;
        case 128:
            reduction_kernel<T, 128>
                <<<gridSize, blockSize>>>(d_input, d_output, n);
            break;
        case 64:
            reduction_kernel<T, 64>
                <<<gridSize, blockSize>>>(d_input, d_output, n);
            break;
        case 32:
            reduction_kernel<T, 32>
                <<<gridSize, blockSize>>>(d_input, d_output, n);
            break;
        default:
            throw std::invalid_argument("Unsupported blockSize");
    }
}

template<typename T>
T GpuReduction<T>::reduceHostArray(const T* h_inData, size_t n) 
{
    //TODO
    throw std::logic_error("Not implemented");
}

// Template methods definitions
template<typename T>
T GpuReduction<T>::reduceDeviceArray(const T* d_inData, size_t n) 
{
    if (n > lastReducedSize) {
        initializeReductionBuffers(n);
        lastReducedSize = n;
    }

    T* d_input = d_reductionBuffer1;
    T* d_output = d_reductionBuffer2;

    int blockSize = 128;//TODO blockSize choosing
    int gridSize = (n + blockSize - 1) / blockSize;
    reduction(d_inData, d_output, n, blockSize, gridSize);
    n = gridSize;
    cudaDeviceSynchronize();

    while (n > 1) { 
        std::swap(d_input, d_output);
        gridSize = (n + blockSize - 1) / blockSize;
        reduction(d_input, d_output, n, blockSize, gridSize);
        n = gridSize;
        cudaDeviceSynchronize();
    }

    T result{};
    //check
    cudaMemcpy(&result, d_output, sizeof(T), cudaMemcpyDeviceToHost);

    return result;
}
*/
