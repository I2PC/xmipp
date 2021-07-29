#pragma once

template <typename T>
class GpuReduction 
{
public:
    GpuReduction();
    ~GpuReduction();
    T reduceDeviceArray(const T* d_inData, size_t n);
    void reduceDeviceArrayAsync(const T* d_inData, size_t n, T* h_output);
    void reset();
private:
    void initializeReductionBuffers(size_t n);
    void reduction(const T* d_input, T* d_output, size_t n,
            size_t blockSize, size_t gridSize);

    T* d_reductionBuffer1 = nullptr;
    T* d_reductionBuffer2 = nullptr;
    void* stream = nullptr;//TODO maybe not worth it, try to reduce all 3 arrays at once
    size_t lastReducedSize = 0;
};
