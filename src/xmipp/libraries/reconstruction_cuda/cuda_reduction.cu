#pragma once

template<typename T, unsigned BLOCK_SIZE>
__global__ void reduction_kernel(const T* g_inData, T* g_outData, unsigned n) 
{
    __shared__ T s_data[BLOCK_SIZE];

    unsigned tIdx = threadIdx.x;
    unsigned sumIdx = blockIdx.x * (2 * BLOCK_SIZE) + threadIdx.x;

    T sum = (sumIdx < n) ? g_inData[sumIdx] : 0;

    if (sumIdx + BLOCK_SIZE < n) {
        sum += g_inData[sumIdx + BLOCK_SIZE];
    }

    s_data[tIdx] = sum;
    __syncthreads();

    if (BLOCK_SIZE >= 512 && tIdx < 256) {
        s_data[tIdx] = sum = sum + s_data[tIdx + 256];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 256 && tIdx < 128) {
        s_data[tIdx] = sum = sum + s_data[tIdx + 128];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 128 && tIdx < 64) {
        s_data[tIdx] = sum = sum + s_data[tIdx + 64];
    }

    __syncthreads();

    if (tIdx < 32) {
        if (BLOCK_SIZE >= 64) {
            s_data[tIdx] = sum = sum + s_data[tIdx + 32];
        }
        for (int offset = 32 / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset, 32);
        }
    }

    if (tIdx == 0) {
        g_outData[blockIdx.x] = sum;
    }
}
