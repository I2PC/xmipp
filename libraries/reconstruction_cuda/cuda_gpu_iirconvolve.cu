// https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/
// TILE_DIM=32, BLOCK_ROWS=8
// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded
// to avoid shared memory bank conflicts.
#include "cuda_basic_math.h"
__global__
void transposeNoBankConflicts(float *odata, const float *idata) {
    __shared__ float tile[32][32 + 1];

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    int width = gridDim.x * 32;

    for (int j = 0; j < 32; j += 8)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;  // transpose block offset
    y = blockIdx.x * 32 + threadIdx.y;

    for (int j = 0; j < 32; j += 8)
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__
void iirConvolve2D_Cardinal_Bspline_3_MirrorOffBound(float* input,
        float* output, size_t xDim, size_t yDim) {
    // assign line to thread
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    float* line = output + (idy * xDim);

    // adjust gain
    float z = sqrtf(3.f) - 2.f;
    float z1 = 1.0 - z;
    float gain = -(z1 * z1) / z;

    // copy original data
    for (int i = 0; i < xDim; i++) {
        line[i] = input[i + (idy * xDim)] * gain;
    }

    // prepare some values
    float sum = (line[0] + powf(z, xDim) * line[xDim - 1]) * (1.f + z) / z;
    z1 = z;
    float z2 = powf(z, 2 * xDim - 2);
    float iz = 1.f / z;
    for (int j = 1; j < (xDim - 1); ++j) {
        sum += (z2 + z1) * line[j];
        z1 *= z;
        z2 *= iz;
    }
    line[0] = sum * z / (1.f - powf(z, 2 * xDim));
    for (int j = 1; j < xDim; ++j) {
        line[j] += z * line[j - 1];
    }
    line[xDim - 1] *= z / (z - 1.f);
    for (int j = xDim - 2; 0 <= j; --j) {
        line[j] = z * (line[j + 1] - line[j]);
    }
}

template<bool normalize>
__global__
void shiftFFT2D(float2* data,
        int noOfImages, size_t dimFFTX, size_t dimX, size_t dimY, float shiftX, float shiftY) {
    // assign pixel to thread
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;

    if (idx >= dimFFTX || idy >= dimY ) return;
    size_t fIndex = idy * dimFFTX + idx; // index within single image
    float normFactor = 1.f / (dimX * dimY);
    int idxCen = idx;
    int halfY = (dimY - 1) / 2;
    int idyCen = (idy <= halfY) ?  idy : idy - dimY ;
    float shift = ((idxCen * shiftX) / dimX) + ((idyCen * shiftY) / dimY);
    float arg = -2 * PI * shift;
    float2 tmp = make_float2(cosf(arg), sinf(arg));

    for (int n = 0; n < noOfImages; n++) {
        size_t index = n * dimFFTX * dimY + fIndex; // index within consecutive images
        float2 tmp2 = data[index];
        if (normalize) {
            tmp2 *= normFactor;
        }
        float2 res;
        res.x = (tmp.x*tmp2.x) - (tmp.y*tmp2.y);
        res.y = (tmp.y*tmp2.x) + (tmp.x*tmp2.y);
        data[index] = res;
    }
}
