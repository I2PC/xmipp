#include "cuda_basic_math.h"
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
    // this transform expects FFT shifted to center. Recalculate the line index
    int halfY = (dimY - 1) / 2; //
    int idyCen = (idy <= halfY) ?  idy : idy - dimY ;
    float shift = ((idx * shiftX) / dimX) + ((idyCen * shiftY) / dimY);
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
