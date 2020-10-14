#include "reconstruction_cuda/cuda_gpu_movie_alignment_correlation.h"
#include "data/fft_settings.h"
#include "core/xmipp_fftw.h"
#include "core/multidim_array.h"
#include <iostream>
#include <gtest/gtest.h>
#include <random>
// MORE INFO HERE: http://code.google.com/p/googletest/wiki/AdvancedGuide
class MovieAlignmentCorrelationGpuTest : public ::testing::Test
{
protected:
    //init metadatas
    virtual void SetUp()
    {
    	// nothing to do
    }
};

TEST_F( MovieAlignmentCorrelationGpuTest, FFTvsIFFTSingleoutOfPlace)
{
    // allocate and prepare data
    size_t x = 25;
    size_t y = 13;
    size_t noOfElements = x * y;

    float* src = new float[noOfElements];
    float* res = new float[noOfElements]();
    for (int i = 0; i < y; i++) {
        for (int j = 0; j < x; j++) {
            int v = (i * x) + j;
            src[v] = (float)v;
        }
    }

    // prepare gpu and cpu memory
    GpuMultidimArrayAtGpu<float> imagesGPU(x, y, 1, 1);
    imagesGPU.copyToGpu(src);
    GpuMultidimArrayAtGpu<std::complex<float>> FFT;

    // perform fft and inverse fft
    mycufftHandle handle_fft;
    mycufftHandle handle_ifft;
    imagesGPU.fft(FFT, handle_fft);
    FFT.ifft(imagesGPU, handle_ifft);
    imagesGPU.copyToCpu(res);

    float normalization = 1 / (float)(x * y);
    float delta = 0.0001;
    for (int i = 0; i < noOfElements; ++i) {
        EXPECT_NEAR(src[i], res[i] * normalization, delta);
    }

    handle_fft.clear();
    handle_ifft.clear();
    delete[] src;
    delete[] res;
}

TEST_F( MovieAlignmentCorrelationGpuTest, FFTvsIFFTManyoutOfPlace)
{
    // allocate and prepare data
    size_t x = 25;
    size_t y = 13;
    size_t batch = 3;
    size_t noOfElements = x * y * batch;

    float* src = new float[noOfElements];
    float* res = new float[noOfElements]();
    for (int n = 0; n < batch; n++) {
        for (int i = 0; i < y; i++) {
            for (int j = 0; j < x; j++) {
                int v = (n * x * y) + (i * x) + j;
                src[v] = (float)v;
            }
        }
    }

    // prepare gpu and cpu memory
    GpuMultidimArrayAtGpu<float> imagesGPU(x, y, 1, batch);
    imagesGPU.copyToGpu(src);
    GpuMultidimArrayAtGpu<std::complex<float>> FFT;

    // perform fft and inverse fft
    mycufftHandle handle_fft;
    mycufftHandle handle_ifft;
    imagesGPU.fft(FFT, handle_fft);
    FFT.ifft(imagesGPU, handle_ifft);
    imagesGPU.copyToCpu(res);

    float normalization = 1 / (float)(x * y);
    float delta = 0.00037f;
    for (int i = 0; i < noOfElements; ++i) {
        EXPECT_NEAR(src[i], res[i] * normalization, delta);
    }

    handle_fft.clear();
    handle_ifft.clear();
    delete[] src;
    delete[] res;
}

GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
