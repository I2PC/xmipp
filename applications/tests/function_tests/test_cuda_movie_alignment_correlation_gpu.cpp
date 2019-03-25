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

TEST_F( MovieAlignmentCorrelationGpuTest, performFFTAndScaleSingleNoScaleNoFilter)
{
    size_t x = 25;
    size_t x_out = x / 2 + 1;
    size_t y = 13;
    size_t y_out = y;
    size_t noOfElements = x * y;
    MultidimArray<float> filter(y,x);
    filter.initConstant(1.f);
    size_t max_elems = x_out * 2 * y_out;

    // prepare data
    float* src = new float[noOfElements];
    float* data = new float[max_elems];
    float* res = new float[noOfElements]();
    for (int i = 0; i < y; i++) {
        for (int j = 0; j < x; j++) {
            int v = (i * x) + j;
            data[v] = (float)v;
            src[v] = (float)v;
        }
    }

    // perform FFT. In the memory we should have FT of the input
    performFFTAndScale(data, 1, x, y, 1, x_out, y_out, filter);
    // transform them to spatial domain
    GpuMultidimArrayAtGpu<float> imagesGPU(x, y, 1, 1);
    GpuMultidimArrayAtGpu<std::complex<float> > FFT(x_out, y_out, 1, 1);
    FFT.copyToGpu((std::complex<float>*)data);

    mycufftHandle handle;
    FFT.ifft(imagesGPU, handle);
    imagesGPU.copyToCpu(res);
    float delta = 0.0001f;
    for (int i = 0; i < noOfElements; ++i) {
        EXPECT_NEAR(src[i], res[i], delta);
    }

    handle.clear();
    delete[] src;
    delete[] data;
    delete[] res;
}

TEST_F( MovieAlignmentCorrelationGpuTest, performFFTAndScaleFewFramesNoBatchNoScaleNoFilter)
{
    size_t x = 25;
    size_t x_out = x / 2 + 1;
    size_t y = 13;
    size_t y_out = y;
    size_t frames = 3;
    size_t noOfElemsSingle = x * y;
    MultidimArray<float> filter(y,x);
    filter.initConstant(1.f);
    size_t maxElemsSingle = x_out * 2 * y_out;

    // prepare data
    float* src = new float[noOfElemsSingle * frames];
    float* data = new float[maxElemsSingle * frames];
    float* res = new float[noOfElemsSingle * frames]();
    for (int n = 0; n < frames; n++) {
        for (int i = 0; i < y; i++) {
            for (int j = 0; j < x; j++) {
                int v = (n * x * y) + (i * x) + j;
                data[v] = (float)v;
                src[v] = (float)v;
            }
        }
    }

    // perform FFT. In the memory we should have FT of the input
    performFFTAndScale(data, frames, x, y, 1, x_out, y_out, filter);
    // transform them to spatial domain
    GpuMultidimArrayAtGpu<float> imagesGPU(x, y, 1, frames);
    GpuMultidimArrayAtGpu<std::complex<float> > FFT(x_out, y_out, 1, frames);
    FFT.copyToGpu((std::complex<float>*)data);

    mycufftHandle handle;
    FFT.ifft(imagesGPU, handle);
    imagesGPU.copyToCpu(res);
    float delta = 0.00037f;
    for (int i = 0; i < noOfElemsSingle * frames; ++i) {
//        std::cout << res[i] << " ";
        EXPECT_NEAR(src[i], res[i], delta);
    }

    handle.clear();
    delete[] src;
    delete[] data;
    delete[] res;
}

TEST_F( MovieAlignmentCorrelationGpuTest, performFFTAndScaleFewFramesBatchNoScaleNoFilter)
{
    size_t x = 25;
    size_t x_out = x / 2 + 1;
    size_t y = 13;
    size_t y_out = y;
    size_t frames = 5;
    size_t batch = 3;
    size_t noOfElemsSingle = x * y;
    MultidimArray<float> filter(y,x);
    filter.initConstant(1.f);
    size_t maxElemsSingle = x_out * 2 * y_out;

    // prepare data
    float* src = new float[noOfElemsSingle * frames];
    float* data = new float[maxElemsSingle * frames];
    float* res = new float[noOfElemsSingle * frames]();
    for (int n = 0; n < frames; n++) {
        for (int i = 0; i < y; i++) {
            for (int j = 0; j < x; j++) {
                int v = (n * x * y) + (i * x) + j;
                data[v] = (float)v;
                src[v] = (float)v;
            }
        }
    }

    // perform FFT. In the memory we should have FT of the input
    performFFTAndScale(data, frames, x, y, batch, x_out, y_out, filter);
    // transform them to spatial domain
    GpuMultidimArrayAtGpu<float> imagesGPU(x, y, 1, frames);
    GpuMultidimArrayAtGpu<std::complex<float> > FFT(x_out, y_out, 1, frames);
    FFT.copyToGpu((std::complex<float>*)data);

    mycufftHandle handle;
    FFT.ifft(imagesGPU, handle);
    imagesGPU.copyToCpu(res);
    float delta = 0.00074f;
    for (int i = 0; i < noOfElemsSingle * frames; ++i) {
//        std::cout << res[i] << " ";
        EXPECT_NEAR(src[i], res[i], delta);
    }

    handle.clear();
    delete[] src;
    delete[] data;
    delete[] res;
}

TEST_F( MovieAlignmentCorrelationGpuTest, performFFTAndScaleSingleNoScaleFilter)
{
    size_t x = 25;
    size_t x_out = x / 2 + 1;
    size_t y = 13;
    size_t y_out = y;
    size_t noOfElements = x * y;
    MultidimArray<float> filter(y,x);
    filter.initConstant(0.5f);
    size_t max_elems = x_out * 2 * y_out;

    // prepare data
    float* src = new float[noOfElements];
    float* data = new float[max_elems];
    float* res = new float[noOfElements]();
    for (int i = 0; i < y; i++) {
        for (int j = 0; j < x; j++) {
            int v = (i * x) + j;
            data[v] = (float)v;
            src[v] = (float)v;
        }
    }

    // perform FFT. In the memory we should have FT of the input
    performFFTAndScale(data, 1, x, y, 1, x_out, y_out, filter);
    // transform them to spatial domain
    GpuMultidimArrayAtGpu<float> imagesGPU(x, y, 1, 1);
    GpuMultidimArrayAtGpu<std::complex<float> > FFT(x_out, y_out, 1, 1);
    FFT.copyToGpu((std::complex<float>*)data);

    mycufftHandle handle;
    FFT.ifft(imagesGPU, handle);
    imagesGPU.copyToCpu(res);
    float correction = 0.5f;
    float delta = 0.0001f;
    for (int i = 0; i < noOfElements; ++i) {
        EXPECT_NEAR(src[i] * correction, res[i], delta);
    }

    handle.clear();
    delete[] src;
    delete[] data;
    delete[] res;
}

GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
