#pragma once

#include "data/dimensions.h"
#include "reconstruction_cuda/gpu.h"
#include "data/fft_settings.h"

template <typename T>
class CUDAFlexAlignScale final
{
public:
    struct Params
    {
        Dimensions raw = Dimensions(0);   // raw movie, that needs to be binned
        Dimensions movie = Dimensions(0); // cropped / binned size
        Dimensions out = Dimensions(0);   // size used for output / scaled frames
        size_t outBatch = 1;
        bool doBinning = false;
        const size_t movieBatch = 1; // always 1
    };

    CUDAFlexAlignScale(const Params &p, const GPU &gpu) : mParams(p), mGpu(gpu), mFT(nullptr), mIT(nullptr), mAux1(nullptr), mAux2(nullptr) {}

    ~CUDAFlexAlignScale();

    size_t estimateBytes()
    {
        return estimateBytesAlloc(false);
    };

    void init()
    {
        estimateBytesAlloc(true);
    }

    void run(T *h_raw, T *h_movie, std::complex<T> *h_out, T *d_filter)
    {
        if (mParams.doBinning)
        {
            runFFTScale(h_raw, getRawSettings(), h_out, d_filter);
            runScaleIFT(h_movie);
        } else {
             runFFTScale(h_movie, getMovieSettings(), h_out, d_filter);
        }
    };

private:
    size_t estimateBytesAlloc(bool alloc);

    void runFFTScale(T *h_in, const FFTSettings<T> &in, std::complex<T> *h_out, T *d_filter);

    void runScaleIFT(T *h_outBin);

    auto getRawSettings() const
    {
        return FFTSettings<T>(mParams.raw, mParams.movieBatch);
    }

    auto getMovieSettings() const
    {
        auto dir = mParams.doBinning ? false : true; // when we do binning, we need to perform IFT, otherwise we do FFT
        return FFTSettings<T>(mParams.movie, mParams.movieBatch, false, dir);
    }

    auto getOutputSettings() const
    {
        return FFTSettings<T>(mParams.out, mParams.outBatch, false, false);
    }

    constexpr std::complex<T> *asCT(void *ptr)
    {
        return reinterpret_cast<std::complex<T> *>(ptr);
    };
    constexpr T *asT(void *ptr)
    {

        return reinterpret_cast<T *>(ptr);
    };

    const Params mParams;
    const GPU &mGpu;

    static constexpr unsigned BLOCK_DIM = 32;

    void *mFT;
    void *mIT;
    void *mAux1;
    void *mAux2;
};
