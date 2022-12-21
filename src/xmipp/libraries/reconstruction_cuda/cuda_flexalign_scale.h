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
        bool doBinning = false;
        Dimensions raw = Dimensions(0);   // raw movie, that needs to be binned
        Dimensions movie = Dimensions(0); // cropped / binned size
        Dimensions out = Dimensions(0);   // size used for output / scaled frames
        size_t batch = 1;
    };

    CUDAFlexAlignScale(const Params &p, const GPU &gpu) : mParams(p), mGpu(gpu) {}

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

    auto getOutputSettings() const
    {
        return FFTSettings<T>(mParams.out, mParams.batch, false, false);
    }

    auto getMovieSettings() const
    {
        auto dir = mParams.doBinning ? false : true; // when we do binning, we need to perform IFT, otherwise we do FFT
        return FFTSettings<T>(mParams.movie, mParams.batch, false, dir);
    }

    void synch() const {
        mGpu.synch();
    }

private:
    size_t estimateBytesAlloc(bool alloc);

    void runFFTScale(T *h_in, const FFTSettings<T> &in, std::complex<T> *h_out, T *d_filter);

    void runScaleIFT(T *h_outBin);

    auto getRawSettings() const
    {
        return FFTSettings<T>(mParams.raw, mParams.batch);
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

    void *mFT = nullptr;
    void *mIT = nullptr;
    void *mAux1 = nullptr;
    void *mAux2 = nullptr;
};
