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

template<typename T>
class CUDAFlexAlignCorrelate final {

public: 
    struct Params
    {
        Dimensions dim = Dimensions(0);   // size and number of signals to process
        size_t bufferSize = 1; // how many signals can we upload to GPU at once (up to 2 buffers are used)
        size_t batch = 1; // for correlation
    };

    CUDAFlexAlignCorrelate(const Params &p, GPU &gpu) : mParams(p), mGpu(gpu) {}

    ~CUDAFlexAlignCorrelate();

    size_t estimateBytes()
    {
        return estimateBytesAlloc(false);
    };

    void init()
    {
        mReady = true;
        estimateBytesAlloc(true);
    }

    void run(std::complex<T> *h_FTs, float *h_pos, float maxDist);

    void synch() const {
        mGpu.synch();
    }

private:
    size_t estimateBytesAlloc(bool alloc);

    FFTSettings<T> getSettings() const
    {
        return FFTSettings<T>(mParams.dim.copyForN(noOfCorrelations()), mParams.batch, false, false);
    }

    size_t noOfCorrelations() const {
        return mParams.dim.n() * (mParams.dim.n() - 1) / 2;
    }

    void computeCorrelations(
        std::complex<T> *d_in1, size_t in1Size, size_t in1Offset, 
        std::complex<T> *d_in2, size_t in2Size, size_t in2Offset,
        T *h_pos, float maxDist);

    void copyInRightOrder(T* h_pos, bool isWithin,
        int iStart, int iStop, int jStart, int jStop, size_t jSize,
        size_t offset1, size_t offset2);

    const Params mParams;
    GPU &mGpu;

    T *d_imgs = nullptr;
    std::complex<T> *d_ffts = nullptr;
    std::complex<T> *d_fftBuffer1 = nullptr;
    std::complex<T> *d_fftBuffer2 = nullptr;
    float *d_indices = nullptr;
    float *d_positions = nullptr;
    void *mIT = nullptr;
    bool mReady = false;

    static constexpr unsigned BLOCK_DIM = 32;
};