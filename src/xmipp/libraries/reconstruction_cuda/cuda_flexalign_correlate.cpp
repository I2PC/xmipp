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

#include "cuda_flexalign_correlate.h"
#include "reconstruction_adapt_cuda/basic_mem_manager.h"
#include "cuda_fft.h"
#include "cuda_gpu_movie_alignment_correlation_kernels.cu"
#include "cuda_single_extrema_finder.h"
#include "cuda_asserts.h"

template<typename T>
CUDAFlexAlignCorrelate<T>::~CUDAFlexAlignCorrelate() {
    CudaFFT<T>::release(reinterpret_cast<cufftHandle *>(mIT));
    BasicMemManager::instance().give(d_ffts);
    BasicMemManager::instance().give(d_fftBuffer1);
    BasicMemManager::instance().give(d_fftBuffer2);
    BasicMemManager::instance().give(d_indices);
    BasicMemManager::instance().give(d_positions);
}


template <typename T>
size_t CUDAFlexAlignCorrelate<T>::estimateBytesAlloc(bool alloc) {
    const auto settings = getSettings();
    auto bIT = CudaFFT<T>().estimatePlanBytes(settings);
    auto bSignals = settings.sBytesBatch();
    auto bFTs = settings.fBytesBatch();        
    auto bBuffer1 = settings.fBytesSingle() * mParams.bufferSize;
    auto bBuffer2 = (mParams.dim.n() == mParams.bufferSize) ? 0 : bBuffer1;
    auto bIndices = settings.batch() * sizeof(float); // index of the maxima
    auto bPositions = settings.batch() * 2 * sizeof(float); // 2D position of the maxima
    if (alloc) {
        mIT = CudaFFT<T>::createPlan(mGpu, settings);
        d_imgs = reinterpret_cast<T*>(BasicMemManager::instance().get(bSignals, MemType::CUDA));
        d_ffts = reinterpret_cast<std::complex<T>*>(BasicMemManager::instance().get(bFTs, MemType::CUDA));
        d_fftBuffer1 = reinterpret_cast<std::complex<T>*>(BasicMemManager::instance().get(bBuffer1, MemType::CUDA));
        d_fftBuffer2 = reinterpret_cast<std::complex<T>*>(BasicMemManager::instance().get(bBuffer2, MemType::CUDA));
        d_indices = reinterpret_cast<float*>(BasicMemManager::instance().get(bIndices, MemType::CUDA));
        d_positions = reinterpret_cast<float*>(BasicMemManager::instance().get(bPositions, MemType::CUDA));
    }
    return bIT + bSignals + bFTs + bBuffer1 + bBuffer2 + bIndices + bPositions;
}

template<typename T>
void CUDAFlexAlignCorrelate<T>::copyInRightOrder(T* h_pos, bool isWithin,
        int iStart, int iStop, int jStart, int jStop, size_t jSize,
        size_t offset1, size_t offset2) {
    constexpr size_t coordinates = 2; // we copy 2D coordinates
    const auto signals = mParams.dim.n();
    size_t counter = 0;
    bool ready = false;
    auto stream = *(cudaStream_t*)mGpu.stream();
    for (int i = iStart; i <= iStop; i++) {
        for (int j = isWithin ? i + 1 : 0; j < jSize; j++) {
            if (!ready) {
                ready = true;
                j = jStart;
                continue; // skip first iteration
            }
            if (ready) {
                size_t actualI = offset1 + i;
                size_t actualJ = offset2 + j;
                size_t toCopy = (i == iStop) ? jStop - j + 1 : jSize - j;
                // imagine correlation in layers, correlation of 0th img with other is first layer, 1st with other is second etc
                // compute sum of images in complete layers
                size_t imgsInPreviousLayers = (((signals - 1) + (signals - actualI)) * (actualI)) / 2;
                size_t imgsInCurrentLayer = actualJ - actualI - 1;
                gpuErrchk(cudaMemcpyAsync(h_pos + (coordinates * (imgsInPreviousLayers + imgsInCurrentLayer)),
                    d_positions + (counter * coordinates),
                    toCopy * coordinates * sizeof(float),
                    cudaMemcpyDeviceToHost, stream));
                counter += toCopy;
                break; // skip to next outer iteration
            }
            if ((iStop == i) && (jStop == j)) {
                return;
            }
        }
    }
}

template<typename T>
void CUDAFlexAlignCorrelate<T>::run(std::complex<T> *h_FTs, float *h_pos, float maxDist) {
    const auto settings = getSettings();
    const auto singleFTPixels = settings.fDim().xy();
    const auto singleFTBytes = settings.fBytesSingle();

    const auto signals = mParams.dim.n();
    const auto buffer2Size = signals == mParams.bufferSize ? 0 : mParams.bufferSize;
    auto buffer1Offset = 0L;

    auto stream = *(cudaStream_t*)mGpu.stream();
    do {
        size_t buffer1ToCopy = std::min(mParams.bufferSize, signals - buffer1Offset);
        size_t inputOffsetBuffer1 = buffer1Offset * singleFTPixels;
        gpuErrchk(cudaMemcpyAsync(d_fftBuffer1, h_FTs + inputOffsetBuffer1,
                buffer1ToCopy * singleFTBytes, cudaMemcpyHostToDevice, stream));

        // compute inter-buffer correlations
        computeCorrelations(d_fftBuffer1, buffer1ToCopy, buffer1Offset,
                d_fftBuffer1, buffer1ToCopy, buffer1Offset,
                h_pos, maxDist);
        size_t buffer2Offset = buffer1Offset + buffer1ToCopy;
        while (buffer2Offset < signals) {
            // copy other buffer
            size_t buffer2ToCopy = std::min(buffer2Size, signals - buffer2Offset);
            size_t inputOffsetBuffer2 = buffer2Offset * singleFTPixels;
            gpuErrchk(cudaMemcpyAsync(d_fftBuffer2, h_FTs + inputOffsetBuffer2,
                    buffer2ToCopy * singleFTBytes, cudaMemcpyHostToDevice, stream));

            computeCorrelations(d_fftBuffer1, buffer1ToCopy, buffer1Offset,
                    d_fftBuffer2, buffer2ToCopy, buffer2Offset,
                    h_pos, maxDist);
            buffer2Offset += buffer2ToCopy;
        }
        buffer1Offset += buffer1ToCopy;
    } while (buffer1Offset < signals);
    gpuErrchk(cudaPeekAtLastError());
}

template<typename T>
void CUDAFlexAlignCorrelate<T>::computeCorrelations(
        std::complex<T> *d_in1, size_t in1Size, size_t in1Offset, 
        std::complex<T> *d_in2, size_t in2Size, size_t in2Offset,
        T *h_pos, float maxDist) {
    const auto isWithin = d_in1 == d_in2; // correlations are computed within the same buffer
    const auto settings = getSettings();
    auto stream = *(cudaStream_t*)mGpu.stream();
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
        dim3 dimGridCorr(ceil(settings.fDim().x()/(float)dimBlock.x), ceil(settings.fDim().y()/(float)dimBlock.y));

    auto counter = 0L;
    int origI = 0;
    int origJ = isWithin ? 0 : -1; // kernel must skip first iteration
    for (int i = 0; i < in1Size; i++) {
        for (int j = isWithin ? i + 1 : 0; j < in2Size; j++) {
            counter++;
            bool isLastIIter = isWithin ? (i == in1Size - 2) : (i == in1Size -1);
            if (counter == settings.batch() || (isLastIIter && (j == in2Size -1)) ) {
                // kernel must perform last iteration
                // compute correlation from input buffers. Result are FFT signals with low frequencies in the middle
                if (std::is_same<T, float>::value) {
                correlate<<<dimGridCorr, dimBlock, 0, stream>>>(
                        reinterpret_cast<float2*>(d_in1), reinterpret_cast<float2*>(d_in2),
                        reinterpret_cast<float2*>(d_ffts), settings.fDim().x(), settings.fDim().y(),
                        isWithin, origI, i, origJ, j, in2Size);
                } else {
                    throw std::logic_error("unsupported type");
                }
                // convert FD to SD
                cudaMemsetAsync(d_imgs, 0, settings.sBytesBatch(), stream); // clean it to ensure that IFFT is properly computed
                CudaFFT<T>::ifft(*reinterpret_cast<cufftHandle *>(mIT), d_ffts, d_imgs);
                cudaMemsetAsync(d_indices, 0, counter * sizeof(float), stream); // clean it
                // look for maxima - results are indices of the max position
                ExtremaFinder::CudaExtremaFinder<T>::sFindMax2DAroundCenter(mGpu, settings.sDim().copyForN(counter), d_imgs, d_indices, nullptr, maxDist);
                // now convert indices to float positions - we reuse the same memory block, but since we had images there, we should have more then enough space for that
                cudaMemsetAsync(d_positions, 0, counter * sizeof(float) * 2, stream); // clean it
                ExtremaFinder::CudaExtremaFinder<T>::sRefineLocation(mGpu, settings.sDim().copyForN(counter), d_indices, d_positions, d_imgs);
                // send results to host
                copyInRightOrder(h_pos, isWithin, origI, i, origJ, j, in2Size, in1Offset, in2Offset);
                origI = i;
                origJ = j;
                counter = 0;
            }
        }
    }
}

template class CUDAFlexAlignCorrelate<float>;