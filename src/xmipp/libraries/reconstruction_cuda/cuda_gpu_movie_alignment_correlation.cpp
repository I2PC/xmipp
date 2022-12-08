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

#include <cuda_runtime_api.h>
#include "reconstruction_cuda/cuda_asserts.h"
#include "cuda_gpu_movie_alignment_correlation.h"
#include "reconstruction_cuda/cuda_basic_math.h"
#include "cuda_gpu_movie_alignment_correlation_kernels.cu"
#include "reconstruction_adapt_cuda/basic_mem_manager.h"
#include "cuda_single_extrema_finder.h"



template<typename T>
size_t GlobAlignmentData<T>::estimateBytes(const FFTSettings<T> &in, const std::optional<FFTSettings<T>> &bin, const FFTSettings<T> &out) {
    size_t planSize = CudaFFT<T>().estimatePlanBytes(in);
    size_t aux = std::max(in.sBytesBatch(), out.fBytesSingle() * in.batch());
    size_t fd = in.fBytesBatch();
    if (bin) {
        planSize += CudaFFT<T>().estimatePlanBytes(bin.value());
        aux = std::max(aux, bin.value().fBytesBatch());
        fd = std::max(std::max(fd, bin.value().sBytesBatch()), out.fBytesSingle() * in.batch());
    }
    return planSize + aux + fd;
}
 

template<typename T>
void GlobAlignmentData<T>::alloc(const FFTSettings<T> &in, const std::optional<FFTSettings<T>> &bin, const FFTSettings<T> &out, const GPU &gpu) {
    {
        plan = CudaFFT<T>::createPlan(gpu, in);
        if (bin) {
            plan_bin = CudaFFT<T>::createPlan(gpu, bin.value());
        }
    }
    {// here we will first frames to be converted to FD, and then scaled frames in FD
        auto bytes = std::max(in.sBytesBatch(), out.fBytesSingle() * in.batch());
        if (bin) {
            plan_bin = CudaFFT<T>::createPlan(gpu, bin.value());
            bytes = std::max(bytes, bin.value().fBytesBatch());
        }
        d_aux = reinterpret_cast<T*>(BasicMemManager::instance().get(bytes, MemType::CUDA));
    } 
    {
        auto bytes = in.fBytesBatch();
        if (bin) {
            bytes = std::max(std::max(bytes, bin.value().sBytesBatch()), out.fBytesSingle() * in.batch());
        }
        d_ft = reinterpret_cast<std::complex<T>*>(BasicMemManager::instance().get(bytes, MemType::CUDA));
    }
}

template<typename T>
void GlobAlignmentData<T>::release() {
    CudaFFT<T>::release(plan);
    CudaFFT<T>::release(plan_bin);
    BasicMemManager::instance().give(d_ft);
    BasicMemManager::instance().give(d_aux);
}

template<typename T>
void CorrelationData<T>::alloc(const FFTSettings<T> &settings, size_t bufferSize, const GPU &gpu) {
    plan = CudaFFT<T>::createPlan(gpu, settings);
    d_ffts = reinterpret_cast<std::complex<T>*>(BasicMemManager::instance().get(settings.fBytesBatch(), MemType::CUDA));
    d_imgs = reinterpret_cast<T*>(BasicMemManager::instance().get(settings.sBytesBatch(), MemType::CUDA));
    d_fftBuffer1 = reinterpret_cast<std::complex<T>*>(BasicMemManager::instance().get(settings.fBytesSingle() * bufferSize, MemType::CUDA));
    if (bufferSize != settings.sDim().n()) {
        d_fftBuffer2 = reinterpret_cast<std::complex<T>*>(BasicMemManager::instance().get(settings.fBytesSingle() * bufferSize, MemType::CUDA));
    } else {
        d_fftBuffer2 = nullptr;
    }
}

template<typename T>
void CorrelationData<T>::release() {
    CudaFFT<T>::release(plan);
    BasicMemManager::instance().give(d_ffts);
    BasicMemManager::instance().give(d_imgs);
    BasicMemManager::instance().give(d_fftBuffer1);
    if (nullptr != d_fftBuffer2) {
        BasicMemManager::instance().give(d_fftBuffer2);
    }
}


template void performFFTAndScale<float>(float *h_in, const FFTSettings<float> &in, 
    std::complex<float> *h_out, const FFTSettings<float> &out, 
    MultidimArray<float> &filter, const GPU &gpu, GlobAlignmentData<float> &aux);
template <typename T>
void performFFTAndScale(T *h_in, const FFTSettings<T> &in, 
    std::complex<T> *h_out, const FFTSettings<T> &out, 
    MultidimArray<T> &filter, const GPU &gpu, GlobAlignmentData<T> &aux)
{
    auto stream = *(cudaStream_t*)gpu.stream();    

    // perform FFT of cropped frames
    gpuErrchk(cudaMemcpyAsync(aux.d_aux, h_in, in.sBytesBatch(), cudaMemcpyHostToDevice, stream));
    // gpuErrchk(cudaMemcpy(d_aux, h_in, in.sBytesBatch(), cudaMemcpyHostToDevice));
    CudaFFT<T>::fft(*aux.plan, aux.d_aux, aux.d_ft);
    // scale frames in FD
    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGrid(ceil(out.fDim().x()/(float)dimBlock.x), ceil(out.fDim().y()/(float)dimBlock.y));
    scaleFFT2D(&dimGrid, &dimBlock,
        aux.d_ft, (std::complex<T>*)aux.d_aux,
        in.batch(), in.fDim().x(), in.fDim().y(),
        out.fDim().x(), out.fDim().y(),
        filter.data, 1.f/in.sDim().xy(), false, gpu);
    gpuErrchk( cudaPeekAtLastError() );
    // copy data out
    gpuErrchk(cudaMemcpyAsync(h_out, aux.d_aux, out.fBytesSingle() * in.batch(), cudaMemcpyDeviceToHost, stream));
    // gpuErrchk(cudaMemcpy(h_out, d_aux, out.fBytesSingle() * in.batch(), cudaMemcpyDeviceToHost));
    
    gpuErrchk(cudaPeekAtLastError());
}

template void runFFTBinScale<float>(float *h_in, const FFTSettings<float> &in, 
    float *h_outBin, const FFTSettings<float> &bin, 
    std::complex<float> *h_out, const FFTSettings<float> &out, 
    MultidimArray<float> &filter, const GPU &gpu, GlobAlignmentData<float> &aux);
template <typename T>
void runFFTBinScale(T *h_in, const FFTSettings<T> &in, 
    T *h_outBin,
    const FFTSettings<T> &bin, 
    std::complex<T> *h_out, const FFTSettings<T> &out, 
    MultidimArray<T> &filter, const GPU &gpu, GlobAlignmentData<T> &aux)
{
    auto stream = *(cudaStream_t*)gpu.stream();    
    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
    {
        // perform FFT
        auto *d_in = aux.d_aux;
        auto *d_out = aux.d_ft;
        gpuErrchk(cudaMemcpyAsync(d_in, h_in, in.sBytesBatch(), cudaMemcpyHostToDevice, stream));
        CudaFFT<T>::fft(*aux.plan, d_in, d_out);
    }
    {
        auto *d_in = aux.d_ft;
        auto *d_out = reinterpret_cast<std::complex<T>*>(aux.d_aux);
        // scale frames in FD to perform binning
        dim3 dimGrid(ceil(bin.fDim().x()/(float)dimBlock.x), ceil(bin.fDim().y()/(float)dimBlock.y));
        auto hack = 0.f; // no idea why I can't just pass nullptr
        scaleFFT2D(&dimGrid, &dimBlock,
            d_in, d_out,
            in.batch(), in.fDim().x(), in.fDim().y(),
            bin.fDim().x(), bin.fDim().y(),
            &hack, 1.f/in.sDim().xy(), false, gpu);
        // scaleFFT2D(&dimGrid, &dimBlock,
            // d_in, aux.d_ft,
            // bin.batch(), bin.fDim().x(), bin.fDim().y(),
            // out.fDim().x(), out.fDim().y(),
            // filter.data, 1.f/bin.sDim().xy(), false, gpu);
        gpuErrchk( cudaPeekAtLastError() );
    }
    {
        auto *d_in = reinterpret_cast<std::complex<T>*>(aux.d_aux);
        // perform IFT and send data back
        auto *d_out = reinterpret_cast<T*>(aux.d_ft);
        CudaFFT<T>::ifft(*aux.plan_bin, d_in, d_out);
        gpuErrchk(cudaMemcpyAsync(h_outBin, d_out, bin.sBytesBatch(), cudaMemcpyDeviceToHost, stream));

        // perform another crop to scale frames
        dim3 dimGrid(ceil(out.fDim().x()/(float)dimBlock.x), ceil(out.fDim().y()/(float)dimBlock.y));
        scaleFFT2D(&dimGrid, &dimBlock,
            d_in, aux.d_ft,
            bin.batch(), bin.fDim().x(), bin.fDim().y(),
            out.fDim().x(), out.fDim().y(),
            filter.data, 1.f/bin.sDim().xy(), false, gpu);
        gpuErrchk( cudaPeekAtLastError() );

        // copy data out
        gpuErrchk(cudaMemcpyAsync(h_out, d_out, out.fBytesSingle() * bin.batch(), cudaMemcpyDeviceToHost, stream));
    }
}

template void performFFTAndScale<float>(float* inOutData, int noOfImgs, int inX,
    int inY, int inBatch, int outFFTX, int outY, MultidimArray<float> &filter);
template<typename T>
void performFFTAndScale(T* inOutData, int noOfImgs, int inX, int inY,
        int inBatch, int outFFTX, int outY, MultidimArray<T> &filter) {
    assert((inX / 2 + 1) >= outFFTX);
    assert(inY >= outY);
    mycufftHandle handle;
    size_t counter = 0;
    std::complex<T>* h_result = (std::complex<T>*)inOutData;
    // since memory where the input images are will be reused, we have to make
    // sure it's big enough
    size_t bytesImgs = inX * inY * inBatch * sizeof(T);
    size_t bytesFFTs = outFFTX * outY * inBatch * sizeof(T) * 2; // complex
    T* d_data = NULL;

    auto h_tmpStore = reinterpret_cast<std::complex<T>*>(BasicMemManager::instance().get(outFFTX * outY * inBatch * sizeof(std::complex<T>), MemType::CUDA_HOST));
    gpuErrchk(cudaMalloc(&d_data, std::max(bytesImgs, bytesFFTs)));
    GpuMultidimArrayAtGpu<T> imagesGPU(inX, inY, 1, inBatch, d_data);
    GpuMultidimArrayAtGpu<std::complex<T> > resultingFFT;

    auto loadImgs = [&](){
        int imgToProcess = std::min(inBatch, noOfImgs - static_cast<int>(counter));
        if (imgToProcess < 1) return (size_t)0;
        T* h_imgLoad = inOutData + counter * inX * inY;
        size_t bytesIn = imgToProcess * inX * inY * sizeof(T);
        gpuErrchk(cudaMemcpy(imagesGPU.d_data, h_imgLoad, bytesIn, cudaMemcpyHostToDevice));
        size_t bytesOut = imgToProcess * outFFTX * outY * sizeof(std::complex<T>);
        return bytesOut;
    };

    while (counter < noOfImgs) {
        size_t bytesToCopyFromGPU;
        if (0 == counter) { // first iteration
            bytesToCopyFromGPU = loadImgs();
        }
        // perform FFT (out of place) and scale
        imagesGPU.fft(resultingFFT, handle);
        dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
        dim3 dimGrid(ceil(outFFTX/(float)dimBlock.x), ceil(outY/(float)dimBlock.y));
        scaleFFT2D(&dimGrid, &dimBlock,
            (std::complex<T>*)resultingFFT.d_data,
            (std::complex<T>*)imagesGPU.d_data,
            inBatch, resultingFFT.Xdim, resultingFFT.Ydim,
            outFFTX, outY, filter.data, 1.f/imagesGPU.yxdim, false);
        gpuErrchk( cudaPeekAtLastError() );

        std::complex<T>* h_imgStore = h_result + counter * outFFTX * outY;
        counter += inBatch;
        // because the spatial data are consecutive, and FFT might be actually bigger than spatial data
        // (i.e. they can overlap), we need to copy processed data to extra location
        gpuErrchk(cudaMemcpy((void*)h_tmpStore, (void*)imagesGPU.d_data, bytesToCopyFromGPU, cudaMemcpyDeviceToHost));
        // get new data from CPU. Now we can overwrite the location, as we have the results already
        size_t tmp = loadImgs();
        // copy results back to CPU, to the original memory location, consecutive
        memcpy(h_imgStore, h_tmpStore, bytesToCopyFromGPU);
        // and update the no of bytes we should process next iteration
        bytesToCopyFromGPU = tmp;
    }
    handle.clear();
    BasicMemManager::instance().give(h_tmpStore);
}

template<>
void scaleFFT2D(void* dimGrid, void* dimBlock, const std::complex<float>* d_inFFT, std::complex<float>* d_outFFT, int noOfFFT, size_t inFFTX, size_t inFFTY, size_t outFFTX, size_t outFFTY,
    float* d_filter, float normFactor, bool center, const GPU &gpu) {
    auto stream = *(cudaStream_t*)gpu.stream();
    if (NULL == d_filter) {
        if ((float)1 == normFactor) {
            if (center) {
                scaleFFT2DKernel<float2, float, false, false, true>
                    <<<*(dim3*)dimGrid, *(dim3*)dimBlock, 0, stream>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, NULL, 1.f);
            } else {
                scaleFFT2DKernel<float2, float, false, false, false>
                    <<<*(dim3*)dimGrid, *(dim3*)dimBlock, 0, stream>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, NULL, 1.f);
            }
        } else { // normalize
            if (center) {
                scaleFFT2DKernel<float2, float, false, true, true>
                    <<<*(dim3*)dimGrid, *(dim3*)dimBlock, 0, stream>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, NULL, normFactor);
            } else {
                scaleFFT2DKernel<float2, float, false, true, false>
                    <<<*(dim3*)dimGrid, *(dim3*)dimBlock, 0, stream>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, NULL, normFactor);
            }
        }
    } else { // apply filter (on output)
        if ((float)1 == normFactor) {
            if (center) {
                scaleFFT2DKernel<float2, float, true, false, true>
                    <<<*(dim3*)dimGrid, *(dim3*)dimBlock, 0, stream>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, d_filter, 1.f);
            } else {
                scaleFFT2DKernel<float2, float, true, false, false>
                    <<<*(dim3*)dimGrid, *(dim3*)dimBlock, 0, stream>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, d_filter, 1.f);
            }
        } else { // normalize
            if (center) {
                scaleFFT2DKernel<float2, float, true, true, true>
                    <<<*(dim3*)dimGrid, *(dim3*)dimBlock, 0, stream>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, d_filter, normFactor);
            } else {
                scaleFFT2DKernel<float2, float, true, true, false>
                    <<<*(dim3*)dimGrid, *(dim3*)dimBlock, 0, stream>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, d_filter, normFactor);
            }
        }
    }
    gpuErrchk(cudaPeekAtLastError());
}

template<>
void scaleFFT2D(void* dimGrid, void* dimBlock, const std::complex<float>* d_inFFT, std::complex<float>* d_outFFT, int noOfFFT, size_t inFFTX, size_t inFFTY, size_t outFFTX, size_t outFFTY,
    float* d_filter, float normFactor, bool center) {
    if (NULL == d_filter) {
        if ((float)1 == normFactor) {
            if (center) {
                scaleFFT2DKernel<float2, float, false, false, true>
                    <<<*(dim3*)dimGrid, *(dim3*)dimBlock>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, NULL, 1.f);
            } else {
                scaleFFT2DKernel<float2, float, false, false, false>
                    <<<*(dim3*)dimGrid, *(dim3*)dimBlock>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, NULL, 1.f);
            }
        } else { // normalize
            if (center) {
                scaleFFT2DKernel<float2, float, false, true, true>
                    <<<*(dim3*)dimGrid, *(dim3*)dimBlock>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, NULL, normFactor);
            } else {
                scaleFFT2DKernel<float2, float, false, true, false>
                    <<<*(dim3*)dimGrid, *(dim3*)dimBlock>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, NULL, normFactor);
            }
        }
    } else { // apply filter (on output)
        if ((float)1 == normFactor) {
            if (center) {
                scaleFFT2DKernel<float2, float, true, false, true>
                    <<<*(dim3*)dimGrid, *(dim3*)dimBlock>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, d_filter, 1.f);
            } else {
                scaleFFT2DKernel<float2, float, true, false, false>
                    <<<*(dim3*)dimGrid, *(dim3*)dimBlock>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, d_filter, 1.f);
            }
        } else { // normalize
            if (center) {
                scaleFFT2DKernel<float2, float, true, true, true>
                    <<<*(dim3*)dimGrid, *(dim3*)dimBlock>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, d_filter, normFactor);
            } else {
                scaleFFT2DKernel<float2, float, true, true, false>
                    <<<*(dim3*)dimGrid, *(dim3*)dimBlock>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, d_filter, normFactor);
            }
        }
    }
    gpuErrchk(cudaPeekAtLastError());
}


template<typename T>
void copyInRightOrder(T* d_imgs, T* h_imgs, int xDim, int yDim, bool isWithin,
        int iStart, int iStop, int jStart, int jStop, size_t jSize,
        size_t offset1, size_t offset2, size_t maxImgs) {
    size_t pixelsPerImage =  2;
    // size_t pixelsPerImage =  xDim * yDim;
    size_t counter = 0;
    bool ready = false;
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
                size_t toCopy = jSize - j;
                // imagine correlation in layers, correlation of 0th img with other is first layer, 1st with other is second etc
                // compute sum of images in complete layers
                size_t imgsInPreviousLayers = (((maxImgs - 1) + (maxImgs - actualI)) * (actualI)) / 2;
                size_t imgsInCurrentLayer = actualJ - actualI - 1;
                gpuErrchk(cudaMemcpy(h_imgs + (pixelsPerImage * (imgsInPreviousLayers + imgsInCurrentLayer)),
                    d_imgs + (counter * pixelsPerImage),
                    toCopy * pixelsPerImage * sizeof(T),
                    cudaMemcpyDeviceToHost));
                // gpuErrchk(cudaMemcpy(h_imgs + 1770+(pixelsPerImage * (imgsInPreviousLayers + imgsInCurrentLayer)),
                //     d_imgs + (counter * pixelsPerImage),
                //     toCopy * pixelsPerImage * sizeof(T),
                //     cudaMemcpyDeviceToHost));
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
void copyInRightOrderNew(T* d_pos, T* h_pos, bool isWithin,
        int iStart, int iStop, int jStart, int jStop, size_t jSize,
        size_t offset1, size_t offset2, size_t maxImgs, const GPU &gpu) {
    size_t coordinates = 2;
    size_t counter = 0;
    bool ready = false;
    auto stream = *(cudaStream_t*)gpu.stream();
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
                size_t toCopy = jSize - j;
                // imagine correlation in layers, correlation of 0th img with other is first layer, 1st with other is second etc
                // compute sum of images in complete layers
                size_t imgsInPreviousLayers = (((maxImgs - 1) + (maxImgs - actualI)) * (actualI)) / 2;
                size_t imgsInCurrentLayer = actualJ - actualI - 1;
                gpuErrchk(cudaMemcpyAsync(h_pos + (coordinates * (imgsInPreviousLayers + imgsInCurrentLayer)),
                    d_pos + (counter * coordinates),
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

template void computeCorrelations<float>(float maxDist, size_t noOfImgs,
    const FFTSettings<float> &settings, std::complex<float>* h_FFTs,
     size_t maxFFTsInBuffer, float *result, CorrelationData<float> &aux, const GPU &gpu);
template<typename T>
void computeCorrelations(float maxDist, size_t noOfImgs, const FFTSettings<T> &settings, std::complex<T> *h_FFTs,
        size_t maxFFTsInBuffer, T *result, CorrelationData<T> &aux, const GPU &gpu) {
    size_t singleFFTPixels = settings.fDim().xy();
    size_t singleFFTBytes = settings.fBytesSingle();

    size_t buffer1Size = maxFFTsInBuffer;
    size_t buffer2Size = noOfImgs == maxFFTsInBuffer ? 0 : maxFFTsInBuffer;
    size_t buffer1Offset = 0;

    auto stream = *(cudaStream_t*)gpu.stream();
    do {
        size_t buffer1ToCopy = std::min(buffer1Size, noOfImgs - buffer1Offset);
        size_t inputOffsetBuffer1 = buffer1Offset * singleFFTPixels;
        gpuErrchk(cudaMemcpyAsync(aux.d_fftBuffer1, h_FFTs + inputOffsetBuffer1,
                buffer1ToCopy * singleFFTBytes, cudaMemcpyHostToDevice, stream));

        // compute inter-buffer correlations
        computeCorrelationsNew(maxDist, noOfImgs, aux.d_fftBuffer1, buffer1ToCopy,
                aux.d_fftBuffer1, buffer1ToCopy,
                buffer1Offset, buffer1Offset, aux.d_ffts, aux.d_imgs,
                *aux.plan, result, settings, gpu);
        size_t buffer2Offset = buffer1Offset + buffer1ToCopy;
        while (buffer2Offset < noOfImgs) {
            // copy other buffer
            size_t buffer2ToCopy = std::min(buffer2Size, noOfImgs - buffer2Offset);
            size_t inputOffsetBuffer2 = buffer2Offset * singleFFTPixels;
            gpuErrchk(cudaMemcpyAsync(aux.d_fftBuffer2, h_FFTs + inputOffsetBuffer2,
                    buffer2ToCopy * singleFFTBytes, cudaMemcpyHostToDevice, stream));

            computeCorrelationsNew(maxDist, noOfImgs, aux.d_fftBuffer1, buffer1ToCopy,
                    aux.d_fftBuffer2, buffer2ToCopy,
                    buffer1Offset, buffer2Offset, aux.d_ffts, aux.d_imgs,
                    *aux.plan, result, settings, gpu);

            buffer2Offset += buffer2ToCopy;
        }

        buffer1Offset += buffer1ToCopy;

    } while (buffer1Offset < noOfImgs);
    gpuErrchk(cudaPeekAtLastError());
}

template<typename T>
void computeCorrelationsNew(float maxDist, size_t noOfImgs, 
        void* d_in1, size_t in1Size, void* d_in2, size_t in2Size,
        size_t in1Offset, size_t in2Offset,
        std::complex<T> *d_ffts, T *d_imgs, cufftHandle &handler,
        T *result, const FFTSettings<T> &settings, const GPU &gpu) {
    bool isWithin = d_in1 == d_in2; // correlation is done within the same buffer

    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGridCorr(ceil(settings.fDim().x()/(float)dimBlock.x), ceil(settings.fDim().y()/(float)dimBlock.y));

    auto stream = *(cudaStream_t*)gpu.stream();

    size_t batchCounter = 0;
    size_t counter = 0;
    int origI = 0;
    int origJ = isWithin ? 0 : -1; // kernel must skip first iteration
    for (int i = 0; i < in1Size; i++) {
        for (int j = isWithin ? i + 1 : 0; j < in2Size; j++) {
            counter++;
            bool isLastIIter = isWithin ? (i == in1Size - 2) : (i == in1Size -1);
            if (counter == settings.batch() || (isLastIIter && (j == in2Size -1)) ) {
                // kernel must perform last iteration
                // compute correlation from input buffers. Result are FFT images
                if (std::is_same<T, float>::value) {
                correlate<<<dimGridCorr, dimBlock, 0, stream>>>((float2*)d_in1, (float2*)d_in2,
                        (float2*)d_ffts, settings.fDim().x(), settings.fDim().y(),
                        isWithin, origI, i, origJ, j, in2Size);
                } else {
                    throw std::logic_error("unsupported type");
                }
                gpuErrchk(cudaPeekAtLastError());
                // convert FFTs to space domain
                CudaFFT<T>::ifft(handler, d_ffts, d_imgs);
                // look for maxima - results are indices of the max position
                auto *indices = reinterpret_cast<float*>(d_ffts);
                ExtremaFinder::CudaExtremaFinder<T>::sFindMax2DAroundCenter(gpu, settings.sDim(), d_imgs, indices, nullptr, maxDist);
                // now convert indices to float positions - we reuse the same memory block, but since we had images there, we should have more then enough space for that
                auto *positions = reinterpret_cast<float*>(d_ffts) + (noOfImgs * (noOfImgs - 1) / 2);
                ExtremaFinder::CudaExtremaFinder<T>::sRefineLocation(gpu, settings.sDim(), indices, positions, d_imgs);
                copyInRightOrderNew(positions, result,
                        isWithin, origI, i, origJ, j, in2Size, in1Offset, in2Offset, noOfImgs, gpu);
                origI = i;
                origJ = j;
                counter = 0;
                batchCounter++;
            }
        }
    }
}


template void computeCorrelations<float>(float maxDist, size_t noOfImgs,
    std::complex<float>* h_FFTs,
    int fftSizeX, int imgSizeX, int fftSizeY, size_t maxFFTsInBuffer,
    int fftBatchSize, float*& result);
template<typename T>
void computeCorrelations(float maxDist, size_t noOfImgs, std::complex<T>* h_FFTs,
        int fftSizeX, int imgSizeX, int fftSizeY, size_t maxFFTsInBuffer,
        int fftBatchSize, T*& result) {

    GpuMultidimArrayAtGpu<std::complex<T> > ffts(fftSizeX, fftSizeY, 1, fftBatchSize);
    GpuMultidimArrayAtGpu<T> imgs(imgSizeX, fftSizeY, 1, fftBatchSize);
    mycufftHandle myhandle;

    int device = -1;
    gpuErrchk(cudaGetDevice(&device));
    
    size_t singleFFTPixels = fftSizeX * fftSizeY;
    size_t singleFFTBytes = singleFFTPixels * sizeof(T) * 2;

    size_t buffer1Size = std::min(maxFFTsInBuffer, noOfImgs);
    void* d_fftBuffer1 = BasicMemManager::instance().get(buffer1Size * singleFFTBytes, MemType::CUDA);

    size_t buffer2Size = std::max((size_t)0,
            std::min(maxFFTsInBuffer, noOfImgs - buffer1Size));
    void* d_fftBuffer2 = BasicMemManager::instance().get(buffer2Size * singleFFTBytes, MemType::CUDA);

    size_t buffer1Offset = 0;
    do {
        size_t buffer1ToCopy = std::min(buffer1Size, noOfImgs - buffer1Offset);
        size_t inputOffsetBuffer1 = buffer1Offset * singleFFTPixels;
        gpuErrchk(cudaMemcpy(d_fftBuffer1, h_FFTs + inputOffsetBuffer1,
                buffer1ToCopy * singleFFTBytes, cudaMemcpyHostToDevice));

        // compute inter-buffer correlations
        computeCorrelations(maxDist, noOfImgs, d_fftBuffer1, buffer1ToCopy,
                d_fftBuffer1, buffer1ToCopy,
                fftBatchSize, buffer1Offset, buffer1Offset, ffts, imgs,
                myhandle, result);
        size_t buffer2Offset = buffer1Offset + buffer1ToCopy;
        while (buffer2Offset < noOfImgs) {
            // copy other buffer
            size_t buffer2ToCopy = std::min(buffer2Size, noOfImgs - buffer2Offset);
            size_t inputOffsetBuffer2 = buffer2Offset * singleFFTPixels;
            gpuErrchk(cudaMemcpy(d_fftBuffer2, h_FFTs + inputOffsetBuffer2,
                    buffer2ToCopy * singleFFTBytes, cudaMemcpyHostToDevice));

            computeCorrelations(maxDist, noOfImgs, d_fftBuffer1, buffer1ToCopy,
                    d_fftBuffer2, buffer2ToCopy,
                    fftBatchSize, buffer1Offset, buffer2Offset, ffts, imgs,
                    myhandle, result);

            buffer2Offset += buffer2ToCopy;
        }

        buffer1Offset += buffer1ToCopy;

    } while (buffer1Offset < noOfImgs);

    BasicMemManager::instance().give(d_fftBuffer1);
    BasicMemManager::instance().give(d_fftBuffer2);

    // cudaFree(d_fftBuffer1);
    // cudaFree(d_fftBuffer2);

    gpuErrchk(cudaPeekAtLastError());
}

template<typename T>
void computeCorrelations(float maxDist, int noOfImgs,
        void* d_in1, size_t in1Size, void* d_in2, size_t in2Size,
        int fftBatchSize, size_t in1Offset, size_t in2Offset,
        GpuMultidimArrayAtGpu<std::complex<T> >& ffts,
            GpuMultidimArrayAtGpu<T>& imgs, mycufftHandle& handler,
            T*& result) {
    bool isWithin = d_in1 == d_in2; // correlation is done within the same buffer

    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGridCorr(ceil(ffts.Xdim/(float)dimBlock.x), ceil(ffts.Ydim/(float)dimBlock.y));

auto gpu = GPU();
gpu.set();


    size_t batchCounter = 0;
    size_t counter = 0;
    int origI = 0;
    int origJ = isWithin ? 0 : -1; // kernel must skip first iteration
    for (int i = 0; i < in1Size; i++) {
        for (int j = isWithin ? i + 1 : 0; j < in2Size; j++) {
            counter++;
            bool isLastIIter = isWithin ? (i == in1Size - 2) : (i == in1Size -1);
            if (counter == fftBatchSize || (isLastIIter && (j == in2Size -1)) ) {
                // kernel must perform last iteration
                // compute correlation from input buffers. Result are FFT images
                if (std::is_same<T, float>::value) {
                correlate<<<dimGridCorr, dimBlock>>>((float2*)d_in1, (float2*)d_in2,
                        (float2*)ffts.d_data, ffts.Xdim, ffts.Ydim,
                        isWithin, origI, i, origJ, j, in2Size);
                } else {
                    throw std::logic_error("unsupported type");
                }
                // convert FFTs to space domain
                ffts.ifft(imgs, handler);
                // look for maxima - results are indices of the max position
                auto *indices = reinterpret_cast<float*>(ffts.d_data);
                auto dim = Dimensions(imgs.Xdim, imgs.Ydim, 1, imgs.Ndim);
                ExtremaFinder::CudaExtremaFinder<T>::sFindMax2DAroundCenter(gpu, dim, (T*)imgs.d_data, indices, nullptr, maxDist);
                // now convert indices to float positions - we reuse the same memory block, but since we had images there, we should have more then enough space for that
                auto *positions = reinterpret_cast<float*>(ffts.d_data) + (noOfImgs * (noOfImgs - 1) / 2);
                ExtremaFinder::CudaExtremaFinder<T>::sRefineLocation(gpu, dim, indices, positions, (T*)imgs.d_data);
                copyInRightOrder(positions, result,
                        0, 0, // FIXME remove
                        isWithin, origI, i, origJ, j, in2Size, in1Offset, in2Offset, noOfImgs);
                origI = i;
                origJ = j;
                counter = 0;
                batchCounter++;
            }
        }
    }
}

template class GlobAlignmentData<float>;
template class CorrelationData<float>;