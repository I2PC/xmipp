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


template<typename T>
size_t GlobAlignmentData<T>::estimateBytes(const FFTSettings<T> &in, const FFTSettings<T> &out) {
    size_t planSize = CudaFFT<T>().estimatePlanBytes(in);
    size_t aux = std::max(in.sBytesBatch(), out.fBytesSingle() * in.batch());
    size_t fd = in.fBytesBatch();
    return planSize + aux + fd;
}
 

template<typename T>
void GlobAlignmentData<T>::alloc(const FFTSettings<T> &in, const FFTSettings<T> &out, const GPU &gpu) {
    plan = CudaFFT<T>::createPlan(gpu, in);
    // here we will first frames to be converted to FD, and then scaled frames in FD
    d_aux = reinterpret_cast<T*>(BasicMemManager::instance().get(std::max(in.sBytesBatch(), out.fBytesSingle() * in.batch()), MemType::CUDA));
    d_ft = reinterpret_cast<std::complex<T>*>(BasicMemManager::instance().get(in.fBytesBatch(), MemType::CUDA));
}

template<typename T>
void GlobAlignmentData<T>::release() {
    CudaFFT<T>::release(plan);
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
    size_t pixelsPerImage =  xDim * yDim;
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
void copyInRightOrderNew(T* d_imgs, T* h_imgs, int xDim, int yDim, bool isWithin,
        int iStart, int iStop, int jStart, int jStop, size_t jSize,
        size_t offset1, size_t offset2, size_t maxImgs, const GPU &gpu) {
    size_t pixelsPerImage =  xDim * yDim;
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
                gpuErrchk(cudaMemcpyAsync(h_imgs + (pixelsPerImage * (imgsInPreviousLayers + imgsInCurrentLayer)),
                    d_imgs + (counter * pixelsPerImage),
                    toCopy * pixelsPerImage * sizeof(T),
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

template void computeCorrelations<float>(size_t centerSize, size_t noOfImgs,
    const FFTSettings<float> &settings, std::complex<float>* h_FFTs,
     size_t maxFFTsInBuffer, float *result, CorrelationData<float> &aux, const GPU &gpu);
template<typename T>
void computeCorrelations(size_t centerSize, size_t noOfImgs, const FFTSettings<T> &settings, std::complex<T> *h_FFTs,
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
        computeCorrelationsNew(centerSize, noOfImgs, aux.d_fftBuffer1, buffer1ToCopy,
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

            computeCorrelationsNew(centerSize, noOfImgs, aux.d_fftBuffer1, buffer1ToCopy,
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
void computeCorrelationsNew(size_t centerSize, size_t noOfImgs, 
        void* d_in1, size_t in1Size, void* d_in2, size_t in2Size,
        size_t in1Offset, size_t in2Offset,
        std::complex<T> *d_ffts, T *d_imgs, cufftHandle &handler,
        T *result, const FFTSettings<T> &settings, const GPU &gpu) {
    bool isWithin = d_in1 == d_in2; // correlation is done within the same buffer

    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGridCorr(ceil(settings.fDim().x()/(float)dimBlock.x), ceil(settings.fDim().y()/(float)dimBlock.y));
    dim3 dimGridCrop(ceil(centerSize/(float)dimBlock.x), ceil(centerSize/(float)dimBlock.y));

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
                // convert FFTs to space domain
                CudaFFT<T>::ifft(handler, d_ffts, d_imgs);
                // crop images in space domain, use memory for FFT to avoid realocation
                cropSquareInCenter<<<dimGridCrop, dimBlock, 0, stream>>>((T*)d_imgs,
                        (T*)d_ffts, settings.sDim().x(), settings.sDim().y(),
                        counter, centerSize, centerSize);
                copyInRightOrderNew((T*)d_ffts, result,
                        centerSize, centerSize,
                        isWithin, origI, i, origJ, j, in2Size, in1Offset, in2Offset, noOfImgs, gpu);
                origI = i;
                origJ = j;
                counter = 0;
                batchCounter++;
            }
        }
    }
}


template void computeCorrelations<float>(size_t centerSize, size_t noOfImgs,
    std::complex<float>* h_FFTs,
    int fftSizeX, int imgSizeX, int fftSizeY, size_t maxFFTsInBuffer,
    int fftBatchSize, float*& result);
template<typename T>
void computeCorrelations(size_t centerSize, size_t noOfImgs, std::complex<T>* h_FFTs,
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
        computeCorrelations(centerSize, noOfImgs, d_fftBuffer1, buffer1ToCopy,
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

            computeCorrelations(centerSize, noOfImgs, d_fftBuffer1, buffer1ToCopy,
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
void computeCorrelations(size_t centerSize, int noOfImgs,
        void* d_in1, size_t in1Size, void* d_in2, size_t in2Size,
        int fftBatchSize, size_t in1Offset, size_t in2Offset,
        GpuMultidimArrayAtGpu<std::complex<T> >& ffts,
            GpuMultidimArrayAtGpu<T>& imgs, mycufftHandle& handler,
            T*& result) {
    bool isWithin = d_in1 == d_in2; // correlation is done within the same buffer

    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGridCorr(ceil(ffts.Xdim/(float)dimBlock.x), ceil(ffts.Ydim/(float)dimBlock.y));
    dim3 dimGridCrop(ceil(centerSize/(float)dimBlock.x), ceil(centerSize/(float)dimBlock.y));

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
                // crop images in space domain, use memory for FFT to avoid realocation
                cropSquareInCenter<<<dimGridCrop, dimBlock>>>((T*)imgs.d_data,
                        (T*)ffts.d_data, imgs.Xdim, imgs.Ydim,
                        counter, centerSize, centerSize);
                copyInRightOrder((T*)ffts.d_data, result,
                        centerSize, centerSize,
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