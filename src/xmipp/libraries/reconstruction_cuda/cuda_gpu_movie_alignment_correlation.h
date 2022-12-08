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

#ifndef CUDA_GPU_MOVIE_ALIGNMENT_CORRELATION
#define CUDA_GPU_MOVIE_ALIGNMENT_CORRELATION

#include <optional>
#include "reconstruction/movie_alignment_gpu_defines.h"
#include "cuFFTAdvisor/utils.h"
#include <vector>
#include "reconstruction_cuda/cuda_xmipp_utils.h"
#include "core/multidim_array.h"
#include <type_traits>
#include <stdexcept>
#include <cassert>
#include "data/fft_settings.h"
#include "reconstruction_cuda/gpu.h"
#include "reconstruction_cuda/cuda_fft.h"



template<typename T>
struct GlobAlignmentData {
    void alloc(const FFTSettings<T> &in, const std::optional<FFTSettings<T>> &bin, const FFTSettings<T> &out, const GPU &gpu);
    void release();
    static size_t estimateBytes(const FFTSettings<T> &in, const std::optional<FFTSettings<T>> &bin, const FFTSettings<T> &out);
    T *d_aux;
    std::complex<T> *d_ft;
    cufftHandle *plan;
    cufftHandle *plan_bin;
};

template<typename T>
struct CorrelationData {
    void alloc(const FFTSettings<T> &settings, size_t bufferSize, const GPU &gpu);
    void release();
    T *d_imgs;
    std::complex<T> *d_ffts;
    std::complex<T> *d_fftBuffer1;
    std::complex<T> *d_fftBuffer2;
    cufftHandle *plan;
};

/**@defgroup ProgAGPUMovieAlign Cuda GPU movie Alignment Correlation
   @ingroup ReconsLibrary */
//@{
/**
 * This function performs FFT and scale (with filter) of the input images
 * @param inOutData input images. Scaled images will be stored here. Must be
 * big enough to store all output images in FT
 * @param noOfImgs no of the input images
 * @param inX X dim of the input image, space domain
 * @param inY Y dim of the input image, space domain
 * @param inBatch no of the images to process in batch (FFT)
 * @param outFFTX X dim of the output image, freq. domain
 * @param outY Y dim of the output image, freq. domain
 * @param filter to apply per each output image (must be size of output)
 */
template<typename T>
void performFFTAndScale(T* inOutData, int noOfImgs, int inX, int inY,
        int inBatch, int outFFTX, int outY, MultidimArray<T> &filter);

template<typename T>
void runFFTBinScale(T *h_in, const FFTSettings<T> &in, 
    T *h_outBin, const FFTSettings<T> &bin, 
    std::complex<T> *h_out, const FFTSettings<T> &out, 
    MultidimArray<T> &filter, const GPU &gpu, GlobAlignmentData<T> &aux);

/**
 * This function performs FFT and scale (with filter) of the input images.
 * Warning: when this funcion returns, some memory transfers still might be pending.
 * Make sure that you wait for them!
 * @param h_in input frames on host
 * @param in settings describing the input data
 * @param h_out result will be stored here
 * @param out settings describing the output data
 * @param filter to apply per each output image (must be size of output)
 * @param gpu where the operation should be performed
 * @param aux memory storages
 */
template <typename T>
void performFFTAndScale(T *h_in, const FFTSettings<T> &in, 
    std::complex<T> *h_out, const FFTSettings<T> &out, 
    MultidimArray<T> &filter, const GPU &gpu, GlobAlignmentData<T> &aux);

/**
 * Perform scale of the Fourier domain. Possibly with filtering, normalization and
 * centering
 * @param dimGrid kernel grid (must be of type dim3)
 * @param dimBlock kernel block (must be of type dim3)
 * @param d_inFFT input Fourier data (GPU)
 * @param d_outFFT output Fourier data (GPU)
 * @param noOfFFT no of input images
 * @param inFFTX X dim of the input Fourier
 * @param inFFTY Y dim of the input Fourier
 * @param outFFTX X dim of the output Fourier
 * @param outFFTY Y dim of the output Fourier
 * @param d_filter to be used, must be of the input size. Can be NULL
 * @param normFactor normalization factor to be used (typically 1/X*Y).
 * Use 1 to avoid normalization
 * @param center the image (after IFFT) using multiplication. Works only for
 * inputs of even size.
 */
template<typename T>
void scaleFFT2D(void* dimGrid, void* dimBlock,
        const std::complex<T>* d_inFFT, std::complex<T>* d_outFFT,
        int noOfFFT, size_t inFFTX, size_t inFFTY, size_t outFFTX, size_t outFFTY,
        T* d_filter, T normFactor, bool center);
template<typename T>
void scaleFFT2D(void* dimGrid, void* dimBlock,
        const std::complex<T>* d_inFFT, std::complex<T>* d_outFFT,
        int noOfFFT, size_t inFFTX, size_t inFFTY, size_t outFFTX, size_t outFFTY,
        T* d_filter, T normFactor, bool center, const GPU &gpu);

/**
 * Function will copy correlation images from GPU in proper order
 * @param d_imgs input images (GPU)
 * @param h_imgs output images (CPU)
 * @param xDim X size of the image
 * @param yDim Y size of the image
 * @param isWithin correlation was done within single buffer
 * @param iStart first index in the first batch to start with
 * @param iStop last index in the first batch to process (included)
 * @param jStart first index in the second batch to process
 * @param jStop last index in the second batch to process (included)
 * @param jSize size of the second batch
 * @param offset1 of the beginning of the first batch
 * @param offset2 of the beginning of the second bath
 * @param maxImgs no of images that are processed all together (e.g. images in movie)
 */
template<typename T>
void copyInRightOrder(T* d_imgs, T* result, int xDim, int yDim, bool isWithin,
        int iStart, int iStop, int jStart, int jStop, size_t jSize,
        size_t offset1, size_t offset2, size_t maxImgs);
template<typename T>
void copyInRightOrderNew(T* d_pos, T* h_pos, bool isWithin,
        int iStart, int iStop, int jStart, int jStop, size_t jSize,
        size_t offset1, size_t offset2, size_t maxImgs, const GPU &gpu);

/**
 * Function performs cross-correlation on input images in fourier space.
 * It does so by loading several images to GPU at the same time and computing
 * correlation on the batch. Result are centers of correlations.
 * Function uses two buffers where the images are loaded.
 * @param maxDist max distance from the center where to look for maxima
 * @param noOfImgs to process
 * @param h_FFTs actual FFTs
 * @param fftSizeX X dim of the input images
 * @param imgSizeX X dim of the original images (in space domain)
 * @param fftSizeY Y dim of the input images
 * @param maxFFTsInBuffer max number of images to have in one buffer at one moment
 * @param fftBatchSize batch size of the IFFT
 * @param result position of the maxima (n X-Y pairs)
 */
template<typename T>
void computeCorrelations(float maxDist, size_t noOfImgs, const FFTSettings<T> &settings, std::complex<T> *h_FFTs,
        size_t maxFFTsInBuffer, T *result, CorrelationData<T> &aux, const GPU &gpu);


/**
 * Function performs cross-correlation on input images in fourier space.
 * It does so by loading several images to GPU at the same time and computing
 * correlation on the batch. 
 * Then it looks for the position of the maxima. They are returned in X-Y positions.
 * Function uses two buffers where the images are loaded.
 * @param maxDist max distance from the center where to look for maxima
 * @param noOfImgs to process
 * @param h_FFTs actual FFTs
 * @param fftSizeX X dim of the input images
 * @param imgSizeX X dim of the original images (in space domain)
 * @param fftSizeY Y dim of the input images
 * @param maxFFTsInBuffer max number of images to have in one buffer at one moment
 * @param fftBatchSize batch size of the IFFT
 * @param result position of the maxima (n X-Y pairs)
 */
template<typename T>
void computeCorrelations(float maxDist, size_t noOfImgs, std::complex<T>* h_FFTs,
        int fftSizeX, int imgSizeX, int fftSizeY, size_t maxFFTsInBuffer,
        int fftBatchSize, T*& result);

/**
 * Function performs cross-correlation of the images (in Fourier domain) stored
 * in the buffers.
 * Then it looks for the position of the maxima. They are returned in X-Y positions.
 * Function uses two buffers where the images are loaded.
 * @param maxDist max distance from the center where to look for maxima
 * @param noOfImgs to process
 * @param d_in1 first buffer with FFTs
 * @param in1Size no of images in first buffer
 * @param d_in2 second buffer with FFTs
 * @param in2Size no of images in second buffer
 * @param fftBatchSize batch size of the IFFT
 * @param in1Offset offset of the first batch
 * @param in2Offset offset of the second batch
 * @param ffts object used for FFTs
 * @param imgs object where resulting correlations are stored
 * @param handler of the FFT transform
 * @param result position of the maxima (n X-Y pairs)
 */
template<typename T>
void computeCorrelations(float maxDist, int noOfImgs,
        void* d_in1, size_t in1Size, void* d_in2, size_t in2Size,
        int fftBatchSize, size_t in1Offset, size_t in2Offset,
        GpuMultidimArrayAtGpu<std::complex<T> >& ffts,
        GpuMultidimArrayAtGpu<T>& imgs, mycufftHandle& handler,
        T*& result);
        template<typename T>
void computeCorrelationsNew(float maxDist, size_t noOfImgs, 
        void* d_in1, size_t in1Size, void* d_in2, size_t in2Size,
        size_t in1Offset, size_t in2Offset,
        std::complex<T> *ffts, T *imgs, cufftHandle &handler,
        T *result, const FFTSettings<T> &settings);
//@}

#endif
