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
#include "reconstruction_cuda/cuda_utils.h"
#include "cuda_shift_aligner.h"
#include "reconstruction_cuda/cuda_gpu_movie_alignment_correlation_kernels.cu"

namespace Alignment {

template<typename T>
void CudaShiftAligner<T>::computeCorrelations2DOneToN(
        std::complex<T> *h_inOut,
        const std::complex<T> *h_ref,
        const FFTSettingsNew<T> &dims) {
    // FIXME make sure that dim is less than 4GB
    // allocate and copy reference signal
    std::complex<T> *d_ref;
    gpuErrchk(cudaMalloc(&d_ref, dims.fBytesSingle()));
    gpuErrchk(cudaMemcpy(d_ref, h_ref, dims.fBytesSingle(),
            cudaMemcpyHostToDevice));

    // allocate memory for other signals
    std::complex<T> *d_inOut;
    gpuErrchk(cudaMalloc(&d_inOut, dims.fBytesBatch()));

    // process other signals
    for (size_t offset = 0; offset < dims.fDim().n(); offset += dims.batch()) {
        // how many signals to process
        size_t toProcess = std::min(dims.batch(), dims.fDim().n() - offset);

        // copy memory
        gpuErrchk(cudaMemcpy(
                d_inOut,
                h_inOut + offset * dims.fDim().xy(),
                toProcess * dims.fBytesSingle(),
                    cudaMemcpyHostToDevice));

        computeCorrelations2DOneToN<false>(// FIXME handle center template
                d_inOut, d_ref,
                dims.fDim().x(), dims.fDim().y(), toProcess);

        // copy data back
        gpuErrchk(cudaMemcpy(
                h_inOut + offset * dims.fDim().xy(),
                d_inOut,
                toProcess * dims.fBytesSingle(),
                cudaMemcpyDeviceToHost));
    }

    // release memory
    gpuErrchk(cudaFree(d_inOut));
    gpuErrchk(cudaFree(d_ref));
}

template<typename T>
std::vector<Point2D<T>> CudaShiftAligner<T>::computeShift2DOneToN(
        T *h_others,
        T *h_ref,
        FFTSettingsNew<T> &dims,
        size_t maxShift) {
    // transform reference signal
    GpuMultidimArrayAtGpu<T> d_ref(dims.sDim().x(), dims.sDim().y());
    d_ref.copyToGpu(h_ref);
    mycufftHandle handleRef;
    GpuMultidimArrayAtGpu<std::complex<T>> d_ref_ft;
    d_ref.fft(d_ref_ft, handleRef);

    // prepare space for FT in batch
    mycufftHandle handleOthers;
    GpuMultidimArrayAtGpu<std::complex<T>> d_others_ft;
    GpuMultidimArrayAtGpu<T> d_others(dims.sDim().x(), dims.sDim().y(), dims.sDim().z(), dims.batch());

    size_t centerSize = maxShift * 2 + 1;
    auto h_centers = new T[centerSize * centerSize * dims.batch()];
    MultidimArray<T> helper(centerSize, centerSize);
    auto origHelperData = helper.data;
    mycufftHandle handleOthersInv; // for inverse transform

    // reserve enough space for shifts
    auto result = std::vector<Point2D<T>>();
    result.reserve(dims.fDim().n());
    // process signals
    for (size_t offset = 0; offset < dims.fDim().n(); offset += dims.batch()) {
        // how many signals to process
        size_t toProcess = std::min(dims.batch(), dims.fDim().n() - offset);

        // copy memory
       gpuErrchk(cudaMemcpy(
               d_others.d_data,
               h_others + offset * dims.sDim().xy(),
               toProcess * dims.sBytesSingle(),
               cudaMemcpyHostToDevice));

        // perform FT
        d_others.fft(d_others_ft, handleOthers);

        // compute shifts
        auto shifts = computeShifts2DOneToN(
                d_others_ft.d_data,
                d_ref_ft.d_data,
                dims.fDim().x(), dims.fDim().y(), toProcess,
                d_others.d_data, handleOthersInv, // reuse data
                dims.sDim().x(),
                h_centers, helper, maxShift);

        // append shifts to existing results
        result.insert(result.end(), shifts.begin(), shifts.end());
    }

    // release memory
    delete[] h_centers;
    // avoid memory leak
    helper.data = origHelperData;
    return result;
}

template<typename T>
std::vector<Point2D<T>> CudaShiftAligner<T>::computeShiftFromCorrelations2D(
        T *h_centers, MultidimArray<T> &helper, size_t nDim,
        size_t centerSize, size_t maxShift) {
    assert(centerSize == (2 * maxShift + 1));
    assert(helper.xdim == helper.ydim);
    assert(helper.xdim == centerSize);
    T x;
    T y;
    auto result = std::vector<Point2D<T>>();
    helper.setXmippOrigin(); // tell the array that the 'center' is in the center
    for (size_t n = 0; n < nDim; ++n) {
        helper.data = h_centers + n * centerSize * centerSize;
        bestShift(helper, x, y, nullptr, maxShift);
        result.emplace_back(x, y);
    }
    // avoid data corruption
    helper.data = nullptr;
    return result;
}

template<typename T>
std::vector<Point2D<T>> CudaShiftAligner<T>::computeShifts2DOneToN(
        std::complex<T> *d_othersF,
        std::complex<T> *d_ref,
        size_t xDimF, size_t yDimF, size_t nDim,
        T *d_othersS, mycufftHandle handle,
        size_t xDimS,
        T *h_centers, MultidimArray<T> &helper, size_t maxShift) {
    // correlate signals and shift FT so that it will be centered after IFT
    computeCorrelations2DOneToN<true>(
            d_othersF, d_ref,
            xDimF, yDimF, nDim);

    // perform IFT
    GpuMultidimArrayAtGpu<std::complex<T>> ft(
            xDimF, yDimF, 1, nDim, d_othersF);
    GpuMultidimArrayAtGpu<T> spatial(
            xDimS, yDimF, 1, nDim, d_othersS);
    ft.ifft(spatial, handle);

    // crop images in spatial domain, use memory for FT to avoid realocation
    // FIXME add check that the resulting centers are small enough to fit the FT
    size_t centerSize = maxShift * 2 + 1;
    dim3 dimBlockCrop(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGridCrop(
            std::ceil(centerSize / (float)dimBlockCrop.x),
            std::ceil(centerSize / (float)dimBlockCrop.y));
    cropSquareInCenter<<<dimGridCrop, dimBlockCrop>>>(
            spatial.d_data, (T*)ft.d_data,
            xDimS, yDimF, nDim, centerSize);
    // FIXME make sure that spatial_d_data is big enough, there is invalid memory access now

    // copy data back
    gpuErrchk(cudaMemcpy(h_centers, ft.d_data,
            nDim * centerSize * centerSize * sizeof(T),
            cudaMemcpyDeviceToHost));

    // unbind the memory, otherwise the destructor would release it
    ft.d_data = nullptr;
    spatial.d_data = nullptr;

    // compute shifts
    return computeShiftFromCorrelations2D(h_centers, helper, nDim, centerSize, maxShift);
}

template<typename T>
std::vector<Point2D<T>> CudaShiftAligner<T>::computeShift2DOneToN(
    std::complex<T> *h_others,
    std::complex<T> *h_ref,
    FFTSettingsNew<T> &dims,
    size_t maxShift) {
    // FIXME make sure that dim is less than 4GB
    // FIXME shift should be less than half of the input
    // allocate and copy reference signal
    std::complex<T> *d_ref;
    gpuErrchk(cudaMalloc(&d_ref, dims.fBytesSingle()));
    gpuErrchk(cudaMemcpy(d_ref, h_ref, dims.fBytesSingle(),
            cudaMemcpyHostToDevice));

    // allocate memory for other signals
    std::complex<T> *d_others;
    gpuErrchk(cudaMalloc(&d_others, dims.fBytesBatch()));
    T *d_spatial;
    gpuErrchk(cudaMalloc(&d_spatial, dims.sBytesBatch()));

    size_t centerSize = maxShift * 2 + 1;
    auto h_centers = new T[centerSize * centerSize * dims.batch()];
    MultidimArray<T> helper(centerSize, centerSize);
    auto origHelperData = helper.data;
    mycufftHandle handle;

    // reserve enough space for shifts
    auto result = std::vector<Point2D<T>>();
    result.reserve(dims.fDim().n());
    // process signals
    for (size_t offset = 0; offset < dims.fDim().n(); offset += dims.batch()) {
        // how many signals to process
        size_t toProcess = std::min(dims.batch(), dims.fDim().n() - offset);

        // copy memory
        gpuErrchk(cudaMemcpy(
                d_others,
                h_others + offset * dims.fDim().xy(),
                toProcess * dims.fBytesSingle(),
                cudaMemcpyHostToDevice));

        // compute shift
        auto shifts = computeShifts2DOneToN(
                d_others,
                d_ref,
                dims.fDim().x(), dims.fDim().y(), toProcess,
                d_spatial, handle,
                dims.sDim().x(),
                h_centers, helper, maxShift);

        // append shifts to existing results
        result.insert(result.end(), shifts.begin(), shifts.end());
    }

    // release memory
    delete[] h_centers;
    gpuErrchk(cudaFree(d_ref));
    gpuErrchk(cudaFree(d_others));
    gpuErrchk(cudaFree(d_spatial));
    // avoid memory leak
    helper.data = origHelperData;

    return result;
}

template<typename T>
template<bool center>
void CudaShiftAligner<T>::computeCorrelations2DOneToN(
        std::complex<T> *d_inOut,
        const std::complex<T> *d_ref,
        size_t xDim, size_t yDim, size_t nDim) {
    // compute kernel size
    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGrid(
            ceil(xDim / (float)dimBlock.x),
            ceil(yDim / (float)dimBlock.y));
    if (std::is_same<T, float>::value) {
        computeCorrelations2DOneToNKernel<float2, center>
            <<<dimGrid, dimBlock>>> (
                (float2*)d_inOut, (float2*)d_ref,
                xDim, yDim, nDim);
    } else if (std::is_same<T, double>::value) {
        computeCorrelations2DOneToNKernel<double2, center>
            <<<dimGrid, dimBlock>>> (
                (double2*)d_inOut, (double2*)d_ref,
                xDim, yDim, nDim);
    } else {
        REPORT_ERROR(ERR_TYPE_INCORRECT, "Not implemented");
    }
}

template class CudaShiftAligner<float>;
//template class CudaShiftAligner<double>;

} /* namespace Alignment */
