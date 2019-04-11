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
void CudaShiftAligner<T>::init2D(AlignType type,
        const FFTSettingsNew<T> &dims, size_t maxShift, bool includingFT) {
    release();

    m_type = type;
    m_dims = dims;
    m_maxShift = maxShift;
    m_includingFT = includingFT;
    m_centerSize = 2 * maxShift + 1;

    check();

    switch (type) {
        case AlignType::OneToN:
            init2DOneToN();
            break;
        default:
            REPORT_ERROR(ERR_NOT_IMPLEMENTED, "This alignment type is not supported yet");
    }

    m_isInit = true;
}

template<typename T>
void CudaShiftAligner<T>::setDefault() {
    m_dims = FFTSettingsNew<T>(0);
    m_maxShift = 0;
    m_centerSize = 0;

    // device memory
    m_d_single_FD = nullptr;
    m_d_batch_FD = nullptr;
    m_d_single_SD = nullptr;
    m_d_batch_SD = nullptr;

    // host memory
    m_h_centers = nullptr;
    m_origHelperData = nullptr;

    // flags
    m_includingFT = false;
    m_isInit = false;
    m_is_d_single_FD_loaded = false;
}

template<typename T>
void CudaShiftAligner<T>::load2DReferenceOneToN(const std::complex<T> *h_ref) {
    auto isReady = (m_isInit && (AlignType::OneToN == m_type));
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGICAL_ERROR, "Not ready to load a reference signal");
    }

    // copy reference to GPU
    gpuErrchk(cudaMemcpy(m_d_single_FD, h_ref, m_dims.fBytesSingle(),
            cudaMemcpyHostToDevice));

    // update state
    m_is_d_single_FD_loaded = true;
}

template<typename T>
void CudaShiftAligner<T>::load2DReferenceOneToN(const T *h_ref) {
    auto isReady = (m_isInit && (AlignType::OneToN == m_type) && m_includingFT);
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGICAL_ERROR, "Not ready to load a reference signal");
    }

    // copy reference to GPU
    gpuErrchk(cudaMemcpy(m_d_single_SD, h_ref, m_dims.sBytesSingle(),
            cudaMemcpyHostToDevice));

    // perform FT
    auto spatial = GpuMultidimArrayAtGpu<T>(m_dims.sDim().x(), m_dims.sDim().y(), 1, 1, m_d_single_SD);
    auto freq = GpuMultidimArrayAtGpu<std::complex<T>>(m_dims.fDim().x(), m_dims.fDim().y(), 1, 1, m_d_single_FD);
    spatial.fft(freq, m_singleToFD);

    // update state
    m_is_d_single_FD_loaded = true;

    // unbind data
    spatial.d_data = nullptr;
    freq.d_data = nullptr;
}

template<typename T>
void CudaShiftAligner<T>::release() {
    // device memory
    gpuErrchk(cudaFree(m_d_single_FD));
    gpuErrchk(cudaFree(m_d_batch_FD));
    gpuErrchk(cudaFree(m_d_single_SD));
    gpuErrchk(cudaFree(m_d_batch_SD));

    // host memory
    delete[] m_h_centers;
    m_helper.data = m_origHelperData;
    m_helper.clear();

    // FT data
    m_singleToFD.clear();
    m_batchToFD.clear();
    m_batchToSD.clear();

    setDefault();
}

template<typename T>
void CudaShiftAligner<T>::init2DOneToN() {
    // allocate space for data in Fourier domain
    gpuErrchk(cudaMalloc(&m_d_single_FD, m_dims.fBytesSingle()));
    gpuErrchk(cudaMalloc(&m_d_batch_FD, m_dims.fBytesBatch()));

    if (m_includingFT) {
        // allocate space for data in Spatial domain
        gpuErrchk(cudaMalloc(&m_d_single_SD, m_dims.sBytesSingle()));
        gpuErrchk(cudaMalloc(&m_d_batch_SD, m_dims.sBytesBatch()));
    }

    // allocate helper objects
    m_h_centers = new T[m_centerSize * m_centerSize * m_dims.batch()];
    m_helper = MultidimArray<T>(m_centerSize, m_centerSize);
    m_origHelperData = m_helper.data;
}

template<typename T>
void CudaShiftAligner<T>::check() {
    if (m_dims.fBytesBatch() >= ((size_t)4 * 1024 * 1014 * 1024)) {
       REPORT_ERROR(ERR_VALUE_INCORRECT, "Batch is bigger than max size (4GB)");
    }
    if ((0 == m_dims.fDim().size())
        || (0 == m_dims.sDim().size())) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "Fourier or Spatial domain dimension is zero (0)");
    }
    if ((m_centerSize > m_dims.sDim().x())
        || m_centerSize > m_dims.sDim().y()) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "The maximum shift (and hence the shift area: 2 * shift + 1) "
                "must be sharply smaller than the smallest dimension");
    }
    if ((0 != (m_dims.sDim().x() % 2))
        || (0 != (m_dims.sDim().y() % 2))) {
        // while performing IFT of the correlation, we center the signal using multiplication
        // in the FD. This, however, works only for even signal.
            REPORT_ERROR(ERR_VALUE_INCORRECT,
                    "The X and Y dimensions have to be multiple of two. Crop your signal");
        }
}

template<typename T>
template<bool center>
void CudaShiftAligner<T>::computeCorrelations2DOneToN(
        std::complex<T> *h_inOut) {
    bool isReady = (m_isInit && (AlignType::OneToN == m_type) && m_is_d_single_FD_loaded);

    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGICAL_ERROR, "Not ready to execute. Call init() before");
    }

    // process signals in batches
    for (size_t offset = 0; offset < m_dims.fDim().n(); offset += m_dims.batch()) {
        // how many signals to process
        size_t toProcess = std::min(m_dims.batch(), m_dims.fDim().n() - offset);

        // copy memory
        gpuErrchk(cudaMemcpy(
                m_d_batch_FD,
                h_inOut + offset * m_dims.fDim().xy(),
                toProcess * m_dims.fBytesSingle(),
                    cudaMemcpyHostToDevice));

        CudaShiftAligner<T>::computeCorrelations2DOneToN<center>(
                m_d_batch_FD, m_d_single_FD,
                m_dims.fDim().x(), m_dims.fDim().y(), toProcess);

        // copy data back
        gpuErrchk(cudaMemcpy(
                h_inOut + offset * m_dims.fDim().xy(),
                m_d_batch_FD,
                toProcess * m_dims.fBytesSingle(),
                cudaMemcpyDeviceToHost));
    }
}

template<typename T>
std::vector<Point2D<T>> CudaShiftAligner<T>::computeShift2DOneToN(
        T *h_others) {
    bool isReady = (m_isInit && (AlignType::OneToN == m_type) && m_is_d_single_FD_loaded);

    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGICAL_ERROR, "Not ready to execute. Call init() before");
    }

    // prepare for FT in batch
    auto d_others = GpuMultidimArrayAtGpu<T>(
            m_dims.sDim().x(), m_dims.sDim().y(), m_dims.sDim().z(), m_dims.batch(), m_d_batch_SD);
    auto d_others_FD = GpuMultidimArrayAtGpu<std::complex<T>>(
            m_dims.fDim().x(), m_dims.fDim().y(), m_dims.fDim().z(), m_dims.batch(), m_d_batch_FD);

    // reserve enough space for shifts
    auto result = std::vector<Point2D<T>>();
    result.reserve(m_dims.fDim().n());
    // process signals
    for (size_t offset = 0; offset < m_dims.fDim().n(); offset += m_dims.batch()) {
        // how many signals to process
        size_t toProcess = std::min(m_dims.batch(), m_dims.fDim().n() - offset);

        // copy memory
       gpuErrchk(cudaMemcpy(
               m_d_batch_SD,
               h_others + offset * m_dims.sDim().xy(),
               toProcess * m_dims.sBytesSingle(),
               cudaMemcpyHostToDevice));

        // perform FT
        d_others.fft(d_others_FD, m_batchToFD);

        // compute shifts
        auto shifts = computeShifts2DOneToN(
                m_d_batch_FD,
                m_d_single_FD,
                m_dims.fDim().x(), m_dims.fDim().y(), toProcess,
                m_d_batch_SD, m_batchToSD,
                m_dims.sDim().x(),
                m_h_centers, m_helper, m_maxShift);

        // append shifts to existing results
        result.insert(result.end(), shifts.begin(), shifts.end());
    }

    // unbind data
    d_others.d_data = nullptr;
    d_others_FD.d_data = nullptr;

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
    size_t centerSize = maxShift * 2 + 1;
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

    // crop images in spatial domain, use memory for FT to avoid reallocation
    dim3 dimBlockCrop(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGridCrop(
            std::ceil(centerSize / (float)dimBlockCrop.x),
            std::ceil(centerSize / (float)dimBlockCrop.y));
    cropSquareInCenter<<<dimGridCrop, dimBlockCrop>>>(
            spatial.d_data, (T*)ft.d_data,
            xDimS, yDimF, nDim, centerSize);

    // copy data back
    gpuErrchk(cudaMemcpy(h_centers, ft.d_data,
            nDim * centerSize * centerSize * sizeof(T),
            cudaMemcpyDeviceToHost));

    // unbind the memory, otherwise the destructor would release it
    ft.d_data = nullptr;
    spatial.d_data = nullptr;

    // compute shifts
    return AShiftAligner<T>::computeShiftFromCorrelations2D(
            h_centers, helper, nDim, centerSize, maxShift);
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

// explicit instantiation
template void CudaShiftAligner<float>::computeCorrelations2DOneToN<false>(std::complex<float>*);
template class CudaShiftAligner<float>;

} /* namespace Alignment */
