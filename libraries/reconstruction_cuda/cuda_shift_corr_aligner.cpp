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
#include "cuda_shift_corr_aligner.h"
#include "reconstruction_cuda/cuda_gpu_movie_alignment_correlation_kernels.cu"

namespace Alignment {

template<typename T>
void CudaShiftCorrAligner<T>::init2D(const GPU &gpu, AlignType type,
        const FFTSettingsNew<T> &settings, size_t maxShift,
        bool includingBatchFT, bool includingSingleFT) {
    release();

    m_type = type;
    m_settingsInv = &settings;
    m_maxShift = maxShift;
    m_includingBatchFT = includingBatchFT;
    m_includingSingleFT = includingSingleFT;
    m_centerSize = 2 * maxShift + 1;
    m_gpu = &gpu;

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
void CudaShiftCorrAligner<T>::setDefault() {
    m_settingsInv = nullptr;
    m_maxShift = 0;
    m_centerSize = 0;
    m_type = AlignType::None;
    m_gpu = nullptr;

    // device memory
    m_d_single_FD = nullptr;
    m_d_batch_FD = nullptr;
    m_d_single_SD = nullptr;
    m_d_batch_SD = nullptr;

    // host memory
    m_h_centers = nullptr;
    m_origHelperData = nullptr;

    // FT plans
    m_singleToFD = nullptr;
    m_batchToFD = nullptr;
    m_batchToSD = nullptr;

    // flags
    m_includingBatchFT = false;
    m_includingSingleFT = false;
    m_is_d_single_FD_loaded = false;
    m_isInit = false;
}

template<typename T>
void CudaShiftCorrAligner<T>::load2DReferenceOneToN(const std::complex<T> *h_ref) {
    auto isReady = (m_isInit && (AlignType::OneToN == m_type));
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to load a reference signal");
    }

    // copy reference to GPU
    gpuErrchk(cudaMemcpyAsync(m_d_single_FD, h_ref, m_settingsInv->fBytesSingle(),
            cudaMemcpyHostToDevice, *(cudaStream_t*)m_gpu->stream()));

    // update state
    m_is_d_single_FD_loaded = true;
}

template<typename T>
void CudaShiftCorrAligner<T>::load2DReferenceOneToN(const T *h_ref) {
    auto isReady = (m_isInit && (AlignType::OneToN == m_type) && m_includingSingleFT);
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to load a reference signal");
    }

    // copy reference to GPU
    gpuErrchk(cudaMemcpyAsync(m_d_single_SD, h_ref, m_settingsInv->sBytesSingle(),
            cudaMemcpyHostToDevice, *(cudaStream_t*)m_gpu->stream()));

    // perform FT
    CudaFFT<T>::fft(*m_singleToFD, m_d_single_SD, m_d_single_FD);

    // update state
    m_is_d_single_FD_loaded = true;
}

template<typename T>
void CudaShiftCorrAligner<T>::release() {
    // device memory
    gpuErrchk(cudaFree(m_d_single_FD));
    gpuErrchk(cudaFree(m_d_batch_FD));
    gpuErrchk(cudaFree(m_d_single_SD));
    gpuErrchk(cudaFree(m_d_batch_SD));

    // host memory
    delete[] m_h_centers;
    m_helper.data = m_origHelperData;
    m_helper.clear();

    // FT plans
    CudaFFT<T>::release(m_singleToFD);
    CudaFFT<T>::release(m_batchToFD);
    CudaFFT<T>::release(m_batchToSD);

    setDefault();
}

template<typename T>
void CudaShiftCorrAligner<T>::init2DOneToN() {
    // allocate space for data in Fourier domain
    gpuErrchk(cudaMalloc(&m_d_single_FD, m_settingsInv->fBytesSingle()));
    gpuErrchk(cudaMalloc(&m_d_batch_FD, m_settingsInv->fBytesBatch()));

    // allocate space for data in Spatial domain
    if (m_includingBatchFT) {
        gpuErrchk(cudaMalloc(&m_d_single_SD, m_settingsInv->sBytesSingle()));
    }
    if (m_includingSingleFT) {
        gpuErrchk(cudaMalloc(&m_d_batch_SD, m_settingsInv->sBytesBatch()));
    }

    // allocate plans
    m_batchToSD = CudaFFT<T>::createPlan(*m_gpu, *m_settingsInv);
    auto settingsForw = m_settingsInv->createInverse();
    if (m_includingBatchFT) {
        m_batchToFD = CudaFFT<T>::createPlan(*m_gpu, settingsForw);
    }
    if (m_includingSingleFT) {
        m_singleToFD = CudaFFT<T>::createPlan(*m_gpu, settingsForw.createSingle());
    }

    // allocate helper objects
    m_h_centers = new T[m_centerSize * m_centerSize * m_settingsInv->batch()];
    m_helper = MultidimArray<T>(m_centerSize, m_centerSize);
    m_origHelperData = m_helper.data;
}

template<typename T>
void CudaShiftCorrAligner<T>::check() {
    m_gpu->forceSet();
    if (m_settingsInv->isForward()) {
        REPORT_ERROR(ERR_VALUE_INCORRECT, "Inverse transform expected");
    }
    if (m_settingsInv->isInPlace()) {
        REPORT_ERROR(ERR_VALUE_INCORRECT, "In-place transform supported");
    }
    if (m_settingsInv->fBytesBatch() >= ((size_t)4 * 1024 * 1014 * 1024)) {
       REPORT_ERROR(ERR_VALUE_INCORRECT, "Batch is bigger than max size (4GB)");
    }
    if ((0 == m_settingsInv->fDim().size())
        || (0 == m_settingsInv->sDim().size())) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "Fourier or Spatial domain dimension is zero (0)");
    }
    if ((m_centerSize > m_settingsInv->sDim().x())
        || m_centerSize > m_settingsInv->sDim().y()) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "The maximum shift (and hence the shift area: 2 * shift + 1) "
                "must be sharply smaller than the smallest dimension");
    }
    if ((0 != (m_settingsInv->sDim().x() % 2))
        || (0 != (m_settingsInv->sDim().y() % 2))) {
        // while performing IFT of the correlation, we center the signal using multiplication
        // in the FD. This, however, works only for even signal.
            REPORT_ERROR(ERR_VALUE_INCORRECT,
                    "The X and Y dimensions have to be multiple of two. Crop your signal");
    }

    switch (m_type) {
        case AlignType::OneToN:
            break;
        default:
            REPORT_ERROR(ERR_VALUE_INCORRECT,
               "This type is not supported.");
    }
}

template<typename T>
template<bool center>
void CudaShiftCorrAligner<T>::computeCorrelations2DOneToN(
        std::complex<T> *h_inOut) {
    bool isReady = (m_isInit && (AlignType::OneToN == m_type) && m_is_d_single_FD_loaded);

    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() before");
    }

    auto stream = *(cudaStream_t*)m_gpu->stream();
    // process signals in batches
    for (size_t offset = 0; offset < m_settingsInv->fDim().n(); offset += m_settingsInv->batch()) {
        // how many signals to process
        size_t toProcess = std::min(m_settingsInv->batch(), m_settingsInv->fDim().n() - offset);

        // copy memory
        gpuErrchk(cudaMemcpyAsync(
                m_d_batch_FD,
                h_inOut + offset * m_settingsInv->fDim().xy(),
                toProcess * m_settingsInv->fBytesSingle(),
                cudaMemcpyHostToDevice, stream));

        CudaShiftCorrAligner<T>::computeCorrelations2DOneToN<center>(
                *m_gpu,
                m_d_batch_FD, m_d_single_FD,
                m_settingsInv->fDim().x(), m_settingsInv->fDim().y(), toProcess);

        // copy data back
        gpuErrchk(cudaMemcpyAsync(
                h_inOut + offset * m_settingsInv->fDim().xy(),
                m_d_batch_FD,
                toProcess * m_settingsInv->fBytesSingle(),
                cudaMemcpyDeviceToHost, stream));
    }
}

template<typename T>
std::vector<Point2D<T>> CudaShiftCorrAligner<T>::computeShift2DOneToN(
        T *h_others) {
    bool isReady = (m_isInit && (AlignType::OneToN == m_type) && m_is_d_single_FD_loaded);

    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() before");
    }

    // reserve enough space for shifts
    auto result = std::vector<Point2D<T>>();
    result.reserve(m_settingsInv->fDim().n());
    // process signals
    for (size_t offset = 0; offset < m_settingsInv->fDim().n(); offset += m_settingsInv->batch()) {
        // how many signals to process
        size_t toProcess = std::min(m_settingsInv->batch(), m_settingsInv->fDim().n() - offset);

        // copy memory
       gpuErrchk(cudaMemcpyAsync(
               m_d_batch_SD,
               h_others + offset * m_settingsInv->sDim().xy(),
               toProcess * m_settingsInv->sBytesSingle(),
               cudaMemcpyHostToDevice, *(cudaStream_t*)m_gpu->stream()));

        // perform FT
       CudaFFT<T>::fft(*m_batchToFD, m_d_batch_SD, m_d_batch_FD);

        // compute shifts
        auto shifts = computeShifts2DOneToN(
                *m_gpu,
                m_d_batch_FD,
                m_d_single_FD,
                m_settingsInv->fDim().x(), m_settingsInv->fDim().y(), toProcess,
                m_d_batch_SD, *m_batchToSD,
                m_settingsInv->sDim().x(),
                m_h_centers, m_helper, m_maxShift);

        // append shifts to existing results
        result.insert(result.end(), shifts.begin(), shifts.end());
    }

    return result;
}

template<typename T>
std::vector<Point2D<T>> CudaShiftCorrAligner<T>::computeShifts2DOneToN(
        const GPU &gpu,
        std::complex<T> *d_othersF,
        std::complex<T> *d_ref,
        size_t xDimF, size_t yDimF, size_t nDim,
        T *d_othersS, cufftHandle plan,
        size_t xDimS,
        T *h_centers, MultidimArray<T> &helper, size_t maxShift) {
    size_t centerSize = maxShift * 2 + 1;
    // correlate signals and shift FT so that it will be centered after IFT
    computeCorrelations2DOneToN<true>(gpu,
            d_othersF, d_ref,
            xDimF, yDimF, nDim);

    // perform IFT
    CudaFFT<T>::ifft(plan, d_othersF, d_othersS);

    auto stream = *(cudaStream_t*)gpu.stream();

    // crop images in spatial domain, use memory for FT to avoid reallocation
    dim3 dimBlockCrop(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGridCrop(
            std::ceil(centerSize / (float)dimBlockCrop.x),
            std::ceil(centerSize / (float)dimBlockCrop.y));
    cropSquareInCenter<<<dimGridCrop, dimBlockCrop, 0, stream>>>(
            d_othersS, (T*)d_othersF,
            xDimS, yDimF, nDim, centerSize);

    // copy data back
    gpuErrchk(cudaMemcpyAsync(h_centers, d_othersF,
            nDim * centerSize * centerSize * sizeof(T),
            cudaMemcpyDeviceToHost, stream));

    gpu.synchStream();

    // compute shifts
    return AShiftAligner<T>::computeShiftFromCorrelations2D(
            h_centers, helper, nDim, centerSize, maxShift);
}

template<typename T>
template<bool center>
void CudaShiftCorrAligner<T>::computeCorrelations2DOneToN(
        const GPU &gpu,
        std::complex<T> *d_inOut,
        const std::complex<T> *d_ref,
        size_t xDim, size_t yDim, size_t nDim) {
    auto stream = *(cudaStream_t*)gpu.stream();
    // compute kernel size
    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGrid(
            ceil(xDim / (float)dimBlock.x),
            ceil(yDim / (float)dimBlock.y));
    if (std::is_same<T, float>::value) {
        computeCorrelations2DOneToNKernel<float2, center>
            <<<dimGrid, dimBlock, 0, stream>>> (
                (float2*)d_inOut, (float2*)d_ref,
                xDim, yDim, nDim);
    } else if (std::is_same<T, double>::value) {
        computeCorrelations2DOneToNKernel<double2, center>
            <<<dimGrid, dimBlock, 0, stream>>> (
                (double2*)d_inOut, (double2*)d_ref,
                xDim, yDim, nDim);
    } else {
        REPORT_ERROR(ERR_TYPE_INCORRECT, "Not implemented");
    }
}

// explicit instantiation
template void CudaShiftCorrAligner<float>::computeCorrelations2DOneToN<false>(std::complex<float>*);
template class CudaShiftCorrAligner<float>;

} /* namespace Alignment */
