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
#include "cuda_shift_corr_estimator.h"
#include "reconstruction_cuda/cuda_gpu_movie_alignment_correlation_kernels.cu"

namespace Alignment {

template<typename T>
void CudaShiftCorrEstimator<T>::init2D(const HW &hw, AlignType type,
        const FFTSettingsNew<T> &settings, size_t maxShift,
        bool includingBatchFT, bool includingSingleFT) {
    release();
    try {
        m_gpu = &dynamic_cast<const GPU&>(hw);
    } catch (std::bad_cast&) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Instance of GPU expected");
    }

    AShiftCorrEstimator<T>::init2D(type, settings, maxShift,
        includingBatchFT, includingSingleFT);

    this->m_isInit = true;
}

template<typename T>
void CudaShiftCorrEstimator<T>::setDefault() {
    AShiftCorrEstimator<T>::setDefault();

    m_gpu = nullptr;

    // device memory
    m_d_single_FD = nullptr;
    m_d_batch_FD = nullptr;
    m_d_single_SD = nullptr;
    m_d_batch_SD = nullptr;

    // host memory
    m_h_centers = nullptr;

    // FT plans
    m_singleToFD = nullptr;
    m_batchToFD = nullptr;
    m_batchToSD = nullptr;
}

template<typename T>
void CudaShiftCorrEstimator<T>::load2DReferenceOneToN(const std::complex<T> *h_ref) {
    auto isReady = (this->m_isInit && (AlignType::OneToN == this->m_type));
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to load a reference signal");
    }

    // copy reference to GPU
    gpuErrchk(cudaMemcpyAsync(m_d_single_FD, h_ref, this->m_settingsInv->fBytesSingle(),
            cudaMemcpyHostToDevice, *(cudaStream_t*)m_gpu->stream()));

    // update state
    m_is_d_single_FD_loaded = true;
}

template<typename T>
void CudaShiftCorrEstimator<T>::load2DReferenceOneToN(const T *h_ref) {
    auto isReady = (this->m_isInit && (AlignType::OneToN == this->m_type) && this->m_includingSingleFT);
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to load a reference signal");
    }

    // copy reference to GPU
    gpuErrchk(cudaMemcpyAsync(m_d_single_SD, h_ref, this->m_settingsInv->sBytesSingle(),
            cudaMemcpyHostToDevice, *(cudaStream_t*)m_gpu->stream()));

    // perform FT
    CudaFFT<T>::fft(*m_singleToFD, m_d_single_SD, m_d_single_FD);

    // update state
    m_is_d_single_FD_loaded = true;
}

template<typename T>
void CudaShiftCorrEstimator<T>::release() {
    // device memory
    gpuErrchk(cudaFree(m_d_single_FD));
    gpuErrchk(cudaFree(m_d_batch_FD));
    gpuErrchk(cudaFree(m_d_single_SD));
    gpuErrchk(cudaFree(m_d_batch_SD));

    // host memory
    delete[] m_h_centers;

    // FT plans
    CudaFFT<T>::release(m_singleToFD);
    CudaFFT<T>::release(m_batchToFD);
    CudaFFT<T>::release(m_batchToSD);

    AShiftCorrEstimator<T>::release();

    CudaShiftCorrEstimator<T>::setDefault();
}

template<typename T>
void CudaShiftCorrEstimator<T>::init2DOneToN() {
    AShiftCorrEstimator<T>::init2DOneToN();
    // allocate space for data in Fourier domain
    gpuErrchk(cudaMalloc(&m_d_single_FD, this->m_settingsInv->fBytesSingle()));
    gpuErrchk(cudaMalloc(&m_d_batch_FD, this->m_settingsInv->fBytesBatch()));

    // allocate host memory
    m_h_centers = new T[this->m_centerSize * this->m_centerSize * this->m_batch]();

    // allocate plans and additional space needed
    m_batchToSD = CudaFFT<T>::createPlan(*m_gpu, *this->m_settingsInv);
    auto settingsForw = this->m_settingsInv->createInverse();
    if (this->m_includingBatchFT) {
        gpuErrchk(cudaMalloc(&m_d_batch_SD, this->m_settingsInv->sBytesBatch()));
        m_batchToFD = CudaFFT<T>::createPlan(*m_gpu, settingsForw);
    }
    if (this->m_includingSingleFT) {
        gpuErrchk(cudaMalloc(&m_d_single_SD, this->m_settingsInv->sBytesSingle()));
        m_singleToFD = CudaFFT<T>::createPlan(*m_gpu, settingsForw.createSingle());
    }
}

template<typename T>
void CudaShiftCorrEstimator<T>::computeCorrelations2DOneToN(
        std::complex<T> *h_inOut, bool center) {
    bool isReady = (this->m_isInit && (AlignType::OneToN == this->m_type) && m_is_d_single_FD_loaded);

    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() before");
    }

    auto stream = *(cudaStream_t*)m_gpu->stream();
    // process signals in batches
    for (size_t offset = 0; offset < this->m_settingsInv->fDim().n(); offset += this->m_settingsInv->batch()) {
        // how many signals to process
        size_t toProcess = std::min(this->m_settingsInv->batch(), this->m_settingsInv->fDim().n() - offset);

        // copy memory
        gpuErrchk(cudaMemcpyAsync(
                m_d_batch_FD,
                h_inOut + offset * this->m_settingsInv->fDim().xy(),
                toProcess * this->m_settingsInv->fBytesSingle(),
                cudaMemcpyHostToDevice, stream));

        auto dims = Dimensions(
                this->m_settingsInv->fDim().x(),
                this->m_settingsInv->fDim().y(),
                1,
                toProcess);
        if (center) {
            sComputeCorrelations2DOneToN<true>(*m_gpu, m_d_batch_FD, m_d_single_FD, dims);
        } else {
            sComputeCorrelations2DOneToN<false>(*m_gpu, m_d_batch_FD, m_d_single_FD, dims);
        }

        // copy data back
        gpuErrchk(cudaMemcpyAsync(
                h_inOut + offset * this->m_settingsInv->fDim().xy(),
                m_d_batch_FD,
                toProcess * this->m_settingsInv->fBytesSingle(),
                cudaMemcpyDeviceToHost, stream));
    }
}

template<typename T>
void CudaShiftCorrEstimator<T>::computeShift2DOneToN(
        T *h_others) {
    bool isReady = (this->m_isInit && (AlignType::OneToN == this->m_type)
            && m_is_d_single_FD_loaded && this->m_includingBatchFT);

    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() before");
    }

    // reserve enough space for shifts
    this->m_shifts2D.reserve(this->m_settingsInv->fDim().n());
    // process signals
    for (size_t offset = 0; offset < this->m_settingsInv->fDim().n(); offset += this->m_settingsInv->batch()) {
        // how many signals to process
        size_t toProcess = std::min(this->m_settingsInv->batch(), this->m_settingsInv->fDim().n() - offset);

        // copy memory
        gpuErrchk(cudaMemcpyAsync(
               m_d_batch_SD,
               h_others + offset * this->m_settingsInv->sDim().xy(),
               toProcess * this->m_settingsInv->sBytesSingle(),
               cudaMemcpyHostToDevice, *(cudaStream_t*)m_gpu->stream()));

        // perform FT
        CudaFFT<T>::fft(*m_batchToFD, m_d_batch_SD, m_d_batch_FD);

        // compute shifts
        auto shifts = computeShifts2DOneToN(
                *m_gpu,
                m_d_batch_FD,
                m_d_batch_SD,
                m_d_single_FD,
                this->m_settingsInv->createSubset(toProcess),
                *m_batchToSD,
                this->m_h_centers,
                this->m_maxShift);

        // append shifts to existing results
        this->m_shifts2D.insert(this->m_shifts2D.end(), shifts.begin(), shifts.end());
    }

    // update state
    this->m_is_shift_computed = true;
}

template<typename T>
std::vector<Point2D<float>> CudaShiftCorrEstimator<T>::computeShifts2DOneToN(
        const GPU &gpu,
        std::complex<T> *d_othersF,
        T *d_othersS,
        std::complex<T> *d_ref,
        const FFTSettingsNew<T> &settings,
        cufftHandle plan,
        T *h_centers,
        size_t maxShift) {
    // we need even input in order to perform the shift (in FD, while correlating) properly
    assert(0 == (settings.sDim().x() % 2));
    assert(0 == (settings.sDim().y() % 2));
    assert(1 == settings.sDim().zPadded());

    size_t centerSize = maxShift * 2 + 1;

    // make sure we have enough memory for centers of the correlation functions
    assert(settings.fBytesBatch() >= (settings.sDim().n() * centerSize * centerSize * sizeof(T)));

    // correlate signals and shift FT so that it will be centered after IFT
    sComputeCorrelations2DOneToN<true>(gpu,
            d_othersF, d_ref,
            settings.fDim());

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
            settings.sDim().x(), settings.sDim().y(), settings.sDim().n(), centerSize, centerSize);

    // copy data back
    gpuErrchk(cudaMemcpyAsync(h_centers, d_othersF,
            settings.sDim().n() * centerSize * centerSize * sizeof(T),
            cudaMemcpyDeviceToHost, stream));

    gpu.synch();

    // compute shifts
    auto result = std::vector<Point2D<float>>();
    AShiftCorrEstimator<T>::findMaxAroundCenter(
            h_centers, Dimensions(centerSize, centerSize, 1, settings.sDim().n()), maxShift, result);
    return result;
}

template<typename T>
void CudaShiftCorrEstimator<T>::computeCorrelations2DOneToN(
        const HW &hw,
        std::complex<T> *inOut,
        const std::complex<T> *ref,
        const Dimensions &dims,
        bool center) {
    const GPU *gpu;
    try {
        gpu = &dynamic_cast<const GPU&>(hw);
    } catch (std::bad_cast&) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Instance of GPU expected");
    }
    if (center) {
        return sComputeCorrelations2DOneToN<true>(*gpu, inOut, ref, dims);
    } else {
        return sComputeCorrelations2DOneToN<false>(*gpu, inOut, ref, dims);
    }
}

template<typename T>
template<bool center>
void CudaShiftCorrEstimator<T>::sComputeCorrelations2DOneToN(
        const GPU &gpu,
        std::complex<T> *d_inOut,
        const std::complex<T> *d_ref,
        const Dimensions &dims) {
    if (center) {
        // we cannot assert xDim, as we don't know if the spatial size was even
        assert(0 == (dims.y() % 2));
    }
    assert(0 < dims.x());
    assert(0 < dims.y());
    assert(1 == dims.z());
    assert(0 < dims.n());

    auto stream = *(cudaStream_t*)gpu.stream();
    // compute kernel size
    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGrid(
            ceil(dims.x() / (float)dimBlock.x),
            ceil(dims.y() / (float)dimBlock.y));
    if (std::is_same<T, float>::value) {
        computeCorrelations2DOneToNKernel<float2, center>
            <<<dimGrid, dimBlock, 0, stream>>> (
                (float2*)d_inOut, (float2*)d_ref,
                dims.x(), dims.y(), dims.n());
    } else if (std::is_same<T, double>::value) {
        computeCorrelations2DOneToNKernel<double2, center>
            <<<dimGrid, dimBlock, 0, stream>>> (
                (double2*)d_inOut, (double2*)d_ref,
                dims.x(), dims.y(), dims.n());
    } else {
        REPORT_ERROR(ERR_TYPE_INCORRECT, "Not implemented");
    }
}

template<typename T>
void CudaShiftCorrEstimator<T>::check() {
    if (this->m_settingsInv->isInPlace()) {
        REPORT_ERROR(ERR_VALUE_INCORRECT, "Only out-of-place transform is supported");
    }
    if ((this->m_centerSize * this->m_centerSize * sizeof(T)) > this->m_settingsInv->fBytesBatch()) {
        // see computeShifts2DOneToN
        REPORT_ERROR(ERR_NOT_IMPLEMENTED, "This implementation is not able to handle cases when"
                "maxShift is more than half of the dimension of the input");
    }
}

// explicit instantiation
template class CudaShiftCorrEstimator<float>;
template class CudaShiftCorrEstimator<double>;

} /* namespace Alignment */
