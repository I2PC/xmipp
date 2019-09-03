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
#include "cuda_rot_polar_estimator.h"
#include "cuda_gpu_polar.cu"
#include "cuda_gpu_movie_alignment_correlation_kernels.cu"

namespace Alignment {

template<typename T>
void CudaRotPolarEstimator<T>::init2D(HW &hw) {
    try {
        m_gpu = &dynamic_cast<GPU&>(hw);
    } catch (std::bad_cast&) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Instance of GPU expected");
    }

    m_firstRing = this->m_dims->x() / 5;
    m_lastRing = (this->m_dims->x() - 1) / 2;
//    m_lastRing = (this->m_dims->x() - 3) / 2; // FIXME DS uncomment // so that we have some edge around the biggest ring
    // all rings have the same number of samples, to make FT easier
    // FIXME DS number of samples is a candidate for tuning, as for 'big' (256) images resulting in 398 samples (2*199)
    m_samples = std::max(1, 2 * (int)(M_PI * m_lastRing)); // keep this even
    // FIXME DS change the names, this doesn't make any sense
    m_logicalSettings = new FFTSettingsNew<T>(m_samples, getNoOfRings(), 1, this->m_dims->n(), this->m_batch);
    m_hwSettings = new FFTSettingsNew<T>(m_samples, 1, 1, this->m_dims->n() * getNoOfRings(), this->m_batch * getNoOfRings());
    m_inverseSettings = new FFTSettingsNew<T>(m_samples, 1, 1, this->m_dims->n(), this->m_batch, false, false);


    gpuErrchk(cudaMalloc(&m_d_ref, m_logicalSettings->fBytesSingle())); // FT of the samples
    gpuErrchk(cudaMalloc(&m_d_batch, this->m_dims->xy() * this->m_batch * sizeof(T))); // Cartesian batch
    gpuErrchk(cudaMalloc(&m_d_batchPolarFD, m_logicalSettings->fBytesBatch())); // FT of the polar samples
    gpuErrchk(cudaMalloc(&m_d_batchPolarOrCorr, m_logicalSettings->sBytesBatch())); // IFT of the polar samples

    m_batchToFD = CudaFFT<T>::createPlan(*m_gpu, *m_hwSettings);
    m_batchToSD = CudaFFT<T>::createPlan(*m_gpu, *m_inverseSettings);

    m_h_batchResult = new T[m_samples * this->m_batch];

    // synch primitives
    m_mutex = new std::mutex();
    m_cv = new std::condition_variable;

    if (std::is_same<T, float>()) { // FIXME DS remove
        m_dataAux.resize(this->m_dims->y(), this->m_dims->x());
    }

    this->m_isInit = true;
}

template<typename T>
void CudaRotPolarEstimator<T>::release() {
    delete m_logicalSettings;
    delete m_hwSettings;

    // device memory
    gpuErrchk(cudaFree(m_d_batch));
    gpuErrchk(cudaFree(m_d_batchPolarOrCorr));
    gpuErrchk(cudaFree(m_d_batchPolarFD));
    gpuErrchk(cudaFree(m_d_ref));

    // FT plans
    CudaFFT<T>::release(m_singleToFD);
    CudaFFT<T>::release(m_batchToFD);
    CudaFFT<T>::release(m_batchToSD);

    delete[] m_h_batchResult;

    // synch primitives
    delete m_mutex;
    delete m_cv;

    m_dataAux.clear(); // FIXME DS remove
    m_refPolarFourierI.clear(); // FIXME DS remove
    delete m_refPlans; // FIXME DS remove
    m_rotCorrAux.clear(); // FIXME DS remove

    ARotationEstimator<T>::release();
    CudaRotPolarEstimator<T>::setDefault();
}

template<typename T>
void CudaRotPolarEstimator<T>::setDefault() {
    m_gpu = nullptr;
    m_logicalSettings = nullptr;
    m_hwSettings = nullptr;

    // device memory
    m_d_batch = nullptr;
    m_d_batchPolarOrCorr = nullptr;
    m_d_batchPolarFD = nullptr;
    m_d_ref = nullptr;

    // host memory
    m_h_batchResult = nullptr;

    // FT plans
    m_singleToFD = nullptr;
    m_batchToFD = nullptr;
    m_batchToSD = nullptr;

    m_firstRing = -1;
    m_lastRing = -1;

    // synch primitives
    m_mutex = nullptr;
    m_cv = nullptr;
    m_isDataReady = false;

    m_dataAux.clear(); // FIXME DS remove
    m_refPolarFourierI.clear(); // FIXME DS remove
    m_refPlans = nullptr; // FIXME DS remove
    m_rotCorrAux.clear(); // FIXME DS remove

    ARotationEstimator<T>::setDefault();
}

template<>
MultidimArray<double> CudaRotPolarEstimator<float>::convert(float *data) { // FIXME remove
    const size_t s = this->m_dims->xyz();
    for (size_t i = 0; i < s; ++i) {
        m_dataAux.data[i] = data[i];
    }
    return m_dataAux;
}

template<>
MultidimArray<double> CudaRotPolarEstimator<double>::convert(double *data) { // FIXME remove
    return MultidimArray<double>(
            this->m_dims->n(), this->m_dims->z(),
            this->m_dims->y(), this->m_dims->x(),
            data);
}

template<typename T> // FIXME DS rework
void CudaRotPolarEstimator<T>::load2DReferenceOneToN(const T *ref) {
    // FIXME DS rework properly
    MultidimArray<double> tmp = convert(const_cast<T*>(ref));
    tmp.setXmippOrigin();
    normalizedPolarFourierTransform(tmp, m_refPolarFourierI, true, // conjugate once
            m_firstRing, m_lastRing, m_refPlans, 1);
    m_rotCorrAux.resize(2 * m_refPolarFourierI.getSampleNoOuterRing() - 1);
    m_aux.local_transformer.setReal(m_rotCorrAux);

    auto settingsSingle = m_logicalSettings->createSingle();
    auto tmp1 = new std::complex<T>[settingsSingle.fDim().size()];
    auto position = tmp1;
//    printf("reference:\n");
    for (auto &a : m_refPolarFourierI.rings) { // rings are Y (rows)
//        for (int x = 0; x < a.nzyxdim; ++x) {
//            std::cout << a.data[x] << " ";
//        }
//        printf("\n");
        std::copy(a.data, a.data + a.nzyxdim, position);
        position += a.nzyxdim;
    }
//    printf("\n");
    auto stream = *(cudaStream_t*)m_gpu->stream();
    // copy memory
    gpuErrchk(cudaMemcpyAsync(
            m_d_ref,
            tmp1,
            settingsSingle.fBytesSingle(),
            cudaMemcpyHostToDevice, stream));
    m_gpu->synch();
    delete[] tmp1;
    this->m_is_ref_loaded = true;
}

template<typename T>
void CudaRotPolarEstimator<T>::sComputeCorrelationsOneToN(
        const GPU &gpu,
        std::complex<T> *d_inOut,
        const std::complex<T> *d_ref,
        const Dimensions &dims,
        int firstRingRadius) {
    auto stream = *(cudaStream_t*)gpu.stream();
    dim3 dimBlock(32);
    dim3 dimGrid(
        ceil(dims.size() / (float)dimBlock.x));
    if (std::is_same<T, float>::value) {
        computePolarCorrelationsSumOneToNKernel<float2>
            <<<dimGrid, dimBlock, 0, stream>>> (
            (float2*)d_inOut, (float2*)d_ref,
            firstRingRadius,
            dims.x(), dims.y(), dims.n());
    } else if (std::is_same<T, double>::value) {
        computePolarCorrelationsSumOneToNKernel<double2>
            <<<dimGrid, dimBlock, 0, stream>>> (
            (double2*)d_inOut, (double2*)d_ref,
            firstRingRadius,
            dims.x(), dims.y(), dims.n());
    } else {
            REPORT_ERROR(ERR_TYPE_INCORRECT, "Not implemented");
    }
}

template<typename T>
void CudaRotPolarEstimator<T>::loadThreadRoutine(T *h_others,
        void *loadStream) {
    auto lStream = (cudaStream_t*)loadStream;
    for (size_t offset = 0; offset < this->m_dims->n(); offset += this->m_batch) {
        // how many signals to process
        size_t toProcess = std::min(this->m_batch, this->m_dims->n() - offset);

        T *h_src = h_others + offset * this->m_dims->xy();
        size_t bytes = toProcess * this->m_dims->xy() * sizeof(T);

        // copy memory
        gpuErrchk(cudaMemcpyAsync(
                m_d_batch,
                h_src,
                bytes,
                cudaMemcpyHostToDevice, *lStream));
        // block until data is loaded
        cudaStreamSynchronize(*lStream);

        // notify processing stream that it can work
        m_isDataReady = true;
        m_cv->notify_all();

        // wait till the data is processed
        std::unique_lock<std::mutex> lk(*m_mutex);
        m_cv->wait(lk, [&]{return !m_isDataReady;});
    }
}

template<typename T>
void CudaRotPolarEstimator<T>::computeRotation2DOneToN(T *h_others) {
    bool isReady = (this->m_isInit && (AlignType::OneToN == this->m_type) && this->m_is_ref_loaded);

    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() and load reference");
    }

    if ( ! GPU::isMemoryPinned(h_others)) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Input memory has to be pinned (page-locked)");
    }

    this->m_rotations2D.reserve(this->m_dims->n());
    auto workstream = *(cudaStream_t*)m_gpu->stream();
    auto loadStream = new cudaStream_t;
    gpuErrchk(cudaStreamCreate(loadStream));

    m_isDataReady = false;

    std::thread loadingThread(&CudaRotPolarEstimator<T>::loadThreadRoutine, this,
            h_others, loadStream);

    // process signals in batches
    for (size_t offset = 0; offset < this->m_dims->n(); offset += this->m_batch) {
        // how many signals to process
        size_t toProcess = std::min(this->m_batch, this->m_dims->n() - offset);

        // block until data is loaded
        std::unique_lock<std::mutex> lk(*m_mutex);
        m_cv->wait(lk, [&]{return m_isDataReady;});

        // call polar transformation kernel
        dim3 dimBlock(32, 32);
        dim3 dimGrid(
            ceil(toProcess * m_samples / (float)dimBlock.x),
            ceil(getNoOfRings() / (float)dimBlock.y));
        polarFromCartesian<T, true>
            <<<dimGrid, dimBlock, 0, workstream>>> (
            m_d_batch, this->m_dims->x(), this->m_dims->y(),
            m_d_batchPolarOrCorr, m_samples, getNoOfRings(), m_firstRing, toProcess);

        // notify that buffer is processed (new will be loaded on background)
        m_gpu->synch();
        m_isDataReady = false;
        m_cv->notify_all();

        // FIXME DS add normalization

        CudaFFT<T>::fft(*m_batchToFD, m_d_batchPolarOrCorr, (std::complex<T>*)m_d_batchPolarFD);

        auto dims = this->m_logicalSettings->fDim().copyForN(toProcess);
        sComputeCorrelationsOneToN(*m_gpu, (std::complex<T>*)m_d_batchPolarFD, m_d_ref, dims, m_firstRing);

        CudaFFT<T>::ifft(*m_batchToSD, (std::complex<T>*)m_d_batchPolarFD, m_d_batchPolarOrCorr);

        // copy data back
        gpuErrchk(cudaMemcpyAsync(
                m_h_batchResult,
                m_d_batchPolarOrCorr,
                m_samples * toProcess * sizeof(T),
                cudaMemcpyDeviceToHost, workstream));

        m_gpu->synch();
        // locate max angle
        for (size_t i = 0; i < toProcess; ++i) {
            size_t posun = i * m_samples;
            size_t index = 0;
            T max = std::numeric_limits<T>::min();
            for (size_t x = 0; x < m_samples; ++x) {
                T v = m_h_batchResult[posun + x];
                if (max < v) {
                    max = v;
                    index = x;
                }
            }
            T angle = index * (T)360 / m_samples;
            this->m_rotations2D.emplace_back(angle);
        }
    }
    loadingThread.join();
    this->m_is_rotation_computed = true;
}



template<typename T>
void CudaRotPolarEstimator<T>::check() {
    ARotationEstimator<T>::check();
    if (this->m_dims->x() != this->m_dims->y()) {
        // because of the rings
        REPORT_ERROR(ERR_ARG_INCORRECT, "This estimator can work only with square signal");
    }
//    if (this->m_dims->x() < 6) {
//        // we need some edge around the biggest ring, to avoid invalid memory access
//        REPORT_ERROR(ERR_ARG_INCORRECT, "The input signal is too small.");
//    } // FIXME DS uncomment
    if (this->m_dims->isPadded()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Padded signal is not supported");
    }
    if (m_logicalSettings->sElemsBatch() > std::numeric_limits<int>::max()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Too big batch. It would cause int overflow in the cuda kernel");
    }
}

// explicit instantiation
template class CudaRotPolarEstimator<float>;
template class CudaRotPolarEstimator<double>;

} /* namespace Alignment */
