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
void CudaRotPolarEstimator<T>::init2D(bool reuse) {
    // FIXME DS implement reuse (properly)
    release();
    auto s = this->getSettings();
    if (2 != s.hw.size()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Two GPU streams are needed");
    }
    try {
        m_workStream = dynamic_cast<GPU*>(s.hw.at(0));
        m_loadStream = dynamic_cast<GPU*>(s.hw.at(1));
    } catch (std::bad_cast&) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Instance of GPU expected");
    }

    m_firstRing = s.refDims.x() / 5;
    m_lastRing = (s.refDims.x() - 3) / 2; // so that we have some edge around the biggest ring
    // all rings have the same number of samples, to make FT easier
    m_samples = std::max(1, 2 * (int)(M_PI * m_lastRing)); // keep this even

    auto batchPolar = FFTSettingsNew<T>(m_samples, 1, 1, // x, y, z
            s.otherDims.n() * getNoOfRings(), // each signal has N rings
            s.batch * getNoOfRings()); // so we have to multiply by that
    // try to tune number of samples. We don't mind using more samples (higher precision)
    // if we can also do it faster!
    auto proposal = CudaFFT<T>::findOptimal(*m_workStream,
        // test small batch, as we want the results as soon as possible (should be around 1s)
        batchPolar.createSubset(std::min((size_t)1000, batchPolar.sDim().n())),
        0, false, 10, false, false);
    if (proposal.has_value()) {
        batchPolar = FFTSettingsNew<T>(proposal.value().sDim().x(), 1, 1, // x, y, z
                    s.otherDims.n() * getNoOfRings(), // each signal has N rings
                    s.batch * getNoOfRings()); // so we have to multiply by that
        m_samples = batchPolar.sDim().x();
    }

    auto singlePolar = FFTSettingsNew<T>(m_samples, 1, 1, // x, y, z
            getNoOfRings(), // each signal has N rings
            getNoOfRings()); // process all at once

    // allocate host memory
    gpuErrchk(cudaMalloc(&m_d_ref, singlePolar.fBytes())); // FT of the samples
    gpuErrchk(cudaMalloc(&m_d_batch, s.otherDims.sizeSingle() * s.batch * sizeof(T))); // Cartesian batch
    gpuErrchk(cudaMalloc(&m_d_batchPolarFD, batchPolar.fBytesBatch())); // FT of the polar samples
    gpuErrchk(cudaMalloc(&m_d_batchPolarOrCorr, batchPolar.sBytesBatch())); // IFT of the polar samples

    // FT plans
    m_singleToFD = CudaFFT<T>::createPlan(*m_workStream, singlePolar);
    m_batchToFD = CudaFFT<T>::createPlan(*m_workStream, batchPolar);
    m_batchToSD = CudaFFT<T>::createPlan(*m_workStream, batchPolar.createInverse());

    // allocate device memory
    m_h_batchResult = new T[m_samples * s.batch];

    // synch primitives
    m_mutex = new std::mutex();
    m_cv = new std::condition_variable();
}

template<typename T>
void CudaRotPolarEstimator<T>::release() {
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

    setDefault();
}

template<typename T>
void CudaRotPolarEstimator<T>::setDefault() {
    m_workStream = nullptr;
    m_loadStream = nullptr;

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
    m_samples = -1;

    // synch primitives
    m_mutex = nullptr;
    m_cv = nullptr;
    m_isDataReady = false;
}

template<typename T>
void CudaRotPolarEstimator<T>::load2DReferenceOneToN(const T *h_ref) {
    auto isReady = this->isInitialized();
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to load a reference signal");
    }

    auto inCart = this->getSettings().refDims.copyForN(1);
    auto outPolar = Dimensions(m_samples, getNoOfRings(), 1, 1);
    m_loadStream->set();
    m_workStream->set();
    auto loadStream = *(cudaStream_t*)m_loadStream->stream();
    auto workStream = *(cudaStream_t*)m_workStream->stream();

    bool unlock = false;
    if ( ! GPU::isMemoryPinned(h_ref)) {
        unlock = true;
        m_loadStream->lockMemory(h_ref, inCart.size() * sizeof(T));
    }

    // copy memory
    gpuErrchk(cudaMemcpyAsync(
            m_d_batch, // reuse memory
            h_ref,
            inCart.size() * sizeof(T),
            cudaMemcpyHostToDevice, loadStream));
    // wait for data and unlock memory, if necessary
    m_loadStream->synch();
    if (unlock) {
        m_loadStream->unlockMemory(h_ref);
    }

    // call polar transformation kernel
    sComputePolarTransform<true>(*m_workStream,
            inCart, m_d_batch,
            outPolar, m_d_batchPolarOrCorr,
            m_firstRing);

    // FIXME DS add normalization
    CudaFFT<T>::fft(*m_singleToFD, m_d_batchPolarOrCorr, m_d_ref);
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
        ceil((dims.x() * dims.n()) / (float)dimBlock.x));
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
void CudaRotPolarEstimator<T>::loadThreadRoutine(T *h_others) {
    m_loadStream->set();
    auto lStream = *(cudaStream_t*)m_loadStream->stream();
    auto s = this->getSettings();
    for (size_t offset = 0; offset < s.otherDims.n(); offset += s.batch) {
        std::unique_lock<std::mutex> lk(*m_mutex);
        // wait till the data is processed
        m_cv->wait(lk, [&]{return !m_isDataReady;});
        // how many signals to process
        size_t toProcess = std::min(s.batch, s.otherDims.n() - offset);

        T *h_src = h_others + offset * s.otherDims.sizeSingle();
        size_t bytes = toProcess * s.otherDims.sizeSingle() * sizeof(T);

        // copy memory
        gpuErrchk(cudaMemcpyAsync(
                m_d_batch,
                h_src,
                bytes,
                cudaMemcpyHostToDevice, lStream));
        // block until data is loaded
        m_loadStream->synch();

        // notify processing stream that it can work
        m_isDataReady = true;
        m_cv->notify_one();
    }
}

template<typename T>
template<bool FULL_CIRCLE>
void CudaRotPolarEstimator<T>::sComputePolarTransform(
        const GPU &gpu,
        const Dimensions &dimIn,
        T * __restrict__ d_in,
        const Dimensions &dimOut,
        T * __restrict__ d_out,
        int posOfFirstRing) {

    dim3 dimBlock(32);
    dim3 dimGrid(
        ceil((dimOut.x() * dimOut.n()) / (float)dimBlock.x));

    auto stream = *(cudaStream_t*)gpu.stream();

    polarFromCartesian<T, FULL_CIRCLE>
        <<<dimGrid, dimBlock, 0, stream>>> (
        d_in, dimIn.x(), dimIn.y(),
        d_out, dimOut.x(), dimOut.y(), dimOut.n(), posOfFirstRing);
}

template<typename T>
template<bool FULL_CIRCLE>
void CudaRotPolarEstimator<T>::sNormalize(
        const GPU &gpu,
        const Dimensions &dim,
        T * __restrict__ d_in,
        T * __restrict__ d_1,
        T * __restrict__ d_2,
        int posOfFirstRing) {
    T *d_outAvg;
    T *d_outStddev;
    size_t elems = dim.x() * dim.n();
    size_t bytes = elems * sizeof(T);
    gpuErrchk(cudaMalloc(&d_outAvg, bytes));
    gpuErrchk(cudaMemset(d_outAvg, 0, bytes));
    gpuErrchk(cudaMalloc(&d_outStddev, bytes));
    gpuErrchk(cudaMemset(d_outStddev, 0, bytes));

    dim3 dimBlock(64);
    dim3 dimGrid(
        ceil((dim.x() * dim.n()) / (float)dimBlock.x));

    auto stream = *(cudaStream_t*)gpu.stream();

    computeSumSumSqr<T, FULL_CIRCLE>
        <<<dimGrid, dimBlock, 0, stream>>> (
        d_in, dim.x(), dim.y(), dim.n(),
        d_outAvg, d_outStddev, posOfFirstRing);

    const T piConst = FULL_CIRCLE ? (2 * PI) : PI;
    // sum of the first n terms of an arithmetic sequence
    // a1 = first radius (posOfFirstRing)
    // an = last radius (dim.y() - 1 + posOfFirstRing)
    // s = n * (a1 + an) / 2
    const T radiiSum = (dim.y() * (2 * posOfFirstRing + dim.y() - 1)) / (T)2;
    T norm = piConst * radiiSum;

    normalize<T>
        <<<dimGrid, dimBlock, 0, stream>>> (
        d_in, dim.x(), dim.y(), dim.n(),
        norm,
        d_outAvg, d_outStddev);

    gpuErrchk(cudaFree(d_outAvg));
    gpuErrchk(cudaFree(d_outStddev));
}

template<typename T>
template<bool FULL_CIRCLE>
std::vector<T> CudaRotPolarEstimator<T>::sFindMaxAngle(
        const Dimensions &dims,
        T *polarCorrelations) {
    assert(dims.is1D());
    auto result = std::vector<T>();
    result.reserve(dims.n());

    // locate max angle
    for (size_t offset = 0; offset < dims.size(); offset += dims.x()) {
        auto start = polarCorrelations + offset;
        auto max = std::max_element(start, start + dims.x());
        auto pos = std::distance(start, max);
        T angle = pos * (FULL_CIRCLE ? (T)360 : (T)180) / dims.x();
        result.emplace_back(angle);
    }
    return result;
}

template<typename T>
void CudaRotPolarEstimator<T>::computeRotation2DOneToN(T *h_others) {
    bool isReady = this->isInitialized() && this->isRefLoaded();

    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() and load reference");
    }
    if ( ! GPU::isMemoryPinned(h_others)) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Input memory has to be pinned (page-locked)");
    }
    m_workStream->set();
    // start loading data at the background
    m_isDataReady = false;
    auto loadingThread = std::thread(&CudaRotPolarEstimator<T>::loadThreadRoutine, this, h_others);

    const bool fullCircle = true;

    auto s = this->getSettings();
    // process signals in batches
    for (size_t offset = 0; offset < s.otherDims.n(); offset += s.batch) {
        // how many signals to process
        size_t toProcess = std::min(s.batch, s.otherDims.n() - offset);
        auto inCart = s.otherDims.copyForN(toProcess);
        auto outPolar = Dimensions(m_samples, getNoOfRings(), 1, toProcess);
        auto outPolarFourier = Dimensions(m_samples / 2 + 1, getNoOfRings(), 1, toProcess);
        auto resSize = Dimensions(m_samples, 1, 1, toProcess);

        {
            // block until data is loaded
            // mutex will be freed once leaving this block
            std::unique_lock<std::mutex> lk(*m_mutex);
            m_cv->wait(lk, [&]{return m_isDataReady;});
            // call polar transformation kernel
            sComputePolarTransform<fullCircle>(*m_workStream,
                    inCart, m_d_batch,
                    outPolar, m_d_batchPolarOrCorr,
                    m_firstRing);

            // notify that buffer is processed (new will be loaded in background)
            m_workStream->synch();
            m_isDataReady = false;
            m_cv->notify_one();
        }

        // FIXME DS normalize
        sNormalize<fullCircle>(*m_workStream, outPolar, m_d_batchPolarOrCorr, nullptr, nullptr, m_firstRing);

        CudaFFT<T>::fft(*m_batchToFD, m_d_batchPolarOrCorr, m_d_batchPolarFD);

        sComputeCorrelationsOneToN(*m_workStream, m_d_batchPolarFD, m_d_ref, outPolarFourier, m_firstRing);

        CudaFFT<T>::ifft(*m_batchToSD, m_d_batchPolarFD, m_d_batchPolarOrCorr);

        // copy data back
        m_workStream->synch();
        auto loadStream = *(cudaStream_t*)m_loadStream->stream();
        gpuErrchk(cudaMemcpyAsync(
                m_h_batchResult,
                m_d_batchPolarOrCorr,
                resSize.size() * sizeof(T),
                cudaMemcpyDeviceToHost, loadStream));
        m_loadStream->synch();

        // extract angles
        auto angles = sFindMaxAngle<true>(resSize, m_h_batchResult);
        this->getRotations2D().insert(this->getRotations2D().end(), angles.begin(), angles.end());
    }
    loadingThread.join();
}

template<typename T>
void CudaRotPolarEstimator<T>::check() {
    auto dims = this->getSettings().refDims;
    if (dims.x() != dims.y()) {
        // because of the rings
        REPORT_ERROR(ERR_ARG_INCORRECT, "This estimator can work only with square signal");
    }
    if (dims.x() < 6) {
        // we need some edge around the biggest ring, to avoid invalid memory access
        REPORT_ERROR(ERR_ARG_INCORRECT, "The input signal is too small.");
    }
    if (dims.isPadded()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Padded signal is not supported");
    }
}

// explicit instantiation
template class CudaRotPolarEstimator<float>;
template class CudaRotPolarEstimator<double>;

} /* namespace Alignment */
