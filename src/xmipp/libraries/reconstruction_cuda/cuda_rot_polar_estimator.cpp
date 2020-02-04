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
void CudaRotPolarEstimator<T>::init2D() {
    release();
    auto s = this->getSettings();
    if (2 != s.hw.size()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Two GPU streams are needed");
    }
    try {
        m_mainStream = dynamic_cast<GPU*>(s.hw.at(0));
        m_backgroundStream = dynamic_cast<GPU*>(s.hw.at(1));
    } catch (std::bad_cast&) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Instance of GPU expected");
    }

    // all rings have the same number of samples, to make FT easier
    m_samples = std::max(1, 2 * (int)(M_PI * s.lastRing)); // keep this even
    auto batchPolar = FFTSettingsNew<T>(m_samples, 1, 1, // x, y, z
            s.otherDims.n() * s.getNoOfRings(), // each signal has N rings
            s.batch * s.getNoOfRings()); // so we have to multiply by that
    if (s.allowTuningOfNumberOfSamples) {
        // try to tune number of samples. We don't mind using more samples (higher precision)
        // if we can also do it faster!
        auto proposal = CudaFFT<T>::findOptimal(*m_mainStream,
            // test small batch, as we want the results as soon as possible (should be around 1s)
            batchPolar.createSubset(std::min((size_t)1000, batchPolar.sDim().n())),
            0, false, 10, false, false);
        if (proposal.has_value()) {
            batchPolar = FFTSettingsNew<T>(proposal.value().sDim().x(), 1, 1, // x, y, z
                        s.otherDims.n() * s.getNoOfRings(), // each signal has N rings
                        s.batch * s.getNoOfRings()); // so we have to multiply by that
            m_samples = batchPolar.sDim().x();
        }
    }

    auto singlePolar = FFTSettingsNew<T>(m_samples, 1, 1, // x, y, z
            s.getNoOfRings(), // each signal has N rings
            s.getNoOfRings()); // process all at once

    // allocate host memory
    gpuErrchk(cudaMalloc(&m_d_ref, singlePolar.fBytes())); // FT of the samples
    if (s.allowDataOverwrite) { // we need space just for reference(s)
        gpuErrchk(cudaMalloc(&m_d_batch, s.otherDims.sizeSingle() * s.refDims.n() * sizeof(T))); // Cartesian reference(s)
    } else { // we need space for the whole batch (which should be bigger than no of references)
        gpuErrchk(cudaMalloc(&m_d_batch, s.otherDims.sizeSingle() * s.batch * sizeof(T))); // Cartesian batch
    }
    gpuErrchk(cudaMalloc(&m_d_batchPolarFD, batchPolar.fBytesBatch())); // FT of the polar samples
    gpuErrchk(cudaMalloc(&m_d_batchPolarOrCorr, batchPolar.sBytesBatch())); // IFT of the polar samples
    size_t sumsBytes = m_samples * s.batch * sizeof(T);
    gpuErrchk(cudaMalloc(&m_d_sumsOrMaxPos, sumsBytes));
    gpuErrchk(cudaMalloc(&m_d_sumsSqr, sumsBytes));

    // FT plans
    m_singleToFD = CudaFFT<T>::createPlan(*m_mainStream, singlePolar);
    m_batchToFD = CudaFFT<T>::createPlan(*m_mainStream, batchPolar);
    auto inversePolar = FFTSettingsNew<T>(m_samples, 1, 1, // x, y, z
            s.batch, s.batch, // while computing correlation, we also sum the rings,
            false,false); // i.e. we end up with batch * samples elements
    m_batchToSD = CudaFFT<T>::createPlan(*m_mainStream, inversePolar);
    gpuErrchk(cudaMalloc(&m_d_batchCorrSumFD, inversePolar.fBytesBatch()));

    // allocate device memory
    m_h_batchMaxPositions = new float[m_samples * s.batch];

    // synch primitives
    m_mutex = new std::mutex();
    m_cv = new std::condition_variable();
}

template<typename T>
void CudaRotPolarEstimator<T>::release() {
    const auto &s = this->getSettings();
    // device memory
    gpuErrchk(cudaFree(m_d_batch));
    gpuErrchk(cudaFree(m_d_batchPolarOrCorr));
    gpuErrchk(cudaFree(m_d_batchPolarFD));
    gpuErrchk(cudaFree(m_d_ref));
    gpuErrchk(cudaFree(m_d_sumsOrMaxPos));
    gpuErrchk(cudaFree(m_d_sumsSqr));
    gpuErrchk(cudaFree(m_d_batchCorrSumFD));

    // FT plans
    CudaFFT<T>::release(m_singleToFD);
    CudaFFT<T>::release(m_batchToFD);
    CudaFFT<T>::release(m_batchToSD);

    delete[] m_h_batchMaxPositions;

    // synch primitives
    delete m_mutex;
    delete m_cv;

    setDefault();
}

template<typename T>
void CudaRotPolarEstimator<T>::setDefault() {
    m_mainStream = nullptr;
    m_backgroundStream = nullptr;

    // device memory
    m_d_batch = nullptr;
    m_d_batchPolarOrCorr = nullptr;
    m_d_batchPolarFD = nullptr;
    m_d_ref = nullptr;
    m_d_sumsOrMaxPos = nullptr;
    m_d_sumsSqr = nullptr;
    m_d_batchCorrSumFD = nullptr;

    // host memory
    m_h_batchMaxPositions = nullptr;

    // FT plans
    m_singleToFD = nullptr;
    m_batchToFD = nullptr;
    m_batchToSD = nullptr;

    m_samples = -1;

    // synch primitives
    m_mutex = nullptr;
    m_cv = nullptr;
    m_isDataReady = false;
}

template<typename T>
void CudaRotPolarEstimator<T>::load2DReferenceOneToN(const T *h_ref) {
    const bool isFullCircle = this->getSettings().fullCircle;
    if (isFullCircle) {
        load2DReferenceOneToN<true>(h_ref);
    } else {
        load2DReferenceOneToN<false>(h_ref);
    }
}

template<typename T>
template<bool FULL_CIRCLE>
void CudaRotPolarEstimator<T>::load2DReferenceOneToN(const T *h_ref) {
    auto isReady = this->isInitialized();
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to load a reference signal");
    }

    auto s = this->getSettings();
    auto inCart = s.refDims.copyForN(1);
    auto outPolar = Dimensions(m_samples, s.getNoOfRings(), 1, 1);
    m_mainStream->set();
    auto stream = *(cudaStream_t*)m_mainStream->stream();

    bool unlock = false;
    if ( ! GPU::isMemoryPinned(h_ref)) {
        unlock = true;
        m_mainStream->lockMemory(h_ref, inCart.size() * sizeof(T));
    }

    // copy memory
    gpuErrchk(cudaMemcpyAsync(
            m_d_batch, // reuse memory
            h_ref,
            inCart.size() * sizeof(T),
            cudaMemcpyHostToDevice, stream));
    if (unlock) {
        m_mainStream->unlockMemory(h_ref);
    }
    // call polar transformation kernel
    sComputePolarTransform<true>(*m_mainStream,
            inCart, m_d_batch,
            outPolar, m_d_batchPolarOrCorr,
            s.firstRing);
    // It seems that the normalization is not necessary (does not change the result
    // of the synthetic tests. It can however prevent e.g. overflow, hence it's kept here)
    // normalize data
//    sNormalize<FULL_CIRCLE>(*m_mainStream,
//            outPolar, m_d_batchPolarOrCorr,
//            m_d_sumsOrMaxPos, m_d_sumsSqr,
//            s.firstRing);
    CudaFFT<T>::fft(*m_singleToFD, m_d_batchPolarOrCorr, m_d_ref);
}

template<typename T>
void CudaRotPolarEstimator<T>::sComputeCorrelationsOneToN(
        const GPU &gpu,
        const std::complex<T> *d_in,
        std::complex<T> *d_out,
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
            (float2*)d_in, (float2*)d_out,
            (float2*)d_ref,
            firstRingRadius,
            dims.x(), dims.y(), dims.n());
    } else if (std::is_same<T, double>::value) {
        computePolarCorrelationsSumOneToNKernel<double2>
            <<<dimGrid, dimBlock, 0, stream>>> (
            (double2*)d_in, (double2*)d_out,
            (double2*)d_ref,
            firstRingRadius,
            dims.x(), dims.y(), dims.n());
    } else {
            REPORT_ERROR(ERR_TYPE_INCORRECT, "Not implemented");
    }
}

template<typename T>
void CudaRotPolarEstimator<T>::loadThreadRoutine(T *others) {
    m_backgroundStream->set();
    auto lStream = *(cudaStream_t*)m_backgroundStream->stream();
    auto s = this->getSettings();
    for (size_t offset = 0; offset < s.otherDims.n(); offset += s.batch) {
        std::unique_lock<std::mutex> lk(*m_mutex);
        // wait till the data is processed
        m_cv->wait(lk, [&]{return !m_isDataReady;});
        // how many signals to process
        size_t toProcess = std::min(s.batch, s.otherDims.n() - offset);

        T *src = others + offset * s.otherDims.sizeSingle();
        size_t bytes = toProcess * s.otherDims.sizeSingle() * sizeof(T);

        auto kind = m_backgroundStream->isGpuPointer(src)
                ? cudaMemcpyDeviceToDevice
                : cudaMemcpyHostToDevice;
        // copy memory
        gpuErrchk(cudaMemcpyAsync(
                m_d_batch,
                src,
                bytes,
                kind, lStream));
        // block until data is loaded
        m_backgroundStream->synch();

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
    assert (dimIn.x() == dimIn.y());
    assert ((dimOut.y() + 1) * 2 <= dimIn.x()); // assert that there's space around the biggest ring
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
        T * __restrict__ d_sums,
        T * __restrict__ d_sumsSqr,
        int posOfFirstRing) {
    dim3 dimBlock(64);
    dim3 dimGrid(
        ceil((dim.x() * dim.n()) / (float)dimBlock.x));

    auto stream = *(cudaStream_t*)gpu.stream();
    // clear the arrays
    size_t bytes = dim.n() * sizeof(T);
    gpuErrchk(cudaMemset(d_sums, 0, bytes));
    gpuErrchk(cudaMemset(d_sumsSqr, 0, bytes));
    computeSumSumSqr<T, FULL_CIRCLE>
        <<<dimGrid, dimBlock, 0, stream>>> (
        d_in, dim.x(), dim.y(), dim.n(),
        d_sums, d_sumsSqr, posOfFirstRing);

    const T piConst = FULL_CIRCLE ? (2 * M_PI) : M_PI;
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
        d_sums, d_sumsSqr);
}

template<typename T>
void CudaRotPolarEstimator<T>::computeRotation2DOneToN(T *others) {
    const bool isFullCircle = this->getSettings().fullCircle;
    if (isFullCircle) {
        computeRotation2DOneToN<true>(others);
    } else {
        computeRotation2DOneToN<false>(others);
    }
}

template<typename T>
template<bool FULL_CIRCLE>
void CudaRotPolarEstimator<T>::waitAndConvert(
        const Dimensions &inCart,
        const Dimensions &outPolar,
        unsigned firstRing) {
    // block until data is loaded
    // mutex will be freed once leaving this block
    std::unique_lock<std::mutex> lk(*m_mutex);
    m_cv->wait(lk, [&]{return m_isDataReady;});
    // call polar transformation kernel
    sComputePolarTransform<FULL_CIRCLE>(*m_mainStream,
            inCart, m_d_batch,
            outPolar, m_d_batchPolarOrCorr,
            firstRing);

    // notify that buffer is processed (new will be loaded in background)
    m_mainStream->synch();
    m_isDataReady = false;
    m_cv->notify_one();
}

template<typename T>
template<bool FULL_CIRCLE>
void CudaRotPolarEstimator<T>::computeRotation2DOneToN(T *others) {
    const auto &s = this->getSettings();
    bool isReady = this->isInitialized() && this->isRefLoaded();
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() and load reference");
    }
    if ( ! GPU::isMemoryPinned(others)) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Input memory has to be pinned (page-locked)");
    }
    if (s.allowDataOverwrite && ( ! m_mainStream->isGpuPointer(others))) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Incompatible parameters: allowDataOverwrite && 'others' data on host");
    }

    m_mainStream->set();
    // make sure that all work (e.g. loading and processing of the reference) is done
    m_mainStream->synch();
    // start loading data at the background
    m_isDataReady = false;
    const bool createLocalCopy = ( ! s.allowDataOverwrite);
    std::thread loadingThread;
    if (createLocalCopy) {
        loadingThread = std::thread(&CudaRotPolarEstimator<T>::loadThreadRoutine, this, others);
    }

    // process signals in batches
    for (size_t offset = 0; offset < s.otherDims.n(); offset += s.batch) {
        // how many signals to process
        size_t toProcess = std::min(s.batch, s.otherDims.n() - offset);
        auto inCart = s.otherDims.copyForN(toProcess);
        auto outPolar = Dimensions(m_samples, s.getNoOfRings(), 1, toProcess);
        auto outPolarFourier = Dimensions(m_samples / 2 + 1, s.getNoOfRings(), 1, toProcess);
        auto resSize = Dimensions(m_samples, 1, 1, toProcess);

        // get proper data
        if (createLocalCopy) {
            waitAndConvert<FULL_CIRCLE>(inCart, outPolar, s.firstRing);
        } else {
            T *in = others + (offset * s.otherDims.sizeSingle());
            sComputePolarTransform<FULL_CIRCLE>(*m_mainStream,
                inCart, in,
                outPolar, m_d_batchPolarOrCorr,
                s.firstRing);
        }

        // It seems that the normalization is not necessary (does not change the result
        // of the synthetic tests. It can however prevent e.g. overflow, hence it's kept here)
//        sNormalize<FULL_CIRCLE>(*m_mainStream,
//                outPolar, m_d_batchPolarOrCorr,
//                m_d_sumsOrMaxPos, m_d_sumsSqr,
//                s.firstRing);

        CudaFFT<T>::fft(*m_batchToFD, m_d_batchPolarOrCorr, m_d_batchPolarFD);

        sComputeCorrelationsOneToN(*m_mainStream, m_d_batchPolarFD, m_d_batchCorrSumFD, m_d_ref, outPolarFourier, s.firstRing);

        CudaFFT<T>::ifft(*m_batchToSD, m_d_batchCorrSumFD, m_d_batchPolarOrCorr);

        // locate maxima for each signal (results are always floats)
        auto d_positions = (float*)m_d_sumsOrMaxPos;
        ExtremaFinder::CudaExtremaFinder<T>::sFindMax(
                *m_mainStream, resSize, m_d_batchPolarOrCorr, d_positions, nullptr);

        // copy data back (they are always float)
        auto stream = *(cudaStream_t*)m_mainStream->stream();
        gpuErrchk(cudaMemcpyAsync(
                m_h_batchMaxPositions,
                d_positions,
                resSize.n() * sizeof(float), // one position per signal
                cudaMemcpyDeviceToHost, stream));
        m_mainStream->synch();

        // convert positions of the maxima to angles
        for (int n = 0; n < resSize.n(); ++n) {
            auto angle = m_h_batchMaxPositions[n] * (FULL_CIRCLE ? (T)360 : (T)180) / resSize.x();
            this->getRotations2D().emplace_back(angle);
        }
    }
    if (createLocalCopy) {
        loadingThread.join();
    }
}

template<typename T>
void CudaRotPolarEstimator<T>::check() {
    auto s = this->getSettings();
    if (s.refDims.x() != s.refDims.y()) {
        // because of the rings
        REPORT_ERROR(ERR_ARG_INCORRECT, "This estimator can work only with square signal");
    }
    if (s.refDims.x() < 6) {
        // we need some edge around the biggest ring, to avoid invalid memory access
        REPORT_ERROR(ERR_ARG_INCORRECT, "The input signal is too small.");
    }
    if (s.refDims.isPadded()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Padded signal is not supported");
    }
    if ((s.lastRing + 1) * 2 >= s.refDims.x()) {
        // bilinear interpolation we use does not check boundaries
        REPORT_ERROR(ERR_ARG_INCORRECT, "Last ring is too big, this would cause invalid memory access");
    }
}

// explicit instantiation
template class CudaRotPolarEstimator<float>;
template class CudaRotPolarEstimator<double>;

} /* namespace Alignment */
