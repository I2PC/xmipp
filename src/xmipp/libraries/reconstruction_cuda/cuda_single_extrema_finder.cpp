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

#include "cuda_single_extrema_finder.cu"
#include "cuda_single_extrema_finder.h"
#include "reconstruction_cuda/cuda_asserts.h"

namespace ExtremaFinder {

template<typename T>
void CudaExtremaFinder<T>::setDefault() {
    m_loadStream = nullptr;
    m_workStream = nullptr;

    // device memory
    m_d_values = nullptr;
    m_d_positions = nullptr;
    m_d_batch = nullptr;

    // synch primitives
    m_mutex = nullptr;
    m_cv = nullptr;
    m_isDataReady = false;

    // host memory
    m_h_batchResult = nullptr;
}

template<typename T>
void CudaExtremaFinder<T>::release() {
    // device memory
    gpuErrchk(cudaFree(m_d_values));
    gpuErrchk(cudaFree(m_d_positions));
    gpuErrchk(cudaFree(m_d_batch));

    // synch primitives
    delete m_mutex;
    delete m_cv;

    // host memory
    if (nullptr != m_h_batchResult) {
        m_loadStream->unpinMemory(m_h_batchResult);
        free(m_h_batchResult);
    }

    setDefault();
}

template<typename T>
void CudaExtremaFinder<T>::check() const {
    // nothing to do
}

template<typename T>
void CudaExtremaFinder<T>::initMax() {
    return initBasic();
}

template<typename T>
void CudaExtremaFinder<T>::findMax(const T *__restrict h_data) {
    auto kernel = [&](const GPU &gpu,
            const Dimensions &dims,
            const T * __restrict__ d_data,
            float * __restrict__ d_positions,
            T * __restrict__ d_values) {
        sFindMax(gpu, dims, d_data, d_positions, d_values);
    };
    return findBasic(h_data, kernel);
}

template<typename T>
bool CudaExtremaFinder<T>::canBeReusedMax(const ExtremaFinderSettings &s) const {
    return canBeReusedBasic(s);
}

template<typename T>
void CudaExtremaFinder<T>::initLowest() {
    return initBasic();
}

template<typename T>
void CudaExtremaFinder<T>::findLowest(const T *__restrict h_data) {
    auto kernel = [&](const GPU &gpu,
            const Dimensions &dims,
            const T * __restrict__ d_data,
            float * __restrict__ d_positions,
            T * __restrict__ d_values) {
        sFindLowest(gpu, dims, d_data, d_positions, d_values);
    };
    return findBasic(h_data, kernel);
}

template<typename T>
bool CudaExtremaFinder<T>::canBeReusedLowest(const ExtremaFinderSettings &s) const {
    return canBeReusedBasic(s);
}

template<typename T>
void CudaExtremaFinder<T>::initMaxAroundCenter() {
    return initBasic();
}

template<typename T>
void CudaExtremaFinder<T>::findMaxAroundCenter(const  T *__restrict__ h_data) {
    auto kernel2D = [&](const GPU &gpu,
            const Dimensions &dims,
            const T * __restrict__ d_data,
            float * __restrict__ d_positions,
            T * __restrict__ d_values) {
        sFindMax2DAroundCenter(gpu, dims, d_data, d_positions, d_values,
                this->getSettings().maxDistFromCenter);
    };
    if (this->getSettings().dims.is2D()) {
        return findBasic(h_data, kernel2D);
    }
    REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Not implemented");
}

template<typename T>
bool CudaExtremaFinder<T>::canBeReusedMaxAroundCenter(const ExtremaFinderSettings &s) const {
    return canBeReusedBasic(s);
}

template<typename T>
void CudaExtremaFinder<T>::initLowestAroundCenter() {
    return initBasic();
}

template<typename T>
void CudaExtremaFinder<T>::findLowestAroundCenter(const  T *__restrict__ h_data) {
    auto kernel2D = [&](const GPU &gpu,
            const Dimensions &dims,
            const T * __restrict__ d_data,
            float * __restrict__ d_positions,
            T * __restrict__ d_values) {
        sFindLowest2DAroundCenter(gpu, dims, d_data, d_positions, d_values,
                this->getSettings().maxDistFromCenter);
    };
    if (this->getSettings().dims.is2D()) {
        return findBasic(h_data, kernel2D);
    }
    REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Not implemented");
}

template<typename T>
bool CudaExtremaFinder<T>::canBeReusedLowestAroundCenter(const ExtremaFinderSettings &s) const {
    return canBeReusedBasic(s);
}

template<typename T>
void CudaExtremaFinder<T>::initBasic() {
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

    size_t bytesBatch = s.dims.sizeSingle() * s.batch * sizeof(T);
    size_t bytesResult = s.batch * std::max(sizeof(T), sizeof(float));
    // device memory
    gpuErrchk(cudaMalloc(&m_d_batch, bytesBatch));
    gpuErrchk(cudaMalloc(&m_d_positions, bytesResult));
    gpuErrchk(cudaMalloc(&m_d_values, bytesResult));

    // synch primitives
    m_mutex = new std::mutex();
    m_cv = new std::condition_variable();

    // host memory
    m_h_batchResult = (T*)memoryUtils::page_aligned_alloc(bytesResult);
    // to make Valgrind happy (otherwise uninitialized memory access)
    memset(m_h_batchResult, 0, bytesResult);
    m_loadStream->pinMemory(m_h_batchResult, bytesResult);
}

template<typename T>
template<typename KERNEL>
void CudaExtremaFinder<T>::findBasic(const T * __restrict__ h_data, const KERNEL &k) {
    bool isReady = this->isInitialized();
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() first");
    }
    if ( ! GPU::isMemoryPinned(h_data)) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Input memory has to be pinned (page-locked)");
    }
    m_workStream->set();
    m_loadStream->set();
    // start loading data at the background
    m_isDataReady = false;
    auto loadingThread = std::thread(&CudaExtremaFinder<T>::loadThreadRoutine, this, h_data);

    auto s = this->getSettings();
    // process signals in batches
    for (size_t offset = 0; offset < s.dims.n(); offset += s.batch) {
        // how many signals to process
        size_t toProcess = std::min(s.batch, s.dims.n() - offset);
        auto batchDims = s.dims.copyForN(toProcess);
        {
            // block until data is loaded
            // mutex will be freed once leaving this block
            std::unique_lock<std::mutex> lk(*m_mutex);
            m_cv->wait(lk, [&]{return m_isDataReady;});
            // call finding kernel
            k(*m_workStream, batchDims, m_d_batch,
                   m_d_positions, m_d_values);

            // notify that buffer is processed (new will be loaded in background)
            m_workStream->synch();
            m_isDataReady = false;
            m_cv->notify_one();
        }
        downloadPositionsFromGPU(offset, toProcess);
        downloadValuesFromGPU(offset, toProcess);
    }
    loadingThread.join();
}

template<typename T>
bool CudaExtremaFinder<T>::canBeReusedBasic(const ExtremaFinderSettings &s) const {
    if (2 != s.hw.size()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Two GPU streams are needed");
    }
    GPU* newWorkStream = nullptr;
    GPU* newLoadStream = nullptr;
    try {
        newWorkStream = dynamic_cast<GPU*>(s.hw.at(0));
        newLoadStream = dynamic_cast<GPU*>(s.hw.at(1));
    } catch (std::bad_cast&) {
        return false;
    }
    auto oldBatch = this->getSettings().batch;
    auto oldBatchSize = this->getSettings().dims.sizeSingle() * oldBatch;
    return ((newWorkStream == m_workStream)
            && (newLoadStream == m_loadStream)
            && (oldBatch >= s.batch)
            && (oldBatchSize >= (s.dims.sizeSingle() * s.batch)));
}

template<typename T>
void CudaExtremaFinder<T>::downloadPositionsFromGPU(size_t offset, size_t count) {
    size_t bytesResult = count * sizeof(float);
    auto lStream = *(cudaStream_t*)m_loadStream->stream();
    auto s = this->getSettings();
    if ((ResultType::Position == s.resultType)
        || (ResultType::Both == s.resultType)) {
        // copy to buffer which is page-aligned, as we will work asynchronously
        gpuErrchk(cudaMemcpyAsync(
            m_h_batchResult,
            m_d_positions,
            bytesResult,
            cudaMemcpyDeviceToHost, lStream));
        m_loadStream->synch();
        memcpy(this->getPositions().data() + offset, m_h_batchResult, bytesResult);
    }
}

template<typename T>
void CudaExtremaFinder<T>::downloadValuesFromGPU(size_t offset, size_t count) {
    size_t bytesResult = count * sizeof(T);
    auto lStream = *(cudaStream_t*)m_loadStream->stream();
    auto s = this->getSettings();
    if ((ResultType::Value == s.resultType)
        || (ResultType::Both == s.resultType)) {
        // copy to buffer which is page-aligned, as we will work asynchronously
        gpuErrchk(cudaMemcpyAsync(
            m_h_batchResult,
            m_d_values,
            bytesResult,
            cudaMemcpyDeviceToHost, lStream));
        m_loadStream->synch();
        memcpy(this->getValues().data() + offset, m_h_batchResult, bytesResult);
    }
}

template<typename T>
void CudaExtremaFinder<T>::loadThreadRoutine(const T *h_data) {
    m_loadStream->set();
    auto lStream = *(cudaStream_t*)m_loadStream->stream();
    auto s = this->getSettings();
    for (size_t offset = 0; offset < s.dims.n(); offset += s.batch) {
        std::unique_lock<std::mutex> lk(*m_mutex);
        // wait till the data is processed
        m_cv->wait(lk, [&]{return !m_isDataReady;});
        // how many signals to process
        size_t toProcess = std::min(s.batch, s.dims.n() - offset);

        const T *h_src = h_data + offset * s.dims.sizeSingle();
        size_t bytes = toProcess * s.dims.sizeSingle() * sizeof(T);

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
size_t CudaExtremaFinder<T>::ceilPow2(size_t x)
{
    if (x <= 1) return 1;
    int power = 2;
    x--;
    while (x >>= 1) power <<= 1;
    return power;
}

template<typename T>
void CudaExtremaFinder<T>::sFindMax(const GPU &gpu,
        const Dimensions &dims,
        const T * __restrict__ d_data,
        float * __restrict__ d_positions,
        T * __restrict__ d_values) {
    return sFindUniversal([] __device__ (T l, T r) { return l > r; },
        std::numeric_limits<T>::lowest(),
        gpu, dims, d_data, d_positions, d_values);
}

template<typename T>
void CudaExtremaFinder<T>::sFindLowest(const GPU &gpu,
        const Dimensions &dims,
        const T * __restrict__ d_data,
        float * __restrict__ d_positions,
        T * __restrict__ d_values) {
    return sFindUniversal([] __device__ (T l, T r) { return l < r; },
        std::numeric_limits<T>::max(),
        gpu, dims, d_data, d_positions, d_values);
}

template<typename T>
template<typename C>
void CudaExtremaFinder<T>::sFindUniversal(
        const C &comp,
        T startVal,
        const GPU &gpu,
        const Dimensions &dims,
        const T * __restrict__ d_data,
        float * __restrict__ d_positions,
        T * __restrict__ d_values) {
    // check input
    assert(dims.sizeSingle() > 0);
    assert(dims.n() > 0);
    assert(nullptr != d_data);
    assert((nullptr != d_positions) || (nullptr != d_values));
    assert(dims.size() <= std::numeric_limits<unsigned>::max()); // indexing overflow in the kernel

    // create threads / blocks
    size_t maxThreads = 512;
    size_t threads = (dims.sizeSingle() < maxThreads) ? ceilPow2(dims.sizeSingle()) : maxThreads;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(dims.n(), 1, 1);
    auto stream = *(cudaStream_t*)gpu.stream();

    // for each thread, we need two variables in shared memory
    size_t smemSize = 2 * threads * sizeof(T);
    switch (threads) {
        case 512:
            return findUniversal<T, 512><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
        case 256:
            return findUniversal<T, 256><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
        case 128:
            return findUniversal<T, 128><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
        case 64:
            return findUniversal<T, 64><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
        case 32:
            return findUniversal<T, 32><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
        case 16:
            return findUniversal<T, 16><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
        case 8:
            return findUniversal<T, 8><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
        case 4:
            return findUniversal<T, 4><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
        case 2:
            return findUniversal<T, 2><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
        case 1:
            return findUniversal<T, 1><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.sizeSingle());
        default: REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Unsupported number of threads");
    }
}

template<typename T>
void CudaExtremaFinder<T>::sFindMax2DAroundCenter(
        const GPU &gpu,
        const Dimensions &dims,
        const T * d_data,
        float * d_positions,
        T * d_values,
        size_t maxDist) {
    return sFindUniversal2DAroundCenter([] __device__ (T l, T r) { return l > r; },
            std::numeric_limits<T>::lowest(),
            gpu, dims, d_data, d_positions, d_values, maxDist);
}

template<typename T>
void CudaExtremaFinder<T>::sFindLowest2DAroundCenter(
        const GPU &gpu,
        const Dimensions &dims,
        const T * d_data,
        float * d_positions,
        T * d_values,
        size_t maxDist) {
    return sFindUniversal2DAroundCenter([] __device__ (T l, T r) { return l < r; },
            std::numeric_limits<T>::max(),
            gpu, dims, d_data, d_positions, d_values, maxDist);
}

template<typename T>
template<typename C>
void CudaExtremaFinder<T>::sFindUniversal2DAroundCenter(
        const C &comp,
        T startVal,
        const GPU &gpu,
        const Dimensions &dims,
        const T * d_data,
        float * d_positions,
        T * d_values,
        size_t maxDist) {
    // check input
    assert(dims.is2D());
    assert( ! dims.isPadded());
    assert(dims.sizeSingle() > 0);
    assert(dims.n() > 0);
    assert(nullptr != d_data);
    assert((nullptr != d_positions) || (nullptr != d_values));
    assert(0 < maxDist);
    int xHalf = dims.x() / 2;
    int yHalf = dims.y() / 2;
    assert((2 * xHalf) > maxDist);
    assert((2 * yHalf) > maxDist);

    // prepare threads / blocks
    size_t maxThreads = 512;
    size_t windowWidth = 2 * maxDist;
    // threads should process a single row of the signal
    size_t threads = (windowWidth < maxThreads) ? ceilPow2(windowWidth) : maxThreads;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(dims.n(), 1, 1);
    auto stream = *(cudaStream_t*)gpu.stream();

    // for each thread, we need two variables in shared memory
    int smemSize = 2 * threads * sizeof(T);
    switch (threads) {
        case 512:
            return findUniversal2DNearCenter<T, 512><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 256:
            return findUniversal2DNearCenter<T, 256><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 128:
            return findUniversal2DNearCenter<T, 128><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 64:
            return findUniversal2DNearCenter<T, 64><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 32:
            return findUniversal2DNearCenter<T, 32><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 16:
            return findUniversal2DNearCenter<T, 16><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 8:
            return findUniversal2DNearCenter<T, 8><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 4:
            return findUniversal2DNearCenter<T, 4><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 2:
            return findUniversal2DNearCenter<T, 2><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        case 1:
            return findUniversal2DNearCenter<T, 1><<< dimGrid, dimBlock, smemSize, stream>>>(
                comp, startVal, d_data, d_positions, d_values, dims.x(), dims.y(), maxDist);
        default: REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Unsupported number of threads");
    }
}

// explicit instantiation
template class CudaExtremaFinder<float>;
template class CudaExtremaFinder<double>;

} /* namespace ExtremaFinder */
