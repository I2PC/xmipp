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
#include "cuda_correlation_computer.h"
#include "cuda_correlation.cu"

template<typename T>
template<bool NORMALIZE>
void CudaCorrelationComputer<T>::computeOneToN() {
    if (NORMALIZE) {
        computeCorrStatOneToNNormalize();
    } else {
        REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Funnily, this should be easier than what we already can do :)");
    }
    storeResultOneToN<NORMALIZE>();
}

template<typename T>
template<bool NORMALIZE>
void CudaCorrelationComputer<T>::storeResultOneToN() {
    auto &res = this->getFiguresOfMerit();
    const size_t noOfSignals = this->getSettings().otherDims.n();
    const size_t elems = this->getSettings().refDims.sizeSingle();
    if (NORMALIZE) { // we must normalize the output
        auto ref = computeStat(m_h_ref_corrRes[0], elems); // single result
        auto others = (ResRaw*) m_h_corrRes; // array of results
        for(size_t n = 0; n < noOfSignals; ++n) {
            auto o = computeStat(others[n], elems);
            T num = others[n].corr - (o.avg * ref.avg * elems);
            T denom = o.stddev * ref.stddev * elems;
            res.at(n) = num / denom;
        }
    } else { // we can directly use the correlation index
        auto others = (ResNorm*) m_h_corrRes; // array of results
        for(size_t n = 0; n < noOfSignals; ++n) {
            res.at(n) = others[n].corr / elems;
        }
    }
}

template<typename T>
template<typename U>
CudaCorrelationComputer<T>::Stat CudaCorrelationComputer<T>::computeStat(U r, size_t norm) {
    Stat s;
    s.avg = r.sum / norm;
    T sumSqrNorm = r.sumSqr / norm;
    s.stddev = sqrt(abs(sumSqrNorm - (s.avg * s.avg)));
    return s;
}

template<typename T>
void CudaCorrelationComputer<T>::compute(T *others) {
    bool isReady = this->isInitialized() && this->isRefLoaded();
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() and load reference");
    }
    if ( ! m_stream->isGpuPointer(others)) {
        REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Processing data from host is not yet supported");
    } else {
        m_d_others = others;
    }

    const auto &s = this->getSettings();
    this->getFiguresOfMerit().resize(s.otherDims.n());
    switch(s.type) {
        case MeritType::OneToN: {
            if (s.normalizeResult) {
                computeOneToN<true>();
            } else {
                computeOneToN<false>();
            }
            break;
        }
        default:
            REPORT_ERROR(ERR_NOT_IMPLEMENTED, "This case is not implemented");
    }
}

template<typename T>
void CudaCorrelationComputer<T>::loadReference(const T *ref) {
    const auto &s = this->getSettings();
    this->setIsRefLoaded(nullptr != ref);

    if (m_stream->isGpuPointer(ref)) {
        REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Only reference data on CPU are currently supported");
    } else {
        size_t bytes = s.refDims.size() * sizeof(T);
        bool hasToPin = ! m_stream->isMemoryPinned(ref);
        if (hasToPin) {
            m_stream->pinMemory(ref, bytes);
        }
        auto stream = *(cudaStream_t*)m_stream->stream();
        // copy data to GPU
        gpuErrchk(cudaMemcpyAsync(
            m_d_ref,
            ref,
            bytes,
            cudaMemcpyHostToDevice, stream));
        if (hasToPin) {
            m_stream->unpinMemory(ref);
        }
    }
    if (s.normalizeResult) {
        computeAvgStddevForRef();
    } else {
        // nothing to do, assume that stddev = 1 and avg = 0 for each signal
    }
}

template<typename T>
void CudaCorrelationComputer<T>::computeCorrStatOneToNNormalize() {
    const auto &dims = this->getSettings().otherDims;
    auto stream = *(cudaStream_t*)m_stream->stream();

    dim3 dimBlock(64);
    dim3 dimGrid(
        ceil((dims.x() * dims.n()) / (float)dimBlock.x));
    // zero-init the result
    size_t bytes = dims.n() * sizeof(ResRaw);
    gpuErrchk(cudaMemset(m_d_corrRes, 0, bytes)); // reuse data

    if (std::is_same<T, float>::value) {
        computeCorrIndexStat2DOneToN<float, float3>
            <<<dimGrid, dimBlock, 0, stream>>> (
            (float*)m_d_ref,
            (float*)m_d_others,
            dims.x(), dims.y(), dims.n(),
            (float3*)m_d_corrRes);
    } else if (std::is_same<T, double>::value) {
        computeCorrIndexStat2DOneToN<double, double3>
            <<<dimGrid, dimBlock, 0, stream>>> (
            (double*)m_d_ref,
            (double*)m_d_others,
            dims.x(), dims.y(), dims.n(),
            (double3*)m_d_corrRes);
    }
    // get result from device
    gpuErrchk(cudaMemcpyAsync(
        m_h_corrRes,
        m_d_corrRes, // reuse data
        bytes,
        cudaMemcpyDeviceToHost, stream));
    m_stream->synch();
}

template<typename T>
void CudaCorrelationComputer<T>::computeAvgStddevForRef() {
    const auto &dims = this->getSettings().refDims;
    auto stream = *(cudaStream_t*)m_stream->stream();

    dim3 dimBlock(64);
    dim3 dimGrid(
        ceil((dims.x() * dims.n()) / (float)dimBlock.x));
    // zero-init the result
    size_t bytes = dims.n() * sizeof(ResRef);
    gpuErrchk(cudaMemset(m_d_corrRes, 0, bytes)); // reuse data
    if (std::is_same<T, float>::value) {
        computeSumSumSqr2D<float, float2>
            <<<dimGrid, dimBlock, 0, stream>>> (
            (float*)m_d_ref,
            dims.x(), dims.y(), dims.n(),
            (float2*)m_d_corrRes);
    } else if (std::is_same<T, double>::value) {
        computeSumSumSqr2D<double, double2>
            <<<dimGrid, dimBlock, 0, stream>>> (
            (double*)m_d_ref,
            dims.x(), dims.y(), dims.n(),
            (double2*)m_d_corrRes);
    }
    // get result from device
    gpuErrchk(cudaMemcpyAsync(
        m_h_ref_corrRes,
        m_d_corrRes, // reuse data
        bytes,
        cudaMemcpyDeviceToHost, stream));
    m_stream->synch();
}

template<typename T>
void CudaCorrelationComputer<T>::initialize(bool doAllocation) {
    const auto &s = this->getSettings();
    if (doAllocation) {
        release();
    }

    m_stream = dynamic_cast<GPU*>(s.hw.at(0));

    if (nullptr == m_stream) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Instance of GPU is expected");
    }
    m_stream->set();
    m_stream->peekLastError();

    if (doAllocation) {
        allocate();
    }
}

template<typename T>
void CudaCorrelationComputer<T>::release() {
    // GPU memory
    gpuErrchk(cudaFree(m_d_ref));
//    gpuErrchk(cudaFree(m_d_others));
    gpuErrchk(cudaFree(m_d_corrRes));
    // CPU memory
    if (this->isInitialized()) { // otherwise these pointer are null and unpin is not happy
        m_stream->unpinMemory(m_h_corrRes);
        m_stream->unpinMemory(m_h_ref_corrRes);
    }
    free(m_h_ref_corrRes);
    free(m_h_corrRes);

    setDefault();
}

template<typename T>
void CudaCorrelationComputer<T>::allocate() {
    using memoryUtils::page_aligned_alloc;
    const auto& s = this->getSettings();

    gpuErrchk(cudaMalloc(&m_d_ref, s.refDims.size() * sizeof(T)));
//    gpuErrchk(cudaMalloc(&m_d_others, s.otherDims.size() * sizeof(T)));

    size_t bytesResOthers;
    size_t bytesResRef;
    if (s.normalizeResult) {
        bytesResOthers = s.otherDims.n() * sizeof(ResRaw);
        bytesResRef = s.refDims.n() * sizeof(ResRef);
    } else {
        bytesResOthers = s.otherDims.n() * sizeof(ResNorm);
        bytesResRef = 1; // we actually don't need anything, but this makes code cleaner
    }
    gpuErrchk(cudaMalloc(&m_d_corrRes, bytesResOthers));

    m_h_ref_corrRes = (ResRef*)page_aligned_alloc(bytesResRef);
    memset(m_h_ref_corrRes, 0, bytesResRef); // to make valgrind happy
    m_stream->pinMemory(m_h_ref_corrRes, bytesResRef);

    m_h_corrRes = page_aligned_alloc(bytesResOthers);
    memset(m_h_corrRes, 0, bytesResOthers); // to make valgrind happy
    m_stream->pinMemory(m_h_corrRes, bytesResOthers);
}

template<typename T>
void CudaCorrelationComputer<T>::setDefault() {
    // GPU memory
    m_d_ref = nullptr;
    m_d_others = nullptr;
    m_d_corrRes = nullptr;
    // CPU memory
    m_h_ref_corrRes = nullptr;
    m_h_corrRes = nullptr;
    // others
    m_stream = nullptr;
}

template<typename T>
bool CudaCorrelationComputer<T>::canBeReused(const MeritSettings &s) const {
    bool result = true;
    if ( ! this->isInitialized()) {
        return false;
    }
    auto &sOrig = this->getSettings();
    result = result && sOrig.type == s.type;
    result = result && (sOrig.otherDims.size() >= s.otherDims.size()); // previously, we needed more space

    return result;
}

template<typename T>
void CudaCorrelationComputer<T>::check() {
    const auto &s = this->getSettings();
    if (1 != s.hw.size()) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Single GPU (stream) expected");
    }
    if ( ! s.normalizeResult) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Non-normalized output not yet supported");
    }
    if (MeritType::OneToN != s.type) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Only 1:N correlations are currently supported");
    }
    if ( ! s.otherDims.is2D()) {
        REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Only 2D input is currently supported");
    }
}

// explicit instantiation
template class CudaCorrelationComputer<float>;
template class CudaCorrelationComputer<double>;
