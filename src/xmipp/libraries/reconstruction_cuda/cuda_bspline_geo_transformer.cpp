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
#include "cuda_bspline_geo_transformer.h"
#include "cuda_geo_linear_interpolator.cu"

template<typename T>
void CudaBSplineGeoTransformer<T>::setSrc(const T *h_data) {
    const auto& s = this->getSettings();
    gpuErrchk(cudaMemcpy(
        m_d_src,
        h_data,
        this->getSettings().dims.size() * sizeof(T),
        cudaMemcpyHostToDevice));
    this->setIsSrcSet(true);
}

template<typename T>
void CudaBSplineGeoTransformer<T>::initialize(bool doAllocation) {
    if (doAllocation) {
        release();
    }
    const auto &s = this->getSettings();
    m_stream = dynamic_cast<GPU*>(s.hw.at(0));

    if (nullptr == m_stream) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Instance of GPU is expected");
    }

    if (doAllocation) {
        allocate();
    }
}

template<typename T>
void CudaBSplineGeoTransformer<T>::check() {
    const auto &s = this->getSettings();
    if (InterpolationDegree::Linear != s.degree) {
        REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Only linear interpolation is available");
    }
    if (s.doWrap) {
        REPORT_ERROR(ERR_NOT_IMPLEMENTED, "Wrapping is not yet implemented");
    }
    if (1 != s.hw.size()) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Single GPU (stream) expected");
    }
    if ( ! s.keepSrcCopy) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "'keepSrcCopy' must be set to true");
    }
}

template<typename T>
void CudaBSplineGeoTransformer<T>::copySrcToDest() {
    bool isReady = this->isInitialized()
            && this->isSrcSet();
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Instance is either not initialized or the 'src' has not been set.");
    }
    gpuErrchk(cudaMemcpy(
        m_d_dest,
        m_d_src,
        this->getSettings().dims.size() * sizeof(T),
        cudaMemcpyDeviceToDevice)); // FIXME DS use proper stream
}

template<typename T>
void CudaBSplineGeoTransformer<T>::setDefault() {
    m_d_src = nullptr;
    m_d_dest = nullptr;
    m_stream = nullptr;
    m_d_matrices = nullptr;
}

template<typename T>
void CudaBSplineGeoTransformer<T>::release() {
    gpuErrchk(cudaFree(m_d_src));
    gpuErrchk(cudaFree(m_d_dest));
    gpuErrchk(cudaFree(m_d_matrices));

    setDefault();
}

template<typename T>
void CudaBSplineGeoTransformer<T>::allocate() {
    const auto& s = this->getSettings();
    const Dimensions dims = s.dims;
    size_t bytes = dims.sizePadded() * sizeof(T);
    gpuErrchk(cudaMalloc(&m_d_src, bytes));
    gpuErrchk(cudaMalloc(&m_d_dest, bytes));

    bytes = dims.n() * 9 * sizeof(float);
    gpuErrchk(cudaMalloc(&m_d_matrices, bytes));
}

template<typename T>
T *CudaBSplineGeoTransformer<T>::interpolate(const std::vector<float> &matrices) {
    const auto& dims = this->getSettings().dims;
    size_t bytes = dims.n() * 9 * sizeof(float);
    gpuErrchk(cudaMemcpy(
            m_d_matrices,
            matrices.data(),
            bytes,
            cudaMemcpyHostToDevice));
    dim3 dimBlock(64, 1, 1);
    dim3 dimGrid(
        std::ceil(dims.x() / (float)dimBlock.x),
        std::ceil(dims.y() / (float)dimBlock.y),
        dims.n());
    interpolateKernel<<<dimGrid, dimBlock>>> (
                    m_d_src, m_d_dest, m_d_matrices,
                    dims.x(), dims.y());
    gpuErrchk(cudaDeviceSynchronize());
    return m_d_dest;
}

template<typename T>
void CudaBSplineGeoTransformer<T>::sum(T *dest, size_t firstN) {
    const auto& dims = this->getSettings().dims.copyForN(firstN);
    dim3 dimBlock(64, 64, 1);
    dim3 dimGrid(
        std::ceil(dims.x() / (float)dimBlock.x),
        std::ceil(dims.y() / (float)dimBlock.y),
        1);
    auto stream = *(cudaStream_t*)m_stream->stream();

    sumKernel<<<dimGrid, dimBlock, 0, stream>>> (
        m_d_dest, m_d_src, // FIXME DS this will damage src data
        dims.x(), dims.y(), dims.n());

    size_t bytes = dims.sizeSingle() * sizeof(T);
    gpuErrchk(cudaMemcpyAsync(
        dest,
        m_d_src,
        bytes,
        cudaMemcpyDeviceToHost, stream));
    m_stream->synch();
}

// explicit instantiation
template class CudaBSplineGeoTransformer<float>;
template class CudaBSplineGeoTransformer<double>;
