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
#include "cuda_geo_linear_interpolator.h"
#include "cuda_geo_linear_interpolator.cu"

template<typename T>
void CudaGeoLinearTransformer<T>::setDefault() {
    m_d_dest = nullptr;
    m_d_src = nullptr;
    m_d_matrices = nullptr;
}

template<typename T>
void CudaGeoLinearTransformer<T>::release() {
    gpuErrchk(cudaFree(m_d_src));
    gpuErrchk(cudaFree(m_d_dest));
    gpuErrchk(cudaFree(m_d_matrices));

    setDefault();
}

template<typename T>
void CudaGeoLinearTransformer<T>::init() {
    size_t bytes = dims.sizePadded() * sizeof(T);
    gpuErrchk(cudaMalloc(&m_d_src, bytes));
    gpuErrchk(cudaMalloc(&m_d_dest, bytes));

    bytes = dims.n() * 9 * sizeof(float);
    gpuErrchk(cudaMalloc(&m_d_matrices, bytes));
}

template<typename T>
void CudaGeoLinearTransformer<T>::createCopyOnGPU(const T *h_data) {
    size_t bytes = dims.sizePadded() * sizeof(T);
    gpuErrchk(cudaMemcpy(
        m_d_src,
        h_data,
        bytes,
        cudaMemcpyHostToDevice));
}

template<typename T>
T *CudaGeoLinearTransformer<T>::interpolate(const std::vector<float> &matrices) {
    size_t bytes = dims.n() * 9 * sizeof(float);
    gpuErrchk(cudaMemcpy(
            m_d_matrices,
            matrices.data(),
            bytes,
            cudaMemcpyHostToDevice));
    dim3 dimBlock(64, 1, 1);
    dim3 dimGrid(
        std::ceil(dims.x() / (float)dimBlock.x),
        dims.n(), 1);
    interpolateKernel<<<dimGrid, dimBlock>>> (
                    m_d_src, m_d_dest, m_d_matrices,
                    dims.x(), dims.y());
    gpuErrchk(cudaDeviceSynchronize());
    return m_d_dest;
}

// explicit instantiation
template class CudaGeoLinearTransformer<float>;
template class CudaGeoLinearTransformer<double>;
