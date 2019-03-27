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
#include "reconstruction_cuda/cuda_basic_math.h"
#include "cuda_shift_aligner.h"
#include "reconstruction_cuda/cuda_gpu_movie_alignment_correlation_kernels.cu"

namespace Alignment {

template<typename T>
void CudaShiftAligner<T>::computeCorrelations2DOneToN(
        std::complex<T> *h_inOut,
        std::complex<T> *h_ref,
        FFTSettingsNew<T> &dims,
        bool copyToHost) {
    // FIXME make sure that dim is less than 4GB
    // allocate and copy reference signal
    std::complex<T> *d_ref;
    gpuErrchk(cudaMalloc(&d_ref, dims.fBytesSingle()));
    gpuErrchk(cudaMemcpy(d_ref, h_ref, dims.fBytesSingle(),
            cudaMemcpyHostToDevice));

    // compute kernel size
    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
    dim3 dimGrid(
            ceil(dims.fDim().x()/(float)dimBlock.x),
            ceil(dims.fDim().y()/(float)dimBlock.y));

    // allocate memory for other signals
    std::complex<T> *d_inOut;
    gpuErrchk(cudaMalloc(&d_inOut, dims.fBytesBatch()));

    // process other signals
    for (size_t offset = 0; offset < dims.fDim().n(); offset += dims.batch()) {
        // how many signals to process
        size_t toProcess = std::min(dims.batch(), dims.fDim().n() - offset);

        // copy memory
        gpuErrchk(cudaMemcpy(
                d_inOut,
                h_inOut + offset * dims.fDim().xy(),
                toProcess * dims.fBytesSingle(),
                    cudaMemcpyHostToDevice));

        computeCorrelations2DOneToN<false>(&dimBlock, &dimGrid, // FIXME handle center template
                d_inOut, d_ref,
                dims.fDim().x(), dims.fDim().y(), toProcess);

        // copy data back if requested
        if (copyToHost) {
            gpuErrchk(cudaMemcpy(
                    h_inOut + offset * dims.fDim().xy(),
                    d_inOut,
                    toProcess * dims.fBytesSingle(),
                    cudaMemcpyDeviceToHost));
        }
    }

    // release memory
    gpuErrchk(cudaFree(d_inOut));
    gpuErrchk(cudaFree(d_ref));
}

template<typename T>
template<bool center>
void CudaShiftAligner<T>::computeCorrelations2DOneToN(
        void *dimBlock, void *dimGrid,
        std::complex<T> *d_inOut,
        std::complex<T> *d_ref,
        size_t xDim, size_t yDim, size_t nDim) {
    if (std::is_same<T, float>::value) {
        computeCorrelations2DOneToNKernel<float2, center>
            <<<*(dim3*)dimGrid, *(dim3*)dimBlock>>> (
                (float2*)d_inOut, (float2*)d_ref,
                xDim, yDim, nDim);
    } else if (std::is_same<T, double>::value) {
        computeCorrelations2DOneToNKernel<double2, center>
            <<<*(dim3*)dimGrid, *(dim3*)dimBlock>>> (
                (double2*)d_inOut, (double2*)d_ref,
                xDim, yDim, nDim);
    } else {
        REPORT_ERROR(ERR_TYPE_INCORRECT, "Not implemented");
    }
}

template class CudaShiftAligner<float>;
template class CudaShiftAligner<double>;

} /* namespace Alignment */
