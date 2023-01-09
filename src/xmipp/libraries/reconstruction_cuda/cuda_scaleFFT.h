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

#pragma once

#include <complex>
#include <type_traits>
#include "reconstruction_cuda/gpu.h"
#include "reconstruction_cuda/cuda_asserts.h"
#include "cuda_scaleFFT_kernels.cu"


template<template<bool...> class Function, bool... Bs> struct ExpandBools
{
  template<typename... Args> static void Expand(Args &&...args)
  {
    return Function<Bs...>::Execute(std::forward<Args>(args)...);
  }

  template<typename... Args> static void Expand(bool b, Args &&...args)
  {
    return b ? ExpandBools<Function, Bs..., true>::Expand(std::forward<Args>(args)...)
             : ExpandBools<Function, Bs..., false>::Expand(std::forward<Args>(args)...);
  }
};

template<bool applyFilter, bool normalize, bool center, bool ignoreLowFreq>
struct ScaleFFT2D
{
    template<typename T>
    static void Execute(dim3 dimGrid, dim3 dimBlock, 
        const std::complex<T> *d_inFFT, std::complex<T> *d_outFFT, 
        int noOfFFT, 
        size_t inFFTX, size_t inFFTY, 
        size_t outFFTX, size_t outFFTY,
        T* d_filter, T normFactor, const GPU &gpu) {
            auto stream = *(cudaStream_t*)gpu.stream();
            if (std::is_same<T, float>::value) {
                return scaleFFT2DKernel<float2, float, applyFilter, normalize, center, ignoreLowFreq>
                    <<<dimGrid, dimBlock, 0, stream>>>(
                        (float2*)d_inFFT, (float2*)d_outFFT,
                        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, d_filter, normFactor);
            }
            else {
                REPORT_ERROR(ERR_TYPE_INCORRECT, "Not implemented");
            }
    }
};


template<typename T>
void scaleFFT2D(dim3 dimGrid, dim3 dimBlock, 
        const std::complex<T> *d_inFFT, std::complex<T> *d_outFFT, 
        int noOfFFT, 
        size_t inFFTX, size_t inFFTY, 
        size_t outFFTX, size_t outFFTY,
        T* d_filter, T normFactor, bool center, bool ignoreLowFreq, const GPU &gpu) {
    ExpandBools<ScaleFFT2D>::Expand((NULL != d_filter), (T)1 != normFactor, center, ignoreLowFreq,
        dimGrid, dimBlock, d_inFFT, d_outFFT, 
        noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY,
        d_filter, normFactor, gpu);
    gpuErrchk(cudaPeekAtLastError());
}

