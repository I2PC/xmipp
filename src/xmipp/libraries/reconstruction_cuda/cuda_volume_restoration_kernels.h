/***************************************************************************
 *
 * Authors:    Martin Horacek (horacek1martin@gmail.com)
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
#ifndef CUDA_VOLUME_RESTORATION_KERNELS
#define CUDA_VOLUME_RESTORATION_KERNELS

#include <complex>

#include "cuda_cdf.h"

namespace Gpu {
/*
 * Because Xmipp currently doesn't allow to inlude cuda_runtime in header files,
 * definitions of functions must be in separate .cpp file and therefore
 * aggregating them in one class allows us to instantiate only the class
 * for double/float instead of instantiating each function
*/
template< typename T >
class VolumeRestorationKernels {
public:
	static void computeWeights(const T* d_Vfiltered1, const T* d_Vfiltered2, T* d_V1r, T* d_V2r, T* d_S, size_t volume_size, const Gpu::CDF<T>& cdf_mN, T weightPower, int weightFun);

	static void filterFourierVolume(const T* d_R2, const std::complex<T>* d_fV, std::complex<T>* d_buffer, size_t volume_size, T w2, T w2Step);

	static void computeAveragePositivity(const T* d_V1, const T* d_V2, T* d_S, size_t volume_size);

	static void computeAveragePositivity(const T* d_V1, const T* d_V2, T* d_S, const int* d_mask, size_t volume_size);

	static void filterS(const T* d_R2, std::complex<T>* d_fVol, size_t volume_size);

	static void maskForCDF(T* __restrict__ d_aux, const T* __restrict__ d_S, const int* __restrict__ d_mask, size_t volume_size);

	static void maskWithNoiseProbability(T* d_V, const Gpu::CDF<T>& cdf_S, const Gpu::CDF<T>& cdf_N, size_t volume_size);

	static void deconvolveRestored(std::complex<T>* d_fVol, std::complex<T>* d_fV1, std::complex<T>* d_fV2, const T* d_R2, T K1, T K2, T lambda, size_t volume_size, size_t fourier_size);

	static void convolveFourierVolume(std::complex<T>* d_fVol, const T* d_R2, T K, size_t volume_size);

	static void normalizeForFFT(T* d_V1, T* d_V2, size_t volume_size);

	static void normalizeForFFT(T* d_V1, size_t volume_size);

	static void restorationSigmaCostError(T& error, const std::complex<T>* d_fVol, const std::complex<T>* d_fV1, const std::complex<T>* d_fV2, const T* __restrict__ d_R2, T K1, T K2, size_t fourier_size);

	static void computeDiffAndAverage(const T* __restrict__ d_V1, const T* __restrict__ d_V2, T* __restrict__ d_S, T* __restrict__ d_N, size_t volume_size);

	static std::pair<T, T> computeAvgStd(const T* __restrict__ d_N, size_t volume_size);

	static std::pair<T, T> computeAvgStdWithMask(const T* __restrict__ d_N, const int* __restrict__ d_mask, size_t mask_size, size_t volume_size);

	static void computeDifference(T* __restrict__ d_V1, T* __restrict__ d_V2, const T* __restrict__ d_S, const T* __restrict__ d_N, T k, size_t volume_size);

	static size_t computeMaskSize(const int* __restrict__ d_mask, size_t volume_size);

	static void multiplyByConstant(T* __restrict__ d_array, T c, size_t volume_size);

};

} // namespace Gpu

#endif // CUDA_VOLUME_RESTORATION_KERNELS