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
#include "cuda_cdf.h"

#include <cuda_runtime_api.h>
#include "cuda_asserts.h"

#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

#include "cuda_cdf.cu"

namespace Gpu {

template< typename T >
CDF<T>::CDF(size_t volume_size, T multConst /* = 1.0 */, T probStep /* = 0.005 */)
: volume_size(volume_size)
, multConst(multConst)
, probStep(probStep)
, Nsteps(round(1.0/probStep)) {
	gpuErrchk( cudaMalloc((void**)&d_V, volume_size * type_size) );
	gpuErrchk( cudaMalloc((void**)&d_x, Nsteps * type_size) );
	gpuErrchk( cudaMalloc((void**)&d_probXLessThanx, Nsteps * type_size) );
}

template< typename T >
CDF<T>::~CDF() {
	gpuErrchk( cudaFree(d_V) );
	gpuErrchk( cudaFree(d_x) );
	gpuErrchk( cudaFree(d_probXLessThanx) );
}

template< typename T >
void CDF<T>::calculateCDF(const T* d_filtered1, const T* d_filtered2) {
	_calculateDifference(d_filtered1, d_filtered2);
	sort();
	_updateProbabilities();
}

template< typename T >
void CDF<T>::calculateCDF(const T* d_S) {
	_calculateSquare(d_S);
	sort();
	_updateProbabilities();
}

template< typename T >
void CDF<T>::_calculateDifference(const T* __restrict__ d_filtered1, const T* __restrict__ d_filtered2) {
	T* __restrict__ d_V = this->d_V;
	const T multConst = this->multConst;

	auto compute_diff = [=] __device__ (int index) {
		T diff = d_filtered1[index] - d_filtered2[index];
		d_V[index] = multConst * diff * diff;
	};

	thrust::for_each_n(thrust::device, thrust::counting_iterator<int>(0), volume_size, compute_diff);
}

template< typename T >
void CDF<T>::_calculateSquare(const T* __restrict__ d_S) {
	T* __restrict__ d_V = this->d_V;

	auto k = [=] __host__ __device__ (int index) {
		T val = d_S[index];
		d_V[index] = val * val;
	};

	thrust::for_each_n(thrust::device, thrust::counting_iterator<int>(0), volume_size, k);
}

template< typename T >
void CDF<T>::sort() {
	thrust::sort(thrust::device, d_V, d_V + volume_size);
}

template< typename T >
void CDF<T>::_updateProbabilities() {
	T* __restrict__ d_V = this->d_V;
	T* __restrict__ kd_x = this->d_x;
	T* __restrict__ kd_probXLessThanx = this->d_probXLessThanx;
	const T kprobStep = this->probStep;
	const size_t k_volume_size = this->volume_size;

	auto k = [d_V, kd_x, kd_probXLessThanx, kprobStep, k_volume_size] __device__ (int index) {
		int i = 0;
		for (T p = kprobStep / 2; p < 1; p += kprobStep, i++) {
			int idx = static_cast<int>(round(p * k_volume_size));
			kd_probXLessThanx[i] = p;
			kd_x[i] = d_V[idx];
		}
	};

	thrust::for_each_n(thrust::device, thrust::counting_iterator<int>(0), 1, k);
}

template class CDF<float>;
template class CDF<double>;

} // namespace Gpu