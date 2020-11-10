/***************************************************************************
 *
 * Authors:     Jan Polak (456647@mail.muni.cz)
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

#include <starpu.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <reconstruction_cuda/cuda_asserts.h>
#include "reconstruct_fourier_codelets.h"
#include "reconstruct_fourier_util.h"

// Implementation of redux methods

inline void memset_to_zero_cuda(void * buffer) {
	uintptr_t cudaPtr = STARPU_VECTOR_GET_PTR(buffer);
	size_t size = STARPU_VECTOR_GET_NX(buffer) * STARPU_VECTOR_GET_ELEMSIZE(buffer);

	gpuErrchk(cudaMemsetAsync((void *) cudaPtr, 0, size, starpu_cuda_get_local_stream()));
	// gpuErrchk(cudaStreamSynchronize(starpu_cuda_get_local_stream())); disabled because codelet is async
}

void func_redux_init_volume_cuda(void* buffers[], void* cl_arg) {
	(void)cl_arg;
	memset_to_zero_cuda(buffers[0]);
}

void func_redux_init_weights_cuda(void* buffers[], void* cl_arg) {
	(void)cl_arg;
	memset_to_zero_cuda(buffers[0]);
}

inline void memset_to_zero_cpu(void * buffer) {
	uintptr_t ptr = STARPU_VECTOR_GET_PTR(buffer);
	size_t size = STARPU_VECTOR_GET_NX(buffer) * STARPU_VECTOR_GET_ELEMSIZE(buffer);
	memset((void *) ptr, 0, size);
}

void func_redux_init_volume_cpu(void* buffers[], void* cl_arg) {
	(void)cl_arg;
	memset_to_zero_cpu(buffers[0]);
}

void func_redux_init_weights_cpu(void* buffers[], void* cl_arg) {
	(void)cl_arg;
	memset_to_zero_cpu(buffers[0]);
}

__global__
void sum_floats(float4 * target, float4 * source, uint length) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		target[i].x += source[i].x;
		target[i].y += source[i].y;
		target[i].z += source[i].z;
		target[i].w += source[i].w;
	}
}

inline void sum_floats_on_cuda(uintptr_t targetCudaPtr, uintptr_t sourceCudaPtr, size_t noOfFloats) {
	// Note: When initializing data in reconstruct_fourier_starpu.cpp#run(),
	// buffer size was padded to multiple of 4, so we can leverage that.

	assert(noOfFloats % 4 == 0);
	size_t noOfFloat4 = noOfFloats / 4;

	dim3 threadsPerBlock(512);
	// Rounding up, we don't know whether noOfFloats is also a multiple of 512 (unlikely), additional checking is done in kernel
	dim3 numBlocks(static_cast<unsigned int>((noOfFloat4 + threadsPerBlock.x - 1) / threadsPerBlock.x));

	sum_floats<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>((float4*) targetCudaPtr, (float4*)(sourceCudaPtr), noOfFloat4);
	gpuErrchk(cudaPeekAtLastError());
	// gpuErrchk(cudaStreamSynchronize(starpu_cuda_get_local_stream())); disabled because codelet is async
}

void func_redux_sum_volume_cuda(void* buffers[], void* cl_arg) {
	(void)cl_arg;
	uintptr_t targetCudaPtr = STARPU_VECTOR_GET_PTR(buffers[0]);
	uintptr_t sourceCudaPtr = STARPU_VECTOR_GET_PTR(buffers[1]);
	size_t size = STARPU_VECTOR_GET_NX(buffers[0]) * 2; // *2 because complex
	sum_floats_on_cuda(targetCudaPtr, sourceCudaPtr, size);
}

void func_redux_sum_weights_cuda(void* buffers[], void* cl_arg) {
	(void)cl_arg;
	uintptr_t targetCudaPtr = STARPU_VECTOR_GET_PTR(buffers[0]);
	uintptr_t sourceCudaPtr = STARPU_VECTOR_GET_PTR(buffers[1]);
	size_t size = STARPU_VECTOR_GET_NX(buffers[0]);
	sum_floats_on_cuda(targetCudaPtr, sourceCudaPtr, size);
}

inline void sum_floats_on_cpu(float* __restrict targetPtr, float* __restrict sourcePtr, size_t noOfFloats) {
	// Should get vectorized
	targetPtr = (float*)XMIPP_ASSUME_ALIGNED(targetPtr, ALIGNMENT);
	sourcePtr = (float*)XMIPP_ASSUME_ALIGNED(sourcePtr, ALIGNMENT);

	for (size_t i = 0; i < noOfFloats; i++) {
		targetPtr[i] += sourcePtr[i];
	}
}

void func_redux_sum_volume_cpu(void* buffers[], void* cl_arg) {
	(void)cl_arg;
	uintptr_t targetPtr = STARPU_VECTOR_GET_PTR(buffers[0]);
	uintptr_t sourcePtr = STARPU_VECTOR_GET_PTR(buffers[1]);
	size_t size = STARPU_VECTOR_GET_NX(buffers[0]) * 2; // *2 because complex
	sum_floats_on_cpu((float*) targetPtr, (float*) sourcePtr, size);
}

void func_redux_sum_weights_cpu(void* buffers[], void* cl_arg) {
	(void)cl_arg;
	uintptr_t targetPtr = STARPU_VECTOR_GET_PTR(buffers[0]);
	uintptr_t sourcePtr = STARPU_VECTOR_GET_PTR(buffers[1]);
	size_t size = STARPU_VECTOR_GET_NX(buffers[0]);
	sum_floats_on_cpu((float*) targetPtr, (float*) sourcePtr, size);
}