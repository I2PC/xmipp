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

#include "reconstruct_fourier_codelets.h"
#include "reconstruct_fourier_timing.h"
#include "reconstruct_fourier_scheduler.h"

double rfs_arch_cost_function(starpu_task * task, starpu_perfmodel_arch * arch, unsigned nimpl) {
	auto* data = (rfs_data*)starpu_sched_ctx_get_policy_data(task->sched_ctx);
	assert(data);

	size_t size = 0;
	if (task->cl->model && task->cl->model->size_base) {
		size = task->cl->model->size_base(task, nimpl);
	}

	double estimate = 0;
	for (int device = 0; device < arch->ndevices; ++device) {
		double newEstimate = data->timing.estimate(task->cl, nimpl, arch->devices[device].type, arch->devices[device].devid, size);
		estimate = STARPU_MAX(estimate, newEstimate);
	}

	return estimate;
}

starpu_perfmodel create_common_perfmodel(const char* symbol) {
	starpu_perfmodel model = {};
	model.type = STARPU_PER_ARCH;
	model.arch_cost_function = rfs_arch_cost_function;
	model.symbol = symbol;
	return model;
}

static size_t load_projections_size_base(starpu_task * task, unsigned nimpl) {
	auto imageDataHandle = STARPU_TASK_GET_HANDLE(task, 1);
	return starpu_vector_get_elemsize(imageDataHandle) * starpu_vector_get_nx(imageDataHandle);
}

static size_t padded_image_to_fft_size_base(starpu_task * task, unsigned nimpl) {
	auto imageDataHandle = STARPU_TASK_GET_HANDLE(task, 0);
	return starpu_vector_get_elemsize(imageDataHandle) * starpu_vector_get_nx(imageDataHandle);
}

static size_t reconstruct_fft_size_base(starpu_task * task, unsigned nimpl) {
	auto fftHandle = STARPU_TASK_GET_HANDLE(task, 0);
	auto traverseSpacesHandle = STARPU_TASK_GET_HANDLE(task, 1);
	return starpu_vector_get_elemsize(fftHandle) * starpu_vector_get_nx(fftHandle) * starpu_matrix_get_nx(traverseSpacesHandle);
}

Codelets::Codelets() {
	// NOTE(jp): This used to be in designated initializers and this whole cpp wasn't necessary.
	// But C++ does not have them, so it has to be done through this ugly circus.

	// Load Projections Codelet
	load_projections.where = STARPU_CPU;
	load_projections.cpu_funcs[0] = func_load_projections;
	load_projections.cpu_funcs_name[0] = "combine_image_func";
	load_projections.nbuffers = 3;
	load_projections.modes[0] = STARPU_W; // LoadedImagesBuffer
	load_projections.modes[1] = STARPU_W; // Image Data Buffer
	load_projections.modes[2] = STARPU_W; // Spaces Buffer
	load_projections.name = "codelet_load_projections";
	static struct starpu_perfmodel load_projections_model = create_common_perfmodel("load_projections_model");
	load_projections_model.size_base = load_projections_size_base;
	load_projections.model = &load_projections_model;
	// cl_arg: LoadProjectionArgs - MUST NOT BE COPIED!!!


	// Padded Image to FFT Codelet
	padded_image_to_fft.where = STARPU_CPU | STARPU_CUDA;
	padded_image_to_fft.cpu_funcs[0] = func_padded_image_to_fft_cpu;
	padded_image_to_fft.cpu_funcs_name[0] = "func_padded_image_to_fft_cpu";
	padded_image_to_fft.cuda_funcs[0] = func_padded_image_to_fft_cuda;
	padded_image_to_fft.cuda_flags[0] = STARPU_CUDA_ASYNC;
	padded_image_to_fft.nbuffers = 4;
	padded_image_to_fft.modes[0] = STARPU_R; // Padded Image Data Buffer
	padded_image_to_fft.modes[1] = STARPU_W; // FFT Buffer
	padded_image_to_fft.modes[2] = STARPU_SCRATCH; // Raw FFT Scratch Area
	padded_image_to_fft.modes[3] = STARPU_R; // LoadedImagesBuffer
	padded_image_to_fft.specific_nodes = 1;
	padded_image_to_fft.nodes[0] = STARPU_SPECIFIC_NODE_LOCAL;
	padded_image_to_fft.nodes[1] = STARPU_SPECIFIC_NODE_LOCAL;
	padded_image_to_fft.nodes[2] = STARPU_SPECIFIC_NODE_LOCAL;
	padded_image_to_fft.nodes[3] = STARPU_SPECIFIC_NODE_CPU;
	padded_image_to_fft.name = "codelet_padded_image_to_fft";
	static struct starpu_perfmodel padded_image_to_fft_model = create_common_perfmodel("padded_image_to_fft_model");
	padded_image_to_fft_model.size_base = padded_image_to_fft_size_base;
	padded_image_to_fft.model = &padded_image_to_fft_model;
	// cl_arg: PaddedImageToFftArgs


	// Reconstruct FFT Codelet
	reconstruct_fft.where = STARPU_CPU | STARPU_CUDA;
	// NOTE: From StarPU/examples/cg/cg_kernels.c it seems that STARPU_SPMD applies only to CPU implementations,
	//       which we rely on. However, the documentation does not say that explicitly, so beware.
	reconstruct_fft.type = starpu_codelet_type::STARPU_SEQ; // starpu_codelet_type::STARPU_SPMD; Disabled for now due to a bug in StarPU 1.3.2 (memory handles are not correctly copied to all SPMD nodes)
	reconstruct_fft.max_parallelism = 1 << 10;// Large number, we don't really care
	reconstruct_fft.cpu_funcs[0] = func_reconstruct_cpu_lookup_interpolation;
	reconstruct_fft.cpu_funcs[1] = func_reconstruct_cpu_dynamic_interpolation; // (Typically about 2x slower than lookup interpolation)
	reconstruct_fft.cpu_funcs_name[0] = "func_reconstruct_cpu_lookup_interpolation";
	reconstruct_fft.cpu_funcs_name[1] = "func_reconstruct_cpu_dynamic_interpolation";
	reconstruct_fft.cuda_funcs[0] = func_reconstruct_cuda;
	reconstruct_fft.cuda_flags[0] = STARPU_CUDA_ASYNC;
	reconstruct_fft.nbuffers = 6;
	reconstruct_fft.modes[0] = STARPU_R; // FFT Buffer
	reconstruct_fft.modes[1] = STARPU_R; // Traverse Spaces Buffer
	reconstruct_fft.modes[2] = STARPU_R; // Blob Table Squared Buffer (only present if fastLateBlobbing is false)
	reconstruct_fft.modes[3] = STARPU_REDUX; // Result Volume Buffer
	reconstruct_fft.modes[4] = STARPU_REDUX; // Result Weights Buffer
	reconstruct_fft.modes[5] = STARPU_R; // LoadedImagesBuffer
	reconstruct_fft.specific_nodes = 1;
	reconstruct_fft.nodes[0] = STARPU_SPECIFIC_NODE_LOCAL;
	reconstruct_fft.nodes[1] = STARPU_SPECIFIC_NODE_LOCAL;
	reconstruct_fft.nodes[2] = STARPU_SPECIFIC_NODE_LOCAL;
	reconstruct_fft.nodes[3] = STARPU_SPECIFIC_NODE_LOCAL;
	reconstruct_fft.nodes[4] = STARPU_SPECIFIC_NODE_LOCAL;
	reconstruct_fft.nodes[5] = STARPU_SPECIFIC_NODE_CPU;
	reconstruct_fft.name = "codelet_reconstruct_fft";
	static struct starpu_perfmodel reconstruct_fft_model = create_common_perfmodel("reconstruct_fft_model");
	reconstruct_fft_model.size_base = reconstruct_fft_size_base;
	reconstruct_fft.model = &reconstruct_fft_model;
	// cl_arg: ReconstructFftArgs

	// Redux volume & weights
	// Init volume
	redux_init_volume.where = STARPU_CPU | STARPU_CUDA;
	redux_init_volume.cpu_funcs[0] = func_redux_init_volume_cpu;
	redux_init_volume.cpu_funcs_name[0] = "func_redux_init_volume_cpu";
	redux_init_volume.cuda_funcs[0] = func_redux_init_volume_cuda;
	redux_init_volume.cuda_flags[0] = STARPU_CUDA_ASYNC;
	redux_init_volume.nbuffers = 1;
	redux_init_volume.modes[0] = STARPU_W;
	redux_init_volume.name = "redux_init_volume";
	static struct starpu_perfmodel redux_init_volume_model = create_common_perfmodel("redux_init_volume_model");
	redux_init_volume.model = &redux_init_volume_model;
	// Init weight
	redux_init_weights.where = STARPU_CPU | STARPU_CUDA;
	redux_init_weights.cpu_funcs[0] = func_redux_init_weights_cpu;
	redux_init_weights.cpu_funcs_name[0] = "func_redux_init_weights_cpu";
	redux_init_weights.cuda_funcs[0] = func_redux_init_weights_cuda;
	redux_init_weights.cuda_flags[0] = STARPU_CUDA_ASYNC;
	redux_init_weights.nbuffers = 1;
	redux_init_weights.modes[0] = STARPU_W;
	redux_init_weights.name = "redux_init_weights";
	static struct starpu_perfmodel redux_init_weights_model = create_common_perfmodel("redux_init_weights_model");
	redux_init_weights.model = &redux_init_weights_model;
	// Sum volume
	redux_sum_volume.where = STARPU_CPU | STARPU_CUDA;
	redux_sum_volume.cpu_funcs[0] = func_redux_sum_volume_cpu;
	redux_sum_volume.cpu_funcs_name[0] = "func_redux_sum_volume_cpu";
	redux_sum_volume.cuda_funcs[0] = func_redux_sum_volume_cuda;
	redux_sum_volume.cuda_flags[0] = STARPU_CUDA_ASYNC;
	redux_sum_volume.nbuffers = 2;
	redux_sum_volume.modes[0] = STARPU_RW;
	redux_sum_volume.modes[1] = STARPU_R;
	redux_sum_volume.name = "redux_sum_volume";
	static struct starpu_perfmodel redux_sum_volume_model = create_common_perfmodel("redux_sum_volume_model");
	redux_sum_volume.model = &redux_sum_volume_model;
	// Sum weight
	redux_sum_weights.where = STARPU_CPU | STARPU_CUDA;
	redux_sum_weights.cpu_funcs[0] = func_redux_sum_weights_cpu;
	redux_sum_weights.cpu_funcs_name[0] = "func_redux_sum_weights_cpu";
	redux_sum_weights.cuda_funcs[0] = func_redux_sum_weights_cuda;
	redux_sum_weights.cuda_flags[0] = STARPU_CUDA_ASYNC;
	redux_sum_weights.nbuffers = 2;
	redux_sum_weights.modes[0] = STARPU_RW;
	redux_sum_weights.modes[1] = STARPU_R;
	redux_sum_weights.name = "redux_sum_weights";
	static struct starpu_perfmodel redux_sum_weights_model = create_common_perfmodel("redux_sum_weights_model");
	redux_sum_weights.model = &redux_sum_weights_model;
}

Codelets codelets;