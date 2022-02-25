/***************************************************************************
 *
 * Authors:     Jan Polak (456647@mail.muni.cz)
 *              Roberto Marabini (roberto@cnb.csic.es)
 *              Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *              Jose Roman Bilbao-Castro (jrbcast@ace.ual.es)
 *              Vahid Abrishami (vabrishami@cnb.csic.es)
 *              David Strelak (davidstrelak@gmail.com)
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

#include "core/bilib/kernel.h"
#include "core/metadata_sql.h"
#include "core/symmetries.h"
#include "core/xmipp_fftw.h"

#include "reconstruct_fourier_starpu.h"

#include <starpu.h>

#include "reconstruct_fourier_codelets.h"
// #include "reconstruct_fourier_scheduler.h"
#include <atomic>
#include "reconstruct_fourier_util.h"
#include "reconstruct_fourier_starpu_util.h"

#include <iostream>
#include <iomanip>

const int DEBUG_SYNCHRONOUS_TASKS = 0;

// Define params
void ProgRecFourierStarPU::defineParams() {
	//usage
	addUsageLine("Generate 3D reconstructions from projections using direct Fourier interpolation with arbitrary geometry.");
	addUsageLine("Kaisser-windows are used for interpolation in Fourier space.");
	//params
	// addParamsLine("   -i <md_file>                  : Metadata file with input projections");
	// addParamsLine("  [-o <volume_file=\"rec_fourier.vol\">] : Filename for output volume");
	addParamsLine("  [-fakeX <size=128>]            : size of the projection");
	addParamsLine("  [--sym <count=1>]              : Nubmer of symmetires to use");
	addParamsLine("  [--fakeNoOfImages <count=1000>]: Nubmer of images / projections to use");
	addParamsLine("  [--padding <proj=1.0> <vol=1.0>] : Padding used for projections and volume");
	addParamsLine("  [--max_resolution <p=0.5>]     : Max resolution (Nyquist=0.5)");
	addParamsLine("  [--weight]                     : Use weights stored in the image metadata");
	addParamsLine("  [--blob <radius=1.9> <order=0> <alpha=15>] : Blob parameters");
	addParamsLine("                                 : radius in pixels, order of Bessel function in blob and parameter alpha");
	addParamsLine("  [--fast]                       : Do the blobbing at the end of the computation.");
	addParamsLine("                                 : Gives slightly different results, but is faster.");
	addParamsLine("  [--table]                      : Instead of using on-demand computed interpolation coefficients, use precomputed table.");
	addParamsLine("  [--batchSize <size=25>]        : Amount of images in single batch. Too many won't fit in memory and will not be able to run as fast, too few will have large overhead.");
	addParamsLine("                                 : Each additional image in batch will require at least imageSide^2 * 32 bytes, e.g.");
	addParamsLine("                                 : cca 52MB per batch of 25 256x256 images or 210MB for 512x512 images.");
	addParamsLine("  [--fourierThreads <num=0>]     : Number of threads used for final fourier transformation. Zero means all available.");
	addExampleLine("For reconstruct enforcing i3 symmetry and using stored weights:", false);
	addExampleLine("   xmipp_starpu_reconstruct_fourier  -i reconstruction.sel --sym i3 --weight");

	/* Buffer sizes:
	 *  paddedImages:
	 *      batchSize * paddedImgSize * paddedImgSize * sizeof(float)
	 *      batchSize * symmetryCount * sizeof(TraverseSpace)       <- Typically in order of few kB per batch
	 *  fft:
	 *      batchSize * fftSizeX * fftSizeY * 2 * sizeof(float)
	 *  fftWorkSpace:
	 *      paddedImgSize * paddedImgSize/2 * sizeof(float) * 2     <- Does not scale with batchSize
	 */
}

// Read arguments ==========================================================
void ProgRecFourierStarPU::readParams() {
	// fn_in = getParam("-i");
	// fn_out = getParam("-o");
	// fn_sym = getParam("--sym");
	params.fakeX = getIntParam("-fakeX");
	params.useTable = checkParam("--table");
	params.noOfSymmetries = getIntParam("--sym");
	params.fakeNoOfImages = getIntParam("--fakeNoOfImages");

	params.padding_factor_proj = getDoubleParam("--padding", 0);
	params.padding_factor_vol = getDoubleParam("--padding", 1);
	params.maxResolution = (float)getDoubleParam("--max_resolution");
	params.do_weights = checkParam("--weight");
	params.blob.radius = getDoubleParam("--blob", 0);
	params.blob.order  = getIntParam("--blob", 1);
	params.blob.alpha  = getDoubleParam("--blob", 2);
	params.fastLateBlobbing = checkParam("--fast");

	int batchSize = getIntParam("--batchSize");
	params.batchSize = batchSize <= 0 ? 25 : static_cast<uint32_t>(batchSize);

	{// Number of threads of final fourier transformation
		int fourierThreads = getIntParam("--fourierThreads");
		if (fourierThreads <= 0) {
			fourierThreads = (int)sysconf(_SC_NPROCESSORS_ONLN); // Default to all
		}
		if (fourierThreads <= 0) {
			fourierThreads = 1;
			std::cerr << "--fourierThreads: Cannot obtain number of cores. Defaulting to one." << std::endl;
		}
		params.fourierTransformThreads = static_cast<uint32_t>(fourierThreads);
	}
}

void ProgRecFourierStarPU::prepareMetaData(const FileName& fn_in, MetaDataVec& SF) {
	// Read the input images
	SF.read(fn_in);
	SF.removeDisabled();
}

uint32_t ProgRecFourierStarPU::computeBatchCount(const ProgRecFourierStarPU::Params &params, const MetaData &SF) {
	return (static_cast<uint32_t>(params.fakeNoOfImages) + params.batchSize - 1) / params.batchSize;
}

void ProgRecFourierStarPU::prepareConstants(const Params& params, const MetaData& SF, const FileName& fn_sym, ComputeConstants& constants) {
	// Ask for memory for the output volume and its Fourier transform
	size_t imageSize = params.fakeX;
	// {
	// 	size_t objId = SF.firstRowId();
	// 	FileName fnImg;
	// 	SF.getValue(MDL_IMAGE, fnImg, objId);
	// 	Image<double> I;
	// 	I.read(fnImg, HEADER);

	// 	imageSize = I().xdim;
	// 	if (imageSize != I().ydim)
	// 		REPORT_ERROR(ERR_MULTIDIM_SIZE, "This algorithm only works for squared images");
	// }

	constants.imgSize = static_cast<int>(imageSize);
	constants.paddedImgSize = static_cast<uint32_t>(imageSize * params.padding_factor_vol);
	{
		uint32_t conserveRows = (uint32_t) ceil(2.0f * constants.paddedImgSize * params.maxResolution);
		// Round up to nearest even number (i.e. divisible by 2)
		constants.maxVolumeIndex = conserveRows + (conserveRows & 1); // second term is 1 iff conserveRows is odd, 0 otherwise
	}

	// Build a table of blob values
	blobtype blobFourier = params.blob;
	blobtype blobNormalized = params.blob;
	blobFourier.radius /= params.padding_factor_vol * imageSize;
	blobNormalized.radius /= params.padding_factor_proj / params.padding_factor_vol;

	double deltaSqrt     = (params.blob.radius * params.blob.radius) / (BLOB_TABLE_SIZE_SQRT - 1);
	double deltaFourier  = (sqrt(3.0) * imageSize / 2.0) / (BLOB_TABLE_SIZE_SQRT-1);
	constants.iDeltaSqrt    = static_cast<float>(1 / deltaSqrt);
	constants.iDeltaFourier = static_cast<float>(1 / deltaFourier);

	// The interpolation kernel must integrate to 1
	constants.iw0 = 1.0 / blob_Fourier_val(0.0, blobNormalized);
	double padXdim3 = params.padding_factor_vol * imageSize;
	padXdim3 = padXdim3 * padXdim3 * padXdim3;
	double blobTableSize = params.blob.radius * sqrt(1.0 / (BLOB_TABLE_SIZE_SQRT-1));
	for (int i = 0; i < BLOB_TABLE_SIZE_SQRT; i++) {
		//use a r*r sample instead of r
		constants.blobTableSqrt[i] = static_cast<float>(blob_val(blobTableSize*sqrt((double)i), params.blob) * constants.iw0);
		constants.fourierBlobTable[i] = blob_Fourier_val(deltaFourier * i, blobFourier) * padXdim3 * constants.iw0;
	}

	{// Get symmetries
		Matrix2D<double> Identity(3, 3);
		Identity.initIdentity();
		constants.R_symmetries.push_back(Identity);
		for (size_t i = 1; i < params.noOfSymmetries;++i) {
			auto s = Matrix2D<double>(3, 3);
			auto m = GenerateMatrix();
			for (size_t j = 0; j < 9; ++j) { s.mdata[j] = m[j];}
			constants.R_symmetries.emplace_back(s);
		}
		// if (!fn_sym.isEmpty()) {
		// 	SymList SL;
		// 	SL.readSymmetryFile(fn_sym);
		// 	constants.R_symmetries.reserve(constants.R_symmetries.size() + SL.symsNo());
		// 	for (int isym = 0; isym < SL.symsNo(); isym++) {
		// 		Matrix2D<double> L(4, 4), R(4, 4);
		// 		SL.getMatrices(isym, L, R);
		// 		R.resize(3, 3);
		// 		constants.R_symmetries.push_back(R);
		// 	}
		// }
	}
}

void ProgRecFourierStarPU::setIO(const FileName &fn_in, const FileName &fn_out) {
	this->fn_in = fn_in;
	this->fn_out = fn_out;
}

// Show ====================================================================
void ProgRecFourierStarPU::show() const {
	std::cout << " =====================================================================" << std::endl;
	std::cout << " Direct 3D reconstruction method using Kaiser windows as interpolators" << std::endl;
	std::cout << " =====================================================================" << std::endl;
	std::cout << " Input selfile             : "  << fn_in << std::endl;
	std::cout << " padding_factor_proj       : "  << params.padding_factor_proj << std::endl;
	std::cout << " padding_factor_vol        : "  << params.padding_factor_vol << std::endl;
	std::cout << " Output volume             : "  << fn_out << std::endl;
	if (!fn_sym.isEmpty())
		std::cout << " Symmetry file for projections : "  << fn_sym << std::endl;
	if (params.do_weights)
		std::cout << " Use weights stored in the image headers or doc file" << std::endl;
	else
		std::cout << " Do NOT use weights" << std::endl;
	std::cout << "\n Interpolation Function"
	          << "\n   blrad                 : "  << params.blob.radius
	          << "\n   blord                 : "  << params.blob.order
	          << "\n   blalpha               : "  << params.blob.alpha
	          << "\n max_resolution          : "  << params.maxResolution
	          << "\n -----------------------------------------------------------------" << std::endl;
}

// Main routine ------------------------------------------------------------

struct ProgRecFourierStarPU::SimpleBatchProvider : ProgRecFourierStarPU::BatchProvider {
	const uint32_t batchCount;
	const bool enableProgressBar;
	uint32_t givenBatches = 0;
	/** Tracked only if enableProgressBar == true */
	std::atomic<uint32_t> completedBatches { 0 };

	SimpleBatchProvider(const uint32_t batchCount, const bool enableProgressBar)
			: batchCount(batchCount)
			, enableProgressBar(enableProgressBar) {
		if (enableProgressBar) {
			init_progress_bar(batchCount);
		}
	}

	uint32_t maxBatches() override {
		return batchCount;
	}

	int32_t nextBatch() override {
		if (givenBatches >= batchCount) {
			return -1;
		} else {
			return givenBatches++;
		}
	}

	void batchCompleted() override {
		if (!enableProgressBar)
			return; // Disabled

		uint32_t batches = ++completedBatches;
		// Update progress bar, but not on every batch, since it would slow us down and spam the CLI too quickly
		if ((batches % 16 /* Stride, keep a power of two so it is fast */) == 0 || batches == batchCount) {
			static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
			pthread_mutex_lock(&mutex);
			progress_bar(batches);
			pthread_mutex_unlock(&mutex);
		}
	}

	virtual ~SimpleBatchProvider() {
		if (enableProgressBar) {
			// To end with pretty progress bar even if we omitted some batches (which shouldn't happen)
			progress_bar(batchCount);
		}
	}
};

void ProgRecFourierStarPU::run() {
	if (verbose) {
		show();
	}

	// prepareMetaData(fn_in, SF);
	prepareConstants(params, SF, fn_sym, computeConstants);

	printf("\nRunning Fourier Reconstruction in Xmipp.\nImage size: %d*%d (%lu)\nBatch: %d\nSymmetries: "
		   "%d\nInterpolation type: %s\nInterpolation coefficient type: %s\n",
		   computeConstants.paddedImgSize,
		   computeConstants.paddedImgSize,
		   params.fakeNoOfImages,
		   params.batchSize,
		   params.noOfSymmetries,
		   params.fastLateBlobbing ? "delayed interpolation" : "immediate interpolation",
		   params.useTable ? "precomputed table" : "dynamic computation"); fflush(stdout);

	initStarPU();

	SimpleBatchProvider batchProvider(computeBatchCount(params, SF), (bool) verbose);



	ComputeStarPUResult result = computeStarPU(params, SF, computeConstants, batchProvider, (bool) verbose);

	// We won't need StarPU anymore
	shutdownStarPU();

	// TODO(jp): This step seems unnecessary (later steps would have to be rewritten to work with flat arrays) (also in MPI version)
	// Convert flat volume and weight arrays into multidimensional arrays and destroy originals
	// ALSO THIS PRINTS data
	// std::complex<float>*** tempVolume = result.createXmippStyleVolume(computeConstants.maxVolumeIndex);
	// float*** tempWeights = result.createXmippStyleWeights(computeConstants.maxVolumeIndex);
	result.destroy();

	// Adjust and save the resulting volume
	// postProcessAndSave(params, computeConstants, fn_out, tempVolume, tempWeights);
}

void ProgRecFourierStarPU::initStarPU() {
	// // Request more workers per CUDA-capable GPU
	// setenv("STARPU_NWORKER_PER_CUDA",
	//        "2" /* seems to work best, 1 leaves GPU idle for fractions of a second (but shouldn't) */,
	//        false /* don't overwrite user specified value */);
	// // Be more reluctant to discard calibration data
	// setenv("STARPU_HISTORY_MAX_ERROR", "200" /*%*/, false);
	// // Do not let users to specify scheduling policy, because we are using our own and StarPU REALLY doesn't like this conflict
	// setenv("STARPU_SCHED", "", true);

	// starpu_conf starpu_configuration;
	// starpu_conf_init(&starpu_configuration);
	// starpu_configuration.sched_policy = &schedulers.reconstruct_fourier;
	CHECK_STARPU(starpu_init(nullptr));//&starpu_configuration));

	// int cpuWorkers[STARPU_NMAXWORKERS];
	// unsigned cpuWorkerCount = starpu_worker_get_ids_by_type(starpu_worker_archtype::STARPU_CPU_WORKER, cpuWorkers, STARPU_NMAXWORKERS);
	// if (cpuWorkerCount > 1) {
	// 	int combinedWorker = starpu_combined_worker_assign_workerid(cpuWorkerCount, cpuWorkers);
	// 	starpu_sched_ctx_add_workers(&combinedWorker, 1, 0);

	// 	schedulers.reconstruct_fourier.add_workers(0, nullptr, 0);// Just force a refresh, not a very clean solution
	// }

	starpu_malloc_set_align(ALIGNMENT);
}

ProgRecFourierStarPU::ComputeStarPUResult ProgRecFourierStarPU::computeStarPU(
		const Params& params, const MetaData& SF, const ComputeConstants& computeConstants, BatchProvider& batches,
		bool verbose) {

	uint32_t maxVolumeIndex = computeConstants.maxVolumeIndex;
	uint32_t paddedImgSize = computeConstants.paddedImgSize;

	// Initialize GPU
	padded_image_to_fft_cuda_initialize(paddedImgSize);
	reconstruct_cuda_initialize_constants(maxVolumeIndex, maxVolumeIndex,
	                                      static_cast<float>(params.blob.radius),
	                                      static_cast<float>(params.blob.alpha),
	                                      computeConstants.iDeltaSqrt, static_cast<float>(computeConstants.iw0),
	                                      1.f / getBessiOrderAlpha(params.blob));

	std::vector<size_t> selFileObjectIds;
	// SF.findObjects(selFileObjectIds);
	const uint32_t fftSizeX = maxVolumeIndex / 2;
	const uint32_t fftSizeY = maxVolumeIndex;

	PaddedImageToFftArgs imageToFftArg = {
			params.maxResolution * params.maxResolution,
			paddedImgSize,
			fftSizeX, fftSizeY
	};

	ReconstructFftArgs reconstructFftArg = {
			static_cast<float>(params.blob.radius),
			maxVolumeIndex,
			params.fastLateBlobbing,
			params.blob.order,
			static_cast<float>(params.blob.alpha),
			static_cast<uint32_t>(computeConstants.R_symmetries.size()),
			fftSizeX, fftSizeY
	};

	starpu_data_handle_t resultVolumeHandle = {0};
	starpu_data_handle_t resultWeightsHandle = {0};

	ComputeStarPUResult result;

	{
		/*
		 * Allocate 3D array (continuous) of given size^3.
		 * Allocated array is cleared (to zero)
		 */
		uint32_t dim = static_cast<uint32_t>(maxVolumeIndex + 1);
		uint32_t dim3 = align(dim * dim * dim, 4); // Make the array size always a multiple of 4, for faster redux operations
		size_t volumeDataSize = dim3 * sizeof(std::complex<float>);
		size_t weightDataSize = dim3 * sizeof(float);
		CHECK_STARPU(starpu_malloc((void **) &result.volumeData, volumeDataSize));
		CHECK_STARPU(starpu_malloc((void **) &result.weightsData, weightDataSize));
		// As a final step, StarPU will redux the data created on GPU into this
		memset(result.volumeData, 0, volumeDataSize);
		memset(result.weightsData, 0, weightDataSize);

		starpu_vector_data_register(&resultVolumeHandle, STARPU_MAIN_RAM, (uintptr_t) result.volumeData, dim3, sizeof(std::complex<float>));
		starpu_vector_data_register(&resultWeightsHandle, STARPU_MAIN_RAM, (uintptr_t) result.weightsData, dim3, sizeof(float));
		starpu_data_set_reduction_methods(resultVolumeHandle, &codelets.redux_sum_volume, &codelets.redux_init_volume);
		starpu_data_set_reduction_methods(resultWeightsHandle, &codelets.redux_sum_weights, &codelets.redux_init_weights);
		starpu_data_set_name(resultVolumeHandle, "Result Volume Data");
		starpu_data_set_name(resultWeightsHandle, "Result Weights Data");
	}

	starpu_data_handle_t blobTableSquaredHandle = {0};
	if (params.fastLateBlobbing) {
		// Blob Table is not used on GPU, but we need to register empty regardless
		starpu_void_data_register(&blobTableSquaredHandle);
	} else {
		starpu_variable_data_register(&blobTableSquaredHandle, STARPU_MAIN_RAM,
		                              reinterpret_cast<uintptr_t>(&computeConstants.blobTableSqrt), sizeof(computeConstants.blobTableSqrt));
	}
	starpu_data_set_name(blobTableSquaredHandle, "Blob Table Squared");

	const uint32_t totalImages = params.fakeNoOfImages;//static_cast<uint32_t>(SF.size());
	const uint32_t maxBatches = batches.maxBatches();

	std::vector<LoadProjectionArgs> loadProjectionArgs;
	// WARNING: Backing arrays of this vector must never reallocate, as we use pointers into it for codelet arguments
	loadProjectionArgs.reserve(maxBatches);

	// Used in a STARPU_SCRATCH mode, so the handle can be shared between all relevant tasks,
	// as the all will get a different memory to work with.
	starpu_data_handle_t fftScratchMemoryHandle = {0};
	// As documented in http://fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data
	uint32_t fftResultX = paddedImgSize/2+1;
	starpu_matrix_data_register(&fftScratchMemoryHandle, -1, 0, fftResultX /* no padding */, fftResultX, paddedImgSize, sizeof(float) * 2);
	starpu_data_set_name(fftScratchMemoryHandle, "Scratch for raw FFT");

	int32_t batch;
	while ((batch = batches.nextBatch()) != -1) {
		const uint32_t batchStart = batch * params.batchSize;
		const uint32_t batchEnd = XMIPP_MIN(batchStart + params.batchSize, totalImages);
		const uint32_t currentBatchSize = static_cast<uint32_t>(batchEnd - batchStart);

		starpu_data_handle_t amountLoadedHandle = {0};
		starpu_variable_data_register(&amountLoadedHandle, -1, 0, sizeof(LoadedImagesBuffer));
		starpu_data_set_name(amountLoadedHandle, "Batch Meta Data");

		starpu_data_handle_t paddedImagesDataHandle = {0};
		starpu_vector_data_register(&paddedImagesDataHandle, -1, 0, currentBatchSize, align(paddedImgSize * paddedImgSize * sizeof(float), ALIGNMENT));
		starpu_data_set_name(paddedImagesDataHandle, "Batch Padded Image Data");

		starpu_data_handle_t traverseSpacesHandle = {0};
		starpu_matrix_data_register(&traverseSpacesHandle, -1, 0, computeConstants.R_symmetries.size(), computeConstants.R_symmetries.size(), currentBatchSize, sizeof(RecFourierProjectionTraverseSpace));
		starpu_data_set_name(traverseSpacesHandle, "Batch Traverse Spaces");

		// Submit the task to load the projections
		{
			const size_t argIndex = loadProjectionArgs.size();

			// Create new LoadProjectionArgs for this batch
			loadProjectionArgs.push_back(LoadProjectionArgs {
					batchStart, batchEnd,
					SF,
					selFileObjectIds,
					params.do_weights,
					computeConstants.R_symmetries,
					maxVolumeIndex, maxVolumeIndex,
					static_cast<float>(params.blob.radius),
					params.fastLateBlobbing,
					paddedImgSize,
					fftSizeX, fftSizeY
			});

			const LoadProjectionArgs& loadProjectionArg = loadProjectionArgs[argIndex];

			starpu_task *loadProjectionsTask = starpu_task_create();
			loadProjectionsTask->name = "LoadProjections";
			loadProjectionsTask->cl = &codelets.load_projections;
			loadProjectionsTask->cl_arg = (void *) &loadProjectionArg;
			loadProjectionsTask->cl_arg_size = sizeof(loadProjectionArg);
			loadProjectionsTask->cl_arg_free = 0; // Do not free! (probably default)
			loadProjectionsTask->handles[0] = amountLoadedHandle;
			loadProjectionsTask->handles[1] = paddedImagesDataHandle;
			loadProjectionsTask->handles[2] = traverseSpacesHandle;
			loadProjectionsTask->synchronous = DEBUG_SYNCHRONOUS_TASKS;

			CHECK_STARPU(starpu_task_submit(loadProjectionsTask));
		}

		// Compute FFT of padded image data and adjust it (crop, shift, normalize)
		starpu_data_handle_t fftHandle = {0};
		starpu_vector_data_register(&fftHandle, -1, 0, currentBatchSize, fftSizeX * fftSizeY * 2 * sizeof(float));
		starpu_data_set_name(fftHandle, "Batch FFT Data");

		{
			starpu_task* paddedImageToFftTask = starpu_task_create();
			paddedImageToFftTask->name = "PaddedImageToFFT";
			paddedImageToFftTask->cl = &codelets.padded_image_to_fft;
			paddedImageToFftTask->handles[0] = paddedImagesDataHandle;
			paddedImageToFftTask->handles[1] = fftHandle;
			paddedImageToFftTask->handles[2] = fftScratchMemoryHandle;
			paddedImageToFftTask->handles[3] = amountLoadedHandle;
			paddedImageToFftTask->cl_arg = &imageToFftArg;
			paddedImageToFftTask->cl_arg_size = sizeof(PaddedImageToFftArgs);
			paddedImageToFftTask->cl_arg_free = 0;
			paddedImageToFftTask->synchronous = DEBUG_SYNCHRONOUS_TASKS;
			CHECK_STARPU(starpu_task_submit(paddedImageToFftTask));
		}

		starpu_data_unregister_submit(paddedImagesDataHandle);

		{// Submit the actual reconstruction
			starpu_task* reconstructFftTask = starpu_task_create();
			reconstructFftTask->name = "ReconstructFFT";
			reconstructFftTask->cl = params.useTable ? &codelets.reconstruct_fft_table : &codelets.reconstruct_fft;
			reconstructFftTask->handles[0] = fftHandle;
			reconstructFftTask->handles[1] = traverseSpacesHandle;
			reconstructFftTask->handles[2] = blobTableSquaredHandle;
			reconstructFftTask->handles[3] = resultVolumeHandle;
			reconstructFftTask->handles[4] = resultWeightsHandle;
			reconstructFftTask->handles[5] = amountLoadedHandle;
			reconstructFftTask->cl_arg = &reconstructFftArg;
			reconstructFftTask->cl_arg_size = sizeof(ReconstructFftArgs);
			reconstructFftTask->cl_arg_free = 0;
			reconstructFftTask->callback_func = BatchProvider::batchCompleted;
			reconstructFftTask->callback_arg = &batches;
			reconstructFftTask->callback_arg_free = 0;
			reconstructFftTask->synchronous = DEBUG_SYNCHRONOUS_TASKS;

			CHECK_STARPU(starpu_task_submit(reconstructFftTask));
		}

		// Mark buffers shared between tasks of a batch for removal
		starpu_data_unregister_submit(fftHandle);
		starpu_data_unregister_submit(traverseSpacesHandle);
		starpu_data_unregister_submit(amountLoadedHandle);
		// don't overload scheduler
		starpu_task_wait_for_n_submitted(200);
	}

	// Mark helper buffers that are shared between all tasks for removal
	starpu_data_unregister_submit(blobTableSquaredHandle);
	starpu_data_unregister_submit(fftScratchMemoryHandle);

	// Complete all processing
	/*while (starpu_task_nsubmitted() > 0) {
		starpu_do_schedule();
		usleep(100000);//100ms
	}*/
	CHECK_STARPU(starpu_task_wait_for_all());

	// Release result buffers synchronously, implicitly causes them to be copied to original memory location
	starpu_data_unregister(resultVolumeHandle);
	starpu_data_unregister(resultWeightsHandle);

	padded_image_to_fft_cuda_deinitialize();

	return result;
}

std::complex<float>*** ProgRecFourierStarPU::ComputeStarPUResult::createXmippStyleVolume(uint32_t maxVolumeIndex) {
	{
		auto original = std::cout.flags();
		// prepare output formatting
		std::cout << "volume data\n";
		std::cout << std::fixed << std::setfill(' ') << std::left << std::setprecision(3) << std::showpos;

		for (size_t z = 0; z <= maxVolumeIndex; ++z)
		{
			for (size_t y = 0; y <= maxVolumeIndex; ++y)
			{
				for (size_t x = 0; x <= maxVolumeIndex; ++x)
				{
					auto index = z * (maxVolumeIndex + 1) * (maxVolumeIndex + 1) + y * (maxVolumeIndex + 1) + x;
					std::cout << std::setw(7) << volumeData[index] << ' ';
				}
				std::cout << '\n';
			}
				std::cout << "---\n";
		}

		std::cout.flags(original);
	}
	std::complex<float>*** tempVolume = NULL;
	// allocate(tempVolume, maxVolumeIndex + 1, maxVolumeIndex + 1, maxVolumeIndex + 1);
	// copyFlatTo3D(tempVolume, this->volumeData, maxVolumeIndex + 1);
	return tempVolume;
}

float*** ProgRecFourierStarPU::ComputeStarPUResult::createXmippStyleWeights(uint32_t maxVolumeIndex) {
		{
		auto original = std::cout.flags();
		// prepare output formatting
		std::cout << "weight data\n";
		std::cout << std::fixed << std::setfill(' ') << std::left << std::setprecision(3) << std::showpos;

		for (size_t z = 0; z <= maxVolumeIndex; ++z)
		{
			for (size_t y = 0; y <= maxVolumeIndex; ++y)
			{
				for (size_t x = 0; x <= maxVolumeIndex; ++x)
				{
					auto index = z * (maxVolumeIndex + 1) * (maxVolumeIndex + 1) + y * (maxVolumeIndex + 1) + x;
					std::cout << std::setw(7) << weightsData[index] << ' ';
				}
				std::cout << '\n';
			}
				std::cout << "---\n";
		}

		std::cout.flags(original);
	}
	float*** tempWeights = NULL;
	// allocate(tempWeights, maxVolumeIndex + 1, maxVolumeIndex + 1, maxVolumeIndex + 1);
	// copyFlatTo3D(tempWeights, this->weightsData, maxVolumeIndex + 1);
	return tempWeights;
}

void ProgRecFourierStarPU::ComputeStarPUResult::destroy() {
	CHECK_STARPU(starpu_free(volumeData));
	CHECK_STARPU(starpu_free(weightsData));
}

void ProgRecFourierStarPU::shutdownStarPU() {
	starpu_shutdown();
}

void ProgRecFourierStarPU::postProcessAndSave(
		const Params& params, const ComputeConstants& computeConstants, const FileName& fn_out,
		std::complex<float> ***tempVolume, float ***tempWeights) {

	const uint32_t maxVolumeIndex = computeConstants.maxVolumeIndex;

	// remove complex conjugate of the intermediate result
	const uint32_t maxVolumeIndexX = maxVolumeIndex/2;// just half of the space is necessary, the rest is complex conjugate
	const uint32_t maxVolumeIndexYZ = maxVolumeIndex;
	mirrorAndCropTempSpaces(tempVolume, tempWeights, maxVolumeIndexX, maxVolumeIndexYZ);

	if (params.fastLateBlobbing) {
		tempVolume = applyBlob(tempVolume, (float) params.blob.radius, (float*)computeConstants.blobTableSqrt, computeConstants.iDeltaSqrt, (int)maxVolumeIndexX, (int)maxVolumeIndexYZ);
		tempWeights = applyBlob(tempWeights, (float) params.blob.radius, (float*)computeConstants.blobTableSqrt, computeConstants.iDeltaSqrt, (int)maxVolumeIndexX, (int)maxVolumeIndexYZ);
	}

	const uint32_t paddedImgSize = computeConstants.paddedImgSize;
	const uint32_t imgSize = computeConstants.imgSize;

	forceHermitianSymmetry(tempVolume, tempWeights, maxVolumeIndexYZ);
	processWeights(tempVolume, tempWeights, maxVolumeIndexX, maxVolumeIndexYZ, params.padding_factor_proj, params.padding_factor_vol, imgSize);
	release(tempWeights, maxVolumeIndexYZ+1, maxVolumeIndexYZ+1);
	MultidimArray< std::complex<double> > VoutFourier(paddedImgSize, paddedImgSize, paddedImgSize/2 + 1); // allocate space for output Fourier transformation

	convertToExpectedSpace(tempVolume, maxVolumeIndexYZ, VoutFourier);
	release(tempVolume, maxVolumeIndexYZ+1, maxVolumeIndexYZ+1);

	// Output volume
	Image<double> Vout(paddedImgSize, paddedImgSize, paddedImgSize);
	MultidimArray<double> &mVout = Vout.data;

	{
		FourierTransformer transformerVol;
		transformerVol.setThreadsNumber(params.fourierTransformThreads);
		transformerVol.fReal = &mVout;
		transformerVol.setFourierAlias(VoutFourier);
		transformerVol.recomputePlanR2C();
		transformerVol.inverseFourierTransform();
	}

	CenterFFT(mVout, false);

	// Correct by the Fourier transform of the blob
	mVout.setXmippOrigin();
	mVout.selfWindow(FIRST_XMIPP_INDEX(imgSize),FIRST_XMIPP_INDEX(imgSize),
	                  FIRST_XMIPP_INDEX(imgSize),LAST_XMIPP_INDEX(imgSize),
	                  LAST_XMIPP_INDEX(imgSize),LAST_XMIPP_INDEX(imgSize));
	double pad_relation = params.padding_factor_proj / params.padding_factor_vol;
	pad_relation = (pad_relation * pad_relation * pad_relation);

	double ipad_relation = 1.0 / pad_relation;
	double meanFactor2 = 0;
	FOR_ALL_ELEMENTS_IN_ARRAY3D(mVout) {
		double radius = sqrt((double)(k*k + i*i + j*j));
		double aux = radius * computeConstants.iDeltaFourier;
		double factor = computeConstants.fourierBlobTable[ROUND(aux)];
		double factor2 = pow(Sinc(radius / (2*imgSize)), 2);
		A3D_ELEM(mVout,k,i,j) /= (ipad_relation * factor2 * factor);
		meanFactor2 += factor2;
	}
	meanFactor2 /= MULTIDIM_SIZE(mVout);
	FOR_ALL_ELEMENTS_IN_ARRAY3D(mVout) {
		A3D_ELEM(mVout, k, i, j) *= meanFactor2;
	}
	Vout.write(fn_out);
}

void ProgRecFourierStarPU::BatchProvider::batchCompleted(void *batchProvider) {
	BatchProvider& self = *((BatchProvider*) batchProvider);
	self.batchCompleted();
}
