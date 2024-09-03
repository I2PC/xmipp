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

#ifndef XMIPP_LIBRARIES_RECONSTRUCT_FOURIER_STARPU_CODELETS_H_
#define XMIPP_LIBRARIES_RECONSTRUCT_FOURIER_STARPU_CODELETS_H_

#include <core/matrix2d.h>
#include <starpu.h>

class MetaData;

/* Contains StarPU codelets and associated structures */

// ============================== Used buffers ==============================

/*  FFTs Buffer
 *  ===========
 *          float['batchSize' * ('fftSizeX' * 'fftSizeY') * 2]
 *              but only 'noOfImages' is valid (*2 since it's complex)
 *  Holds 'right side of the centered FFT', i.e. only unique values, with high frequencies in the corners
 */

/* Spaces Buffer
 *      RecFourierProjectionTraverseSpace['batchSize' * 'noOfSymmetries']
 *          but only 'noOfImages' is valid
 * Traverse spaces for each FFT, 'describing' the FFTs/paddedImages
 * NOTE: RecFourierProjectionTraverseSpace is a trivial POD struct from <reconstruction/reconstruct_fourier_projection_traverse_space.h>
 */

/* Image Data Buffer (only before FFTs Buffer is ready)
 * =================
 *      double[`batchSize` * (`paddedImageSize` * `paddedImageSize`)]
 *          but only 'noOfImages' is valid
 * Contains  images to be transformed into FFTs.
 */


/* Result Volume Buffer
 * ====================
 * std::complex<float>[maxVolumeIndexYZ+1][maxVolumeIndexYZ+1][maxVolumeIndexYZ+1]
 */

/* Result Weights Buffer
 * =====================
 * float[maxVolumeIndexYZ+1][maxVolumeIndexYZ+1][maxVolumeIndexYZ+1]
 */

/* Blob Table Squared Buffer
 * =========================
 * float[BLOB_TABLE_SIZE_SQRT]
 */

/** Content of a buffer, which holds information about the amount of images in other buffers, which can be variable. */
struct LoadedImagesBuffer {
	/** Amount of valid images in buffers.
	 * 'noOfImages' <= 'batchSize' */
	uint32_t noOfImages;
};

// ============================== Argument structures ==============================

/** MUST NOT BE COPIED!!! */
struct LoadProjectionArgs {
	const uint32_t batchStart, batchEnd;
	const MetaData& selFile;
	const std::vector<size_t>& imageObjectIndices;

	const bool useWeights;
	const std::vector <Matrix2D<double> >& rSymmetries;
	const uint32_t maxVolumeIndexX, maxVolumeIndexYZ;
	const float blobRadius;
	const bool fastLateBlobbing;

	/** Size of source bitmap images. (Regardless of their actual size, they are all padded to this size) */
	uint32_t paddedImageSize;
	/** Dimensions of FFTs. */
	uint32_t fftSizeX, fftSizeY;
};

struct PaddedImageToFftArgs {
	float maxResolutionSqr;

	/** Size of source bitmap images. (Regardless of their actual size, they are all padded to this size) */
	uint32_t paddedImageSize;
	/** Dimensions of FFTs. */
	uint32_t fftSizeX, fftSizeY;
};

struct ReconstructFftArgs {
	float blobRadius;
	uint32_t maxVolIndexYZ;
	bool fastLateBlobbing;
	int blobOrder;
	float blobAlpha;

	/** Amount of symmetries in the image. See Spaces Buffer. */
	uint32_t noOfSymmetries;
	/** Dimensions of FFTs. */
	uint32_t fftSizeX, fftSizeY;
};

// ============================== Codelets ==============================

extern void func_load_projections(void* buffers[], void* cl_arg);

void padded_image_to_fft_cuda_initialize(uint32_t paddedImageSize);
void padded_image_to_fft_cuda_deinitialize();

extern void func_padded_image_to_fft_cpu(void **buffers, void *cl_arg);
extern void func_padded_image_to_fft_cuda(void **buffers, void *cl_arg);

/** Copy constants used for calculation to GPU memory. Blocking operation. */
void reconstruct_cuda_initialize_constants(
		int maxVolIndexX, int maxVolIndexYZ,
		float blobRadius, float blobAlpha,
		float iDeltaSqrt, float iw0, float oneOverBessiOrderAlpha);
extern void func_reconstruct_cuda(void* buffers[], void* cl_arg);

extern void func_reconstruct_cpu_lookup_interpolation(void* buffers[], void* cl_arg);
extern void func_reconstruct_cpu_dynamic_interpolation(void* buffers[], void* cl_arg);


extern void func_redux_sum_volume_cuda(void* buffers[], void* cl_arg);
extern void func_redux_init_volume_cuda(void* buffers[], void* cl_arg);
extern void func_redux_sum_weights_cuda(void* buffers[], void* cl_arg);
extern void func_redux_init_weights_cuda(void* buffers[], void* cl_arg);

extern void func_redux_sum_volume_cpu(void* buffers[], void* cl_arg);
extern void func_redux_init_volume_cpu(void* buffers[], void* cl_arg);
extern void func_redux_sum_weights_cpu(void* buffers[], void* cl_arg);
extern void func_redux_init_weights_cpu(void* buffers[], void* cl_arg);

struct Codelets {

	/** Initial data loading. Loads data for a single batch. */
	starpu_codelet load_projections{0};
	/** Crops and shifts loaded and fourier-transformed images. */
	starpu_codelet padded_image_to_fft{0};
	/** Reconstructs FFTs into volume field. */
	starpu_codelet reconstruct_fft{0};

	/** Redux-init for reconstruct_fft volume and weights */
	starpu_codelet redux_init_volume{0}, redux_init_weights{0};
	/** Redux for reconstruct_fft volume and weights (sum) */
	starpu_codelet redux_sum_volume{0}, redux_sum_weights{0};

	/** Initializes the codelets. */
	Codelets();
};

extern Codelets codelets;

#endif //XMIPP_LIBRARIES_RECONSTRUCT_FOURIER_STARPU_CODELETS_H_
