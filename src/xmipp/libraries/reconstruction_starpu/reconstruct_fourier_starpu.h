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

#ifndef __RECONSTRUCT_FOURIER_GPU_H
#define __RECONSTRUCT_FOURIER_GPU_H

#include "data/blobs.h"
#include "core/xmipp_filename.h"
#include "reconstruction/recons.h" // Used only as a base class for this program's class
#include "reconstruct_fourier_defines.h"
#include "core/metadata_vec.h"


/**@defgroup FourierReconstruction Fourier reconstruction
   @ingroup ReconsLibrary */
//@{
class ProgRecFourierStarPU : public ProgReconsBase {
protected:

	/** Input file name */
	FileName fn_in;

	/** Output file name */
	FileName fn_out;

	/** File with symmetries */
	FileName fn_sym;


	/** Parameters, always loaded at startup. */
	struct Params {
		int noOfSymmetries;

		size_t fakeNoOfImages;

		size_t fakeX;

		bool useTable;
		/** Projection padding Factor */
		double padding_factor_proj;

		/** Volume padding Factor */
		double padding_factor_vol;

		/** Max resolution in Angstroms */
		float maxResolution;

		/** Flag whether to use the weights in the image metadata */
		bool do_weights;

		/** Definition of the blob */
		struct blobtype blob;

		/** If true, blobbing is done at the end of the computation. This is less accurate, but faster. */
		bool fastLateBlobbing;

		/** Number of threads to be used for final fourier transformation */
		uint32_t fourierTransformThreads;

		/** Size of loading buffer (i.e. max number of projection loaded in one buffer).
		 * Images are processed in batches of this size. */
		uint32_t batchSize;
	} params;

	/** SelFile containing all projections */
	MetaDataVec SF;

	/** Constants needed to perform computations. (Not loaded for MPI master, which does not do any computations.) */
	struct ComputeConstants {
		/** variable used for blob table values calculation */
		double iw0;

		/** Table with blob values, linear sampling */
		double fourierBlobTable[BLOB_TABLE_SIZE_SQRT];

		/** Table with blob values, squared sampling */
		float blobTableSqrt[BLOB_TABLE_SIZE_SQRT];

		/** Inverse of the delta and deltaFourier used in the tables */
		float iDeltaFourier, iDeltaSqrt;

		/** Vector with R symmetry matrices */
		std::vector <Matrix2D<double> > R_symmetries;

		/** Size of the original projection, must be a square */
		uint32_t imgSize;

		/** Size of the image including padding. Image must be a square */
		uint32_t paddedImgSize;

		/** maximal index in the temporal volumes */
		uint32_t maxVolumeIndex;
	} computeConstants;

	/** Internal, here just to have access rights. */
	struct CompleteBatchTasks;

public:
	ProgRecFourierStarPU() {
		// Check that there is no logical problem in the defines used by program. If found, error is thrown.
		if (!((TILE == 1) || (BLOCK_DIM % TILE == 0))) {
			REPORT_ERROR(ERR_PARAM_INCORRECT,"TILE has to be set to 1(one) or BLOCK_DIM has to be a multiple of TILE");
		}
		if ((SHARED_IMG == 1) && (TILE != 1)) {
			REPORT_ERROR(ERR_PARAM_INCORRECT,"TILE cannot be used while SHARED_IMG is active");
		}
		if (TILE >= BLOCK_DIM) {
			REPORT_ERROR(ERR_PARAM_INCORRECT,"TILE must be smaller than BLOCK_DIM");
		}
		if ((SHARED_BLOB_TABLE == 1) && (PRECOMPUTE_BLOB_VAL == 0)) {
			REPORT_ERROR(ERR_PARAM_INCORRECT,"PRECOMPUTE_BLOB_VAL must be set to 1(one) when SHARED_BLOB_TABLE is active");
		}
	}

	/** Specify supported command line arguments */
	void defineParams() override;

	/** Read arguments from command line */
	void readParams() override;

	/** Show basic info to standard output */
	void show() const override;

	/** Run the image processing.
	 * Method will load data, process them and store result to final destination. */
	void run() override;

	/** Functions of common reconstruction interface */
	void setIO(const FileName &fn_in, const FileName &fn_out) override;

protected:

	/** Implements a potentially lazy collection of batches for computeStarPU.
	 * Standard, single machine operation will implement this with a simple counter,
	 * distributed version may retrieve the batches over the network, etc.
	 *
	 * Also tracks the progress, via batchCompleted(), possibly showing and updating a progress bar.
	 * While each batch may take a different amount of time, it is a good and fast approximation. */
	struct BatchProvider {
		/** Max amount of batches this source may ever return.
		 * Can be used to pre-allocate working structures.
		 * Will always return the same number. */
		virtual uint32_t maxBatches() = 0;
		/** Retrieve a next batch to work on.
		 * This call may block the thread while it waits for some batches to complete.
		 * @return batch number or -1 to signify that there are no more batches */
		virtual int32_t nextBatch() = 0;
		/** Called after each batch is completed. */
		virtual void batchCompleted() = 0;

		/** StarPU C proxy for batchCompleted().
		 * @param progressTracker is a pointer to BatchProvider instance. */
		static void batchCompleted(void *batchProvider);
	};

	/** Load meta data, selection file with a list of images to process.
	 * @return total batch count */
	static void prepareMetaData(const FileName& fn_in, MetaDataVec& SF);

	static uint32_t computeBatchCount(const Params& params, const MetaData& SF);

	/** Load or compute constants needed for the computation. */
	static void prepareConstants(const Params& params, const MetaData& SF, const FileName& fn_sym, ComputeConstants& constants);

	/** Initialize StarPU. Separate method to allow MPI override */
	static void initStarPU();

	/** Result of computeStarPU */
	struct ComputeStarPUResult {
		/**
		 * 3D volume holding the cropped (without high frequencies) Fourier space.
		 * Lowest frequencies are in the center, i.e. Fourier space creates a sphere within a cube.
		 *
		 * On GPU reinterpreted as array of `float2`,
		 * or `float` with each two successive values represent one complex number.
		 */
		std::complex<float>* volumeData = nullptr;

		/** 3D volume holding the weights of the Fourier coefficients stored in tempVolume. */
		float* weightsData = nullptr;

		/** Copies data from volumeData to a new Xmipp-style 3D array
		 * (accessible by [x][y][z] operator, with each level individually allocated).
		 * Data must not be destroyed, this does not destroy it. */
		std::complex<float>*** createXmippStyleVolume(uint32_t maxVolumeIndex);

		/** Copies data from weightsData to a new Xmipp-style 3D array.
		 * Data must not be destroyed, this does not destroy it. */
		float*** createXmippStyleWeights(uint32_t maxVolumeIndex);

		/** Must be called to free volumeData and weightsData. */
		void destroy();
	};

	static ComputeStarPUResult computeStarPU(
			const Params& params, const MetaData& SF, const ComputeConstants& computeConstants,
			BatchProvider& batches, bool verbose);

	/** Shutdown StarPU. Separate method to allow MPI override */
	static void shutdownStarPU();

	/**
	 * Method will take data stored in tempVolume and tempWeights,
	 * crops it in  X axis, calculates IFFT and stores the result
	 * to final destination.
	 */
	static void postProcessAndSave(const Params& params, const ComputeConstants& computeConstants, const FileName& fn_out,
			std::complex<float> ***tempVolume, float ***tempWeights);

private:
	struct SimpleBatchProvider;
};
//@}
#endif
