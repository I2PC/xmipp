/***************************************************************************
 *
 * Authors:     Jan Polak (456647@mail.muni.cz)
 *              (some code derived from other Xmipp programs by other authors)
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

#include "core/transformations.h"
#include "core/xmipp_fftw.h"
#include "data/fourier_projection.h"

#include "reconstruct_fourier_codelets.h"
#include "reconstruct_fourier_starpu_util.h"

/**
 * @param shiftMatrix 3x3 matrix for 2D affine transformations
 * @param orientationMatrix 3x3 matrix for 3D rotation
 */
static void loadMatrices(const MDRow& row, Matrix2D<double>& shiftMatrix, Matrix2D<double>& orientationMatrix) {
	if (!row.containsLabel(MDL_TRANSFORM_MATRIX)) {
		geo2TransformationMatrix(row, shiftMatrix, /*only_apply_shifts*/ true);
	} else {
		//TODO This ignores only_apply_shifts!
		String matrixStr;
		row.getValue(MDL_TRANSFORM_MATRIX, matrixStr);
		string2TransformationMatrix(matrixStr, shiftMatrix, 3);
	}

	// Compute the coordinate axes associated to this projection
	double rot = 0, tilt = 0, psi = 0;
	row.getValue(MDL_ANGLE_ROT, rot);
	row.getValue(MDL_ANGLE_TILT, tilt);
	row.getValue(MDL_ANGLE_PSI, psi);
	Euler_angles2matrix(rot, tilt, psi, orientationMatrix);
	orientationMatrix = orientationMatrix.transpose();
}

void func_load_projections(void* buffers[], void* cl_arg) {
	const LoadProjectionArgs& arg = *static_cast<LoadProjectionArgs*>(cl_arg);

	uint32_t& amountLoaded = ((LoadedImagesBuffer*) STARPU_VARIABLE_GET_PTR(buffers[0]))->noOfImages;
	float* outImageData = (float*)STARPU_VECTOR_GET_PTR(buffers[1]);
	const size_t outImageDataStride = STARPU_VECTOR_GET_ELEMSIZE(buffers[1]) / sizeof(float);
	auto* outSpaces = (RecFourierProjectionTraverseSpace*)STARPU_MATRIX_GET_PTR(buffers[2]);

	MultidimArray<float> transformedImageData;
	MultidimArray<float> paddedImageData(arg.paddedImageSize, arg.paddedImageSize); // Declared here so that internal allocated memory can be reused

	uint32_t traverseSpaceIndex = 0;
	uint32_t projectionIndex = 0;

	for (uint32_t projectionInBatch = arg.batchStart; projectionInBatch < arg.batchEnd; projectionInBatch++) {
		const size_t imageObjectIndex = arg.imageObjectIndices[projectionInBatch];

		// Read projection from selfile, read also angles and shifts if present but only apply shifts
		Projection proj;
		auto row = arg.selFile.getRow(imageObjectIndex);

		double headerWeight = 1;
		row->getValue(MDL_WEIGHT, headerWeight);
		if (arg.useWeights && headerWeight == 0.f) {
			continue;
		}

		{
			FileName name;
			row->getValue(MDL_IMAGE, name);
			proj.read(name);
		}

		// NOTE(jp): Weight has to be read after the image data is read, it cannot be inferred just from the selFile,
		// as the image file may contain weight on its own.
		const float weight =  arg.useWeights ? static_cast<float>(proj.weight() * headerWeight) : 1.0f;
		if (weight == 0.f) {
			continue;
		}

		Matrix2D<double> shiftMatrix(3, 3);
		Matrix2D<double> orientationMatrix(3, 3);
		loadMatrices(*row, shiftMatrix, orientationMatrix);

		transformedImageData.resizeNoCopy(proj());
		// FIXME following line is a current bottleneck, as it calls BSpline interpolation
		applyGeometry(xmipp_transformation::BSPLINE3, transformedImageData, proj.data, shiftMatrix, xmipp_transformation::IS_NOT_INV, xmipp_transformation::WRAP);

		paddedImageData.initZeros(arg.paddedImageSize, arg.paddedImageSize);
		paddedImageData.setXmippOrigin();
		transformedImageData.setXmippOrigin();
		FOR_ALL_ELEMENTS_IN_ARRAY2D(transformedImageData)
				A2D_ELEM(paddedImageData,i,j) = static_cast<float>(A2D_ELEM(transformedImageData, i, j));

#if 0
		{// Debug dump
			FILE* debug_file = fopen("debug_new.bin", "w");
			uint32_t header[] = {
					1234567890,
					sizeof(float),
					(uint32_t) paddedImageData.xdim,
					(uint32_t) paddedImageData.ydim,
					(uint32_t) paddedImageData.zdim,
					1
			};
			fwrite(&header, sizeof(header[0]), sizeof(header)/sizeof(header[0]), debug_file);
			fwrite(paddedImageData.data, paddedImageData.getSize(), sizeof(float), debug_file);
			fclose(debug_file);
		}
#endif

		// CenterFFT = center for fft (not fft itself)
		// NOTE(jp): I am not sure why is this done. It seems to flip some signs in the result of FFT according to some pattern.
		CenterFFT(paddedImageData, true);

		// NOTE(jp): No `continue` skipping after this point, indices to outputs are in lockstep
		memcpy(outImageData + projectionIndex * outImageDataStride, paddedImageData.data, paddedImageData.getSize() * sizeof(float));

		// Prepare transforms for all symmetries
		for (const Matrix2D<double>& symmetry : arg.rSymmetries) {
			RecFourierProjectionTraverseSpace& space = outSpaces[traverseSpaceIndex++];
			space.weight = weight;
			space.projectionIndex = projectionIndex; // "index to some array where the respective projection is stored"

			Matrix2D<double> A_SL = symmetry * orientationMatrix;
			Matrix2D<double> A_SLInv = A_SL.inv();
			float transf[3][3];
			float transfInv[3][3];
			A_SL.convertTo(transf);
			A_SLInv.convertTo(transfInv);

			computeTraverseSpace(arg.fftSizeX, arg.fftSizeY,
			                     transf, transfInv, space,
			                     arg.maxVolumeIndexX, arg.maxVolumeIndexYZ, arg.fastLateBlobbing, arg.blobRadius);
		}

		projectionIndex++;
	}

	amountLoaded = projectionIndex;
}
