/***************************************************************************
 *
 * Authors:    David Herreros Calero dherreros@cnb.csic.es
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

#include "forward_art_zernike3d_subtomos.h"
#include <core/transformations.h>
#include <core/xmipp_image_extension.h>
#include <core/xmipp_image_generic.h>
#include <data/projection.h>
#include <data/mask.h>
#include <numeric>

// Empty constructor =======================================================
ProgForwardArtZernike3DSubtomos::ProgForwardArtZernike3DSubtomos()
{
	resume = false;
	produces_a_metadata = true;
	each_image_produces_an_output = false;
	showOptimization = false;
}

ProgForwardArtZernike3DSubtomos::~ProgForwardArtZernike3DSubtomos() = default;

// Read arguments ==========================================================
void ProgForwardArtZernike3DSubtomos::readParams()
{
	XmippMetadataProgram::readParams();
	fnVolR = getParam("--ref");
	fnMaskR = getParam("--mask");
	fnOutDir = getParam("--odir");
	RmaxDef = getIntParam("--RDef");
	phaseFlipped = checkParam("--phaseFlipped");
	useCTF = checkParam("--useCTF");
	Ts = getDoubleParam("--sampling");
	L1 = getIntParam("--l1");
	L2 = getIntParam("--l2");
	t1 = getDoubleParam("--t1");
	t2 = getDoubleParam("--t2");
	blob_r = getDoubleParam("--blobr");
	loop_step = getIntParam("--step");
	sigma = getDoubleParam("--sigma");
	useZernike = checkParam("--useZernike");
	lambda = getDoubleParam("--regularization");
	resume = checkParam("--resume");
	niter = getIntParam("--niter");
	save_iter = getIntParam("--save_iter");
	sort_last_N = getIntParam("--sort_last");
	// fnDone = fnOutDir + "/sphDone.xmd";
	fnVolO = fnOutDir + "/Refined.vol";
}

// Show ====================================================================
void ProgForwardArtZernike3DSubtomos::show()
{
	if (!verbose)
		return;
	XmippMetadataProgram::show();
	std::cout
		<< "Output directory:          " << fnOutDir << std::endl
		<< "Reference volume:          " << fnVolR << std::endl
		<< "Reference mask:            " << fnMaskR << std::endl
		<< "Sampling:                  " << Ts << std::endl
		<< "Max. Radius Deform.        " << RmaxDef << std::endl
		<< "Zernike Degree:            " << L1 << std::endl
		<< "SH Degree:                 " << L2 << std::endl
		<< "Blob radius:               " << blob_r << std::endl
		<< "Step:                      " << loop_step << std::endl
		<< "Correct CTF:               " << useCTF << std::endl
		<< "Correct heretogeneity:     " << useZernike << std::endl
		<< "Phase flipped:             " << phaseFlipped << std::endl
		<< "Regularization:            " << lambda << std::endl
		<< "Number of iterations:      " << niter << std::endl
		<< "Save every # iterations:   " << save_iter << std::endl;
}

// usage ===================================================================
void ProgForwardArtZernike3DSubtomos::defineParams()
{
	addUsageLine("Template-based canonical volume reconstruction through Zernike3D coefficients");
	defaultComments["-i"].clear();
	defaultComments["-i"].addComment("Metadata with initial alignment");
	defaultComments["-o"].clear();
	defaultComments["-o"].addComment("Refined volume");
	XmippMetadataProgram::defineParams();
	addParamsLine("  [--ref <volume=\"\">]        : Reference volume");
	addParamsLine("  [--mask <m=\"\">]            : Mask reference volume");
	addParamsLine("  [--odir <outputDir=\".\">]   : Output directory");
	addParamsLine("  [--sampling <Ts=1>]          : Sampling rate (A/pixel)");
	addParamsLine("  [--RDef <r=-1>]              : Maximum radius of the deformation (px). -1=Half of volume size");
	addParamsLine("  [--l1 <l1=3>]                : Degree Zernike Polynomials=1,2,3,...");
	addParamsLine("  [--l2 <l2=2>]                : Harmonical depth of the deformation=1,2,3,...");
	addParamsLine("  [--blobr <b=4>]              : Blob radius for forward mapping splatting");
	addParamsLine("  [--step <step=1>]            : Voxel index step");
	addParamsLine("  [--sigma <step=0.25>]        : Gaussian sigma");
	addParamsLine("  [--useZernike]               : Correct heterogeneity with Zernike3D coefficients");
	addParamsLine("  [--useCTF]                   : Correct CTF during ART reconstruction");
	addParamsLine("  [--phaseFlipped]             : Input images have been phase flipped");
	addParamsLine("  [--regularization <l=0.01>]  : ART regularization weight");
	addParamsLine("  [--niter <n=1>]              : Number of ART iterations");
	addParamsLine("  [--save_iter <s=0>]          : Save intermidiate volume after #save_iter iterations");
	addParamsLine("  [--sort_last <N=2>]          : The algorithm sorts projections in the most orthogonally possible way. ");
	addParamsLine("                               : The most orthogonal way is defined as choosing the projection which maximizes the ");
	addParamsLine("                               : dot product with the N previous inserted projections. Use -1 to sort with all  ");
	addParamsLine("                               : previous projections");
	addParamsLine("  [--t1 <t1=-60>]              : First tilt angle range");
	addParamsLine("  [--t2 <t2=60>]               : Second tilt angle range");
	addParamsLine("  [--resume]                   : Resume processing");
	addExampleLine("A typical use is:", false);
	addExampleLine("xmipp_art_zernike3d -i anglesFromContinuousAssignment.xmd --ref reference.vol -o assigned_anglesAndDeformations.xmd --l1 3 --l2 2");
}

// // Produce side information ================================================
// void ProgForwardArtZernike3DSubtomos::createWorkFiles() {
// 	// w_i = 1 / getInputMd()->size();
// 	if (resume && fnDone.exists()) {
// 		MetaDataDb done(fnDone);
// 		done.read(fnDone);
// 		getOutputMd() = done;
// 		auto *candidates = getInputMd();
// 		MetaDataDb toDo(*candidates);
// 		toDo.subtraction(done, MDL_IMAGE);
// 		toDo.write(fnOutDir + "/sphTodo.xmd");
// 		*candidates = toDo;
// 	}
// }

void ProgForwardArtZernike3DSubtomos::preProcess()
{

	// Check that metadata has all information neede
	if (!getInputMd()->containsLabel(MDL_ANGLE_ROT) ||
		!getInputMd()->containsLabel(MDL_ANGLE_TILT) ||
		!getInputMd()->containsLabel(MDL_ANGLE_PSI))
	{
		REPORT_ERROR(ERR_MD_MISSINGLABEL, "Input metadata projection angles are missing. Exiting...");
	}

	if (fnVolR != "")
	{
		V.read(fnVolR);
	}
	else
	{
		FileName fn_first_image;
		Image<double> first_image;
		getInputMd()->getRow(1)->getValue(MDL_IMAGE, fn_first_image);
		first_image.read(fn_first_image);
		size_t Xdim_first = XSIZE(first_image());
		V().initZeros(Xdim_first, Xdim_first, Xdim_first);
	}
	V().setXmippOrigin();

	Xdim = XSIZE(V());

	Vout().initZeros(V());
	Vout().setXmippOrigin();

	if (resume && fnVolO.exists())
	{
		Vrefined.read(fnVolO);
	}
	else
	{
		Vrefined() = V();
	}
	// Vrefined().initZeros(V());
	Vrefined().setXmippOrigin();

	if (RmaxDef < 0)
		RmaxDef = Xdim / 2;

	// Transformation matrix
	A.initIdentity(4);

	// CTF Filter
	FilterCTF.FilterBand = CTFINV;
	FilterCTF.FilterShape = RAISED_COSINE;
	FilterCTF.ctf.enable_CTFnoise = false;
	FilterCTF.ctf.enable_CTF = true;
	// FilterCTF.ctf.produceSideInfo();

	// Area where Zernike3D basis is computed (and volume is updated)
	// Read Reference mask if avalaible (otherwise sphere of radius RmaxDef is used)
	Mask mask;
	mask.type = BINARY_CIRCULAR_MASK;
	mask.mode = INNER_MASK;
	if (fnMaskR != "")
	{
		Image<double> aux;
		aux.read(fnMaskR);
		typeCast(aux(), Vmask);
		Vmask.setXmippOrigin();
		double Rmax2 = RmaxDef * RmaxDef;
		for (int k = STARTINGZ(Vmask); k <= FINISHINGZ(Vmask); k++)
		{
			for (int i = STARTINGY(Vmask); i <= FINISHINGY(Vmask); i++)
			{
				for (int j = STARTINGX(Vmask); j <= FINISHINGX(Vmask); j++)
				{
					double r2 = k * k + i * i + j * j;
					if (r2 >= Rmax2)
						A3D_ELEM(Vmask, k, i, j) = 0;
				}
			}
		}
	}
	else
	{
		mask.R1 = RmaxDef;
		mask.generate_mask(V());
		Vmask = mask.get_binary_mask();
		Vmask.setXmippOrigin();
	}

	// Area Zernike3D in 2D
	mask.R1 = RmaxDef;
	mask.generate_mask(XSIZE(V()), XSIZE(V()));
	mask2D = mask.get_binary_mask();
	mask2D.setXmippOrigin();

	vecSize = 0;
	numCoefficients(L1, L2, vecSize);
	fillVectorTerms(L1, L2, vL1, vN, vL2, vM);

	// createWorkFiles();
	initX = STARTINGX(Vrefined());
	endX = FINISHINGX(Vrefined());
	initY = STARTINGY(Vrefined());
	endY = FINISHINGY(Vrefined());
	initZ = STARTINGZ(Vrefined());
	endZ = FINISHINGZ(Vrefined());

	sigma4 = 4 * sigma;  // Test 3
	double size_vg = CEIL(sigma4 * sqrt(2) * 50);
	gaussian3d.initZeros(size_vg, size_vg, size_vg);
	gaussian3d.setXmippOrigin();
	FOR_ALL_ELEMENTS_IN_ARRAY3D(gaussian3d)
	A3D_ELEM(gaussian3d, k, i, j) = gaussian1D(sqrt(k*k + i*i + j*j) / 50.0, sigma);
	gaussian3d *= gaussian1D(0, sigma);

	// This could also be a valid option, but we need a large volume to enclose the gaussian + 
	// its missing wedge properly. Since the convolution is a linear operator, it is better to apply 
	// it once the volume has been formed by standard gaussians
	// Needed gaussian dimensions CEIL(sigma4 * 2.5 * 50)
	// applyMissingWedge(gaussian3d);

	Image<double> W;
	W() = gaussian3d;
	FileName fnMask=fnOutDir+"gaussian_mw.mrc";
	W.write(fnMask);

}

void ProgForwardArtZernike3DSubtomos::finishProcessing()
{
	// XmippMetadataProgram::finishProcessing();
	recoverVol();
	Vout.write(fnVolO);
}

// Predict =================================================================
//#define DEBUG
void ProgForwardArtZernike3DSubtomos::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
	flagEnabled = 1;

	int img_enabled;
	rowIn.getValue(MDL_ENABLED, img_enabled);
	if (img_enabled == -1) return;

	// double auxRot, auxTilt, auxPsi, auxShiftX, auxShiftY;
	rowIn.getValue(MDL_ANGLE_ROT, rot);
	rowIn.getValue(MDL_ANGLE_TILT, tilt);
	rowIn.getValue(MDL_ANGLE_PSI, psi);
	rowIn.getValueOrDefault(MDL_SHIFT_X, shiftX, 0.0);
	rowIn.getValueOrDefault(MDL_SHIFT_Y, shiftY, 0.0);
	rowIn.getValueOrDefault(MDL_SHIFT_Z, shiftZ, 0.0);
	std::vector<double> vectortemp;
	if (useZernike)
	{
		rowIn.getValue(MDL_SPH_COEFFICIENTS, vectortemp);
		std::vector<double> vec(vectortemp.begin(), vectortemp.end());
		clnm = vec;
	}
	rowIn.getValueOrDefault(MDL_FLIP, flip, false);

	if ((rowIn.containsLabel(MDL_CTF_DEFOCUSU) || rowIn.containsLabel(MDL_CTF_MODEL)) && useCTF)
	{
		// std::cout << "Applying CTF" << std::endl;
		hasCTF = true;
		FilterCTF.ctf.readFromMdRow(rowIn, false);
		FilterCTF.ctf.Tm = Ts;
		FilterCTF.ctf.produceSideInfo();
	}
	else
		hasCTF = false;
	MAT_ELEM(A, 0, 3) = shiftX;
	MAT_ELEM(A, 1, 3) = shiftY;
	MAT_ELEM(A, 2, 3) = shiftZ;
	MAT_ELEM(A, 0, 0) = 1;
	MAT_ELEM(A, 0, 1) = 0;
	MAT_ELEM(A, 0, 2) = 0;
	MAT_ELEM(A, 1, 0) = 0;
	MAT_ELEM(A, 1, 1) = 1;
	MAT_ELEM(A, 1, 2) = 0;
	MAT_ELEM(A, 2, 0) = 0;
	MAT_ELEM(A, 2, 1) = 0;
	MAT_ELEM(A, 2, 2) = 1;


	if (verbose >= 2)
		std::cout << "Processing " << fnImg << std::endl;

	I.read(fnImg);
	I().setXmippOrigin();

	// Forward Model
	artModel<Direction::Forward>();
	// forwardModel();

	// ART update
	artModel<Direction::Backward>();
	// updateART();
}
#undef DEBUG

// void ProgForwardArtZernike3DSubtomos::checkPoint() {
// 	getOutputMd().write(fnDone);
// 	// Vrefined.write(fnVolO);
// }

void ProgForwardArtZernike3DSubtomos::numCoefficients(int l1, int l2, int &vecSize)
{
	for (int h = 0; h <= l2; h++)
	{
		int numSPH = 2 * h + 1;
		int count = l1 - h + 1;
		int numEven = (count >> 1) + (count & 1 && !(h & 1));
		if (h % 2 == 0)
		{
			vecSize += numSPH * numEven;
		}
		else
		{
			vecSize += numSPH * (l1 - h + 1 - numEven);
		}
	}
}

void ProgForwardArtZernike3DSubtomos::fillVectorTerms(int l1, int l2, Matrix1D<int> &vL1, Matrix1D<int> &vN,
											  Matrix1D<int> &vL2, Matrix1D<int> &vM)
{
	int idx = 0;
	vL1.initZeros(vecSize);
	vN.initZeros(vecSize);
	vL2.initZeros(vecSize);
	vM.initZeros(vecSize);
	for (int h = 0; h <= l2; h++)
	{
		int totalSPH = 2 * h + 1;
		int aux = std::floor(totalSPH / 2);
		for (int l = h; l <= l1; l += 2)
		{
			for (int m = 0; m < totalSPH; m++)
			{
				VEC_ELEM(vL1, idx) = l;
				VEC_ELEM(vN, idx) = h;
				VEC_ELEM(vL2, idx) = h;
				VEC_ELEM(vM, idx) = m - aux;
				idx++;
			}
		}
	}
}

// void ProgForwardArtZernike3DSubtomos::updateCTFImage(double defocusU, double defocusV, double angle)
// {
// 	ctf.K=1; // get pure CTF with no envelope
// 	ctf.produceSideInfo();
// }

void ProgForwardArtZernike3DSubtomos::splattingAtPos(std::array<double, 3> r, double weight,
											 MultidimArray<double> &mP, MultidimArray<double> &mW,
											 MultidimArray<double> &mV)
{
// Find the part of the volume that must be updated
	double x_pos = r[0];
	double y_pos = r[1];
	double z_pos = r[2];
	auto &mG = gaussian3d;
	int k0 = XMIPP_MAX(FLOOR(z_pos - sigma4), STARTINGZ(mV));
	int kF = XMIPP_MIN(CEIL(z_pos + sigma4), FINISHINGZ(mV));
	int i0 = XMIPP_MAX(FLOOR(y_pos - sigma4), STARTINGY(mV));
	int iF = XMIPP_MIN(CEIL(y_pos + sigma4), FINISHINGY(mV));
	int j0 = XMIPP_MAX(FLOOR(x_pos - sigma4), STARTINGX(mV));
	int jF = XMIPP_MIN(CEIL(x_pos + sigma4), FINISHINGX(mV));
	for (int k = k0; k <= kF; k++)
	{
		int k_int = ROUND(50 * (k - z_pos));
		for (int i = i0; i <= iF; i++)
		{
			int i_int = ROUND(50 * (i - y_pos));
			for (int j = j0; j <= jF; j++)
			{
				// double mod = sqrt((x_pos - j) * (x_pos - j) + y2 + k2);
				// // A3D_ELEM(Vdeformed(),k, i, j) += weight * blob_val(rdiff.module(), blob);
				// A3D_ELEM(mP, k, i, j) += weight * kaiser_value(mod, blob.radius, blob.alpha, blob.order);
				int j_int = ROUND(50 * (j - x_pos));
				if (!gaussian3d.outside(k_int, i_int, j_int))
				{
					double gw = A3D_ELEM(mG, k_int, i_int, j_int);
					A3D_ELEM(mP, k, i, j) += weight * gw;
					A3D_ELEM(mW, k, i, j) += gw * gw;
				}
			}
		}
	}
}

void ProgForwardArtZernike3DSubtomos::updateVoxel(std::array<double, 3> r, double &voxel, MultidimArray<double> &mV)
{
	// Find the part of the volume that must be updated
	double x_pos = r[0];
	double y_pos = r[1];
	double z_pos = r[2];
	double hsigma4 = 4 * sqrt(2);  // 1.5 * sqrt(2)
	double hsigma = 1;  // 1.5 / 4
	int k0 = XMIPP_MAX(FLOOR(z_pos - hsigma4), STARTINGZ(mV));
	int kF = XMIPP_MIN(CEIL(z_pos + hsigma4), FINISHINGZ(mV));
	int i0 = XMIPP_MAX(FLOOR(y_pos - hsigma4), STARTINGY(mV));
	int iF = XMIPP_MIN(CEIL(y_pos + hsigma4), FINISHINGY(mV));
	int j0 = XMIPP_MAX(FLOOR(x_pos - hsigma4), STARTINGX(mV));
	int jF = XMIPP_MIN(CEIL(x_pos + hsigma4), FINISHINGX(mV));
	auto alpha = blob.alpha;
	auto order = blob.order;
	// Perform splatting at this position r
	// ? Probably we can loop only a quarter of the region and use the symmetry to make this faster?
	for (int k = k0; k <= kF; k++)
	{
		for (int i = i0; i <= iF; i++)
		{
			for (int j = j0; j <= jF; j++)
			{
				A3D_ELEM(mV, k, i, j) += voxel * gaussian1D(k-z_pos,hsigma)*
                            					 gaussian1D(i-y_pos,hsigma)*
                            					 gaussian1D(j-x_pos,hsigma);
			}
		}
	}
}

void ProgForwardArtZernike3DSubtomos::recoverVol()
{
	// Find the part of the volume that must be updated
	auto &mVout = Vout();
	const auto &mV = Vrefined();
	mVout.initZeros(mV);

	const auto lastZ = FINISHINGZ(mV);
	const auto lastY = FINISHINGY(mV);
	const auto lastX = FINISHINGX(mV);
	// const int step = DIRECTION == Direction::Forward ? loop_step : 1;
	const int step = loop_step;
	auto pos = std::array<double, 3>{};

	for (int k = STARTINGZ(mV); k <= lastZ; k++)
	{
		for (int i = STARTINGY(mV); i <= lastY; i++)
		{
			for (int j = STARTINGX(mV); j <= lastX; j++)
			{
				if (A3D_ELEM(Vmask, k, i, j) == 1)
				{
					pos[0] = j;
					pos[1] = i;
					pos[2] = k;
					updateVoxel(pos, A3D_ELEM(mV, k, i, j), mVout);
				}
			}
		}
	}
}

void ProgForwardArtZernike3DSubtomos::run()
{
	FileName fnImg, fnImgOut, fullBaseName;
	getOutputMd().clear(); //this allows multiple runs of the same Program object

	//Perform particular preprocessing
	preProcess();

	startProcessing();

	sortOrthogonal();

	if (!oroot.empty())
	{
		if (oext.empty())
			oext = oroot.getFileFormat();
		oextBaseName = oext;
		fullBaseName = oroot.removeFileFormat();
		baseName = fullBaseName.getBaseName();
		pathBaseName = fullBaseName.getDir();
	}

	size_t objId;
	size_t objIndex;
	current_save_iter = 1;
	num_images = 1;
	for (current_iter = 0; current_iter < niter; current_iter++)
	{
		std::cout << "Running iteration " << current_iter + 1 << " with lambda=" << lambda << std::endl;
		objId = 0;
		objIndex = 0;
		time_bar_done = 0;
		FOR_ALL_ELEMENTS_IN_ARRAY1D(ordered_list)
		{
			objId = A1D_ELEM(ordered_list, i) + 1;
			++objIndex;
			auto rowIn = getInputMd()->getRow(objId);
			if (rowIn == nullptr) continue;
			rowIn->getValue(image_label, fnImg);

			if (fnImg.empty())
				break;

			fnImgOut = fnImg;

			MDRowVec rowOut;

			// if (each_image_produces_an_output)
			// {
			// 	if (!oroot.empty()) // Compose out name to save as independent images
			// 	{
			// 		if (oext.empty()) // If oext is still empty, then use ext of indep input images
			// 		{
			// 			if (input_is_stack)
			// 				oextBaseName = "spi";
			// 			else
			// 				oextBaseName = fnImg.getFileFormat();
			// 		}

			// 		if (!baseName.empty() )
			// 			fnImgOut.compose(fullBaseName, objIndex, oextBaseName);
			// 		else if (fnImg.isInStack())
			// 			fnImgOut.compose(pathBaseName + (fnImg.withoutExtension()).getDecomposedFileName(), objIndex, oextBaseName);
			// 		else
			// 			fnImgOut = pathBaseName + fnImg.withoutExtension()+ "." + oextBaseName;
			// 	}
			// 	else if (!fn_out.empty() )
			// 	{
			// 		if (single_image)
			// 			fnImgOut = fn_out;
			// 		else
			// 			fnImgOut.compose(objIndex, fn_out); // Compose out name to save as stacks
			// 	}
			// 	else
			// 		fnImgOut = fnImg;
			// 	setupRowOut(fnImg, *rowIn.get(), fnImgOut, rowOut);
			// }
			// else if (produces_a_metadata)
			// 	setupRowOut(fnImg, *rowIn.get(), fnImgOut, rowOut);

			processImage(fnImg, fnImgOut, *rowIn.get(), rowOut);

			// if (each_image_produces_an_output || produces_a_metadata)
			// 	getOutputMd().addRow(rowOut);

			checkPoint();
			showProgress();

			// Save refined volume every num_images
			if (current_save_iter == save_iter && save_iter > 0)
			{
				recoverVol();
				// Mask mask;
				// mask.type = BINARY_CIRCULAR_MASK;
				// mask.mode = INNER_MASK;
				// mask.R1 = RmaxDef - 2;
				// mask.generate_mask(Vrefined());
				// mask.apply_mask(Vrefined(), Vrefined());
				// Vrefined.write(fnVolO.removeAllExtensions() + "it" + std::to_string(current_iter + 1) + "proj" + std::to_string(num_images) + ".mrc");
				Vout.write(fnVolO.removeAllExtensions() + "_partial.mrc");
				current_save_iter = 1;
			}
			current_save_iter++;
			num_images++;
		}
		num_images = 1;
		current_save_iter = 1;

		recoverVol();
		Vout.write(fnVolO.removeAllExtensions() + "_iter" + std::to_string(current_iter + 1) + ".mrc");

		// Vrefined().threshold("below", 0, 0);
		// Mask mask;
		// mask.type = BINARY_CIRCULAR_MASK;
		// mask.mode = INNER_MASK;
		// mask.R1 = RmaxDef - 2;
		// mask.generate_mask(Vrefined());
		// mask.apply_mask(Vrefined(), Vrefined());
	}
	wait();

	/* Generate name to save mdOut when output are independent images. It uses as prefix
     * the dirBaseName in order not overwriting files when repeating same command on
     * different directories. If baseName is set it is used, otherwise, input name is used.
     * Then, the suffix _oext is added.*/
	if (fn_out.empty())
	{
		if (!oroot.empty())
		{
			if (!baseName.empty())
				fn_out = findAndReplace(pathBaseName, "/", "_") + baseName + "_" + oextBaseName + ".xmd";
			else
				fn_out = findAndReplace(pathBaseName, "/", "_") + fn_in.getBaseName() + "_" + oextBaseName + ".xmd";
		}
		else if (input_is_metadata) /// When nor -o neither --oroot is passed and want to overwrite input metadata
			fn_out = fn_in;
	}

	finishProcessing();

	postProcess();

	/* Reset the default values of the program in case
     * to be reused.*/
	init();
}

void ProgForwardArtZernike3DSubtomos::sortOrthogonal()
{
	int i, j;
	size_t numIMG = getInputMd()->size();
	MultidimArray<short> chosen(numIMG);
	MultidimArray<double> product(numIMG);
	double min_prod = MAXDOUBLE;
	;
	int min_prod_proj = 0;
	std::vector<double> rot;
	std::vector<double> tilt;
	Matrix2D<double> v(numIMG, 3);
	Matrix2D<double> euler;
	getInputMd()->getColumnValues(MDL_ANGLE_ROT, rot);
	getInputMd()->getColumnValues(MDL_ANGLE_TILT, tilt);

	// Initialization
	ordered_list.resize(numIMG);
	for (i = 0; i < numIMG; i++)
	{
		Matrix1D<double> z;
		// Initially no image is chosen
		A1D_ELEM(chosen, i) = 0;

		// Compute the Euler matrix for each image and keep only
		// the third row of each one
		Euler_angles2matrix(rot[i], tilt[i], 0., euler);
		euler.getRow(2, z);
		v.setRow(i, z);
	}

	// Pick first projection as the first one to be presented
	i = 0;
	A1D_ELEM(chosen, i) = 1;
	A1D_ELEM(ordered_list, 0) = i;

	// Choose the rest of projections
	std::cout << "Sorting projections orthogonally...\n"
			  << std::endl;
	Matrix1D<double> rowj, rowi_1, rowi_N_1;
	for (i = 1; i < numIMG; i++)
	{
		// Compute the product of not already chosen vectors with the just
		// chosen one, and select that which has minimum product
		min_prod = MAXDOUBLE;
		v.getRow(A1D_ELEM(ordered_list, i - 1), rowi_1);
		if (sort_last_N != -1 && i > sort_last_N)
			v.getRow(A1D_ELEM(ordered_list, i - sort_last_N - 1), rowi_N_1);
		for (j = 0; j < numIMG; j++)
		{
			if (!A1D_ELEM(chosen, j))
			{
				v.getRow(j, rowj);
				A1D_ELEM(product, j) += ABS(dotProduct(rowi_1, rowj));
				if (sort_last_N != -1 && i > sort_last_N)
					A1D_ELEM(product, j) -= ABS(dotProduct(rowi_N_1, rowj));
				if (A1D_ELEM(product, j) < min_prod)
				{
					min_prod = A1D_ELEM(product, j);
					min_prod_proj = j;
				}
			}
		}

		// Store the chosen vector and mark it as chosen
		A1D_ELEM(ordered_list, i) = min_prod_proj;
		A1D_ELEM(chosen, min_prod_proj) = 1;
	}
}

template <ProgForwardArtZernike3DSubtomos::Direction DIRECTION>
void ProgForwardArtZernike3DSubtomos::artModel()
{
	if (DIRECTION == Direction::Forward)
	{
		Image<double> I_shifted;
		P().initZeros((int)XSIZE(I()), (int)XSIZE(I()), (int)XSIZE(I()));
		P().setXmippOrigin();
		W().initZeros((int)XSIZE(I()), (int)XSIZE(I()), (int)XSIZE(I()));
		W().setXmippOrigin();
		Idiff().initZeros((int)XSIZE(I()), (int)XSIZE(I()), (int)XSIZE(I()));
		Idiff().setXmippOrigin();
		I_shifted().initZeros((int)XSIZE(I()), (int)XSIZE(I()), (int)XSIZE(I()));
		I_shifted().setXmippOrigin();

		if (useZernike)
			zernikeModel<true, Direction::Forward>();
		else
			zernikeModel<false, Direction::Forward>();

		if (hasCTF)
		{
			// updateCTFImage(defocusU, defocusV, defocusAngle);
			// FilterCTF.ctf = ctf;
			if (phaseFlipped)
				FilterCTF.correctPhase();
			FilterCTF.generateMask(I());
			FilterCTF.applyMaskSpace(I());
		}
		if (flip)
		{
			MAT_ELEM(A, 0, 0) *= -1;
			MAT_ELEM(A, 0, 1) *= -1;
			MAT_ELEM(A, 0, 2) *= -1;
		}

		applyMissingWedge(P());

		applyGeometry(xmipp_transformation::LINEAR, I_shifted(), I(), A,
					  xmipp_transformation::IS_NOT_INV, xmipp_transformation::DONT_WRAP, 0.);

		// P.write(fnOutDir + "/PPPtheo.xmp");
		// I_shifted.write(fnOutDir + "/PPPexp.xmp");
		// std::cout << "Press any key" << std::endl;
		// char c; std::cin >> c;

		// Compute difference image and divide by weights
		double error = 0.0f;
		double N = 0.0f;
		const auto &mP = P();
		const auto &mW = W();
		const auto &mIsh = I_shifted();
		auto &mId = Idiff();
		FOR_ALL_ELEMENTS_IN_ARRAY3D(I())
		{
			if (A3D_ELEM(mW, k, i, j) > 0)
			{
				auto diffVal = A3D_ELEM(mIsh, k, i, j) - A3D_ELEM(mP, k, i, j);
				A3D_ELEM(mId, k, i, j) = lambda * (diffVal) / XMIPP_MAX(A3D_ELEM(mW, k, i, j), 1.0);
				error += (diffVal) * (diffVal);
				N++;
			}
		}

		if (verbose >= 3)
		{
			P.write(fnOutDir + "/PPPtheo.mrc");
			W.write(fnOutDir + "/PPPweight.mrc");
			Idiff.write(fnOutDir + "/PPPcorr.mrc");
			I.write(fnOutDir + "/PPPsubtomo.mrc");
			std::cout << "Press any key" << std::endl;
			char c;
			std::cin >> c;
		}

		// Creo que Carlos no usa un RMSE si no un MSE
		error = std::sqrt(error / N);
		std::cout << "Error for image " << num_images << " in iteration " << current_iter + 1 << " : " << error << std::endl;
	}
	else if (DIRECTION == Direction::Backward)
	{
		if (useZernike)
			zernikeModel<true, Direction::Backward>();
		else
			zernikeModel<false, Direction::Backward>();
	}
}

template <bool USESZERNIKE, ProgForwardArtZernike3DSubtomos::Direction DIRECTION>
void ProgForwardArtZernike3DSubtomos::zernikeModel()
{
	auto &mId = Idiff();
	auto &mV = Vrefined();
	auto &mP = P();
	auto &mW = W();
	const size_t idxY0 = USESZERNIKE ? (clnm.size() / 3) : 0;
	const size_t idxZ0 = USESZERNIKE ? (2 * idxY0) : 0;
	const double RmaxF = USESZERNIKE ? RmaxDef : 0;
	const double RmaxF2 = USESZERNIKE ? (RmaxF * RmaxF) : 0;
	const double iRmaxF = USESZERNIKE ? (1.0 / RmaxF) : 0;
	// Rotation Matrix
	constexpr size_t matrixSize = 3;
	const Matrix2D<double> R = [this]()
	{
		auto tmp = Matrix2D<double>();
		tmp.initIdentity(matrixSize);
		Euler_angles2matrix(rot, tilt, psi, tmp, false);
		return tmp;
	}();

	// auto l2Mask = std::vector<size_t>();
	// for (size_t idx = 0; idx < idxY0; idx++) {
	//   if (0 == VEC_ELEM(vL2, idx)) {
	//     l2Mask.emplace_back(idx);
	//   }

	// }
	const auto lastZ = FINISHINGZ(mV);
	const auto lastY = FINISHINGY(mV);
	const auto lastX = FINISHINGX(mV);
	const int step = DIRECTION == Direction::Forward ? loop_step : 1;
	// const int step = loop_step;

	for (int k = STARTINGZ(mV); k <= lastZ; k += step)
	{
		for (int i = STARTINGY(mV); i <= lastY; i += step)
		{
			for (int j = STARTINGX(mV); j <= lastX; j += step)
			{
				double gx = 0.0f, gy = 0.0f, gz = 0.0f;
				if (A3D_ELEM(Vmask, k, i, j) > 0.01)
				{
					if (USESZERNIKE) {
						auto k2 = k * k;
						auto kr = k * iRmaxF;
						auto k2i2 = k2 + i * i;
						auto ir = i * iRmaxF;
						auto r2 = k2i2 + j * j;
						auto jr = j * iRmaxF;
						auto rr = sqrt(r2) * iRmaxF;
						for (size_t idx = 0; idx < idxY0; idx++)
						{
							auto l1 = VEC_ELEM(vL1, idx);
							auto n = VEC_ELEM(vN, idx);
							auto l2 = VEC_ELEM(vL2, idx);
							auto m = VEC_ELEM(vM, idx);
							if (rr > 0 || l2 == 0)
							{
								auto zsph = ZernikeSphericalHarmonics(l1, n, l2, m, jr, ir, kr, rr);
								gx += clnm[idx] * (zsph);
								gy += clnm[idx + idxY0] * (zsph);
								gz += clnm[idx + idxZ0] * (zsph);
							}
						}
					}

					auto r_x = j + gx;
					auto r_y = i + gy;
					auto r_z = k + gz;

					if (DIRECTION == Direction::Forward)
					{
						auto pos = std::array<double, 3>{};
						pos[0] = R.mdata[0] * r_x + R.mdata[1] * r_y + R.mdata[2] * r_z;
						pos[1] = R.mdata[3] * r_x + R.mdata[4] * r_y + R.mdata[5] * r_z;
						pos[2] = R.mdata[6] * r_x + R.mdata[7] * r_y + R.mdata[8] * r_z;
						double voxel_mV = A3D_ELEM(mV, k, i, j);
						splattingAtPos(pos, voxel_mV, mP, mW, mV);
					}
					else if (DIRECTION == Direction::Backward)
					{
						// auto pos = std::array<double, 3>{};
						auto pos_x = R.mdata[0] * r_x + R.mdata[1] * r_y + R.mdata[2] * r_z;
						auto pos_y = R.mdata[3] * r_x + R.mdata[4] * r_y + R.mdata[5] * r_z;
						auto pos_z = R.mdata[6] * r_x + R.mdata[7] * r_y + R.mdata[8] * r_z;
						// pos[0] = j;
						// pos[1] = i;
						// pos[2] = k;
						double voxel = mId.interpolatedElement3D(pos_x, pos_y, pos_z);
						A3D_ELEM(mV, k, i, j) += voxel;
						// updateVoxel(pos, voxel, mV);
					}
				}
			}
		}
	}
}

void ProgForwardArtZernike3DSubtomos::applyMissingWedge(MultidimArray<double> &mV)
{
	// gaussian_mw = gaussian3d;
	// gaussian_mw.setXmippOrigin();
	MultidimArray<std::complex<double>> fg, Fourier;
	MultidimArray<int> maskWedge;
	Matrix2D<double> R;
    R.initIdentity(3);
	// Euler_angles2matrix(static_cast<double>(rot), static_cast<double>(tilt), static_cast<double>(psi), R, false);
    transformer.FourierTransform(mV, fg, false);
	transformer.getFourierAlias(Fourier);
	maskWedge.initZeros(Fourier);
	maskWedge.setXmippOrigin();
	BinaryWedgeMask(maskWedge, t1, t2, R, true);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fg)
	DIRECT_MULTIDIM_ELEM(fg,n) *= DIRECT_MULTIDIM_ELEM(maskWedge,n);
	transformer.inverseFourierTransform();
}