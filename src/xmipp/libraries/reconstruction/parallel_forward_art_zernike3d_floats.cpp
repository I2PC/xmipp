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

#include "parallel_forward_art_zernike3d_floats.h"
#include <core/transformations.h>
#include <core/xmipp_image_extension.h>
#include <core/xmipp_image_generic.h>
#include <data/projection.h>
#include <data/mask.h>
#include <numeric>
#include "data/cpu.h"
#include <fstream>
#include <iterator>

// Empty constructor =======================================================
ProgParallelForwardArtZernike3DFloats::ProgParallelForwardArtZernike3DFloats()
{
	resume = false;
	produces_a_metadata = true;
	each_image_produces_an_output = false;
	showOptimization = false;
}

ProgParallelForwardArtZernike3DFloats::~ProgParallelForwardArtZernike3DFloats() = default;

// Read arguments ==========================================================
void ProgParallelForwardArtZernike3DFloats::readParams()
{
	XmippMetadataProgram::readParams();
	fnVolR = getParam("--ref");
	fnMaskR = getParam("--mask");
	fnMaskRecR = getParam("--recmask");
	fnOutDir = getParam("--odir");
	RmaxDef = getIntParam("--RDef");
	phaseFlipped = checkParam("--phaseFlipped");
	useCTF = checkParam("--useCTF");
	Ts = getDoubleParam("--sampling");
	L1 = getIntParam("--l1");
	L2 = getIntParam("--l2");
	loop_step = getIntParam("--step");
	useZernike = checkParam("--useZernike");
	lambda = getDoubleParam("--regularization");
	resume = checkParam("--resume");
	niter = getIntParam("--niter");
	save_iter = getIntParam("--save_iter");
	sort_last_N = getIntParam("--sort_last");
	FileName outPath = getParam("-o");
	outPath = outPath.afterLastOf("/");
	fnVolO = fnOutDir + "/" + outPath;

	std::string aux;
	aux = getParam("--sigma");
	// Transform string of values separated by white spaces into substrings stored in a vector
	std::stringstream ss(aux);
	std::istream_iterator<std::string> begin(ss);
	std::istream_iterator<std::string> end;
	std::vector<std::string> vstrings(begin, end);
	sigma.resize(vstrings.size());
	std::transform(vstrings.begin(), vstrings.end(), sigma.begin(), [](const std::string& val)
	{
    	return std::stod(val);
	});

	// Parallelization
	int threads = getIntParam("--thr");
	if (0 >= threads)
	{
		threads = CPU::findCores();
	}
	m_threadPool.resize(threads);
}

// Show ====================================================================
void ProgParallelForwardArtZernike3DFloats::show()
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
		<< "Step:                      " << loop_step << std::endl
		<< "Correct CTF:               " << useCTF << std::endl
		<< "Correct heretogeneity:     " << useZernike << std::endl
		<< "Phase flipped:             " << phaseFlipped << std::endl
		<< "Regularization:            " << lambda << std::endl
		<< "Number of iterations:      " << niter << std::endl
		<< "Save every # iterations:   " << save_iter << std::endl;
}

// usage ===================================================================
void ProgParallelForwardArtZernike3DFloats::defineParams()
{
	addUsageLine("Template-based canonical volume reconstruction through Zernike3D coefficients");
	defaultComments["-i"].clear();
	defaultComments["-i"].addComment("Metadata with initial alignment");
	defaultComments["-o"].clear();
	defaultComments["-o"].addComment("Refined volume");
	XmippMetadataProgram::defineParams();
	addParamsLine("  [--ref <volume=\"\">]        : Reference volume");
	addParamsLine("  [--mask <m=\"\">]            : Mask reference volume");
	addParamsLine("  [--recmask <m=\"\">]         : Mask determining reconstruction area");
	addParamsLine("  [--odir <outputDir=\".\">]   : Output directory");
	addParamsLine("  [--sampling <Ts=1>]          : Sampling rate (A/pixel)");
	addParamsLine("  [--RDef <r=-1>]              : Maximum radius of the deformation (px). -1=Half of volume size");
	addParamsLine("  [--l1 <l1=3>]                : Degree Zernike Polynomials=1,2,3,...");
	addParamsLine("  [--l2 <l2=2>]                : Harmonical depth of the deformation=1,2,3,...");
	addParamsLine("  [--blobr <b=4>]              : Blob radius for forward mapping splatting");
	addParamsLine("  [--step <step=1>]            : Voxel index step");
	addParamsLine("  [--sigma <Matrix1D=\"2\">]   : Gaussian sigma");
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
	addParamsLine("  [--resume]                   : Resume processing");
	addParamsLine("  [--thr <N=-1>]                      : Maximal number of the processing CPU threads");
	addExampleLine("A typical use is:", false);
	addExampleLine("xmipp_forward_art_zernike3d -i anglesFromContinuousAssignment.xmd --ref reference.vol -o assigned_anglesAndDeformations.xmd --l1 3 --l2 2");
}

void ProgParallelForwardArtZernike3DFloats::preProcess()
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

	p_busy_elem.resize(Xdim*Xdim);
    for (auto& p : p_busy_elem) {
        p = std::make_unique<std::atomic<PrecisionType*>>(nullptr);
    }

	w_busy_elem.resize(Xdim*Xdim);
    for (auto& p : w_busy_elem) {
        p = std::make_unique<std::atomic<PrecisionType*>>(nullptr);
    }

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
	Vrefined().setXmippOrigin();

	if (RmaxDef < 0)
		RmaxDef = Xdim / 2;

	// Transformation matrix
	A.initIdentity(3);

	// CTF Filter
	FilterCTF.FilterBand = CTFINV;
	FilterCTF.FilterShape = RAISED_COSINE;
	FilterCTF.ctf.enable_CTFnoise = false;
	FilterCTF.ctf.enable_CTF = true;

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


	// Mask determining reconstruction area
	if (fnMaskRecR != "")
	{
		Image<double> aux;
		aux.read(fnMaskRecR);
		typeCast(aux(), VRecMask);
		VRecMask.setXmippOrigin();
		double Rmax2 = RmaxDef * RmaxDef;
		for (int k = STARTINGZ(VRecMask); k <= FINISHINGZ(VRecMask); k++)
		{
			for (int i = STARTINGY(VRecMask); i <= FINISHINGY(VRecMask); i++)
			{
				for (int j = STARTINGX(VRecMask); j <= FINISHINGX(VRecMask); j++)
				{
					double r2 = k * k + i * i + j * j;
					if (r2 >= Rmax2)
						A3D_ELEM(VRecMask, k, i, j) = 0;
				}
			}
		}
	}
	else
	{
		mask.R1 = RmaxDef;
		mask.generate_mask(V());
		VRecMask = mask.get_binary_mask();
		VRecMask.setXmippOrigin();
	}

	// Spherical mask
	mask.R1 = RmaxDef;
	mask.generate_mask(V());
	sphMask = mask.get_binary_mask();
	sphMask.setXmippOrigin();

	// Init P and W vector of images
	P.resize(sigma.size());
	W.resize(sigma.size());

	// Area Zernike3D in 2D
	mask.R1 = RmaxDef;
	mask.generate_mask(XSIZE(V()), XSIZE(V()));
	mask2D = mask.get_binary_mask();
	mask2D.setXmippOrigin();

	vecSize = 0;
	numCoefficients(L1, L2, vecSize);
	fillVectorTerms(L1, L2, vL1, vN, vL2, vM);

	initX = STARTINGX(Vrefined());
	endX = FINISHINGX(Vrefined());
	initY = STARTINGY(Vrefined());
	endY = FINISHINGY(Vrefined());
	initZ = STARTINGZ(Vrefined());
	endZ = FINISHINGZ(Vrefined());

	filter.FilterBand=LOWPASS;
	filter.FilterShape=REALGAUSSIANZ;
	filter2.FilterBand=LOWPASS;
	filter2.FilterShape=REALGAUSSIANZ2;
}

void ProgParallelForwardArtZernike3DFloats::finishProcessing()
{
	recoverVol();
	Vout.write(fnVolO);
}

// Predict =================================================================
void ProgParallelForwardArtZernike3DFloats::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
	flagEnabled = 1;

	int img_enabled;
	rowIn.getValue(MDL_ENABLED, img_enabled);
	if (img_enabled == -1) return;

	rowIn.getValue(MDL_ANGLE_ROT, rot);
	rowIn.getValue(MDL_ANGLE_TILT, tilt);
	rowIn.getValue(MDL_ANGLE_PSI, psi);
	rowIn.getValueOrDefault(MDL_SHIFT_X, shiftX, 0.0);
	rowIn.getValueOrDefault(MDL_SHIFT_Y, shiftY, 0.0);
	std::vector<PrecisionType> vectortemp;
	if (useZernike)
	{
		rowIn.getValue(MDL_SPH_COEFFICIENTS, vectortemp);
		std::vector<PrecisionType> vec(vectortemp.begin(), vectortemp.end());
		clnm = vec;
	}
	rowIn.getValueOrDefault(MDL_FLIP, flip, false);

	if ((rowIn.containsLabel(MDL_CTF_DEFOCUSU) || rowIn.containsLabel(MDL_CTF_MODEL)) && useCTF)
	{
		hasCTF = true;
		FilterCTF.ctf.readFromMdRow(rowIn, false);
		FilterCTF.ctf.Tm = Ts;
		FilterCTF.ctf.produceSideInfo();
	}
	else
		hasCTF = false;
	MAT_ELEM(A, 0, 2) = shiftX;
	MAT_ELEM(A, 1, 2) = shiftY;
	MAT_ELEM(A, 0, 0) = 1;
	MAT_ELEM(A, 0, 1) = 0;
	MAT_ELEM(A, 1, 0) = 0;
	MAT_ELEM(A, 1, 1) = 1;

	if (verbose >= 2)
		std::cout << "Processing " << fnImg << std::endl;

	I.read(fnImg);
	I().setXmippOrigin();

	// Forward Model
	artModel<Direction::Forward>();

	// ART update
	artModel<Direction::Backward>();
}

void ProgParallelForwardArtZernike3DFloats::numCoefficients(int l1, int l2, int &vecSize)
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

void ProgParallelForwardArtZernike3DFloats::fillVectorTerms(int l1, int l2, Matrix1D<int> &vL1, Matrix1D<int> &vN,
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

void ProgParallelForwardArtZernike3DFloats::splattingAtPos(std::array<PrecisionType, 2> r, PrecisionType weight,
											 MultidimArray<PrecisionType> &mP, MultidimArray<PrecisionType> &mW)
{
	int i = round(r[1]);
	int j = round(r[0]);
	if (!mP.outside(i, j))
	{
		int idy = (i)-STARTINGY(mP);
		int idx = (j)-STARTINGX(mP);
		int idn = (idy) * (mP).xdim + (idx);
		while ((*p_busy_elem[idn]) == &A2D_ELEM(mP, i, j));
		(*p_busy_elem[idn]).exchange(&A2D_ELEM(mP, i, j));
		(*w_busy_elem[idn]).exchange(&A2D_ELEM(mW, i, j));
		A2D_ELEM(mP, i, j) += weight;
		A2D_ELEM(mW, i, j) += 1.0;
		(*p_busy_elem[idn]).exchange(nullptr);
		(*w_busy_elem[idn]).exchange(nullptr);
	}
}

void ProgParallelForwardArtZernike3DFloats::recoverVol()
{
	// Find the part of the volume that must be updated
	auto &mVout = Vout();
	const auto &mV = Vrefined();
	mVout.initZeros(mV);

	mVout = mV;
}

void ProgParallelForwardArtZernike3DFloats::run()
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
	current_image = 1;
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
			rowIn->getValue(MDL_ITEM_ID, num_images);
			if (verbose > 2)
				std::cout << "Current image ID:  " << num_images << std::endl;

			if (fnImg.empty())
				break;

			fnImgOut = fnImg;

			MDRowVec rowOut;

			processImage(fnImg, fnImgOut, *rowIn.get(), rowOut);

			checkPoint();
			showProgress();

			// Save refined volume every num_images
			if (current_save_iter == save_iter && save_iter > 0)
			{
				recoverVol();
				Vout.write(fnVolO.removeAllExtensions() + "_partial.mrc");
				current_save_iter = 1;
			}
			current_save_iter++;
			current_image++;
		}
		current_image = 1;
		current_save_iter = 1;

		recoverVol();
		Vout.write(fnVolO.removeAllExtensions() + "_iter" + std::to_string(current_iter + 1) + ".mrc");
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

void ProgParallelForwardArtZernike3DFloats::sortOrthogonal()
{
	int i, j;
	size_t numIMG = getInputMd()->size();
	MultidimArray<short> chosen(numIMG);
	MultidimArray<double> product(numIMG);
	double min_prod = MAXFLOAT;
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
		min_prod = MAXFLOAT;
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

MultidimArray<PrecisionType> ProgParallelForwardArtZernike3DFloats::useFilterPrecision(FourierFilter &filter, MultidimArray<PrecisionType> precisionImage)
{
	MultidimArray<double> doubleImage;
	MultidimArray<PrecisionType> outputImage;
	typeCast(precisionImage, doubleImage);
	filter.generateMask(doubleImage);
	filter.applyMaskSpace(doubleImage);
	typeCast(doubleImage, outputImage);
	return outputImage;
}

template <ProgParallelForwardArtZernike3DFloats::Direction DIRECTION>
void ProgParallelForwardArtZernike3DFloats::artModel()
{
	if (DIRECTION == Direction::Forward)
	{
		Image<double> I_shifted;
		for (int i=0; i<sigma.size(); i++)
		{
			P[i]().initZeros((int)XSIZE(I()), (int)XSIZE(I()));
			P[i]().setXmippOrigin();
			W[i]().initZeros((int)XSIZE(I()), (int)XSIZE(I()));
			W[i]().setXmippOrigin();
		}
		Idiff().initZeros((int)XSIZE(I()), (int)XSIZE(I()));
		Idiff().setXmippOrigin();
		I_shifted().initZeros((int)XSIZE(I()), (int)XSIZE(I()));
		I_shifted().setXmippOrigin();

		if (useZernike)
			zernikeModel<true, Direction::Forward>();
		else
			zernikeModel<false, Direction::Forward>();

		for (int i=0; i<sigma.size(); i++)
		{
			filter.w1=sigma[i];
			filter2.w1=sigma[i];
			P[i] = useFilterPrecision(filter, P[i]());
			W[i] = useFilterPrecision(filter2, W[i]());
		}

		if (hasCTF)
		{
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

		applyGeometry(xmipp_transformation::LINEAR, I_shifted(), I(), A,
					  xmipp_transformation::IS_NOT_INV, xmipp_transformation::DONT_WRAP, 0.);

		// Compute difference image and divide by weights
		double error = 0.0;
		double N = 0.0;
		const auto &mIsh = I_shifted();
		auto &mId = Idiff();
		double c = 1.0;
		FOR_ALL_ELEMENTS_IN_ARRAY2D(I())
		{
			auto diffVal = A2D_ELEM(mIsh, i, j);
			double sumMw = 0.0;
			for (int ids = 0; ids < sigma.size(); ids++)
			{
				const auto &mP = P[ids]();
				const auto &mW = W[ids]();
				const auto sg = sigma[ids];
				if (sigma.size() > 1)
					c = sg * sg;
				diffVal -= c * A2D_ELEM(mP, i, j);
				sumMw += c * c * A2D_ELEM(mW, i, j);
			}
			if (sumMw > 0.0)
			{
				A2D_ELEM(mId, i, j) = lambda * (diffVal) / XMIPP_MAX(sumMw, 1.0);
				error += (diffVal) * (diffVal);
				N++;
			}
		}

		if (verbose >= 3)
		{
			for (int ids = 0; ids < sigma.size(); ids++)
			{
				P[ids].write(fnOutDir + "/PPPtheo_sigma" + std::to_string(sigma[ids]) + ".xmp");
				W[ids].write(fnOutDir + "/PPPweight_sigma" + std::to_string(sigma[ids]) + ".xmp");
			}
			Idiff.write(fnOutDir + "/PPPcorr.xmp");
			std::cout << "Press any key" << std::endl;
			char c;
			std::cin >> c;
		}

		/* Creo que Carlos no usa un RMSE si no un MSE
		 * Translates to: I think Carlos does not use a RMSE but a MSE. */
		error = std::sqrt(error / N);
		if (verbose >= 2)
			std::cout << "Error for image " << num_images << " (" << current_image << ") in iteration " << current_iter + 1 << " : " << error << std::endl;
	}
	else if (DIRECTION == Direction::Backward)
	{
		if (useZernike)
			zernikeModel<true, Direction::Backward>();
		else
			zernikeModel<false, Direction::Backward>();
	}
}

template <bool USESZERNIKE, ProgParallelForwardArtZernike3DFloats::Direction DIRECTION>
void ProgParallelForwardArtZernike3DFloats::zernikeModel()
{
	auto &mV = Vrefined();

	const auto lastZ = FINISHINGZ(mV);
	const int step = DIRECTION == Direction::Forward ? loop_step : 1;

	// Parallelization
	auto futures = std::vector<std::future<void>>();
	futures.reserve(mV.zdim);
	auto routine_forward = [this](int thrId, int k) {
		forwardModel(k, USESZERNIKE);
    };

	auto routine_backward = [this](int thrId, int k) {
		backwardModel(k, USESZERNIKE);
    };

	for (int k = STARTINGZ(mV); k <= lastZ; k += step)
	{
		if (DIRECTION == Direction::Forward)
			futures.emplace_back(m_threadPool.push(routine_forward, k));
		else if (DIRECTION == Direction::Backward)
			futures.emplace_back(m_threadPool.push(routine_backward, k));
	}

	for (auto &f : futures)
	{
        f.get();
    }
}

void ProgParallelForwardArtZernike3DFloats::forwardModel(int k, bool usesZernike)
{
	auto &mV = Vrefined();
	const size_t idxY0 = usesZernike ? (clnm.size() / 3) : 0;
	const size_t idxZ0 = usesZernike ? (2 * idxY0) : 0;
	const PrecisionType RmaxF = usesZernike ? RmaxDef : 0;
	const PrecisionType iRmaxF = usesZernike ? (1.0f / RmaxF) : 0;
	// Rotation Matrix
	constexpr size_t matrixSize = 3;
	const Matrix2D<PrecisionType> R = [this]()
	{
		auto tmp = Matrix2D<PrecisionType>();
		tmp.initIdentity(matrixSize);
		Euler_angles2matrix(rot, tilt, psi, tmp, false);
		return tmp;
	}();

	const auto lastY = FINISHINGY(mV);
	const auto lastX = FINISHINGX(mV);
	const int step = loop_step;
	for (int i = STARTINGY(mV); i <= lastY; i += step)
	{
		for (int j = STARTINGX(mV); j <= lastX; j += step)
		{
			PrecisionType gx = 0.0, gy = 0.0, gz = 0.0;
			if (A3D_ELEM(VRecMask, k, i, j) != 0)
			{
				int img_idx = 0;
				if (sigma.size() > 1)
				{
					PrecisionType sigma_mask = A3D_ELEM(VRecMask, k, i, j);
					auto it = find(sigma.begin(), sigma.end(), sigma_mask);
					img_idx = it - sigma.begin();
				}
				auto &mP = P[img_idx]();
				auto &mW = W[img_idx]();
				if (usesZernike)
				{
					auto k2 = k * k;
					auto kr = k * iRmaxF;
					auto k2i2 = k2 + i * i;
					auto ir = i * iRmaxF;
					auto r2 = k2i2 + j * j;
					auto jr = j * iRmaxF;
					auto rr = SQRT(r2) * iRmaxF;
					for (size_t idx = 0; idx < idxY0; idx++)
					{
						auto l1 = VEC_ELEM(vL1, idx);
						auto n = VEC_ELEM(vN, idx);
						auto l2 = VEC_ELEM(vL2, idx);
						auto m = VEC_ELEM(vM, idx);
						if (rr > 0 || l2 == 0)
						{
							PrecisionType zsph = ZernikeSphericalHarmonics(l1, n, l2, m, jr, ir, kr, rr);
							gx += clnm[idx] * (zsph);
							gy += clnm[idx + idxY0] * (zsph);
							gz += clnm[idx + idxZ0] * (zsph);
						}
					}
				}

				auto r_x = j + gx;
				auto r_y = i + gy;
				auto r_z = k + gz;

				auto pos = std::array<PrecisionType, 2>{};
				pos[0] = R.mdata[0] * r_x + R.mdata[1] * r_y + R.mdata[2] * r_z;
				pos[1] = R.mdata[3] * r_x + R.mdata[4] * r_y + R.mdata[5] * r_z;
				PrecisionType voxel_mV = A3D_ELEM(mV, k, i, j);
				splattingAtPos(pos, voxel_mV, mP, mW);
			}
		}
	}
}

void ProgParallelForwardArtZernike3DFloats::backwardModel(int k, bool usesZernike)
{
	auto &mId = Idiff();
	auto &mV = Vrefined();
	const size_t idxY0 = usesZernike ? (clnm.size() / 3) : 0;
	const size_t idxZ0 = usesZernike ? (2 * idxY0) : 0;
	const PrecisionType RmaxF = usesZernike ? RmaxDef : 0;
	const PrecisionType iRmaxF = usesZernike ? (1.0f / RmaxF) : 0;
	// Rotation Matrix
	constexpr size_t matrixSize = 3;
	const Matrix2D<PrecisionType> R = [this]()
	{
		auto tmp = Matrix2D<PrecisionType>();
		tmp.initIdentity(matrixSize);
		Euler_angles2matrix(rot, tilt, psi, tmp, false);
		return tmp;
	}();

	const auto lastY = FINISHINGY(mV);
	const auto lastX = FINISHINGX(mV);
	const int step = 1;
	for (int i = STARTINGY(mV); i <= lastY; i += step)
	{
		for (int j = STARTINGX(mV); j <= lastX; j += step)
		{
			PrecisionType gx = 0.0, gy = 0.0, gz = 0.0;
			if (A3D_ELEM(sphMask, k, i, j) != 0)
			{
				if (usesZernike)
				{
					auto k2 = k * k;
					auto kr = k * iRmaxF;
					auto k2i2 = k2 + i * i;
					auto ir = i * iRmaxF;
					auto r2 = k2i2 + j * j;
					auto jr = j * iRmaxF;
					auto rr = SQRT(r2) * iRmaxF;
					for (size_t idx = 0; idx < idxY0; idx++)
					{
						auto l1 = VEC_ELEM(vL1, idx);
						auto n = VEC_ELEM(vN, idx);
						auto l2 = VEC_ELEM(vL2, idx);
						auto m = VEC_ELEM(vM, idx);
						if (rr > 0 || l2 == 0)
						{
							PrecisionType zsph = ZernikeSphericalHarmonics(l1, n, l2, m, jr, ir, kr, rr);
							gx += clnm[idx] * (zsph);
							gy += clnm[idx + idxY0] * (zsph);
							gz += clnm[idx + idxZ0] * (zsph);
						}
					}
				}

				auto r_x = j + gx;
				auto r_y = i + gy;
				auto r_z = k + gz;

				auto pos_x = R.mdata[0] * r_x + R.mdata[1] * r_y + R.mdata[2] * r_z;
				auto pos_y = R.mdata[3] * r_x + R.mdata[4] * r_y + R.mdata[5] * r_z;
				PrecisionType voxel = mId.interpolatedElement2D(pos_x, pos_y);
				A3D_ELEM(mV, k, i, j) += voxel;
			}
		}
	}
}
