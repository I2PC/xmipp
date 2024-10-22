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

#include "forward_art_zernike3d_gpu.h"
#include <core/transformations.h>
#include <core/xmipp_image_extension.h>
#include <core/xmipp_image_generic.h>
#include <data/mask.h>
#include <data/projection.h>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include "data/cpu.h"

// Empty constructor =======================================================
ProgForwardArtZernike3DGPU::ProgForwardArtZernike3DGPU()
{
	produces_a_metadata = true;
	each_image_produces_an_output = false;
}

ProgForwardArtZernike3DGPU::~ProgForwardArtZernike3DGPU() = default;

// Read arguments ==========================================================
void ProgForwardArtZernike3DGPU::readParams()
{
	XmippMetadataProgram::readParams();
	fnVolR = getParam("--ref");
	fnMaskRF = getParam("--maskf");
	fnMaskRB = getParam("--maskb");
	fnOutDir = getParam("--odir");
	RmaxDef = getIntParam("--RDef");
	phaseFlipped = checkParam("--phaseFlipped");
	useCTF = checkParam("--useCTF");
	Ts = getDoubleParam("--sampling");
	L1 = getIntParam("--l1");
	L2 = getIntParam("--l2");
	ltv = getDoubleParam("--ltv");
	ltk = getDoubleParam("--ltk");
	ll1 = getDoubleParam("--ll1");
	lst = getDoubleParam("--lst");
	mr = getIntParam("--mr");
	dSize = getIntParam("--dSize");
	loop_step = getIntParam("--step");
	useZernike = checkParam("--useZernike");
	lambda = getDoubleParam("--regularization");
	resume = checkParam("--resume");
	removeNegValues = checkParam("--onlyPositive");
	niter = getIntParam("--niter");
	debug_iter = checkParam("--debug_iter");
	save_iter = getIntParam("--save_iter");
	sort_last_N = getIntParam("--sort_last");
	sort_random = checkParam("--sort_random");
	fnSym = getParam("--sym");
	FileName outPath = getParam("-o");
	outPath = outPath.afterLastOf("/");
	fnVolO = fnOutDir + "/" + outPath;

	std::string aux_sigma;
	aux_sigma = getParam("--sigma");
	// Transform string of values separated by white spaces into substrings stored in a vector
	std::stringstream ss(aux_sigma);
	std::istream_iterator<std::string> begins(ss);
	std::istream_iterator<std::string> ends;
	std::vector<std::string> vstrings(begins, ends);
	sigma.resize(vstrings.size());
	std::transform(
		vstrings.begin(), vstrings.end(), sigma.begin(), [](const std::string &val) { return std::stod(val); });
}

// Show ====================================================================
void ProgForwardArtZernike3DGPU::show() const
{
	if (!verbose)
		return;
	XmippMetadataProgram::show();
	std::cout << "Output directory:          " << fnOutDir << std::endl
			  << "Reference volume:          " << fnVolR << std::endl
			  << "Forward model mask:        " << fnMaskRF << std::endl
			  << "Backward model mask:       " << fnMaskRB << std::endl
			  << "Sampling:                  " << Ts << std::endl
			  << "Max. Radius Deform.        " << RmaxDef << std::endl
			  << "Zernike Degree:            " << L1 << std::endl
			  << "SH Degree:                 " << L2 << std::endl
			  << "Step:                      " << loop_step << std::endl
	          << "Symmetry group:            " << fnSym << std::endl
			  << "Correct CTF:               " << useCTF << std::endl
			  << "Correct heretogeneity:     " << useZernike << std::endl
			  << "Remove negative values:    " << removeNegValues << std::endl
			  << "Phase flipped:             " << phaseFlipped << std::endl
			  << "Regularization:            " << lambda << std::endl
			  << "Number of iterations:      " << niter << std::endl
			  << "Save every # iterations:   " << save_iter << std::endl;
}

// usage ===================================================================
void ProgForwardArtZernike3DGPU::defineParams()
{
	addUsageLine("Template-based canonical volume reconstruction through Zernike3D coefficients");
	defaultComments["-i"].clear();
	defaultComments["-i"].addComment("Metadata with initial alignment");
	defaultComments["-o"].clear();
	defaultComments["-o"].addComment("Refined volume");
	XmippMetadataProgram::defineParams();
	addParamsLine("  [--ref <volume=\"\">]        : Reference volume");
	addParamsLine("  [--maskf <m=\"\">]           : ART forward model reconstruction mask");
	addParamsLine("  [--maskb <m=\"\">]           : ART backward model reconstruction mask");
	addParamsLine("  [--odir <outputDir=\".\">]   : Output directory");
	addParamsLine("  [--sampling <Ts=1>]          : Sampling rate (A/pixel)");
	addParamsLine("  [--RDef <r=-1>]              : Maximum radius of the deformation (px). -1=Half of volume size");
	addParamsLine("  [--l1 <l1=3>]                : Degree Zernike Polynomials=1,2,3,...");
	addParamsLine("  [--l2 <l2=2>]                : Harmonical depth of the deformation=1,2,3,...");
	addParamsLine("  [--blobr <b=4>]              : Blob radius for forward mapping splatting");
	addParamsLine("  [--step <step=1>]            : Voxel index step");
	addParamsLine("  [--sigma <Matrix1D=\"2\">]   : Gaussian sigma");
	addParamsLine("  [--mr <mr=0>]                : Muliresolution levels");
	addParamsLine("  [--dSize <ds=0>]             : Muliresolution size");
	addParamsLine("  [--ltv <ltv=1e-4>]           : Total variation regualrization");
	addParamsLine("  [--ltk <ltv=1e-4>]           : Tikhonov regualrization");
	addParamsLine("  [--ll1 <ll1=1e-4>]           : L1 regualrization");
	addParamsLine("  [--lst <ll1=1e-4>]           : Soft threshold regualrization");
	addParamsLine("  [--sym <sym=c1>]             : Symmetry to be considered during the reconstruction");
	addParamsLine("  [--useZernike]               : Correct heterogeneity with Zernike3D coefficients");
	addParamsLine("  [--useCTF]                   : Correct CTF during ART reconstruction");
	addParamsLine("  [--phaseFlipped]             : Input images have been phase flipped");
	addParamsLine("  [--regularization <l=0.01>]  : ART regularization weight");
	addParamsLine("  [--niter <n=1>]              : Number of ART iterations");
	addParamsLine("  [--debug_iter]               : Save volume after each ART iteration");
	addParamsLine("  [--onlyPositive]             : Remove negative values from generated volumes");
	addParamsLine("  [--save_iter <s=0>]          : Save intermidiate volume after #save_iter iterations");
	addParamsLine(
		"  [--sort_last <N=2>]          : The algorithm sorts projections in the most orthogonally possible way. ");
	addParamsLine("  [--sort_random]              : Random sort of projections");
	addParamsLine(
		"                               : The most orthogonal way is defined as choosing the projection which "
		"maximizes the ");
	addParamsLine(
		"                               : dot product with the N previous inserted projections. Use -1 to sort with "
		"all  ");
	addParamsLine("                               : previous projections");
	addParamsLine("  [--resume]                   : Resume processing");
	addExampleLine("A typical use is:", false);
	addExampleLine(
		"xmipp_cuda_forward_art_zernike3d -i anglesFromContinuousAssignment.xmd --ref reference.vol -o "
		"assigned_anglesAndDeformations.xmd --l1 3 --l2 2");
}

void ProgForwardArtZernike3DGPU::preProcess()
{

	// Check that metadata has all information neede
	if (!getInputMd()->containsLabel(MDL_ANGLE_ROT) || !getInputMd()->containsLabel(MDL_ANGLE_TILT)
		|| !getInputMd()->containsLabel(MDL_ANGLE_PSI)) {
		REPORT_ERROR(ERR_MD_MISSINGLABEL, "Input metadata projection angles are missing. Exiting...");
	}

	if (fnVolR != "") {
		V.read(fnVolR);
	} else {
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

	VZero().initZeros(V());
	VZero().setXmippOrigin();

	if (resume && fnVolO.exists()) {
		Vrefined.read(fnVolO);
	} else {
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
	if (fnMaskRF != "") {
		Image<double> aux;
		aux.read(fnMaskRF);
		typeCast(aux(), VRecMaskF);
		VRecMaskF.setXmippOrigin();
		double Rmax2 = RmaxDef * RmaxDef;
		for (int k = STARTINGZ(VRecMaskF); k <= FINISHINGZ(VRecMaskF); k++) {
			for (int i = STARTINGY(VRecMaskF); i <= FINISHINGY(VRecMaskF); i++) {
				for (int j = STARTINGX(VRecMaskF); j <= FINISHINGX(VRecMaskF); j++) {
					double r2 = k * k + i * i + j * j;
					if (r2 >= Rmax2)
						A3D_ELEM(VRecMaskF, k, i, j) = 0;
				}
			}
		}
	} else {
		mask.R1 = RmaxDef;
		mask.generate_mask(V());
		VRecMaskF = mask.get_binary_mask();
		VRecMaskF.setXmippOrigin();
	}


	// Mask determining reconstruction area
	if (fnMaskRB != "") {
		Image<double> aux;
		aux.read(fnMaskRB);
		typeCast(aux(), VRecMaskB);
		VRecMaskB.setXmippOrigin();
		double Rmax2 = RmaxDef * RmaxDef;
		for (int k = STARTINGZ(VRecMaskB); k <= FINISHINGZ(VRecMaskB); k++) {
			for (int i = STARTINGY(VRecMaskB); i <= FINISHINGY(VRecMaskB); i++) {
				for (int j = STARTINGX(VRecMaskB); j <= FINISHINGX(VRecMaskB); j++) {
					double r2 = k * k + i * i + j * j;
					if (r2 >= Rmax2)
						A3D_ELEM(VRecMaskB, k, i, j) = 0;
				}
			}
		}
	} else {
		mask.R1 = RmaxDef;
		mask.generate_mask(V());
		VRecMaskB = mask.get_binary_mask();
		VRecMaskB.setXmippOrigin();
	}

	// Init P and W vector of images
	P.resize(sigma.size());
	W.resize(sigma.size());

	// Area Zernike3D in 2D
	mask.R1 = RmaxDef;
	mask.generate_mask(XSIZE(V()), XSIZE(V()));
	mask2D = mask.get_binary_mask();
	mask2D.setXmippOrigin();

	vecSize = 0;
	numCoefficients(L1, L2);
	fillVectorTerms(L1, L2);

	initX = STARTINGX(Vrefined());
	endX = FINISHINGX(Vrefined());
	initY = STARTINGY(Vrefined());
	endY = FINISHINGY(Vrefined());
	initZ = STARTINGZ(Vrefined());
	endZ = FINISHINGZ(Vrefined());

	filter.FilterBand = LOWPASS;
	filter.FilterShape = REALGAUSSIANZ;
	filter2.FilterBand = LOWPASS;
	filter2.FilterShape = REALGAUSSIANZ2;

	// Multiresolution filter
	MultidimArray<double> aux_img;
	aux_img.initZeros(Xdim, Xdim);
	aux_img.setXmippOrigin();
	filterMR.FilterBand = LOWPASS;
	filterMR.FilterShape = REALGAUSSIAN;
	filterMR.w1 = 1.0;
	filterMR.do_generate_3dmask = false;
	filterMR.generateMask(aux_img);

	if (mr != 0)
	{	
		lmr = 1.0 / static_cast<double>(mr);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(filterMR.maskFourierd)
		{
			DIRECT_MULTIDIM_ELEM(filterMR.maskFourierd,n) = 0.0;
		}

		for (int i = 1; i <= mr; i++)
		{
			FourierFilter aux_filter;
			aux_filter.FilterBand = LOWPASS;
			aux_filter.FilterShape = REALGAUSSIAN;
			aux_filter.w1 = static_cast<double>(i);
			aux_filter.do_generate_3dmask = false;
			aux_filter.generateMask(aux_img);

			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(filterMR.maskFourierd)
			{
				DIRECT_MULTIDIM_ELEM(filterMR.maskFourierd,n) += DIRECT_MULTIDIM_ELEM(aux_filter.maskFourierd,n);
			}

		}	
	}	
	else
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(filterMR.maskFourierd)
		{
			DIRECT_MULTIDIM_ELEM(filterMR.maskFourierd,n) = 1.0;
		}
		lmr = 0.0;
	}		 


	// Symmetry list
	Matrix2D<double> L(3, 3), R(3, 3);
	L.initIdentity(3);
	R.initIdentity(3);
	SymList SL_aux;
	SL_aux.readSymmetryFile(fnSym);
	LV.push_back(L);
	RV.push_back(R);
	for (int isym = 0; isym < SL_aux.symsNo(); isym++) {
		SL_aux.getMatrices(isym, L, R, false);
		LV.push_back(L);
		RV.push_back(R);
	}

	// Create GPU interface
	const cuda_forward_art_zernike3D::Program<PrecisionType>::ConstantParameters parameters = {
		.VRecMaskF = VRecMaskF,
		.VRecMaskB = VRecMaskB,
		.Vrefined = Vrefined,
		.VZero = VZero,
		.vL1 = vL1,
		.vN = vN,
		.vL2 = vL2,
		.vM = vM,
		.sigma = sigma,
		.RmaxDef = RmaxDef,
		.loopStep = loop_step,
		.filterMR = filterMR.maskFourierd,
	};
	try {
		cudaProgram = std::make_unique<cuda_forward_art_zernike3D::Program<PrecisionType>>(parameters);
	} catch (const std::runtime_error &e) {
		std::cerr << e.what() << '\n';
		std::exit(EXIT_FAILURE);
	}
}

void ProgForwardArtZernike3DGPU::finishProcessing()
{
	recoverVol();
	Vout.write(fnVolO);
}

// Predict =================================================================
void ProgForwardArtZernike3DGPU::processImage(const FileName &fnImg,
											  const FileName &fnImgOut,
											  const MDRow &rowIn,
											  MDRow &rowOut)
{
	flagEnabled = 1;

	int img_enabled;
	rowIn.getValue(MDL_ENABLED, img_enabled);
	if (img_enabled == -1)
		return;

	rowIn.getValue(MDL_ANGLE_ROT, rot);
	rowIn.getValue(MDL_ANGLE_TILT, tilt);
	rowIn.getValue(MDL_ANGLE_PSI, psi);
	rowIn.getValueOrDefault(MDL_SHIFT_X, shiftX, 0.0);
	rowIn.getValueOrDefault(MDL_SHIFT_Y, shiftY, 0.0);
	if (useZernike) {
		std::vector<double> vectortemp;
		rowIn.getValue(MDL_SPH_COEFFICIENTS, vectortemp);
		std::vector<PrecisionType> vec(vectortemp.begin(), vectortemp.end());
		clnm = vec;
	}
	else {
		std::vector<PrecisionType> vec(39, 0.0);
		clnm = vec;
	} 
	rowIn.getValueOrDefault(MDL_FLIP, flip, false);

	if ((rowIn.containsLabel(MDL_CTF_DEFOCUSU) || rowIn.containsLabel(MDL_CTF_MODEL)) && useCTF) {
		hasCTF = true;
		FilterCTF.ctf.readFromMdRow(rowIn, false);
		FilterCTF.ctf.Tm = Ts;
		FilterCTF.ctf.produceSideInfo();
	} else
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

	auto rot_dbl = static_cast<double>(rot);
	auto tilt_dbl = static_cast<double>(tilt);
	auto psi_dbl = static_cast<double>(psi);
	double new_rot, new_tilt, new_psi;

	for (int isym = 0; isym < RV.size(); isym++)
	{

		Euler_apply_transf(LV[isym], RV[isym], rot_dbl, tilt_dbl, psi_dbl, new_rot, new_tilt, new_psi);

		rot = static_cast<PrecisionType>(new_rot);
		tilt = static_cast<PrecisionType>(new_tilt);
		psi = static_cast<PrecisionType>(new_psi);

		// Forward Model
		artModel<Direction::Forward>();

		// ART update
		artModel<Direction::Backward>();
	}
}

void ProgForwardArtZernike3DGPU::numCoefficients(int l1, int l2)
{
	for (int h = 0; h <= l2; h++) {
		int numSPH = 2 * h + 1;
		int count = l1 - h + 1;
		int numEven = (count >> 1) + (count & 1 && !(h & 1));
		if (h % 2 == 0) {
			vecSize += numSPH * numEven;
		} else {
			vecSize += numSPH * (l1 - h + 1 - numEven);
		}
	}
}

void ProgForwardArtZernike3DGPU::fillVectorTerms(int l1,
												 int l2)
{
	int idx = 0;
	vL1.initZeros(vecSize);
	vN.initZeros(vecSize);
	vL2.initZeros(vecSize);
	vM.initZeros(vecSize);
	for (int h = 0; h <= l2; h++) {
		int totalSPH = 2 * h + 1;
		int aux = std::floor(totalSPH / 2);
		for (int l = h; l <= l1; l += 2) {
			for (int m = 0; m < totalSPH; m++) {
				VEC_ELEM(vL1, idx) = l;
				VEC_ELEM(vN, idx) = h;
				VEC_ELEM(vL2, idx) = h;
				VEC_ELEM(vM, idx) = m - aux;
				idx++;
			}
		}
	}
}

void ProgForwardArtZernike3DGPU::recoverVol()
{
	cudaProgram->recoverVolumeFromGPU(Vrefined);
	// Find the part of the volume that must be updated
	auto &mVout = Vout();
	const auto &mV = Vrefined();
	mVout.initZeros(mV);

	mVout = mV;

	// Remove negative values
	if (removeNegValues)
	{
		for (int k=STARTINGZ(mVout); k<=FINISHINGZ(mVout); k++)
			for (int i=STARTINGY(mVout); i<=FINISHINGY(mVout); i++)
				for (int j=STARTINGX(mVout); j<=FINISHINGX(mVout); j++)
				{
					if (A3D_ELEM(mVout, k, i, j) < 0.0)
						A3D_ELEM(mVout, k, i, j) = 0.0;
				}
	}

}

void ProgForwardArtZernike3DGPU::run()
{
	FileName fnImg, fnImgOut, fullBaseName;
	getOutputMd().clear();	//this allows multiple runs of the same Program object

	//Perform particular preprocessing
	preProcess();

	startProcessing();

	if (sort_random)
	{
		std::vector<size_t> imgs_id(getInputMd()->size());
		std::iota(imgs_id.begin(), imgs_id.end(), 0);
		auto rng = std::default_random_engine {};
		std::shuffle(std::begin(imgs_id), std::end(imgs_id), rng);
		ordered_list.initZeros(getInputMd()->size());
		for (size_t i=0; i < XSIZE(ordered_list); i++)
			A1D_ELEM(ordered_list, i) = imgs_id[i];
	}
	else
		sortOrthogonal();

	if (!oroot.empty()) {
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
	for (current_iter = 0; current_iter < niter; current_iter++) {
		std::cout << "Running iteration " << current_iter + 1 << " with lambda=" << lambda << std::endl;
		objId = 0;
		objIndex = 0;
		time_bar_done = 0;
		FOR_ALL_ELEMENTS_IN_ARRAY1D(ordered_list)
		{
			objId = A1D_ELEM(ordered_list, i) + 1;
			++objIndex;
			auto rowIn = getInputMd()->getRow(objId);
			if (rowIn == nullptr)
				continue;
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
			if (current_save_iter == save_iter && save_iter > 0) {
				recoverVol();
				Vout.write(fnVolO.removeAllExtensions() + "_partial.mrc");
				current_save_iter = 1;
			}
			current_save_iter++;
			current_image++;
		}
		current_image = 1;
		current_save_iter = 1;

		if (debug_iter) {
			recoverVol();
			Vout.write(fnVolO.removeAllExtensions() + "_iter" + std::to_string(current_iter + 1) + ".mrc");
		}
	}
	wait();

	/* Generate name to save mdOut when output are independent images. It uses as prefix
     * the dirBaseName in order not overwriting files when repeating same command on
     * different directories. If baseName is set it is used, otherwise, input name is used.
     * Then, the suffix _oext is added.*/
	if (fn_out.empty()) {
		if (!oroot.empty()) {
			if (!baseName.empty())
				fn_out = findAndReplace(pathBaseName, "/", "_") + baseName + "_" + oextBaseName + ".xmd";
			else
				fn_out = findAndReplace(pathBaseName, "/", "_") + fn_in.getBaseName() + "_" + oextBaseName + ".xmd";
		} else if (input_is_metadata)  /// When nor -o neither --oroot is passed and want to overwrite input metadata
			fn_out = fn_in;
	}

	finishProcessing();

	postProcess();

	/* Reset the default values of the program in case
     * to be reused.*/
	init();
}

void ProgForwardArtZernike3DGPU::sortOrthogonal()
{
	int i, j;
	size_t numIMG = getInputMd()->size();
	MultidimArray<short> chosen(numIMG);
	MultidimArray<double> product(numIMG);
	double min_prod = MAXFLOAT;
	;
	int min_prod_proj = 0;
	std::vector<double> rotV;
	std::vector<double> tiltV;
	Matrix2D<double> v(numIMG, 3);
	Matrix2D<double> euler;
	getInputMd()->getColumnValues(MDL_ANGLE_ROT, rotV);
	getInputMd()->getColumnValues(MDL_ANGLE_TILT, tiltV);

	// Initialization
	ordered_list.resize(numIMG);
	for (i = 0; i < numIMG; i++) {
		Matrix1D<double> z;
		// Initially no image is chosen
		A1D_ELEM(chosen, i) = 0;

		// Compute the Euler matrix for each image and keep only
		// the third row of each one
		Euler_angles2matrix(rotV[i], tiltV[i], 0., euler);
		euler.getRow(2, z);
		v.setRow(i, z);
	}

	// Pick first projection as the first one to be presented
	i = 0;
	A1D_ELEM(chosen, i) = 1;
	A1D_ELEM(ordered_list, 0) = i;

	// Choose the rest of projections
	std::cout << "Sorting projections orthogonally...\n" << std::endl;
	Matrix1D<double> rowj, rowi_1, rowi_N_1;
	for (i = 1; i < numIMG; i++) {
		// Compute the product of not already chosen vectors with the just
		// chosen one, and select that which has minimum product
		min_prod = MAXFLOAT;
		v.getRow(A1D_ELEM(ordered_list, i - 1), rowi_1);
		if (sort_last_N != -1 && i > sort_last_N)
			v.getRow(A1D_ELEM(ordered_list, i - sort_last_N - 1), rowi_N_1);
		for (j = 0; j < numIMG; j++) {
			if (!A1D_ELEM(chosen, j)) {
				v.getRow(j, rowj);
				A1D_ELEM(product, j) += ABS(dotProduct(rowi_1, rowj));
				if (sort_last_N != -1 && i > sort_last_N)
					A1D_ELEM(product, j) -= ABS(dotProduct(rowi_N_1, rowj));
				if (A1D_ELEM(product, j) < min_prod) {
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

MultidimArray<PrecisionType> ProgForwardArtZernike3DGPU::useFilterPrecision(FourierFilter &filter,
																			MultidimArray<PrecisionType> precisionImage)
{
	MultidimArray<double> doubleImage;
	MultidimArray<PrecisionType> outputImage;
	typeCast(precisionImage, doubleImage);
	filter.generateMask(doubleImage);
	filter.applyMaskSpace(doubleImage);
	typeCast(doubleImage, outputImage);
	return outputImage;
}

template<ProgForwardArtZernike3DGPU::Direction DIRECTION>
void ProgForwardArtZernike3DGPU::artModel()
{
	if (DIRECTION == Direction::Forward) {
		Image<double> I_shifted;
		for (int i = 0; i < sigma.size(); i++) {
			P[i]().initZeros((int)XSIZE(I()), (int)XSIZE(I()));
			P[i]().setXmippOrigin();
			W[i]().initZeros((int)XSIZE(I()), (int)XSIZE(I()));
			W[i]().setXmippOrigin();
		}
		Idiff().initZeros((int)XSIZE(I()), (int)XSIZE(I()));
		Idiff().setXmippOrigin();
		I_shifted().initZeros((int)XSIZE(I()), (int)XSIZE(I()));
		I_shifted().setXmippOrigin();
		Iws().initZeros((int)XSIZE(I()), (int)XSIZE(I()));
		Iws().setXmippOrigin();

		if (useZernike)
			zernikeModel<true, Direction::Forward>();
		else
			zernikeModel<false, Direction::Forward>();

		for (int i = 0; i < sigma.size(); i++) {
			filter.w1 = sigma[i];
			filter2.w1 = sigma[i];
			P[i] = useFilterPrecision(filter, P[i]());
			W[i] = useFilterPrecision(filter2, W[i]());
		}

		if (hasCTF) {
			if (phaseFlipped)
				FilterCTF.correctPhase();
			FilterCTF.generateMask(I());
			FilterCTF.applyMaskSpace(I());
		}
		if (flip) {
			MAT_ELEM(A, 0, 0) *= -1;
			MAT_ELEM(A, 0, 1) *= -1;
			MAT_ELEM(A, 0, 2) *= -1;
		}

		applyGeometry(xmipp_transformation::LINEAR,
					  I_shifted(),
					  I(),
					  A,
					  xmipp_transformation::IS_NOT_INV,
					  xmipp_transformation::DONT_WRAP,
					  0.);

		// Compute difference image and divide by weights
		double error = 0.0;
		double N = 0.0;
		const auto &mIsh = I_shifted();
		auto &mId = Idiff();
		auto &mIws = Iws();
		double c = 1.0;
		FOR_ALL_ELEMENTS_IN_ARRAY2D(I())
		{
			auto diffVal = A2D_ELEM(mIsh, i, j);
			double sumMw = 0.0;
			for (int ids = 0; ids < sigma.size(); ids++) {
				const auto &mP = P[ids]();
				const auto &mW = W[ids]();
				const auto sg = sigma[ids];
				if (sigma.size() > 1)
					c = sg * sg;
				diffVal -= c * A2D_ELEM(mP, i, j);
				sumMw += c * c * A2D_ELEM(mW, i, j);
			}
			if (sumMw > 0.0) {
				A2D_ELEM(mId, i, j) = lambda * (diffVal);
				A2D_ELEM(mIws, i, j) = XMIPP_MAX(sumMw, 1.0);
				error += (diffVal) * (diffVal);
				N++;
			}
		}

		if (verbose >= 3) {
			for (int ids = 0; ids < sigma.size(); ids++) {
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
			std::cout << "Error for image " << num_images << " (" << current_image << ") in iteration "
					  << current_iter + 1 << " : " << error << std::endl;
	} else if (DIRECTION == Direction::Backward) {
		if (useZernike)
			zernikeModel<true, Direction::Backward>();
		else
			zernikeModel<false, Direction::Backward>();
	}
}

template<bool USESZERNIKE, ProgForwardArtZernike3DGPU::Direction DIRECTION>
void ProgForwardArtZernike3DGPU::zernikeModel()
{
	cuda_forward_art_zernike3D::Program<PrecisionType>::AngleParameters angles = {
		.rot = rot,
		.tilt = tilt,
		.psi = psi,
	};

	cuda_forward_art_zernike3D::Program<PrecisionType>::DynamicParameters parameters = {
		.clnm = clnm,
		.P = P,
		.W = W,
		.Idiff = Idiff,
		.Iws = Iws,
		.angles = angles,
		.ltv = ltv,
		.ltk = ltk,
		.ll1 = ll1,
		.lst = lst,
		.lmr = lmr,
		.dSize = dSize,
		.loopStep = static_cast<PrecisionType>(loop_step),
		.lambda = lambda,
	};

	try {
		if (DIRECTION == Direction::Forward)
			cudaProgram->runForwardKernel<USESZERNIKE>(parameters);
		else if (DIRECTION == Direction::Backward)
			cudaProgram->runBackwardKernel<USESZERNIKE>(parameters);
	} catch (const std::runtime_error &e) {
		std::cerr << e.what() << '\n';
		std::exit(EXIT_FAILURE);
	}
}
