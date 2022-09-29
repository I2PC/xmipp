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

#include "art_zernike3d.h"
#include <core/transformations.h>
#include <core/xmipp_image_extension.h>
#include <core/xmipp_image_generic.h>
#include <data/projection.h>
#include <data/mask.h>
#include <numeric>

// Empty constructor =======================================================
ProgArtZernike3D::ProgArtZernike3D()
{
	resume = false;
    produces_a_metadata = true;
    each_image_produces_an_output = false;
    showOptimization = false;
}

ProgArtZernike3D::~ProgArtZernike3D() = default;

// Read arguments ==========================================================
void ProgArtZernike3D::readParams()
{
	XmippMetadataProgram::readParams();
	fnVolR = getParam("--ref");
	fnOutDir = getParam("--odir");
    RmaxDef = getIntParam("--RDef");
    phaseFlipped = checkParam("--phaseFlipped");
	useCTF = checkParam("--useCTF");
	Ts = getDoubleParam("--sampling");
    L1 = getIntParam("--l1");
	L2 = getIntParam("--l2");
	useZernike = checkParam("--useZernike");
    lambda = getDoubleParam("--regularization");
	resume = checkParam("--resume");
	niter = getIntParam("--niter");
	save_iter = getIntParam("--save_iter");
	sort_last_N = getIntParam("--sort_last");
	fnVolO = fnOutDir + "/Refined.vol";
	keep_input_columns = true;
}

// Show ====================================================================
void ProgArtZernike3D::show()
{
    if (!verbose)
        return;
	XmippMetadataProgram::show();
    std::cout
    << "Output directory:          "   << fnOutDir 		   << std::endl
    << "Reference volume:          "   << fnVolR           << std::endl
	<< "Sampling:                  "   << Ts               << std::endl
    << "Max. Radius Deform.        "   << RmaxDef          << std::endl
    << "Zernike Degree:            "   << L1               << std::endl
    << "SH Degree:                 "   << L2               << std::endl
	<< "Correct CTF:               "   << useCTF           << std::endl
	<< "Correct heretogeneity:     "   << useZernike       << std::endl
    << "Phase flipped:             "   << phaseFlipped     << std::endl
    << "Regularization:            "   << lambda           << std::endl
	<< "Number of iterations:      "   << niter            << std::endl
	<< "Save every # iterations:   "   << save_iter        << std::endl
    ;
}

// usage ===================================================================
void ProgArtZernike3D::defineParams()
{
    addUsageLine("Template-based canonical volume reconstruction through Zernike3D coefficients");
	defaultComments["-i"].clear();
	defaultComments["-i"].addComment("Metadata with initial alignment");
	defaultComments["-o"].clear();
	defaultComments["-o"].addComment("Refined volume");
    XmippMetadataProgram::defineParams();
    addParamsLine("  [--ref <volume=\"\">]       : Reference volume");
	addParamsLine("  [--odir <outputDir=\".\">]   : Output directory");
	addParamsLine("  [--sampling <Ts=1>]          : Sampling rate (A/pixel)");
    addParamsLine("  [--RDef <r=-1>]              : Maximum radius of the deformation (px). -1=Half of volume size");
    addParamsLine("  [--l1 <l1=3>]                : Degree Zernike Polynomials=1,2,3,...");
	addParamsLine("  [--l2 <l2=2>]                : Harmonical depth of the deformation=1,2,3,...");
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
    addExampleLine("A typical use is:",false);
    addExampleLine("xmipp_art_zernike3d -i anglesFromContinuousAssignment.xmd --ref reference.vol -o assigned_anglesAndDeformations.xmd --l1 3 --l2 2");
}

void ProgArtZernike3D::preProcess()
{

	// Check that metadata has all information neede
	if (!getInputMd()->containsLabel(MDL_ANGLE_ROT) || 
		!getInputMd()->containsLabel(MDL_ANGLE_TILT) ||
		!getInputMd()->containsLabel(MDL_ANGLE_PSI))
	{
		REPORT_ERROR(ERR_MD_MISSINGLABEL,"Input metadata projection angles are missing. Exiting...");
	}

	if (fnVolR != "")
	{
    V.read(fnVolR);
	}
	else 
	{
		FileName fn_first_image;
		Image<float> first_image;
		getInputMd()->getRow(1)->getValue(MDL_IMAGE,fn_first_image);
		first_image.read(fn_first_image);
		size_t Xdim_first = XSIZE(first_image());
		V().initZeros(Xdim_first, Xdim_first, Xdim_first);

	}
    V().setXmippOrigin();

    Xdim=XSIZE(V());

	if (resume && fnVolO.exists()) {
		Vrefined.read(fnVolO);
	} else {
		Vrefined() = V();
	}
	Vrefined().setXmippOrigin();

	if (RmaxDef<0)
		RmaxDef = Xdim/2;

    // Transformation matrix
    A.initIdentity(3);

	// CTF Filter
	FilterCTF.FilterBand = CTFINV;
	FilterCTF.FilterShape = CTFINV;
	FilterCTF.ctf.enable_CTFnoise = false;
	FilterCTF.ctf.enable_CTF = true;

	// Area where Zernike3D basis is computed (and volume is updated)
	Mask mask;
	mask.type = BINARY_CIRCULAR_MASK;
	mask.mode = INNER_MASK;
	mask.R1 = RmaxDef;
	mask.generate_mask(V());
	Vmask = mask.get_binary_mask();
	Vmask.setXmippOrigin();

	// Area Zernike3D in 2D
	mask.generate_mask(XSIZE(V()), XSIZE(V()));
	mask2D = mask.get_binary_mask();
	mask2D.setXmippOrigin();

	vecSize = 0;
	numCoefficients(L1,L2,vecSize);
    fillVectorTerms(L1,L2,vL1,vN,vL2,vM);

    // createWorkFiles();
	initX = STARTINGX(Vrefined());
	endX = FINISHINGX(Vrefined());
	initY = STARTINGY(Vrefined());
	endY = FINISHINGY(Vrefined());		
	initZ = STARTINGZ(Vrefined());
	endZ = FINISHINGZ(Vrefined());
}

void ProgArtZernike3D::finishProcessing() {
	Vrefined.write(fnVolO);
}

// Predict =================================================================
void ProgArtZernike3D::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
	rowOut=rowIn; // FIXME have a look if this is needed. I don't think so, or see how to do this automatically in xmipp_metadata_program.cpp
	flagEnabled=1;

	rowIn.getValue(MDL_ANGLE_ROT,rot);
	rowIn.getValue(MDL_ANGLE_TILT,tilt);
	rowIn.getValue(MDL_ANGLE_PSI,psi);
	rowIn.getValueOrDefault(MDL_SHIFT_X,shiftX,0.0);
	rowIn.getValueOrDefault(MDL_SHIFT_Y,shiftY,0.0);
	std::vector<double> vectortemp;
	if (useZernike) {
		rowIn.getValue(MDL_SPH_COEFFICIENTS,vectortemp);
		clnm.initZeros(vectortemp.size()-8);
		for(int i=0; i < vectortemp.size()-8; i++){
			VEC_ELEM(clnm,i) = static_cast<float>(vectortemp[i]);
		}
		removeOverdeformation();
	}
	rowIn.getValueOrDefault(MDL_FLIP,flip, false);
	
	if ((rowIn.containsLabel(MDL_CTF_DEFOCUSU) || rowIn.containsLabel(MDL_CTF_MODEL)) && useCTF)
	{
		hasCTF=true;
		FilterCTF.ctf.readFromMdRow(rowIn, false);
		FilterCTF.ctf.Tm = Ts;
		FilterCTF.ctf.produceSideInfo();
	}
	else
		hasCTF=false;
	MAT_ELEM(A,0,2)=shiftX;
	MAT_ELEM(A,1,2)=shiftY;
	MAT_ELEM(A,0,0)=1;
	MAT_ELEM(A,0,1)=0;
	MAT_ELEM(A,1,0)=0;
	MAT_ELEM(A,1,1)=1;

	if (verbose>=2)
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

void ProgArtZernike3D::numCoefficients(int l1, int l2, int &vecSize)
{
    for (int h=0; h<=l2; h++)
    {
        int numSPH = 2*h+1;
        int count=l1-h+1;
        int numEven=(count>>1)+(count&1 && !(h&1));
        if (h%2 == 0) {
            vecSize += numSPH*numEven;
		}
        else {
        	vecSize += numSPH*(l1-h+1-numEven);
		}
    }
}

void ProgArtZernike3D::fillVectorTerms(int l1, int l2, Matrix1D<int> &vL1, Matrix1D<int> &vN, 
									          Matrix1D<int> &vL2, Matrix1D<int> &vM)
{
    int idx = 0;
	vL1.initZeros(vecSize);
	vN.initZeros(vecSize);
	vL2.initZeros(vecSize);
	vM.initZeros(vecSize);
    for (int h=0; h<=l2; h++) {
        int totalSPH = 2*h+1;
        int aux = std::floor(totalSPH/2);
        for (int l=h; l<=l1; l+=2) {
            for (int m=0; m<totalSPH; m++) {
                VEC_ELEM(vL1,idx) = l;
                VEC_ELEM(vN,idx) = h;
                VEC_ELEM(vL2,idx) = h;
                VEC_ELEM(vM,idx) = m-aux;
                idx++;
            }
        }
    }
}

template<bool INTERPOLATE>
float ProgArtZernike3D::weightsInterpolation3D(float x, float y, float z, std::array<float, 8> &w) {
	int x0 = FLOOR(x);
	float fx0 = x - x0;
	int x1 = x0 + 1;
	float fx1 = x1 - x;

	int y0 = FLOOR(y);
	float fy0 = y - y0;
	int y1 = y0 + 1;
	float fy1 = y1 - y;

	int z0 = FLOOR(z);
	float fz0 = z - z0;
	int z1 = z0 + 1;
	float fz1 = z1 - z;

	w[0] = fx1 * fy1 * fz1;  // w000 (x0,y0,z0)
	w[1] = fx1 * fy1 * fz0;  // w001 (x0,y0,z1)
	w[2] = fx1 * fy0 * fz1;  // w010 (x0,y1,z0)
	w[3] = fx1 * fy0 * fz0;  // w011 (x0,y1,z1)
	w[4] = fx0 * fy1 * fz1;  // w100 (x1,y0,z0)
	w[5] = fx0 * fy1 * fz0;  // w101 (x1,y0,z1)
	w[6] = fx0 * fy0 * fz1;  // w110 (x1,y1,z0)
	w[7] = fx0 * fy0 * fz0;  // w111 (x1,y1,z1)

	if (INTERPOLATE) {
		const auto &mVr = Vrefined();
		if (x0 < initX || y0 < initY || z0 < initZ || x1 > endX ||
			y1 > endY || z1 > endZ) {
		return NAN;
		} else {
		return A3D_ELEM(mVr, z0, y0, x0) * w[0] +
				A3D_ELEM(mVr, z1, y0, x0) * w[1] +
				A3D_ELEM(mVr, z0, y1, x0) * w[2] +
				A3D_ELEM(mVr, z1, y1, x0) * w[3] +
				A3D_ELEM(mVr, z0, y0, x1) * w[4] +
				A3D_ELEM(mVr, z1, y0, x1) * w[5] +
				A3D_ELEM(mVr, z0, y1, x1) * w[6] +
				A3D_ELEM(mVr, z1, y1, x1) * w[7];
		}
	} else {
		return NAN;
	}
}

void ProgArtZernike3D::removeOverdeformation() {
	int pos = 3*vecSize;
	size_t idxY0=(VEC_XSIZE(clnm))/3;
	size_t idxZ0=2*idxY0;

	Matrix2D<float> R, R_inv;
	R.initIdentity(3);
	R_inv.initIdentity(3);
    Euler_angles2matrix(rot, tilt, psi, R, false);
    R_inv = R.inv();
	Matrix1D<float> c;
	c.initZeros(3);
	for (size_t idx=0; idx<idxY0; idx++) {
		XX(c) = VEC_ELEM(clnm,idx); YY(c) = VEC_ELEM(clnm,idx+idxY0); ZZ(c) = VEC_ELEM(clnm,idx+idxZ0);
		c = R * c;
		ZZ(c) = 0.0f;
		c = R_inv * c;
		VEC_ELEM(clnm,idx) = XX(c); VEC_ELEM(clnm,idx+idxY0) = YY(c); VEC_ELEM(clnm,idx+idxZ0) = ZZ(c);
	}
}

void ProgArtZernike3D::run()
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
        oextBaseName   = oext;
        fullBaseName   = oroot.removeFileFormat();
        baseName       = fullBaseName.getBaseName();
        pathBaseName   = fullBaseName.getDir();
    }

	size_t objId;
	size_t objIndex;
	current_save_iter = 1;
	num_images = 1;
	for (current_iter=0; current_iter<niter; current_iter++) {
		std::cout << "Running iteration " << current_iter+1 << " with lambda=" << lambda << std::endl;
		objId = 0;
		objIndex = 0;
		time_bar_done = 0;
		FOR_ALL_ELEMENTS_IN_ARRAY1D(ordered_list)
		{
			objId = A1D_ELEM(ordered_list,i) + 1;
			++objIndex;
			auto rowIn = getInputMd()->getRow(objId);
			rowIn->getValue(image_label, fnImg);

			if (fnImg.empty())
				break;

			fnImgOut = fnImg;

			MDRowVec rowOut;

			if (each_image_produces_an_output)
			{
				if (!oroot.empty()) // Compose out name to save as independent images
				{
					if (oext.empty()) // If oext is still empty, then use ext of indep input images
					{
						if (input_is_stack)
							oextBaseName = "spi";
						else
							oextBaseName = fnImg.getFileFormat();
					}

					if (!baseName.empty() )
						fnImgOut.compose(fullBaseName, objIndex, oextBaseName);
					else if (fnImg.isInStack())
						fnImgOut.compose(pathBaseName + (fnImg.withoutExtension()).getDecomposedFileName(), objIndex, oextBaseName);
					else
						fnImgOut = pathBaseName + fnImg.withoutExtension()+ "." + oextBaseName;
				}
				else if (!fn_out.empty() )
				{
					if (single_image)
						fnImgOut = fn_out;
					else
						fnImgOut.compose(objIndex, fn_out); // Compose out name to save as stacks
				}
				else
					fnImgOut = fnImg;
				setupRowOut(fnImg, *rowIn.get(), fnImgOut, rowOut);
			}
			else if (produces_a_metadata)
				setupRowOut(fnImg, *rowIn.get(), fnImgOut, rowOut);

			processImage(fnImg, fnImgOut, *rowIn.get(), rowOut);

			if (each_image_produces_an_output || produces_a_metadata)
				getOutputMd().addRow(rowOut);

			checkPoint();
			showProgress();

			// Save refined volume every num_images
			if (current_save_iter == save_iter && save_iter > 0) {
				Mask mask;
				mask.type = BINARY_CIRCULAR_MASK;
				mask.mode = INNER_MASK;
				mask.R1 = RmaxDef - 2;
				mask.generate_mask(Vrefined());
				mask.apply_mask(Vrefined(), Vrefined());
				Vrefined.write(fnVolO.removeAllExtensions() + "it" + std::to_string(current_iter+1) + "proj" + std::to_string(num_images) + ".mrc");
				current_save_iter = 1;
			}
			current_save_iter++;
			num_images++;
		}
		num_images = 1;
		current_save_iter = 1;

		Mask mask;
		mask.type = BINARY_CIRCULAR_MASK;
		mask.mode = INNER_MASK;
		mask.R1 = RmaxDef - 2;
		mask.generate_mask(Vrefined());
		mask.apply_mask(Vrefined(), Vrefined());
	}
    wait();

    /* Generate name to save mdOut when output are independent images. It uses as prefix
     * the dirBaseName in order not overwriting files when repeating same command on
     * different directories. If baseName is set it is used, otherwise, input name is used.
     * Then, the suffix _oext is added.*/
    if (fn_out.empty() )
    {
        if (!oroot.empty())
        {
            if (!baseName.empty() )
                fn_out = findAndReplace(pathBaseName,"/","_") + baseName + "_" + oextBaseName + ".xmd";
            else
                fn_out = findAndReplace(pathBaseName,"/","_") + fn_in.getBaseName() + "_" + oextBaseName + ".xmd";
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

void ProgArtZernike3D::sortOrthogonal() {
	int i, j;
	size_t numIMG = getInputMd()->size();
	MultidimArray<short> chosen(numIMG);
	chosen.initZeros(numIMG);
	MultidimArray<double> product(numIMG);
	product.initZeros(numIMG);
	double min_prod = MAXFLOAT;;
    int min_prod_proj = 0;
	std::vector<double> rot;
	std::vector<double> tilt;
	Matrix2D<double> v(numIMG, 3);
	v.initZeros(numIMG, 3);
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
    std::cout << "Sorting projections orthogonally...\n" << std::endl;
    Matrix1D<double> rowj, rowi_1, rowi_N_1;
    for (i = 1; i < numIMG; i++)
    {
        // Compute the product of not already chosen vectors with the just
        // chosen one, and select that which has minimum product
		min_prod = MAXFLOAT;
        v.getRow(A1D_ELEM(ordered_list, i - 1),rowi_1);
        if (sort_last_N != -1 && i > sort_last_N)
            v.getRow(A1D_ELEM(ordered_list, i - sort_last_N - 1),rowi_N_1);
        for (j = 0; j < numIMG; j++)
        {
            if (!A1D_ELEM(chosen, j))
            {
                v.getRow(j,rowj);
                A1D_ELEM(product, j) += ABS(dotProduct(rowi_1,rowj));
                if (sort_last_N != -1 && i > sort_last_N)
                    A1D_ELEM(product, j) -= ABS(dotProduct(rowi_N_1,rowj));
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

template <ProgArtZernike3D::Direction DIRECTION>
void ProgArtZernike3D::artModel()
{
	if (DIRECTION == Direction::Forward)
	{
		Image<float> I_shifted;
		P().initZeros((int)XSIZE(I()), (int)XSIZE(I()));
		P().setXmippOrigin();
		W().initZeros((int)XSIZE(I()), (int)XSIZE(I()));
		W().setXmippOrigin();
		Idiff().initZeros((int)XSIZE(I()), (int)XSIZE(I()));
		Idiff().setXmippOrigin();
		I_shifted().initZeros((int)XSIZE(I()), (int)XSIZE(I()));
		I_shifted().setXmippOrigin();

		if (useZernike)
			zernikeModel<true, Direction::Forward>();
		else
			zernikeModel<false, Direction::Forward>();

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
					  xmipp_transformation::IS_NOT_INV, xmipp_transformation::DONT_WRAP, 0.f);

		// Compute difference image and divide by weights
		float error = 0.0f;
		float N = 0.0f;
		const auto &mP = P();
		const auto &mW = W();
		const auto &mIsh = I_shifted();
		auto &mId = Idiff();
		FOR_ALL_ELEMENTS_IN_ARRAY2D(I())
		{
			if (A2D_ELEM(mask2D, i, j) == 1)
			{
				auto diffVal = A2D_ELEM(mIsh, i, j) - A2D_ELEM(mP, i, j);
				A2D_ELEM(mId, i, j) = lambda * (diffVal) / XMIPP_MAX(A2D_ELEM(mW, i, j), 1.0);
				error += (diffVal) * (diffVal);
				N++;
			}
		}
		// Creo que Carlos no usa un RMSE si no un MSE
		error = std::sqrt(error / N);
		std::cout << "Error for image " << num_images << " in iteration " << current_iter+1 << " : " << error << std::endl;
	}
	else if (DIRECTION == Direction::Backward)
	{
		if (useZernike)
			zernikeModel<true, Direction::Backward>();
		else
			zernikeModel<false, Direction::Backward>();
	}	
}

template<bool USESZERNIKE, ProgArtZernike3D::Direction DIRECTION>
void ProgArtZernike3D::zernikeModel() {
	const auto &mV = V();
	const size_t idxY0 = USESZERNIKE ? (VEC_XSIZE(clnm) / 3) : 0;
	const size_t idxZ0 = USESZERNIKE ? (2 * idxY0) : 0;
	const float RmaxF = USESZERNIKE ? RmaxDef : 0;
	const float RmaxF2 = USESZERNIKE ? (RmaxF * RmaxF) : 0;
	const float iRmaxF = USESZERNIKE ? (1.0 / RmaxF) : 0;
    // Rotation Matrix
	constexpr size_t matrixSize = 3;
    const Matrix2D<float> R = [this](){
		auto tmp = Matrix2D<float>();
		tmp.initIdentity(matrixSize);
		Euler_angles2matrix(rot, tilt, psi, tmp, false);
		return tmp.inv();
	}();

	const auto lastZ = FINISHINGZ(mV);
	const auto lastY = FINISHINGY(mV);
	const auto lastX = FINISHINGX(mV);
	for (int k=STARTINGZ(mV); k<=lastZ; ++k)
	{
		for (int i=STARTINGY(mV); i<=lastY; ++i)
		{
			for (int j=STARTINGX(mV); j<=lastX; ++j)
			{
				if (A3D_ELEM(Vmask,k,i,j) != 1) { continue; }
				auto pos = std::array<float, 3>{};
				pos[0] = R.mdata[0] * j + R.mdata[1] * i + R.mdata[2] * k;
				pos[1] = R.mdata[3] * j + R.mdata[4] * i + R.mdata[5] * k;
				pos[2] = R.mdata[6] * j + R.mdata[7] * i + R.mdata[8] * k;

				float gx=0.0f, gy=0.0f, gz=0.0f;
				if (USESZERNIKE)
				{
					auto k2 = pos[2] * pos[2];
					auto kr = pos[2] * iRmaxF;
					auto k2i2 = k2 + pos[1] * pos[1];
					auto ir = pos[1] * iRmaxF;
					auto r2 = k2i2 + pos[0] * pos[0];
					auto jr = pos[0] * iRmaxF;
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
							gx += VEC_ELEM(clnm, idx) * (zsph);
							gy += VEC_ELEM(clnm, idx + idxY0) * (zsph);
							gz += VEC_ELEM(clnm, idx + idxZ0) * (zsph);
						}
					}
				}
				// }
				auto r_x = pos[0] + gx;
				auto r_y = pos[1] + gy;
				auto r_z = pos[2] + gz;

				auto w = std::array<float, 8>{};
				if (DIRECTION == Direction::Forward)
				{
					auto voxel = weightsInterpolation3D<true>(r_x, r_y, r_z, w);
					if (!isnan(voxel))
					{
						A2D_ELEM(P(), i, j) += voxel;
						A2D_ELEM(W(), i, j) +=  std::inner_product(w.begin(), w.end(), w.begin(), static_cast<float>(0));
					}
				}
				else if (DIRECTION == Direction::Backward)
				{
					int x0 = FLOOR(r_x);
					auto x1 = x0 + 1;
					int y0 = FLOOR(r_y);
					auto y1 = y0 + 1;
					int z0 = FLOOR(r_z);
					auto z1 = z0 + 1;
					float Idiff_val = A2D_ELEM(Idiff(), i, j);
					weightsInterpolation3D<false>(r_x, r_y, r_z, w);
					if (!Vrefined().outside(z0, y0, x0))
						A3D_ELEM(Vrefined(), z0, y0, x0) += Idiff_val * w[0];
					if (!Vrefined().outside(z1, y0, x0))
						A3D_ELEM(Vrefined(), z1, y0, x0) += Idiff_val * w[1];
					if (!Vrefined().outside(z0, y1, x0))
						A3D_ELEM(Vrefined(), z0, y1, x0) += Idiff_val * w[2];
					if (!Vrefined().outside(z1, y1, x0))
						A3D_ELEM(Vrefined(), z1, y1, x0) += Idiff_val * w[3];
					if (!Vrefined().outside(z0, y0, x1))
						A3D_ELEM(Vrefined(), z0, y0, x1) += Idiff_val * w[4];
					if (!Vrefined().outside(z1, y0, x1))
						A3D_ELEM(Vrefined(), z1, y0, x1) += Idiff_val * w[5];
					if (!Vrefined().outside(z0, y1, x1))
						A3D_ELEM(Vrefined(), z0, y1, x1) += Idiff_val * w[6];
					if (!Vrefined().outside(z1, y1, x1))
						A3D_ELEM(Vrefined(), z1, y1, x1) += Idiff_val * w[7];
				}
			}
		}
	}
}
