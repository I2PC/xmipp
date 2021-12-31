/***************************************************************************
 *
 * Authors:    Carlos Oscar Sanchez Sorzano (coss@cnb.csic.es)
 *             David Herreros Calero (dherreros@cnb.csic.es)
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

#include "forward_zernike_images.h"
#include "core/transformations.h"
#include "core/xmipp_image_extension.h"
#include "core/xmipp_image_generic.h"
#include "data/projection.h"
#include "data/mask.h"

// Empty constructor =======================================================
ProgForwardZernikeImages::ProgForwardZernikeImages() : Rerunable("")
{
	resume = false;
    produces_a_metadata = true;
    each_image_produces_an_output = false;
    showOptimization = false;
}

ProgForwardZernikeImages::~ProgForwardZernikeImages() = default;

// Read arguments ==========================================================
void ProgForwardZernikeImages::readParams()
{
	XmippMetadataProgram::readParams();
	fnVolR = getParam("--ref");
	fnMaskR = getParam("--mask");
	fnOutDir = getParam("--odir");
    maxShift = getDoubleParam("--max_shift");
    maxAngularChange = getDoubleParam("--max_angular_change");
    maxResol = getDoubleParam("--max_resolution");
    Ts = getDoubleParam("--sampling");
    Rmax = getIntParam("--Rmax");
    RmaxDef = getIntParam("--RDef");
    optimizeAlignment = checkParam("--optimizeAlignment");
    optimizeDeformation = checkParam("--optimizeDeformation");
	optimizeDefocus = checkParam("--optimizeDefocus");
    phaseFlipped = checkParam("--phaseFlipped");
	useCTF = checkParam("--useCTF");
    L1 = getIntParam("--l1");
	L2 = getIntParam("--l2");
	loop_step = getIntParam("--step");
    lambda = getDoubleParam("--regularization");
	image_mode = getIntParam("--image_mode");
	resume = checkParam("--resume");
	Rerunable::setFileName(fnOutDir + "/sphDone.xmd");
	blob_r = getDoubleParam("--blobr");
	keep_input_columns = true;
}

// Show ====================================================================
void ProgForwardZernikeImages::show()
{
    if (!verbose)
        return;
	XmippMetadataProgram::show();
    std::cout
    << "Output directory:    " << fnOutDir 			 << std::endl
    << "Reference volume:    " << fnVolR             << std::endl
	<< "Reference mask:      " << fnMaskR            << std::endl
    << "Max. Shift:          " << maxShift           << std::endl
    << "Max. Angular Change: " << maxAngularChange   << std::endl
    << "Max. Resolution:     " << maxResol           << std::endl
    << "Sampling:            " << Ts                 << std::endl
    << "Max. Radius:         " << Rmax               << std::endl
    << "Max. Radius Deform.  " << RmaxDef            << std::endl
    << "Zernike Degree:      " << L1                 << std::endl
    << "SH Degree:           " << L2                 << std::endl
	<< "Step:                " << loop_step          << std::endl
	<< "Correct CTF:         " << useCTF             << std::endl
    << "Optimize alignment:  " << optimizeAlignment  << std::endl
    << "Optimize deformation:" << optimizeDeformation<< std::endl
	<< "Optimize defocus:    " << optimizeDefocus    << std::endl
    << "Phase flipped:       " << phaseFlipped       << std::endl
    << "Regularization:      " << lambda             << std::endl
	<< "Blob radius:         " << blob_r             << std::endl
	<< "Image mode:          " << image_mode         << std::endl
    ;
}

// usage ===================================================================
void ProgForwardZernikeImages::defineParams()
{
    addUsageLine("Make a continuous angular assignment with deformations");
	defaultComments["-i"].clear();
	defaultComments["-i"].addComment("Metadata with initial alignment");
	defaultComments["-o"].clear();
	defaultComments["-o"].addComment("Metadata with the angular alignment and deformation parameters");
    XmippMetadataProgram::defineParams();
    addParamsLine("   --ref <volume>              : Reference volume");
	addParamsLine("  [--mask <m=\"\">]            : Reference volume");
	addParamsLine("  [--odir <outputDir=\".\">]   : Output directory");
    addParamsLine("  [--max_shift <s=-1>]         : Maximum shift allowed in pixels");
    addParamsLine("  [--max_angular_change <a=5>] : Maximum angular change allowed (in degrees)");
    addParamsLine("  [--max_resolution <f=4>]     : Maximum resolution (A)");
    addParamsLine("  [--sampling <Ts=1>]          : Sampling rate (A/pixel)");
    addParamsLine("  [--Rmax <R=-1>]              : Maximum radius (px). -1=Half of volume size");
    addParamsLine("  [--RDef <r=-1>]              : Maximum radius of the deformation (px). -1=Half of volume size");
    addParamsLine("  [--l1 <l1=3>]                : Degree Zernike Polynomials=1,2,3,...");
	addParamsLine("  [--l2 <l2=2>]                : Harmonical depth of the deformation=1,2,3,...");
	addParamsLine("  [--step <step=1>]            : Voxel index step");
	addParamsLine("  [--useCTF]                   : Correct CTF");
    addParamsLine("  [--optimizeAlignment]        : Optimize alignment");
    addParamsLine("  [--optimizeDeformation]      : Optimize deformation");
	addParamsLine("  [--optimizeDefocus]          : Optimize defocus");
    addParamsLine("  [--phaseFlipped]             : Input images have been phase flipped");
    addParamsLine("  [--regularization <l=0.01>]  : Regularization weight");
	addParamsLine("  [--blobr <b=4>]              : Blob radius for forward mapping splatting");
	addParamsLine("  [--image_mode <im=-1>]       : Image mode (single, pairs, triplets). By default, it will be automatically identified.");
	addParamsLine("  [--resume]                   : Resume processing");
    addExampleLine("A typical use is:",false);
    addExampleLine("xmipp_angular_sph_alignment -i anglesFromContinuousAssignment.xmd --ref reference.vol -o assigned_anglesAndDeformations.xmd --optimizeAlignment --optimizeDeformation --depth 1");
}

void ProgForwardZernikeImages::preProcess()
{
    V.read(fnVolR);
    V().setXmippOrigin();
    Xdim=XSIZE(V());
    Vdeformed().initZeros(V());
	Vdeformed().setXmippOrigin();
    // sumV=V().sum();

	// Check execution mode (single, pair, or triplet)
	if (image_mode > 0)
	{
		num_images = image_mode;
	}
	else{
		num_images = 1; // Single image
		if (getInputMd()->containsLabel(MDL_IMAGE1) && !getInputMd()->containsLabel(MDL_IMAGE2))
		{
			num_images = 2; // Pair
		}
		else if (getInputMd()->containsLabel(MDL_IMAGE1) && getInputMd()->containsLabel(MDL_IMAGE2))
		{
			num_images = 3; // Triplet
		}
	}

	// Preallocate vectors (Size depends on image number)
	fnImage.resize(num_images, "");
	I.resize(num_images); Ifiltered.resize(num_images); Ifilteredp.resize(num_images);
	P.resize(num_images);
	old_rot.resize(num_images, 0.); old_tilt.resize(num_images, 0.); old_psi.resize(num_images, 0.);
	deltaRot.resize(num_images, 0.); deltaTilt.resize(num_images, 0.); deltaPsi.resize(num_images, 0.);
	old_shiftX.resize(num_images, 0.); old_shiftY.resize(num_images, 0.);
	deltaX.resize(num_images, 0.); deltaY.resize(num_images, 0.);
	old_defocusU.resize(num_images, 0.); old_defocusV.resize(num_images, 0.); old_defocusAngle.resize(num_images, 0.);
	deltaDefocusU.resize(num_images, 0.); deltaDefocusV.resize(num_images, 0.); deltaDefocusAngle.resize(num_images, 0.);
	currentDefocusU.resize(num_images, 0.); currentDefocusV.resize(num_images, 0.); currentAngle.resize(num_images, 0.);

	switch (num_images)
	{
	case 2:
		Ifilteredp[0]().initZeros(Xdim,Xdim); Ifilteredp[1]().initZeros(Xdim,Xdim);
    	Ifilteredp[0]().setXmippOrigin(); Ifilteredp[1]().setXmippOrigin();
		P[0]().initZeros(Xdim,Xdim); P[1]().initZeros(Xdim,Xdim);
		break;
	case 3:
		Ifilteredp[0]().initZeros(Xdim,Xdim); Ifilteredp[1]().initZeros(Xdim,Xdim); Ifilteredp[2]().initZeros(Xdim,Xdim);
    	Ifilteredp[0]().setXmippOrigin(); Ifilteredp[1]().setXmippOrigin(); Ifilteredp[2]().setXmippOrigin();
		P[0]().initZeros(Xdim,Xdim); P[1]().initZeros(Xdim,Xdim); P[2]().initZeros(Xdim,Xdim);
		break;
	
	default:
		Ifilteredp[0]().initZeros(Xdim,Xdim);
    	Ifilteredp[0]().setXmippOrigin();
		P[0]().initZeros(Xdim,Xdim);
		break;
	}

	if (RmaxDef<0)
		RmaxDef = Xdim/2;

	// Read Reference mask if avalaible (otherwise sphere of radius RmaxDef is used)
	Mask mask;
	mask.type = BINARY_CIRCULAR_MASK;
	mask.mode = INNER_MASK;
	if (fnMaskR != "") {
		Image<double> aux;
		aux.read(fnMaskR);
		typeCast(aux(), V_mask);
		V_mask.setXmippOrigin();
		double Rmax2 = RmaxDef*RmaxDef;
		for (int k=STARTINGZ(V_mask); k<=FINISHINGZ(V_mask); k++) {
			for (int i=STARTINGY(V_mask); i<=FINISHINGY(V_mask); i++) {
				for (int j=STARTINGX(V_mask); j<=FINISHINGX(V_mask); j++) {
					double r2 = k*k + i*i + j*j;
					if (r2>=Rmax2)
						A3D_ELEM(V_mask,k,i,j) = 0;
				}
			}
		}
	}
	else {
		mask.R1 = RmaxDef;
		mask.generate_mask(V());
		V_mask = mask.get_binary_mask();
		V_mask.setXmippOrigin();
	}

	// Total Volume Mass (Inside Mask)
	sumV = 0.0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V_mask) {
		if (DIRECT_MULTIDIM_ELEM(V_mask,n) == 1) {
			sumV += DIRECT_MULTIDIM_ELEM(V(),n);
		}
	}

    // Construct mask
    if (Rmax<0)
    	Rmax=Xdim/2;
    mask.R1 = Rmax;
    mask.generate_mask(Xdim,Xdim);
    mask2D=mask.get_binary_mask();

    // Low pass filter
    filter.FilterBand=LOWPASS;
    filter.w1=Ts/maxResol;
    filter.raised_w=0.02;

    // Transformation matrix
    A1.initIdentity(3);
	A2.initIdentity(3);
	A3.initIdentity(3);

	// CTF Filter
	FilterCTF1.FilterBand = CTF;
	FilterCTF1.ctf.enable_CTFnoise = false;
	FilterCTF1.ctf.produceSideInfo();
	FilterCTF2.FilterBand = CTF;
	FilterCTF2.ctf.enable_CTFnoise = false;
	FilterCTF2.ctf.produceSideInfo();
	FilterCTF3.FilterBand = CTF;
	FilterCTF3.ctf.enable_CTFnoise = false;
	FilterCTF3.ctf.produceSideInfo();

	vecSize = 0;
	numCoefficients(L1,L2,vecSize);
    fillVectorTerms(L1,L2,vL1,vN,vL2,vM);

    createWorkFiles();

	// Blob
	blob.radius = blob_r;   // Blob radius in voxels
	blob.order  = 2;        // Order of the Bessel function
    blob.alpha  = 3.6;      // Smoothness parameter

}

void ProgForwardZernikeImages::finishProcessing() {
	XmippMetadataProgram::finishProcessing();
	rename(Rerunable::getFileName().c_str(), (fnOutDir + fn_out).c_str());
}

// #define DEBUG
double ProgForwardZernikeImages::transformImageSph(double *pclnm)
{
	const MultidimArray<double> &mV=V();
	FOR_ALL_ELEMENTS_IN_MATRIX1D(clnm)
		VEC_ELEM(clnm,i)=pclnm[i+1];
	double deformation=0.0;
	totalDeformation=0.0;

	P[0]().initZeros(Xdim,Xdim);
	P[0]().setXmippOrigin();
	double currentRot=old_rot[0] + deltaRot[0];
	double currentTilt=old_tilt[0] + deltaTilt[0];
	double currentPsi=old_psi[0] + deltaPsi[0];
	deformVol(P[0](), mV, deformation, currentRot, currentTilt, currentPsi);

	double cost=0.0;
	const MultidimArray<int> &mMask2D=mask2D;
	double corr2 = 0.0;
	double corr3 = 0.0;

	switch (num_images)
	{
	case 2:
	{
		P[1]().initZeros(Xdim,Xdim);
		P[1]().setXmippOrigin();
		currentRot = old_rot[1] + deltaRot[1];
		currentTilt = old_tilt[1] + deltaTilt[1];
		currentPsi = old_psi[1] + deltaPsi[1];
		deformVol(P[1](), mV, deformation, currentRot, currentTilt, currentPsi);

		if (old_flip)
		{
			MAT_ELEM(A1, 0, 0) *= -1;
			MAT_ELEM(A1, 0, 1) *= -1;
			MAT_ELEM(A1, 0, 2) *= -1;
			MAT_ELEM(A2, 0, 0) *= -1;
			MAT_ELEM(A2, 0, 1) *= -1;
			MAT_ELEM(A2, 0, 2) *= -1;
		}
		applyGeometry(xmipp_transformation::LINEAR, Ifilteredp[0](), Ifiltered[0](), A1, 
					  xmipp_transformation::IS_NOT_INV, xmipp_transformation::DONT_WRAP, 0.);
		applyGeometry(xmipp_transformation::LINEAR, Ifilteredp[1](), Ifiltered[1](), A2, 
					  xmipp_transformation::IS_NOT_INV, xmipp_transformation::DONT_WRAP, 0.);
		// filter.applyMaskSpace(P[1]());
		const MultidimArray<double> mP2 = P[1]();
		MultidimArray<double> &mI2filteredp = Ifilteredp[1]();
		corr2 = correlationIndex(mI2filteredp, mP2, &mMask2D);
	}
		break;

	case 3:
	{
		P[1]().initZeros(Xdim,Xdim);
		P[1]().setXmippOrigin();
		currentRot = old_rot[1] + deltaRot[1];
		currentTilt = old_tilt[1] + deltaTilt[1];
		currentPsi = old_psi[1] + deltaPsi[1];
		deformVol(P[1](), mV, deformation, currentRot, currentTilt, currentPsi);

		P[2]().initZeros(Xdim,Xdim);
		P[2]().setXmippOrigin();
		currentRot = old_rot[2] + deltaRot[2];
		currentTilt = old_tilt[2] + deltaTilt[2];
		currentPsi = old_psi[2] + deltaPsi[2];
		deformVol(P[2](), mV, deformation, currentRot, currentTilt, currentPsi);

		if (old_flip)
		{
			MAT_ELEM(A1, 0, 0) *= -1;
			MAT_ELEM(A1, 0, 1) *= -1;
			MAT_ELEM(A1, 0, 2) *= -1;
			MAT_ELEM(A2, 0, 0) *= -1;
			MAT_ELEM(A2, 0, 1) *= -1;
			MAT_ELEM(A2, 0, 2) *= -1;
			MAT_ELEM(A3, 0, 0) *= -1;
			MAT_ELEM(A3, 0, 1) *= -1;
			MAT_ELEM(A3, 0, 2) *= -1;
		}
		applyGeometry(xmipp_transformation::LINEAR, Ifilteredp[0](), Ifiltered[0](), A1, 
					  xmipp_transformation::IS_NOT_INV, xmipp_transformation::DONT_WRAP, 0.);
		applyGeometry(xmipp_transformation::LINEAR, Ifilteredp[1](), Ifiltered[1](), A2, 
				      xmipp_transformation::IS_NOT_INV, xmipp_transformation::DONT_WRAP, 0.);
		applyGeometry(xmipp_transformation::LINEAR, Ifilteredp[2](), Ifiltered[2](), A3, 
					  xmipp_transformation::IS_NOT_INV, xmipp_transformation::DONT_WRAP, 0.);
		// filter.applyMaskSpace(P[1]());
		// filter.applyMaskSpace(P[2]());
		const MultidimArray<double> mP2 = P[1]();
		const MultidimArray<double> mP3 = P[2]();
		MultidimArray<double> &mI2filteredp = Ifilteredp[1]();
		MultidimArray<double> &mI3filteredp = Ifilteredp[2]();
		corr2 = correlationIndex(mI2filteredp, mP2, &mMask2D);
		corr3 = correlationIndex(mI3filteredp, mP3, &mMask2D);
	}
		break;
	
	default:
		if (old_flip)
		{
			MAT_ELEM(A1, 0, 0) *= -1;
			MAT_ELEM(A1, 0, 1) *= -1;
			MAT_ELEM(A1, 0, 2) *= -1;
		}
		applyGeometry(xmipp_transformation::LINEAR,Ifilteredp[0](),Ifiltered[0](),A1,
					  xmipp_transformation::IS_NOT_INV,xmipp_transformation::DONT_WRAP,0.);
		break;
	}

	if (hasCTF)
    {
    	double defocusU=old_defocusU[0]+deltaDefocusU[0];
    	double defocusV=old_defocusV[0]+deltaDefocusV[0];
    	double angle=old_defocusAngle[0]+deltaDefocusAngle[0];
    	if (defocusU!=currentDefocusU[0] || defocusV!=currentDefocusV[0] || angle!=currentAngle[0]) {
    		updateCTFImage(defocusU,defocusV,angle);
		}
		// FilterCTF1.ctf = ctf;
		FilterCTF1.generateMask(P[0]());
		if (phaseFlipped)
			FilterCTF1.correctPhase();
		FilterCTF1.applyMaskSpace(P[0]());
	}
	filter.applyMaskSpace(P[0]());


	const MultidimArray<double> mP1=P[0]();
	MultidimArray<double> &mI1filteredp=Ifilteredp[0]();
	double corr1=correlationIndex(mI1filteredp,mP1,&mMask2D);

	switch (num_images)
	{
	case 2:
		cost=-(corr1+corr2+corr3) / 2.0;
		break;
	case 3:
		cost=-(corr1+corr2+corr3) / 3.0;
		break;	
	default:
		cost=-(corr1+corr2+corr3);
		break;
	}

#ifdef DEBUG
	std::cout << "A=" << A << std::endl;
	Image<double> save;
	save()=P();
	save.write("PPPtheo.xmp");
	save()=Ifilteredp();
	save.write("PPPfilteredp.xmp");
	save()=Ifiltered();
	save.write("PPPfiltered.xmp");
	// Vdeformed.write("PPPVdeformed.vol");
	std::cout << "Cost=" << cost << " deformation=" << deformation << std::endl;
	std::cout << "Press any key" << std::endl;
	char c; std::cin >> c;
#endif

   if (showOptimization)
   {
		std::cout << "A1=" << A1 << std::endl;
		Image<double> save;
		// save()=P1();
		// save.write("PPPtheo1.xmp");
		// save()=I1filteredp();
		// save.write("PPPfilteredp1.xmp");
		// save()=I1filtered();
		// save.write("PPPfiltered1.xmp");
		save()=P[0]();
		save.write("PPPtheo1.xmp");
		save()=Ifilteredp[0]();
		save.write("PPPfilteredp1.xmp");
		save()=Ifiltered[0]();
		save.write("PPPfiltered1.xmp");

		switch (num_images)
		{
			case 2:
			{
				std::cout << "A2=" << A2 << std::endl;
				save()=P[1]();
				save.write("PPPtheo2.xmp");
				save()=Ifilteredp[1]();
				save.write("PPPfilteredp2.xmp");
				save()=Ifiltered[1]();
				save.write("PPPfiltered2.xmp");
			}
				break;

			case 3:
			{
				std::cout << "A2=" << A2 << std::endl;
				save()=P[1]();
				save.write("PPPtheo2.xmp");
				save()=Ifilteredp[1]();
				save.write("PPPfilteredp2.xmp");
				save()=Ifiltered[1]();
				save.write("PPPfiltered2.xmp");

				std::cout << "A3=" << A3 << std::endl;
				save()=P[2]();
				save.write("PPPtheo3.xmp");
				save()=Ifilteredp[2]();
				save.write("PPPfilteredp3.xmp");
				save()=Ifiltered[2]();
				save.write("PPPfiltered3.xmp");
			}
				break;
			
			default:
				break;
		}
		Vdeformed.write("PPPVdeformed.vol");
		// std::cout << "Cost=" << cost << " corr1=" << corr1 << " corr2=" << corr2 << std::endl;
		// std::cout << "Cost=" << cost << " corr1=" << corr1 << std::endl;
		std::cout << "Deformation=" << totalDeformation << std::endl;
		std::cout << "Press any key" << std::endl;
		char c; std::cin >> c;
    }

    double massDiff=std::abs(sumV-sumVd)/sumV;
    double retval=cost+lambda*(deformation + massDiff);
	if (showOptimization)
		std::cout << cost << " " << deformation << " " << lambda*deformation << " " << sumV << " " << sumVd << " " << massDiff << " " << retval << std::endl;
	return retval;
}

double continuousZernikeCost(double *x, void *_prm)
{
	ProgForwardZernikeImages *prm=(ProgForwardZernikeImages *)_prm;
    int idx = 3*(prm->vecSize);
	// TODO: Optimize parameters for each image (not sharing)
	// prm->deltaDefocusU[0]=x[idx+6]; prm->deltaDefocusU[1]=x[idx+6]; 
	// prm->deltaDefocusV[0]=x[idx+7]; prm->deltaDefocusV[1]=x[idx+7]; 
	// prm->deltaDefocusAngle[0]=x[idx+8]; prm->deltaDefocusAngle[1]=x[idx+8];

	switch (prm->num_images)
	{
	case 2:
		prm->deltaX[0] = x[idx + 1];
		prm->deltaY[0] = x[idx + 3];
		prm->deltaRot[0] = x[idx + 5];
		prm->deltaTilt[0] = x[idx + 7];
		prm->deltaPsi[0] = x[idx + 9];
		// prm->deltaDefocusU[0]=x[idx + 11];
		// prm->deltaDefocusV[0]=x[idx + 13];
		// prm->deltaDefocusAngle[0]=x[idx + 15];

		prm->deltaX[1] = x[idx + 2];
		prm->deltaY[1] = x[idx + 4];
		prm->deltaRot[1] = x[idx + 6];
		prm->deltaTilt[1] = x[idx + 8];
		prm->deltaPsi[1] = x[idx + 10];
		// prm->deltaDefocusU[1]=x[idx + 12];
		// prm->deltaDefocusV[1]=x[idx + 14];
		// prm->deltaDefocusAngle[1]=x[idx + 16];

		MAT_ELEM(prm->A1, 0, 2) = prm->old_shiftX[0] + prm->deltaX[0];
		MAT_ELEM(prm->A1, 1, 2) = prm->old_shiftY[0] + prm->deltaY[0];
		MAT_ELEM(prm->A1, 0, 0) = 1;
		MAT_ELEM(prm->A1, 0, 1) = 0;
		MAT_ELEM(prm->A1, 1, 0) = 0;
		MAT_ELEM(prm->A1, 1, 1) = 1;

		MAT_ELEM(prm->A2, 0, 2) = prm->old_shiftX[1] + prm->deltaX[1];
		MAT_ELEM(prm->A2, 1, 2) = prm->old_shiftY[1] + prm->deltaY[1];
		MAT_ELEM(prm->A2, 0, 0) = 1;
		MAT_ELEM(prm->A2, 0, 1) = 0;
		MAT_ELEM(prm->A2, 1, 0) = 0;
		MAT_ELEM(prm->A2, 1, 1) = 1;
		break;

	case 3:
		prm->deltaX[0] = x[idx + 1];
		prm->deltaY[0] = x[idx + 4];
		prm->deltaRot[0] = x[idx + 7];
		prm->deltaTilt[0] = x[idx + 10];
		prm->deltaPsi[0] = x[idx + 13];
		// prm->deltaDefocusU[0]=x[idx + 16];
		// prm->deltaDefocusV[0]=x[idx + 19];
		// prm->deltaDefocusAngle[0]=x[idx + 22];

		prm->deltaX[1] = x[idx + 2];
		prm->deltaY[1] = x[idx + 5];
		prm->deltaRot[1] = x[idx + 8];
		prm->deltaTilt[1] = x[idx + 11];
		prm->deltaPsi[1] = x[idx + 14];
		// prm->deltaDefocusU[1]=x[idx + 17];
		// prm->deltaDefocusV[1]=x[idx + 20];
		// prm->deltaDefocusAngle[1]=x[idx + 23];

		prm->deltaX[2] = x[idx + 3];
		prm->deltaY[2] = x[idx + 6];
		prm->deltaRot[2] = x[idx + 9];
		prm->deltaTilt[2] = x[idx + 12];
		prm->deltaPsi[2] = x[idx + 15];
		// prm->deltaDefocusU[2]=x[idx + 18];
		// prm->deltaDefocusV[2]=x[idx + 21];
		// prm->deltaDefocusAngle[2]=x[idx + 24];

		MAT_ELEM(prm->A1, 0, 2) = prm->old_shiftX[0] + prm->deltaX[0];
		MAT_ELEM(prm->A1, 1, 2) = prm->old_shiftY[0] + prm->deltaY[0];
		MAT_ELEM(prm->A1, 0, 0) = 1;
		MAT_ELEM(prm->A1, 0, 1) = 0;
		MAT_ELEM(prm->A1, 1, 0) = 0;
		MAT_ELEM(prm->A1, 1, 1) = 1;

		MAT_ELEM(prm->A2, 0, 2) = prm->old_shiftX[1] + prm->deltaX[1];
		MAT_ELEM(prm->A2, 1, 2) = prm->old_shiftY[1] + prm->deltaY[1];
		MAT_ELEM(prm->A2, 0, 0) = 1;
		MAT_ELEM(prm->A2, 0, 1) = 0;
		MAT_ELEM(prm->A2, 1, 0) = 0;
		MAT_ELEM(prm->A2, 1, 1) = 1;

		MAT_ELEM(prm->A3, 0, 2) = prm->old_shiftX[2] + prm->deltaX[2];
		MAT_ELEM(prm->A3, 1, 2) = prm->old_shiftY[2] + prm->deltaY[2];
		MAT_ELEM(prm->A3, 0, 0) = 1;
		MAT_ELEM(prm->A3, 0, 1) = 0;
		MAT_ELEM(prm->A3, 1, 0) = 0;
		MAT_ELEM(prm->A3, 1, 1) = 1;
		break;

	
	default:
		prm->deltaX[0] = x[idx + 1];
		prm->deltaY[0] = x[idx + 2];
		prm->deltaRot[0] = x[idx + 3];
		prm->deltaTilt[0] = x[idx + 4];
		prm->deltaPsi[0] = x[idx + 5];
		prm->deltaDefocusU[0]=x[idx + 6];
		prm->deltaDefocusV[0]=x[idx + 7];
		prm->deltaDefocusAngle[0]=x[idx + 8];

		MAT_ELEM(prm->A1, 0, 2) = prm->old_shiftX[0] + prm->deltaX[0];
		MAT_ELEM(prm->A1, 1, 2) = prm->old_shiftY[0] + prm->deltaY[0];
		MAT_ELEM(prm->A1, 0, 0) = 1;
		MAT_ELEM(prm->A1, 0, 1) = 0;
		MAT_ELEM(prm->A1, 1, 0) = 0;
		MAT_ELEM(prm->A1, 1, 1) = 1;
		break;
	}

	return prm->transformImageSph(x);

	// return prm->transformImageSph(x,prm->old_rot+deltaRot, prm->old_tilt+deltaTilt, prm->old_psi+deltaPsi,
	// 		prm->A, deltaDefocusU, deltaDefocusV, deltaDefocusAngle);
}

// Predict =================================================================
//#define DEBUG
void ProgForwardZernikeImages::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
    Matrix1D<double> steps;
	// totalSize = 3*num_Z_coeff + num_images * 2 shifts + num_images * 3 angles + num_images * 3 CTF
	algn_params = num_images * 5;
	ctf_params = num_images * 3;
    int totalSize = 3*vecSize + algn_params + ctf_params;
	p.initZeros(totalSize);
	clnm.initZeros(totalSize);

	flagEnabled=1;

	rowIn.getValueOrDefault(MDL_IMAGE,       fnImage[0], "");
	rowIn.getValueOrDefault(MDL_ANGLE_ROT,   old_rot[0], 0.0);
	rowIn.getValueOrDefault(MDL_ANGLE_TILT,  old_tilt[0], 0.0);
	rowIn.getValueOrDefault(MDL_ANGLE_PSI,   old_psi[0], 0.0);
	rowIn.getValueOrDefault(MDL_SHIFT_X,     old_shiftX[0], 0.0);
	rowIn.getValueOrDefault(MDL_SHIFT_Y,     old_shiftY[0], 0.0);
	I[0].read(fnImage[0]); 
	I[0]().setXmippOrigin();
	Ifiltered[0]() = I[0](); 
	filter.applyMaskSpace(Ifiltered[0]());
	
	switch (num_images)
	{
	case 2:
		rowIn.getValueOrDefault(MDL_IMAGE1,      fnImage[1], "");
		rowIn.getValueOrDefault(MDL_ANGLE_ROT2,  old_rot[1], 0.0);
		rowIn.getValueOrDefault(MDL_ANGLE_TILT2, old_tilt[1], 0.0);
		rowIn.getValueOrDefault(MDL_ANGLE_PSI2,  old_psi[1], 0.0);
		rowIn.getValueOrDefault(MDL_SHIFT_X2,    old_shiftX[1], 0.0);
		rowIn.getValueOrDefault(MDL_SHIFT_Y2,    old_shiftY[1], 0.0);
		I[1].read(fnImage[1]);
		I[1]().setXmippOrigin();
		Ifiltered[1]() = I[1](); 
		filter.applyMaskSpace(Ifiltered[1]());

		if (verbose >= 2)
			std::cout << "Processing Pair (" << fnImage[0] << "," << fnImage[1] << ")" << std::endl;
		break;

	case 3:
		rowIn.getValueOrDefault(MDL_IMAGE1,      fnImage[1], "");
		rowIn.getValueOrDefault(MDL_ANGLE_ROT2,  old_rot[1], 0.0);
		rowIn.getValueOrDefault(MDL_ANGLE_TILT2, old_tilt[1], 0.0);
		rowIn.getValueOrDefault(MDL_ANGLE_PSI2,  old_psi[1], 0.0);
		rowIn.getValueOrDefault(MDL_SHIFT_X2,    old_shiftX[1], 0.0);
		rowIn.getValueOrDefault(MDL_SHIFT_Y2,    old_shiftY[1], 0.0);
		I[1].read(fnImage[1]);
		I[1]().setXmippOrigin();
		Ifiltered[1]() = I[1](); 
		filter.applyMaskSpace(Ifiltered[1]());

		rowIn.getValueOrDefault(MDL_IMAGE2,      fnImage[2], "");
		rowIn.getValueOrDefault(MDL_ANGLE_ROT3,  old_rot[2], 0.0);
		rowIn.getValueOrDefault(MDL_ANGLE_TILT3, old_tilt[2], 0.0);
		rowIn.getValueOrDefault(MDL_ANGLE_PSI3,  old_psi[2], 0.0);
		rowIn.getValueOrDefault(MDL_SHIFT_X3,    old_shiftX[2], 0.0);
		rowIn.getValueOrDefault(MDL_SHIFT_Y3,    old_shiftY[2], 0.0);
		I[2].read(fnImage[2]);
		I[2]().setXmippOrigin();
		Ifiltered[2]() = I[2](); 
		filter.applyMaskSpace(Ifiltered[2]());

		if (verbose >= 2)
			std::cout << "Processing Triplet (" << fnImage[0] << "," << fnImage[1] << "," << fnImage[2] << ")" << std::endl;
		break;
	
	default:
		if (verbose >= 2)
			std::cout << "Processing Image (" << fnImage[0] << ")" << std::endl;
		break;
	}

	if (rowIn.containsLabel(MDL_FLIP))
    	rowIn.getValue(MDL_FLIP,old_flip);
	else
		old_flip = false;
	
	// FIXME: Add defocus per image and make CTF correction available
	if ((rowIn.containsLabel(MDL_CTF_DEFOCUSU) || rowIn.containsLabel(MDL_CTF_MODEL)) && useCTF)
	{
		hasCTF=true;
		FilterCTF1.ctf.readFromMdRow(rowIn);
		FilterCTF1.ctf.Tm = Ts;
		FilterCTF1.ctf.produceSideInfo();
		old_defocusU[0]=FilterCTF1.ctf.DeltafU;
		old_defocusV[0]=FilterCTF1.ctf.DeltafV;
		old_defocusAngle[0]=FilterCTF1.ctf.azimuthal_angle;
	}
	else
		hasCTF=false;

	for (int h=1;h<=L2;h++)
	{
		if (verbose>=2)
		{
			std::cout<<std::endl;
			std::cout<<"------------------------------ Basis Degrees: ("<<L1<<","<<h<<") ----------------------------"<<std::endl;
		}
        steps.clear();
        steps.initZeros(totalSize);

		// Optimize
		double cost=-1;
		try
		{
			cost=1e38;
			int iter;
			if (optimizeAlignment)
			{
				int init = steps.size() - algn_params - ctf_params;
				int end = steps.size() - ctf_params;
				for (int i = init; i < end; i++)
					steps(i)=1.;
			}
			if (optimizeDefocus) 
			{
				int init = steps.size() - ctf_params;
				int end = steps.size();
				for (int i = init; i < end; i++)
					steps(i)=1.;
			}
			if (optimizeDeformation)
			{
		        minimizepos(L1,h,steps);
			}
			steps_cp = steps;
			powellOptimizer(p, 1, totalSize, &continuousZernikeCost, this, 0.01, cost, iter, steps, verbose>=2);

			if (verbose>=3)
			{
				showOptimization = true;
				continuousZernikeCost(p.adaptForNumericalRecipes(),this);
				showOptimization = false;
			}

			if (cost>0)
			{
				flagEnabled=-1;
				p.initZeros();
			}
			cost=-cost;
			correlation=cost;
			if (verbose>=2)
			{
				std::cout<<std::endl;
				for (int j=1;j<4;j++)
				{
					switch (j)
					{
					case 1:
						std::cout << "X Coefficients=(";
						break;
					case 2:
						std::cout << "Y Coefficients=(";
						break;
					case 3:
						std::cout << "Z Coefficients=(";
						break;
					}
					for (int i=(j-1)*vecSize;i<j*vecSize;i++)
					{
						std::cout << p(i);
						if (i<j*vecSize-1)
							std::cout << ",";
					}
					std::cout << ")" << std::endl;
				}
                std::cout << "Radius=" << RmaxDef << std::endl;
				std::cout << " Dshift=(" << p(totalSize-5) << "," << p(totalSize-4) << ") "
						  << "Drot=" << p(totalSize-3) << " Dtilt=" << p(totalSize-2) 
                          << " Dpsi=" << p(totalSize-1) << std::endl;
				std::cout << " Total deformation=" << totalDeformation << std::endl;
				std::cout<<std::endl;
			}
		}
		catch (XmippError XE)
		{
			std::cerr << XE << std::endl;
			std::cerr << "Warning: Cannot refine " << fnImg << std::endl;
			flagEnabled=-1;
		}
	}

	if (num_images == 1)
	{
		rotateCoefficients();
	}

	//AJ NEW
	writeImageParameters(rowOut);
	//END AJ

}
#undef DEBUG

void ProgForwardZernikeImages::writeImageParameters(MDRow &row) {
	int pos = 3*vecSize;
	if (flagEnabled==1) {
		row.setValue(MDL_ENABLED, 1);
	}
	else {
		row.setValue(MDL_ENABLED, -1);
	}

	switch (num_images)
	{
	case 2:
		row.setValue(MDL_ANGLE_ROT,   old_rot[0]+p(pos+4));
		row.setValue(MDL_ANGLE_ROT2,  old_rot[1]+p(pos+5));
		row.setValue(MDL_ANGLE_TILT,  old_tilt[0]+p(pos+6));
		row.setValue(MDL_ANGLE_TILT2, old_tilt[1]+p(pos+7));
		row.setValue(MDL_ANGLE_PSI,   old_psi[0]+p(pos+8));
		row.setValue(MDL_ANGLE_PSI2,  old_psi[1]+p(pos+9));
		row.setValue(MDL_SHIFT_X,     old_shiftX[0]+p(pos));
		row.setValue(MDL_SHIFT_X2,    old_shiftX[1]+p(pos+1));
		row.setValue(MDL_SHIFT_Y,     old_shiftY[0]+p(pos+2));
		row.setValue(MDL_SHIFT_Y2,    old_shiftY[1]+p(pos+3));
		break;

	case 3:
		row.setValue(MDL_ANGLE_ROT,   old_rot[0]+p(pos+6));
		row.setValue(MDL_ANGLE_ROT2,  old_rot[1]+p(pos+7));
		row.setValue(MDL_ANGLE_ROT3,  old_rot[2]+p(pos+8));
		row.setValue(MDL_ANGLE_TILT,  old_tilt[0]+p(pos+9));
		row.setValue(MDL_ANGLE_TILT2, old_tilt[1]+p(pos+10));
		row.setValue(MDL_ANGLE_TILT3, old_tilt[2]+p(pos+11));
		row.setValue(MDL_ANGLE_PSI,   old_psi[0]+p(pos+12));
		row.setValue(MDL_ANGLE_PSI2,  old_psi[1]+p(pos+13));
		row.setValue(MDL_ANGLE_PSI3,  old_psi[2]+p(pos+14));
		row.setValue(MDL_SHIFT_X,     old_shiftX[0]+p(pos));
		row.setValue(MDL_SHIFT_X2,    old_shiftX[1]+p(pos+1));
		row.setValue(MDL_SHIFT_X3,    old_shiftX[2]+p(pos+2));
		row.setValue(MDL_SHIFT_Y,     old_shiftY[0]+p(pos+3));
		row.setValue(MDL_SHIFT_Y2,    old_shiftY[1]+p(pos+4));
		row.setValue(MDL_SHIFT_Y3,    old_shiftY[2]+p(pos+5));
		break;
	
	default:
		row.setValue(MDL_ANGLE_ROT,   old_rot[0]+p(pos+2));
		row.setValue(MDL_ANGLE_TILT,  old_tilt[0]+p(pos+3));
		row.setValue(MDL_ANGLE_PSI,   old_psi[0]+p(pos+4));
		row.setValue(MDL_SHIFT_X,     old_shiftX[0]+p(pos));
		row.setValue(MDL_SHIFT_Y,     old_shiftY[0]+p(pos+1));
		break;
	}

	row.setValue(MDL_SPH_DEFORMATION, totalDeformation);
	std::vector<double> vectortemp;
	size_t end_clnm = VEC_XSIZE(clnm)-algn_params-ctf_params;
	for (int j = 0; j < end_clnm; j++) {
		vectortemp.push_back(clnm(j));
	}
	row.setValue(MDL_SPH_COEFFICIENTS, vectortemp);
	row.setValue(MDL_COST, correlation);
}

void ProgForwardZernikeImages::checkPoint() {
	MDRowVec rowAppend;
	MetaDataVec checkPoint;
	getOutputMd().getRow(rowAppend, getOutputMd().lastRowId());
	checkPoint.addRow(rowAppend);
	checkPoint.append(Rerunable::getFileName());
}

void ProgForwardZernikeImages::numCoefficients(int l1, int l2, int &vecSize)
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

void ProgForwardZernikeImages::minimizepos(int L1, int l2, Matrix1D<double> &steps)
{
    int size = 0;
	numCoefficients(L1,l2,size);
    int totalSize = (steps.size()-algn_params-ctf_params)/3;
    for (int idx=0; idx<size; idx++) {
        VEC_ELEM(steps,idx) = 1.;
        VEC_ELEM(steps,idx+totalSize) = 1.;
		if (num_images > 1)
		{
			VEC_ELEM(steps,idx+2*totalSize) = 1.;
		}
    }	
}

void ProgForwardZernikeImages::fillVectorTerms(int l1, int l2, Matrix1D<int> &vL1, Matrix1D<int> &vN, 
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

void ProgForwardZernikeImages::updateCTFImage(double defocusU, double defocusV, double angle)
{
	FilterCTF1.ctf.K=1; // get pure CTF with no envelope
	currentDefocusU[0]=FilterCTF1.ctf.DeltafU=defocusU;
	currentDefocusV[0]=FilterCTF1.ctf.DeltafV=defocusV;
	currentAngle[0]=FilterCTF1.ctf.azimuthal_angle=angle;
	FilterCTF1.ctf.produceSideInfo();
}

void ProgForwardZernikeImages::rotateCoefficients() {
	int pos = 3*vecSize;
	size_t idxY0=(VEC_XSIZE(clnm)-algn_params-ctf_params)/3;
	size_t idxZ0=2*idxY0;

	double rot = old_rot[0]+p(pos+2);
	double tilt = old_tilt[0]+p(pos+3);
	double psi = old_psi[0]+p(pos+4);

	Matrix2D<double> R;
	R.initIdentity(3);
    Euler_angles2matrix(rot, tilt, psi, R, false);
    R = R.inv();
	Matrix1D<double> c;
	c.initZeros(3);
	for (size_t idx=0; idx<idxY0; idx++) {
		XX(c) = VEC_ELEM(clnm,idx); YY(c) = VEC_ELEM(clnm,idx+idxY0); ZZ(c) = VEC_ELEM(clnm,idx+idxZ0);
		c = R * c;
		VEC_ELEM(clnm,idx) = XX(c); VEC_ELEM(clnm,idx+idxY0) = YY(c); VEC_ELEM(clnm,idx+idxZ0) = ZZ(c);
	}
}

void ProgForwardZernikeImages::deformVol(MultidimArray<double> &mP, const MultidimArray<double> &mV, double &def,
                                        double rot, double tilt, double psi)
{
	size_t idxY0=(VEC_XSIZE(clnm)-algn_params-ctf_params)/3;
	double Ncount=0.0;
    double modg=0.0;
	double diff2=0.0;

	def=0.0;
	size_t idxZ0=2*idxY0;
	sumVd=0.0;
	double RmaxF=RmaxDef;
	double RmaxF2=RmaxF*RmaxF;
	double iRmaxF=1.0/RmaxF;
    // Rotation Matrix
    Matrix2D<double> R, R_inv;
    R.initIdentity(3);
    Euler_angles2matrix(rot, tilt, psi, R, false);
    R_inv = R.inv();
    // Matrix1D<double> p, pos, c, c_rot, w;
	// p.initZeros(3);
    // pos.initZeros(3);
	// c.initZeros(3);
	// c_rot.initZeros(3);
	// w.initZeros(8);
	// Vdeformed().initZeros(mV);

	auto stepsMask = std::vector<size_t>();
    for (size_t idx = 0; idx < idxY0; idx++) {
      if (1 == VEC_ELEM(steps_cp, idx)) {
        stepsMask.emplace_back(idx);
      }
	}

	// TODO: Poner primero i y j en el loop, acumular suma y guardar al final
	const auto lastZ = FINISHINGZ(mV);
	const auto lastY = FINISHINGY(mV);
	const auto lastX = FINISHINGX(mV);
	for (int k=STARTINGZ(mV); k<=lastZ; k+=loop_step)
	{
		for (int i=STARTINGY(mV); i<=lastY; i+=loop_step)
		{
			for (int j=STARTINGX(mV); j<=lastX; j+=loop_step)
			{
				if (A3D_ELEM(V_mask,k,i,j) == 1) {
					// ZZ(p) = k; YY(p) = i; XX(p) = j;
					// pos = R_inv * pos;
					// pos = R * pos;
					double gx=0.0, gy=0.0, gz=0.0;
					double k2=k*k;
					double kr=k*iRmaxF;
					double k2i2=k2+i*i;
					double ir=i*iRmaxF;
					double r2=k2i2+j*j;
					double jr=j*iRmaxF;
					double rr=sqrt(r2)*iRmaxF;
					for (auto idx : stepsMask) {
						auto l1 = VEC_ELEM(vL1,idx);
						auto n = VEC_ELEM(vN,idx);
						auto l2 = VEC_ELEM(vL2,idx);
						auto m = VEC_ELEM(vM,idx);
						auto zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
						auto c = std::array<double, 3>{};
						// XX(c_rot) = VEC_ELEM(clnm,idx); YY(c_rot) = VEC_ELEM(clnm,idx+idxY0); ZZ(c_rot) = VEC_ELEM(clnm,idx+idxZ0);
						if (num_images == 1)
						{
							double c_x = VEC_ELEM(clnm,idx);
							double c_y = VEC_ELEM(clnm,idx+idxY0);
							// double c_z = VEC_ELEM(clnm,idx+idxZ0);
							c[0] = R_inv.mdata[0] * c_x + R_inv.mdata[1] * c_y;
							c[1] = R_inv.mdata[3] * c_x + R_inv.mdata[4] * c_y;
							c[2] = R_inv.mdata[6] * c_x + R_inv.mdata[7] * c_y;
						}
						else {
							c[0] = VEC_ELEM(clnm,idx);
							c[1] = VEC_ELEM(clnm,idx+idxY0);
							c[2] = VEC_ELEM(clnm,idx+idxZ0);
						}
						if (rr>0 || l2==0) {
							gx += c[0]  *(zsph);
							gy += c[1]  *(zsph);
							gz += c[2]  *(zsph);
						}
					}
					// XX(p) += gx; YY(p) += gy; ZZ(p) += gz;
					// XX(pos) = 0.0; YY(pos) = 0.0; ZZ(pos) = 0.0;
					// for (size_t i = 0; i < R.mdimy; i++)
					// 	for (size_t j = 0; j < R.mdimx; j++)
					// 		VEC_ELEM(pos, i) += MAT_ELEM(R, i, j) * VEC_ELEM(p, j);

					auto pos = std::array<double, 3>{};
					double r_x = j + gx;
					double r_y = i + gy;
					double r_z = k + gz;
					pos[0] = R.mdata[0] * r_x + R.mdata[1] * r_y + R.mdata[2] * r_z;
					pos[1] = R.mdata[3] * r_x + R.mdata[4] * r_y + R.mdata[5] * r_z;
					pos[2] = R.mdata[6] * r_x + R.mdata[7] * r_y + R.mdata[8] * r_z;
					
					double voxel_mV = A3D_ELEM(mV,k,i,j);
					splattingAtPos(pos, voxel_mV, mP, mV);
					// int x0 = FLOOR(XX(pos));
					// int x1 = x0 + 1;
					// int y0 = FLOOR(YY(pos));
					// int y1 = y0 + 1;
					// int z0 = FLOOR(ZZ(pos));
					// int z1 = z0 + 1;
					// double voxel_mV = A3D_ELEM(mV,k,i,j);
					// w = weightsInterpolation3D(XX(pos),YY(pos),ZZ(pos));
					// if (!mV.outside(z0, y0, x0)) {
					// 	A2D_ELEM(mP,y0,x0) += VEC_ELEM(w,0) * voxel_mV;
					// 	A3D_ELEM(Vdeformed(),z0,y0,x0) += VEC_ELEM(w,0) * voxel_mV;
					// }
					// if (!mV.outside(z1,y0,x0)) {
					// 	A2D_ELEM(mP,y0,x0) += VEC_ELEM(w,1) * voxel_mV;
					// 	A3D_ELEM(Vdeformed(),z1,y0,x0) += VEC_ELEM(w,1) * voxel_mV;
					// }
					// if (!mV.outside(z0,y1,x0)) {
					// 	A2D_ELEM(mP,y1,x0) += VEC_ELEM(w,2) * voxel_mV;
					// 	A3D_ELEM(Vdeformed(),z0,y1,x0) += VEC_ELEM(w,2) * voxel_mV;
					// }
					// if (!mV.outside(z1,y1,x0)) {
					// 	A2D_ELEM(mP,y1,x0) += VEC_ELEM(w,3) * voxel_mV;
					// 	A3D_ELEM(Vdeformed(),z1,y1,x0) += VEC_ELEM(w,3) * voxel_mV;
					// }
					// if (!mV.outside(z0,y0,x1)) {
					// 	A2D_ELEM(mP,y0,x1) += VEC_ELEM(w,4) * voxel_mV;
					// 	A3D_ELEM(Vdeformed(),z0,y0,x1) += VEC_ELEM(w,4) * voxel_mV;
					// }
					// if (!mV.outside(z1,y0,x1)) {
					// 	A2D_ELEM(mP,y0,x1) += VEC_ELEM(w,5) * voxel_mV;
					// 	A3D_ELEM(Vdeformed(),z1,y0,x1) += VEC_ELEM(w,5) * voxel_mV;
					// }
					// if (!mV.outside(z0,y1,x1)) {
					// 	A2D_ELEM(mP,y1,x1) += VEC_ELEM(w,6) * voxel_mV;
					// 	A3D_ELEM(Vdeformed(),z0,y1,x1) += VEC_ELEM(w,6) * voxel_mV;
					// }
					// if (!mV.outside(z1,y1,x1)) {
					// 	A2D_ELEM(mP,y1,x1) += VEC_ELEM(w,7) * voxel_mV;
					// 	A3D_ELEM(Vdeformed(),z1,y1,x1) += VEC_ELEM(w,7) * voxel_mV;
					// }
					sumVd += voxel_mV;
					modg += gx*gx+gy*gy+gz*gz;
					Ncount++;
				}
			}
		}
	}

    // def=sqrt(modg/Ncount);
	def = sqrt(modg/Ncount);
	totalDeformation = def;
}

// void ProgForwardZernikeImages::removePixels() {
// 	for (int i=1+STARTINGY(I1()); i<=FINISHINGY(I1()); i+=loop_step) {
// 		for (int j=1+STARTINGX(I1()); j<=FINISHINGX(I1()); j+=loop_step) {
// 			A2D_ELEM(I1(),i,j) = 0.0;
// 		}
// 	}
// }

Matrix1D<double> ProgForwardZernikeImages::weightsInterpolation3D(double x, double y, double z) {
	Matrix1D<double> w;
	w.initZeros(8);

	int x0 = FLOOR(x);
	double fx0 = x - x0;
	int x1 = x0 + 1;
	double fx1 = x1 - x;

	int y0 = FLOOR(y);
	double fy0 = y - y0;
	int y1 = y0 + 1;
	double fy1 = y1 - y;

	int z0 = FLOOR(z);
	double fz0 = z - z0;
	int z1 = z0 + 1;
	double fz1 = z1 - z;

	VEC_ELEM(w,0) = fx1 * fy1 * fz1;  // w000 (x0,y0,z0)
	VEC_ELEM(w,1) = fx1 * fy1 * fz0;  // w001 (x0,y0,z1)
	VEC_ELEM(w,2) = fx1 * fy0 * fz1;  // w010 (x0,y1,z0)
	VEC_ELEM(w,3) = fx1 * fy0 * fz0;  // w011 (x0,y1,z1)
	VEC_ELEM(w,4) = fx0 * fy1 * fz1;  // w100 (x1,y0,z0)
	VEC_ELEM(w,5) = fx0 * fy1 * fz0;  // w101 (x1,y0,z1)
	VEC_ELEM(w,6) = fx0 * fy0 * fz1;  // w110 (x1,y1,z0)
	VEC_ELEM(w,7) = fx0 * fy0 * fz0;  // w111 (x1,y1,z1)

	return w;
}

void ProgForwardZernikeImages::splattingAtPos(std::array<double, 3> r, double weight, MultidimArray<double> &mP, const MultidimArray<double> &mV) {
	// Find the part of the volume that must be updated
	double x_pos = r[0];
	double y_pos = r[1];
	// double z_pos = r[2];
	// int k0 = XMIPP_MAX(FLOOR(z_pos - blob_r), STARTINGZ(mV));
	// int kF = XMIPP_MIN(CEIL(z_pos + blob_r), FINISHINGZ(mV));
	int i0 = XMIPP_MAX(FLOOR(y_pos - blob_r), STARTINGY(mV));
	int iF = XMIPP_MIN(CEIL(y_pos + blob_r), FINISHINGY(mV));
	int j0 = XMIPP_MAX(FLOOR(x_pos - blob_r), STARTINGX(mV));
	int jF = XMIPP_MIN(CEIL(x_pos + blob_r), FINISHINGX(mV));
	// Perform splatting at this position r
	// ? Probably we can loop only a quarter of the region and use the symmetry to make this faster?
	for (int i = i0; i <= iF; i++)
	{
		double y2 = (y_pos - i) * (y_pos - i);
		for (int j = j0; j <= jF; j++)
		{
			double mod = sqrt((x_pos - j) * (x_pos - j) + y2);
			// A3D_ELEM(Vdeformed(),k, i, j) += weight * blob_val(rdiff.module(), blob);
			A2D_ELEM(mP,i,j) += weight * kaiser_proj(mod, blob.radius, blob.alpha, blob.order);
		}
	}
	// The old version (following commented code) gives slightly different results
	// Matrix1D<double> rdiff(3);
	// for (int k = k0; k <= kF; k++)
	// {
	// 	double k2 = (z_pos - k) * (z_pos - k);
	// 	for (int i = i0; i <= iF; i++)
	// 	{
	// 		double y2 = (y_pos - i) * (y_pos - i);
	// 		for (int j = j0; j <= jF; j++)
	// 		{
	// 			double mod = sqrt((x_pos - j) * (x_pos - j) + y2 + k2);
	// 			// A3D_ELEM(Vdeformed(),k, i, j) += weight * blob_val(rdiff.module(), blob);
	// 			A2D_ELEM(mP, i, j) += weight * kaiser_value(mod, blob.radius, blob.alpha, blob.order);
	// 		}
	// 	}
	// }
}