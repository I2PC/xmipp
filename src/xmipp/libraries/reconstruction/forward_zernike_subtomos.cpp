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

#include "forward_zernike_subtomos.h"
#include "core/transformations.h"
#include "core/xmipp_image_extension.h"
#include "core/xmipp_image_generic.h"
#include "data/projection.h"
#include "data/mask.h"

// Empty constructor =======================================================
ProgForwardZernikeSubtomos::ProgForwardZernikeSubtomos() : Rerunable("")
{
	resume = false;
    produces_a_metadata = true;
    each_image_produces_an_output = false;
    showOptimization = false;
}

ProgForwardZernikeSubtomos::~ProgForwardZernikeSubtomos() = default;

// Read arguments ==========================================================
void ProgForwardZernikeSubtomos::readParams()
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
	resume = checkParam("--resume");
	Rerunable::setFileName(fnOutDir + "/sphDone.xmd");
	blob_r = getDoubleParam("--blobr");
	t1 = getDoubleParam("--t1");
	t2 = getDoubleParam("--t2");
	keep_input_columns = true;
}

// Show ====================================================================
void ProgForwardZernikeSubtomos::show()
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
    ;
}

// usage ===================================================================
void ProgForwardZernikeSubtomos::defineParams()
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
	addParamsLine("  [--t1 <t1=-60>]              : First tilt angle range");
	addParamsLine("  [--t2 <t2=60>]               : Second tilt angle range");
	addParamsLine("  [--resume]                   : Resume processing");
    addExampleLine("A typical use is:",false);
    addExampleLine("xmipp_forward_zernike_subtomos -i anglesFromContinuousAssignment.xmd --ref reference.vol -o assigned_anglesAndDeformations.xmd --optimizeAlignment --optimizeDeformation --depth 1");
}

void ProgForwardZernikeSubtomos::preProcess()
{
    V.read(fnVolR);
    V().setXmippOrigin();
    Xdim=XSIZE(V());
    Vdeformed().initZeros(V());
	Vdeformed().setXmippOrigin();
    // sumV=V().sum();

	Ifilteredp().initZeros(Xdim,Xdim,Xdim);
	Ifilteredp().setXmippOrigin();
	P().initZeros(Xdim,Xdim,Xdim);

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
	sumV = 0;
	for (int k = STARTINGZ(V()); k <= FINISHINGZ(V()); k += loop_step)
	{
		for (int i = STARTINGY(V()); i <= FINISHINGY(V()); i += loop_step)
		{
			for (int j = STARTINGX(V()); j <= FINISHINGX(V()); j += loop_step)
			{
				if (A3D_ELEM(V_mask, k, i, j) == 1)
				{
					sumV++;
				}
			}
		}
	}

	// Construct mask
    if (Rmax<0)
    	Rmax=Xdim/2;
    mask.R1 = Rmax;
    mask.generate_mask(Xdim,Xdim,Xdim);
    mask3D=mask.get_binary_mask();

    // Low pass filter
    filter.FilterBand=LOWPASS;
	filter.do_generate_3dmask = true;
    filter.w1=Ts/maxResol;
    filter.raised_w=0.02;

	// MW filter
	// filterMW.FilterBand=WEDGE;
	// filterMW.FilterShape=WEDGE;
	// filterMW.do_generate_3dmask = true;
	// filterMW.t1 = t1;
	// filterMW.t2 = t2;
	// filterMW.rot=filterMW.tilt=filterMW.psi=0.0;
	filterMW.FilterBand=LOWPASS;
	filterMW.FilterShape=WEDGE_RC;
	filterMW.do_generate_3dmask = true;
	filterMW.t1 = t1;
	filterMW.t2 = t2;
	filterMW.rot=filterMW.tilt=filterMW.psi=0.0;
	filterMW.w1=Ts/maxResol;
    filterMW.raised_w=0.02;

    // Transformation matrix
    A.initIdentity(4);

	// CTF Filter
	FilterCTF.FilterBand = CTF;
	FilterCTF.do_generate_3dmask = true;
	FilterCTF.ctf.enable_CTFnoise = false;
	FilterCTF.ctf.produceSideInfo();

	vecSize = 0;
	numCoefficients(L1,L2,vecSize);
    fillVectorTerms(L1,L2,vL1,vN,vL2,vM);

    createWorkFiles();

	// Blob
	blob.radius = blob_r;   // Blob radius in voxels
	blob.order  = 2;        // Order of the Bessel function
    blob.alpha  = 7.05;     // Smoothness parameter

}

void ProgForwardZernikeSubtomos::finishProcessing() {
	XmippMetadataProgram::finishProcessing();
	rename(Rerunable::getFileName().c_str(), (fnOutDir + fn_out).c_str());
}

// #define DEBUG
double ProgForwardZernikeSubtomos::transformImageSph(double *pclnm)
{
	const MultidimArray<double> &mV=V();
	idx_z_clnm.clear();
	z_clnm_diff.clear();
	FOR_ALL_ELEMENTS_IN_MATRIX1D(clnm)
	{
		VEC_ELEM(clnm,i)=pclnm[i+1];
		if (VEC_ELEM(clnm,i) != VEC_ELEM(prev_clnm,i))
		{
			idx_z_clnm.push_back(i);
			z_clnm_diff.push_back(VEC_ELEM(clnm, i) - VEC_ELEM(prev_clnm, i));
			// std::cout << i << std::endl;
		}
	}
	double deformation=0.0;
	totalDeformation=0.0;

	P().initZeros(Xdim,Xdim,Xdim);
	P().setXmippOrigin();
	double currentRot=old_rot + deltaRot;
	double currentTilt=old_tilt + deltaTilt;
	double currentPsi=old_psi + deltaPsi;
	deformVol(P(), mV, deformation, currentRot, currentTilt, currentPsi);

	double cost=0.0;
	const MultidimArray<int> &mMask3D=mask3D;

	if (old_flip)
	{
		MAT_ELEM(A, 0, 0) *= -1;
		MAT_ELEM(A, 0, 1) *= -1;
		MAT_ELEM(A, 0, 2) *= -1;
	}

	// Image<double> save;
	// save()=Ifiltered();
	// save.write("PPPbefore.mrc");

	applyGeometry(xmipp_transformation::LINEAR,Ifilteredp(),Ifiltered(),A,
					xmipp_transformation::IS_NOT_INV,xmipp_transformation::DONT_WRAP,0.);

	// Image<double> save;
	// save()=P();
	// save.write("PPPtheo.mrc");

	// save()=Ifilteredp();
	// save.write("PPPexp.mrc");

	// std::cout << "Press any key" << std::endl;
	// char c; std::cin >> c;

	filterMW.generateMask(P());
	filterMW.applyMaskSpace(P());

	if (hasCTF)
    {
    	double defocusU=old_defocusU+deltaDefocusU;
    	double defocusV=old_defocusV+deltaDefocusV;
    	double angle=old_defocusAngle+deltaDefocusAngle;
    	if (defocusU!=currentDefocusU || defocusV!=currentDefocusV || angle!=currentAngle) {
    		updateCTFImage(defocusU,defocusV,angle);
		}
		// FilterCTF1.ctf = ctf;
		FilterCTF.generateMask(P());
		if (phaseFlipped)
			FilterCTF.correctPhase();
		FilterCTF.applyMaskSpace(P());
	}
	// filter.applyMaskSpace(P());


	const MultidimArray<double> &mP=P();
	MultidimArray<double> &mIfilteredp=Ifilteredp();
	double corr1=correlationIndex(mIfilteredp,mP,&mMask3D);

	cost=-corr1;

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
		std::cout << "A=" << A << std::endl;
		Image<double> save;
		save()=P();
		save.write("PPPtheo.mrc");
		save()=Ifilteredp();
		save.write("PPPfilteredp.mrc");
		save()=Ifiltered();
		save.write("PPPfiltered.mrc");
		Vdeformed.write("PPPVdeformed.vol");
		std::cout << "Deformation=" << totalDeformation << std::endl;
		std::cout << "Press any key" << std::endl;
		char c; std::cin >> c;
    }

    // double massDiff=std::abs(sumV-sumVd)/sumV;
	prev_clnm = clnm;
    double retval=cost+lambda*abs(deformation - prior_deformation);
	if (showOptimization)
		std::cout << cost << " " << deformation << " " << lambda*deformation << " " << sumV << " " << retval << std::endl;
	return retval;
}

double continuousZernikeSubtomoCost(double *x, void *_prm)
{
	ProgForwardZernikeSubtomos *prm=(ProgForwardZernikeSubtomos *)_prm;
    int idx = 3*(prm->vecSize);
	// TODO: Optimize parameters for each image (not sharing)
	// prm->deltaDefocusU[0]=x[idx+6]; prm->deltaDefocusU[1]=x[idx+6]; 
	// prm->deltaDefocusV[0]=x[idx+7]; prm->deltaDefocusV[1]=x[idx+7]; 
	// prm->deltaDefocusAngle[0]=x[idx+8]; prm->deltaDefocusAngle[1]=x[idx+8];

	prm->deltaX = x[idx + 1];
	prm->deltaY = x[idx + 2];
	prm->deltaZ = x[idx + 3];
	prm->deltaRot = x[idx + 4];
	prm->deltaTilt = x[idx + 5];
	prm->deltaPsi = x[idx + 6];
	prm->deltaDefocusU=x[idx + 7];
	prm->deltaDefocusV=x[idx + 8];
	prm->deltaDefocusAngle=x[idx + 9];

	MAT_ELEM(prm->A, 0, 3) = prm->old_shiftX + prm->deltaX;
	MAT_ELEM(prm->A, 1, 3) = prm->old_shiftY + prm->deltaY;
	MAT_ELEM(prm->A, 2, 3) = prm->old_shiftZ + prm->deltaZ;
	MAT_ELEM(prm->A, 0, 0) = 1;
	MAT_ELEM(prm->A, 0, 1) = 0;
	MAT_ELEM(prm->A, 0, 2) = 0;
	MAT_ELEM(prm->A, 1, 0) = 0;
	MAT_ELEM(prm->A, 1, 1) = 1;
	MAT_ELEM(prm->A, 1, 2) = 0;
	MAT_ELEM(prm->A, 2, 0) = 0;
	MAT_ELEM(prm->A, 2, 1) = 0;
	MAT_ELEM(prm->A, 2, 2) = 1;
	// MAT_ELEM(prm->A, 3, 0) = 0;
	// MAT_ELEM(prm->A, 3, 1) = 0;
	// MAT_ELEM(prm->A, 3, 2) = 0;
	// MAT_ELEM(prm->A, 3, 3) = 1;

	return prm->transformImageSph(x);

	// return prm->transformImageSph(x,prm->old_rot+deltaRot, prm->old_tilt+deltaTilt, prm->old_psi+deltaPsi,
	// 		prm->A, deltaDefocusU, deltaDefocusV, deltaDefocusAngle);
}

// Predict =================================================================
//#define DEBUG
void ProgForwardZernikeSubtomos::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
    Matrix1D<double> steps;
	// totalSize = 3*num_Z_coeff + 3 shifts + 3 angles + 3 CTF
    int totalSize = 3*vecSize + 9;
	p.initZeros(totalSize);
	clnm.initZeros(totalSize);
	prev_clnm.initZeros(totalSize);

	// Init positions and deformation field
	vpos.initZeros(sumV, 8);
	df.initZeros(sumV, 3);

	flagEnabled=1;

	rowIn.getValueOrDefault(MDL_IMAGE,       fnImage, "");
	rowIn.getValueOrDefault(MDL_ANGLE_ROT,   old_rot, 0.0);
	rowIn.getValueOrDefault(MDL_ANGLE_TILT,  old_tilt, 0.0);
	rowIn.getValueOrDefault(MDL_ANGLE_PSI,   old_psi, 0.0);
	rowIn.getValueOrDefault(MDL_SHIFT_X,     old_shiftX, 0.0);
	rowIn.getValueOrDefault(MDL_SHIFT_Y,     old_shiftY, 0.0);
	rowIn.getValueOrDefault(MDL_SHIFT_Z,     old_shiftZ, 0.0);
	I.read(fnImage); 
	I().setXmippOrigin();
	Ifiltered() = I(); 
	filter.applyMaskSpace(Ifiltered());
	rotatePositions(old_rot, old_tilt, old_psi);
	
	if (verbose >= 2)
		std::cout << "Processing Image (" << fnImage << ")" << std::endl;

	if (rowIn.containsLabel(MDL_FLIP))
    	rowIn.getValue(MDL_FLIP,old_flip);
	else
		old_flip = false;

	prior_deformation = 0.0;
	if (rowIn.containsLabel(MDL_SPH_COEFFICIENTS))
	{
		std::vector<double> vectortemp;
		rowIn.getValue(MDL_SPH_COEFFICIENTS, vectortemp);
		rowIn.getValueOrDefault(MDL_SPH_DEFORMATION, prior_deformation, 0.0);
		for (int i=0; i<3*vecSize; i++)
		{
			clnm[i] = vectortemp[i];
		}
		// if (optimizeDeformation)
		// 	rotateCoefficients<Direction::ROTATE>();
		p = clnm;
		prev_clnm = clnm;
		preComputeDF();
	}	
	
	// FIXME: Add defocus per image and make CTF correction available
	if ((rowIn.containsLabel(MDL_CTF_DEFOCUSU) || rowIn.containsLabel(MDL_CTF_MODEL)) && useCTF)
	{
		hasCTF=true;
		FilterCTF.ctf.readFromMdRow(rowIn);
		FilterCTF.ctf.Tm = Ts;
		FilterCTF.ctf.produceSideInfo();
		old_defocusU=FilterCTF.ctf.DeltafU;
		old_defocusV=FilterCTF.ctf.DeltafV;
		old_defocusAngle=FilterCTF.ctf.azimuthal_angle;
	}
	else
		hasCTF=false;

	// If deformation is not optimized, do a single iteration
	//? Si usamos priors es mejor ir poco a poco, ir poco a poco pero usar todos los coeffs cada vez (mas lento)
	//? O dar solo una iteracion con todos los coeffs?
	int h = 1;
	// if (!optimizeDeformation)
	// if (rowIn.containsLabel(MDL_SPH_COEFFICIENTS))
	// 	h = L2;

	for (;h<=L2;h++)
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
		correlation = 0.0;
		try
		{
			cost=1e38;
			int iter;
			if (optimizeAlignment)
			{
				int init = steps.size() - 9;
				int end = steps.size() - 3;
				for (int i = init; i < end; i++)
					steps(i)=1.;
			}
			if (optimizeDefocus) 
			{
				int init = steps.size() - 3;
				int end = steps.size();
				for (int i = init; i < end; i++)
					steps(i)=1.;
			}
			if (optimizeDeformation)
			{
		        minimizepos(L1,h,steps);
			}
			steps_cp = steps;
			powellOptimizer(p, 1, totalSize, &continuousZernikeSubtomoCost, this, 0.1, cost, iter, steps, verbose>=2);

			if (verbose>=3)
			{
				showOptimization = true;
				continuousZernikeSubtomoCost(p.adaptForNumericalRecipes(),this);
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
		catch (XmippError &XE)
		{
			std::cerr << XE.what() << std::endl;
			std::cerr << "Warning: Cannot refine " << fnImg << std::endl;
			flagEnabled=-1;
		}
	}

	// if (optimizeDeformation)
	// {
	// 	rotateCoefficients<Direction::UNROTATE>();
	// }

	//AJ NEW
	writeImageParameters(rowOut);
	//END AJ

}
#undef DEBUG

void ProgForwardZernikeSubtomos::writeImageParameters(MDRow &row) {
	int pos = 3*vecSize;
	if (flagEnabled==1) {
		row.setValue(MDL_ENABLED, 1);
	}
	else {
		row.setValue(MDL_ENABLED, -1);
	}
	
	row.setValue(MDL_ANGLE_ROT,   old_rot+p(pos+3));
	row.setValue(MDL_ANGLE_TILT,  old_tilt+p(pos+4));
	row.setValue(MDL_ANGLE_PSI,   old_psi+p(pos+5));
	row.setValue(MDL_SHIFT_X,     old_shiftX+p(pos));
	row.setValue(MDL_SHIFT_Y,     old_shiftY+p(pos+1));
	row.setValue(MDL_SHIFT_Z,     old_shiftZ+p(pos+2));

	row.setValue(MDL_SPH_DEFORMATION, totalDeformation);
	std::vector<double> vectortemp;
	size_t end_clnm = VEC_XSIZE(clnm)-9;
	for (int j = 0; j < end_clnm; j++) {
		vectortemp.push_back(clnm(j));
	}
	row.setValue(MDL_SPH_COEFFICIENTS, vectortemp);
	// row.setValue(MDL_COST, correlation);
}

void ProgForwardZernikeSubtomos::checkPoint() {
	MDRowVec rowAppend;
	MetaDataVec checkPoint;
	getOutputMd().getRow(rowAppend, getOutputMd().lastRowId());
	checkPoint.addRow(rowAppend);
	checkPoint.append(Rerunable::getFileName());
}

void ProgForwardZernikeSubtomos::numCoefficients(int l1, int l2, int &vecSize)
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

void ProgForwardZernikeSubtomos::minimizepos(int L1, int l2, Matrix1D<double> &steps)
{
    int size = 0;
	int prevSize = 0;
	numCoefficients(L1,l2,size);
	numCoefficients(L1,l2-1,prevSize);
    int totalSize = (steps.size()-9)/3;
	if (l2 > 1)
	{
		for (int idx = prevSize; idx < size; idx++)
		{
			VEC_ELEM(steps, idx) = 1.;
			VEC_ELEM(steps, idx + totalSize) = 1.;
			VEC_ELEM(steps, idx + 2 * totalSize) = 1.;
		}
	}
	else
	{
		for (int idx = 0; idx < size; idx++)
		{
			VEC_ELEM(steps, idx) = 1.;
			VEC_ELEM(steps, idx + totalSize) = 1.;
			VEC_ELEM(steps, idx + 2 * totalSize) = 1.;
		}
	}
}

void ProgForwardZernikeSubtomos::fillVectorTerms(int l1, int l2, Matrix1D<int> &vL1, Matrix1D<int> &vN, 
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

void ProgForwardZernikeSubtomos::updateCTFImage(double defocusU, double defocusV, double angle)
{
	FilterCTF.ctf.K=1; // get pure CTF with no envelope
	currentDefocusU=FilterCTF.ctf.DeltafU=defocusU;
	currentDefocusV=FilterCTF.ctf.DeltafV=defocusV;
	currentAngle=FilterCTF.ctf.azimuthal_angle=angle;
	FilterCTF.ctf.produceSideInfo();
}

template<ProgForwardZernikeSubtomos::Direction DIRECTION>
void ProgForwardZernikeSubtomos::rotateCoefficients() {
	int pos = 3*vecSize;
	size_t idxY0=(VEC_XSIZE(clnm)-9)/3;
	size_t idxZ0=2*idxY0;

	double rot = old_rot+p(pos+2);
	double tilt = old_tilt+p(pos+3);
	double psi = old_psi+p(pos+4);

	Matrix2D<double> R;
	R.initIdentity(3);
    Euler_angles2matrix(rot, tilt, psi, R, false);
	if (DIRECTION == Direction::UNROTATE)
    	R = R.inv();
	Matrix1D<double> c;
	c.initZeros(3);
	for (size_t idx=0; idx<idxY0; idx++) {
		XX(c) = VEC_ELEM(clnm,idx); YY(c) = VEC_ELEM(clnm,idx+idxY0); ZZ(c) = VEC_ELEM(clnm,idx+idxZ0);
		c = R * c;
		VEC_ELEM(clnm,idx) = XX(c); VEC_ELEM(clnm,idx+idxY0) = YY(c); VEC_ELEM(clnm,idx+idxZ0) = ZZ(c);
	}
}

void ProgForwardZernikeSubtomos::deformVol(MultidimArray<double> &mP, const MultidimArray<double> &mV, double &def,
                                        double rot, double tilt, double psi)
{
	size_t idxY0=(VEC_XSIZE(clnm)-9)/3;
	double Ncount=0.0;
    double modg=0.0;
	double diff2=0.0;

	def=0.0;
	size_t idxZ0=2*idxY0;
	// sumVd=0.0;
	double RmaxF=RmaxDef;
	double RmaxF2=RmaxF*RmaxF;
	double iRmaxF=1.0/RmaxF;
    // Rotation Matrix
    // Matrix2D<double> R, R_inv;
	// Matrix2D<double> R;
    // R.initIdentity(3);
    // Euler_angles2matrix(rot, tilt, psi, R, false);
	// Matrix2D<double> R_inv = R.inv();
    // R_inv = R.inv();

	// auto stepsMask = std::vector<size_t>();
	// if (optimizeDeformation)
	// {
	// 	for (size_t idx = 0; idx < idxY0; idx++)
	// 	{
	// 		if (1 == VEC_ELEM(steps_cp, idx))
	// 		{
	// 			stepsMask.emplace_back(idx);
	// 		}
	// 	}
	// }
	// else {
	// 	for (size_t idx = 0; idx < idxY0; idx++)
	// 	{
	// 		stepsMask.emplace_back(idx);
	// 	}
	// }

	auto sz = idx_z_clnm.size();
	Matrix1D<int> l1, l2, n, m, idx_v;

	if (!idx_z_clnm.empty())
	{
		l1.initZeros(sz);
		l2.initZeros(sz);
		n.initZeros(sz);
		m.initZeros(sz);
		idx_v.initZeros(sz);
		for (auto j=0; j<sz; j++)
		{
			auto idx = idx_z_clnm[j];
			if (idx >= idxY0 && idx < idxZ0)
				idx -= idxY0;
			else if (idx >= idxZ0)
				idx -= idxZ0;

			VEC_ELEM(idx_v,j) = idx;
			VEC_ELEM(l1,j) = VEC_ELEM(vL1, idx);
			VEC_ELEM(n,j) = VEC_ELEM(vN, idx);
			VEC_ELEM(l2,j) = VEC_ELEM(vL2, idx);
			VEC_ELEM(m,j) = VEC_ELEM(vM, idx);
		}
	}

	// // TODO: Poner primero i y j en el loop, acumular suma y guardar al final
	// const auto lastZ = FINISHINGZ(mV);
	// const auto lastY = FINISHINGY(mV);
	// const auto lastX = FINISHINGX(mV);
	// for (int k=STARTINGZ(mV); k<=lastZ; k+=loop_step)
	// {
	// 	for (int i=STARTINGY(mV); i<=lastY; i+=loop_step)
	// 	{
	// 		for (int j=STARTINGX(mV); j<=lastX; j+=loop_step)
	// 		{
	// 			if (A3D_ELEM(V_mask,k,i,j) == 1) {
	// 				// ZZ(p) = k; YY(p) = i; XX(p) = j;
	// 				// pos = R_inv * pos;
	// 				// pos = R * pos;
	// 				double gx=0.0, gy=0.0, gz=0.0;
	// 				double k2=k*k;
	// 				double kr=k*iRmaxF;
	// 				double k2i2=k2+i*i;
	// 				double ir=i*iRmaxF;
	// 				double r2=k2i2+j*j;
	// 				double jr=j*iRmaxF;
	// 				double rr=sqrt(r2)*iRmaxF;
	// 				for (auto idx : stepsMask) {
	// 					auto l1 = VEC_ELEM(vL1,idx);
	// 					auto n = VEC_ELEM(vN,idx);
	// 					auto l2 = VEC_ELEM(vL2,idx);
	// 					auto m = VEC_ELEM(vM,idx);
	// 					auto zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
	// 					auto c = std::array<double, 3>{};
	// 					// XX(c_rot) = VEC_ELEM(clnm,idx); YY(c_rot) = VEC_ELEM(clnm,idx+idxY0); ZZ(c_rot) = VEC_ELEM(clnm,idx+idxZ0);
	// 					// if (num_images == 1 && optimizeDeformation)
	// 					// {
	// 					// 	//? Hacer algun check para ahorrarnos cuentas si no usamos priors (en ese 
	// 					// 	//? caso podemos usar las lineas comentadas)
	// 					// 	double c_x = VEC_ELEM(clnm,idx);
	// 					// 	double c_y = VEC_ELEM(clnm,idx+idxY0);
	// 					// 	// c[0] = R_inv.mdata[0] * c_x + R_inv.mdata[1] * c_y;
	// 					// 	// c[1] = R_inv.mdata[3] * c_x + R_inv.mdata[4] * c_y;
	// 					// 	// c[2] = R_inv.mdata[6] * c_x + R_inv.mdata[7] * c_y;
	// 					// 	double c_z = VEC_ELEM(clnm, idx + idxZ0);
	// 					// 	c[0] = R_inv.mdata[0] * c_x + R_inv.mdata[1] * c_y + R_inv.mdata[2] * c_z;
	// 					// 	c[1] = R_inv.mdata[3] * c_x + R_inv.mdata[4] * c_y + R_inv.mdata[5] * c_z;
	// 					// 	c[2] = R_inv.mdata[6] * c_x + R_inv.mdata[7] * c_y + R_inv.mdata[8] * c_z;
	// 					// }
	// 					// else {
	// 					c[0] = VEC_ELEM(clnm,idx);
	// 					c[1] = VEC_ELEM(clnm,idx+idxY0);
	// 					c[2] = VEC_ELEM(clnm,idx+idxZ0);
	// 					// }
	// 					if (rr>0 || l2==0) {
	// 						gx += c[0]  *(zsph);
	// 						gy += c[1]  *(zsph);
	// 						gz += c[2]  *(zsph);
	// 					}
	// 				}

	// 				auto pos = std::array<double, 3>{};
	// 				double r_x = j + gx;
	// 				double r_y = i + gy;
	// 				double r_z = k + gz;
	// 				pos[0] = R.mdata[0] * r_x + R.mdata[1] * r_y + R.mdata[2] * r_z;
	// 				pos[1] = R.mdata[3] * r_x + R.mdata[4] * r_y + R.mdata[5] * r_z;
	// 				pos[2] = R.mdata[6] * r_x + R.mdata[7] * r_y + R.mdata[8] * r_z;
					
	// 				double voxel_mV = A3D_ELEM(mV,k,i,j);
	// 				splattingAtPos(pos, voxel_mV, mP, mV);
	// 				if (!mV.outside(pos[2], pos[1], pos[0]))
	// 					sumVd += voxel_mV;
	// 				modg += gx*gx+gy*gy+gz*gz;
	// 				Ncount++;
	// 			}
	// 		}
	// 	}
	// }

	const auto &mVpos = vpos;
	const auto lastY = FINISHINGY(mVpos);
	for (int i=STARTINGY(mVpos); i<=lastY; i++)
	{
		double &gx = A2D_ELEM(df, i, 0);
		double &gy = A2D_ELEM(df, i, 1);
		double &gz = A2D_ELEM(df, i, 2);
		double r_x = A2D_ELEM(mVpos, i, 0);
		double r_y = A2D_ELEM(mVpos, i, 1);
		double r_z = A2D_ELEM(mVpos, i, 2);
		double xr = A2D_ELEM(mVpos, i, 3);
		double yr = A2D_ELEM(mVpos, i, 4);
		double zr = A2D_ELEM(mVpos, i, 5);
		double rr = A2D_ELEM(mVpos, i, 6);

		if (!idx_z_clnm.empty())
		{
			for (auto j = 0; j < sz; j++)
			{	
				auto idx = VEC_ELEM(idx_v, j);
				auto aux_l2 = VEC_ELEM(l2, j);
				auto zsph = ZernikeSphericalHarmonics(VEC_ELEM(l1, j), VEC_ELEM(n, j),
													  aux_l2, VEC_ELEM(m, j), xr, yr, zr, rr);

				auto diff_c_x = VEC_ELEM(clnm, idx) - VEC_ELEM(prev_clnm, idx);
				auto diff_c_y = VEC_ELEM(clnm, idx + idxY0) - VEC_ELEM(prev_clnm, idx + idxY0);
				auto diff_c_z = VEC_ELEM(clnm, idx + idxZ0) - VEC_ELEM(prev_clnm, idx + idxZ0);
				// auto i_diff_c_x = R_inv.mdata[0] * diff_c_x + R_inv.mdata[1] * diff_c_y;
				// auto i_diff_c_y = R_inv.mdata[3] * diff_c_x + R_inv.mdata[4] * diff_c_y;
				// auto i_diff_c_z = R_inv.mdata[6] * diff_c_x + R_inv.mdata[7] * diff_c_y;
				if (rr > 0 || aux_l2 == 0)
				{
					gx += diff_c_x * (zsph);
					gy += diff_c_y * (zsph);
					gz += diff_c_z * (zsph);
				}
			}
		}

		auto r_gx = R.mdata[0] * gx + R.mdata[1] * gy + R.mdata[2] * gz;
		auto r_gy = R.mdata[3] * gx + R.mdata[4] * gy + R.mdata[5] * gz;
		auto r_gz = R.mdata[6] * gx + R.mdata[7] * gy + R.mdata[8] * gz;

		auto pos = std::array<double, 3>{};
		pos[0] = r_x + r_gx;
		pos[1] = r_y + r_gy;
		pos[2] = r_z + r_gz;

		double voxel_mV = A2D_ELEM(mVpos, i, 7);
		splattingAtPos(pos, voxel_mV, mP, mV);

		modg += gx * gx + gy * gy + gz * gz;
		Ncount++;
	}

    // def=sqrt(modg/Ncount);
	def = sqrt(modg/Ncount);
	totalDeformation = def;
}

Matrix1D<double> ProgForwardZernikeSubtomos::weightsInterpolation3D(double x, double y, double z) {
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

void ProgForwardZernikeSubtomos::splattingAtPos(std::array<double, 3> r, double weight, MultidimArray<double> &mP, const MultidimArray<double> &mV) {
	// Find the part of the volume that must be updated
	// double x_pos = r[0];
	// double y_pos = r[1];
	// double z_pos = r[2];
	// // int k0 = XMIPP_MAX(FLOOR(z_pos - blob_r), STARTINGZ(mV));
	// // int kF = XMIPP_MIN(CEIL(z_pos + blob_r), FINISHINGZ(mV));
	// // int i0 = XMIPP_MAX(FLOOR(y_pos - blob_r), STARTINGY(mV));
	// // int iF = XMIPP_MIN(CEIL(y_pos + blob_r), FINISHINGY(mV));
	// // int j0 = XMIPP_MAX(FLOOR(x_pos - blob_r), STARTINGX(mV));
	// // int jF = XMIPP_MIN(CEIL(x_pos + blob_r), FINISHINGX(mV));
	// int k0 = XMIPP_MAX(FLOOR(z_pos - sigma4), STARTINGZ(mV));
	// int kF = XMIPP_MIN(CEIL(z_pos + sigma4), FINISHINGZ(mV));
	// int i0 = XMIPP_MAX(FLOOR(y_pos - sigma4), STARTINGY(mV));
	// int iF = XMIPP_MIN(CEIL(y_pos + sigma4), FINISHINGY(mV));
	// int j0 = XMIPP_MAX(FLOOR(x_pos - sigma4), STARTINGX(mV));
	// int jF = XMIPP_MIN(CEIL(x_pos + sigma4), FINISHINGX(mV));
	// int size = gaussianProjectionTable.size();
	// for (int k = k0; k <= kF; k++)
	// {
	// 	double k2 = (z_pos - k) * (z_pos - k);
	// 	for (int i = i0; i <= iF; i++)
	// 	{
	// 		double y2 = (y_pos - i) * (y_pos - i);
	// 		for (int j = j0; j <= jF; j++)
	// 		{
	// 			// double mod = sqrt((x_pos - j) * (x_pos - j) + y2 + k2);
	// 			// // A3D_ELEM(Vdeformed(),k, i, j) += weight * blob_val(rdiff.module(), blob);
	// 			// A3D_ELEM(mP, k, i, j) += weight * kaiser_value(mod, blob.radius, blob.alpha, blob.order);
	// 			double mod = sqrt((x_pos - j) * (x_pos - j) + y2 + k2);
	// 			double didx = mod * 1000;
	// 			int idx = ROUND(didx);
	// 			if (idx < size)
	// 			{
	// 				double gw = gaussianProjectionTable.vdata[idx];
	// 				A3D_ELEM(mP, k, i, j) += weight * gw;
	// 			}
	// 		}
	// 	}
	// }

	double x_pos = r[0];
	double y_pos = r[1];
	double z_pos = r[2];
	int i0 = XMIPP_MAX(CEIL(y_pos - loop_step), STARTINGY(mV));
	int iF = XMIPP_MIN(FLOOR(y_pos + loop_step), FINISHINGY(mV));
	int j0 = XMIPP_MAX(CEIL(x_pos - loop_step), STARTINGX(mV));
	int jF = XMIPP_MIN(FLOOR(x_pos + loop_step), FINISHINGX(mV));
	int k0 = XMIPP_MAX(CEIL(z_pos - loop_step), STARTINGZ(mV));
	int kF = XMIPP_MIN(FLOOR(z_pos + loop_step), FINISHINGZ(mV));

	double m = 1. / loop_step;
	for (int k = k0; k <= kF; k++)
	{
		double z_val = 1. - m * ABS(k - z_pos);
		for (int i = i0; i <= iF; i++)
		{
			double y_val = 1. - m * ABS(i - y_pos);
			for (int j = j0; j <= jF; j++)
			{
				double x_val = 1. - m * ABS(j - x_pos);
				A3D_ELEM(mP, k, i, j) += weight * x_val * y_val * z_val;
			}
		}
	}
}

void ProgForwardZernikeSubtomos::rotatePositions(double rot, double tilt, double psi) 
{
	// Matrix2D<double> R;
    R.initIdentity(3);
    Euler_angles2matrix(rot, tilt, psi, R, false);

	const MultidimArray<double> &mV=V();

	const auto lastZ = FINISHINGZ(mV);
	const auto lastY = FINISHINGY(mV);
	const auto lastX = FINISHINGX(mV);
	size_t count = 0;
	double iRmaxF=1.0/RmaxDef;
	for (int k=STARTINGZ(mV); k<=lastZ; k+=loop_step)
	{
		for (int i=STARTINGY(mV); i<=lastY; i+=loop_step)
		{
			for (int j=STARTINGX(mV); j<=lastX; j+=loop_step)
			{
				if (A3D_ELEM(V_mask,k,i,j) == 1) {
					double x = j;
					double y = i;
					double z = k;
					double r_x = R.mdata[0] * x + R.mdata[1] * y + R.mdata[2] * z;
					double r_y = R.mdata[3] * x + R.mdata[4] * y + R.mdata[5] * z;
					double r_z = R.mdata[6] * x + R.mdata[7] * y + R.mdata[8] * z;

					A2D_ELEM(vpos, count, 0) = r_x;
					A2D_ELEM(vpos, count, 1) = r_y;
					A2D_ELEM(vpos, count, 2) = r_z;
					A2D_ELEM(vpos, count, 3) = j * iRmaxF;
					A2D_ELEM(vpos, count, 4) = i * iRmaxF;
					A2D_ELEM(vpos, count, 5) = z * iRmaxF;
					A2D_ELEM(vpos, count, 6) = sqrt(x*x + y*y + z*z) * iRmaxF;
					A2D_ELEM(vpos, count, 7) = A3D_ELEM(mV, k, i, j);

					count++;
				}
			}
		}
	}
} 


void ProgForwardZernikeSubtomos::preComputeDF()
{	
	size_t idxY0=(VEC_XSIZE(clnm)-9)/3;
	size_t idxZ0=2*idxY0;
	// Matrix2D<double> R_inv = R.inv();
	const auto &mVpos = vpos;
	const auto lastY = FINISHINGY(mVpos);
	for (int i=STARTINGY(mVpos); i<=lastY; i++)
	{
		double &gx = A2D_ELEM(df, i, 0);
		double &gy = A2D_ELEM(df, i, 1);
		double &gz = A2D_ELEM(df, i, 2);
		double r_x = A2D_ELEM(mVpos, i, 0);
		double r_y = A2D_ELEM(mVpos, i, 1);
		double r_z = A2D_ELEM(mVpos, i, 2);
		double xr = A2D_ELEM(mVpos, i, 3);
		double yr = A2D_ELEM(mVpos, i, 4);
		double zr = A2D_ELEM(mVpos, i, 5);
		double rr = A2D_ELEM(mVpos, i, 6);

		if (!idx_z_clnm.empty())
		{
			for (int idx = 0; idx < idxY0; idx++)
			{
				auto aux_l2 = VEC_ELEM(vL2, idx);
				auto zsph = ZernikeSphericalHarmonics(VEC_ELEM(vL1, idx), VEC_ELEM(vN, idx),
													  aux_l2, VEC_ELEM(vM, idx), xr, yr, zr, rr);
				auto c_x = VEC_ELEM(clnm, idx);
				auto c_y = VEC_ELEM(clnm, idx + idxY0);
				auto c_z = VEC_ELEM(clnm, idx + idxZ0);
				// auto i_c_x = R_inv.mdata[0] * c_x + R_inv.mdata[1] * c_y + R_inv.mdata[2] * c_z;
				// auto i_c_y = R_inv.mdata[3] * c_x + R_inv.mdata[4] * c_y + R_inv.mdata[5] * c_z;
				// auto i_c_z = R_inv.mdata[6] * c_x + R_inv.mdata[7] * c_y + R_inv.mdata[8] * c_z;
				if (rr > 0 || aux_l2 == 0)
				{
					gx += c_x * (zsph);
					gy += c_y * (zsph);
					gz += c_z * (zsph);
				}
			}
		}
	}	
}