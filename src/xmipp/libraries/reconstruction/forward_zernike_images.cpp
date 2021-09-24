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
ProgForwardZernikeImages::ProgForwardZernikeImages()
{
	resume = false;
    produces_a_metadata = true;
    each_image_produces_an_output = false;
    ctfImage = NULL;
    showOptimization = false;
}

ProgForwardZernikeImages::~ProgForwardZernikeImages()
{
	delete ctfImage;
}

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
    L1 = getIntParam("--l1");
	L2 = getIntParam("--l2");
	loop_step = getIntParam("--step");
    lambda = getDoubleParam("--regularization");
	resume = checkParam("--resume");
	fnDone = fnOutDir + "/sphDone.xmd";
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
    << "Optimize alignment:  " << optimizeAlignment  << std::endl
    << "Optimize deformation:" << optimizeDeformation<< std::endl
	<< "Optimize defocus;    " << optimizeDefocus    << std::endl
    << "Phase flipped:       " << phaseFlipped       << std::endl
    << "Regularization:      " << lambda             << std::endl
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
    addParamsLine("  [--optimizeAlignment]        : Optimize alignment");
    addParamsLine("  [--optimizeDeformation]      : Optimize deformation");
	addParamsLine("  [--optimizeDefocus]          : Optimize defocus");
    addParamsLine("  [--phaseFlipped]             : Input images have been phase flipped");
    addParamsLine("  [--regularization <l=0.01>]  : Regularization weight");
	addParamsLine("  [--resume]                   : Resume processing");
    addExampleLine("A typical use is:",false);
    addExampleLine("xmipp_angular_sph_alignment -i anglesFromContinuousAssignment.xmd --ref reference.vol -o assigned_anglesAndDeformations.xmd --optimizeAlignment --optimizeDeformation --depth 1");
}

// Produce side information ================================================
void ProgForwardZernikeImages::createWorkFiles() {
	if (resume && fnDone.exists()) {
		MetaDataDb done(fnDone);
		done.read(fnDone);
		getOutputMd() = done;
		auto *candidates = getInputMd();
		MetaDataDb toDo(*candidates);
		toDo.subtraction(done, MDL_IMAGE);
		toDo.write(fnOutDir + "/sphTodo.xmd");
		*candidates = toDo;
	}
}

void ProgForwardZernikeImages::preProcess()
{
    V.read(fnVolR);
    V().setXmippOrigin();
    Xdim=XSIZE(V());
    Vdeformed().initZeros(V());
	Vdeformed().setXmippOrigin();
    // sumV=V().sum();

	// Filter input and reference volumes according to the values of sigma
	FourierFilter filter_v;
    filter_v.FilterShape = REALGAUSSIAN;
    filter_v.FilterBand = LOWPASS;
	filter_v.generateMask(V());
	// This filtered version of the input volume is needed to interpolate more accurately VD
	V_f = V;
	filter_v.w1 = 2;
	filter_v.do_generate_3dmask = true;
	filter_v.applyMaskSpace(V_f());

    Ifilteredp().initZeros(Xdim,Xdim);
    Ifilteredp().setXmippOrigin();

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
    A.initIdentity(3);

	// CTF Filter
	FilterCTF.FilterBand = CTF;
	FilterCTF.ctf.enable_CTFnoise = false;
	FilterCTF.ctf.produceSideInfo();

	vecSize = 0;
	numCoefficients(L1,L2,vecSize);
    fillVectorTerms(L1,L2,vL1,vN,vL2,vM);

    createWorkFiles();
}

void ProgForwardZernikeImages::finishProcessing() {
	XmippMetadataProgram::finishProcessing();
	rename(fnDone.c_str(), fn_out.c_str());
}

// #define DEBUG
double ProgForwardZernikeImages::tranformImageSph(double *pclnm, double rot, double tilt, double psi,
		Matrix2D<double> &A, double deltaDefocusU, double deltaDefocusV, double deltaDefocusAngle)
{
	const MultidimArray<double> &mV=V();
	FOR_ALL_ELEMENTS_IN_MATRIX1D(clnm)
		VEC_ELEM(clnm,i)=pclnm[i+1];
	double deformation=0.0;
	totalDeformation=0.0;
	P().initZeros((int)XSIZE(I()),(int)XSIZE(I()));
    P().setXmippOrigin();
	deformVol(P(), mV, deformation, rot, tilt, psi);
	// P.write("PPPtheo_ori.xmp");
	// Vdeformed.write("PPPVdeformed.vol");
	// std::cout << "Press any key" << std::endl;
	// char c; std::cin >> c;
	if (hasCTF)
    {
    	double defocusU=old_defocusU+deltaDefocusU;
    	double defocusV=old_defocusV+deltaDefocusV;
    	double angle=old_defocusAngle+deltaDefocusAngle;
    	if (defocusU!=currentDefocusU || defocusV!=currentDefocusV || angle!=currentAngle) {
    		updateCTFImage(defocusU,defocusV,angle);
		}
		FilterCTF.ctf = ctf;
		FilterCTF.generateMask(P());
		if (phaseFlipped)
			FilterCTF.correctPhase();
		FilterCTF.applyMaskSpace(P());
	}
    double cost=0;
	if (old_flip)
	{
		MAT_ELEM(A,0,0)*=-1;
		MAT_ELEM(A,0,1)*=-1;
		MAT_ELEM(A,0,2)*=-1;
	}

	applyGeometry(LINEAR,Ifilteredp(),Ifiltered(),A,IS_NOT_INV,DONT_WRAP,0.);
	filter.applyMaskSpace(P());
	const MultidimArray<double> mP=P();
	const MultidimArray<int> &mMask2D=mask2D;
	MultidimArray<double> &mIfilteredp=Ifilteredp();
	double corr=correlationIndex(mIfilteredp,mP,&mMask2D);
	cost=-corr;

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
		save.write("PPPtheo.xmp");
		save()=Ifilteredp();
		save.write("PPPfilteredp.xmp");
		save()=Ifiltered();
		save.write("PPPfiltered.xmp");
		Vdeformed.write("PPPVdeformed.vol");
		std::cout << "Cost=" << cost << " corr=" << corr << std::endl;
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
	double deltax=x[idx+1];
	double deltay=x[idx+2];
	double deltaRot=x[idx+3];
	double deltaTilt=x[idx+4];
	double deltaPsi=x[idx+5];
	double deltaDefocusU=x[idx+6];
	double deltaDefocusV=x[idx+7];
	double deltaDefocusAngle=x[idx+8];
	if (prm->maxShift>0 && deltax*deltax+deltay*deltay>prm->maxShift*prm->maxShift)
		return 1e38;
	if (prm->maxAngularChange>0 && (fabs(deltaRot)>prm->maxAngularChange || fabs(deltaTilt)>prm->maxAngularChange || fabs(deltaPsi)>prm->maxAngularChange))
		return 1e38;

	MAT_ELEM(prm->A,0,2)=prm->old_shiftX+deltax;
	MAT_ELEM(prm->A,1,2)=prm->old_shiftY+deltay;
	MAT_ELEM(prm->A,0,0)=1;
	MAT_ELEM(prm->A,0,1)=0;
	MAT_ELEM(prm->A,1,0)=0;
	MAT_ELEM(prm->A,1,1)=1;

	return prm->tranformImageSph(x,prm->old_rot+deltaRot, prm->old_tilt+deltaTilt, prm->old_psi+deltaPsi,
			prm->A, deltaDefocusU, deltaDefocusV, deltaDefocusAngle);
}

// Predict =================================================================
//#define DEBUG
void ProgForwardZernikeImages::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
    Matrix1D<double> steps;
    int totalSize = 3*vecSize+8;
	p.resize(totalSize);
	clnm.initZeros(totalSize);

	flagEnabled=1;

	rowIn.getValueOrDefault(MDL_ANGLE_ROT, old_rot, 0.0);
	rowIn.getValueOrDefault(MDL_ANGLE_TILT, old_tilt, 0.0);
	rowIn.getValueOrDefault(MDL_ANGLE_PSI, old_psi, 0.0);
	rowIn.getValueOrDefault(MDL_SHIFT_X, old_shiftX, 0.0);
	rowIn.getValueOrDefault(MDL_SHIFT_Y, old_shiftY, 0.0);
	if (rowIn.containsLabel(MDL_FLIP))
    	rowIn.getValue(MDL_FLIP,old_flip);
	else
		old_flip = false;
	
	if ((rowIn.containsLabel(MDL_CTF_DEFOCUSU) || rowIn.containsLabel(MDL_CTF_MODEL)))
	{
		hasCTF=true;
		ctf.readFromMdRow(rowIn);
		ctf.Tm = Ts;
		ctf.produceSideInfo();
		old_defocusU=ctf.DeltafU;
		old_defocusV=ctf.DeltafV;
		old_defocusAngle=ctf.azimuthal_angle;
	}
	else
		hasCTF=false;

	if (verbose>=2)
		std::cout << "Processing " << fnImg << std::endl;
	I.read(fnImg);
	I().setXmippOrigin();
	if (loop_step > 1) {
		removePixels();
	}

	Ifiltered()=I();
	filter.applyMaskSpace(Ifiltered());

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
				steps(totalSize-8)=steps(totalSize-7)=steps(totalSize-6)=steps(totalSize-5)=steps(totalSize-4)=1.;
			if (optimizeDefocus)
				steps(totalSize-3)=steps(totalSize-2)=steps(totalSize-1)=1.;
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

	// rotateCoefficients();

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
	row.setValue(MDL_ANGLE_ROT,   old_rot+p(pos+2));
	row.setValue(MDL_ANGLE_TILT,  old_tilt+p(pos+3));
	row.setValue(MDL_ANGLE_PSI,   old_psi+p(pos+4));
	row.setValue(MDL_SHIFT_X,     old_shiftX+p(pos+0));
	row.setValue(MDL_SHIFT_Y,     old_shiftY+p(pos+1));
	row.setValue(MDL_SPH_DEFORMATION, totalDeformation);
	std::vector<double> vectortemp;
	for (int j = 0; j < VEC_XSIZE(clnm); j++) {
		vectortemp.push_back(clnm(j));
	}
	row.setValue(MDL_SPH_COEFFICIENTS, vectortemp);
	row.setValue(MDL_COST, correlation);
}

void ProgForwardZernikeImages::checkPoint() {
	getOutputMd().append(fnDone);
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
    int totalSize = (steps.size()-8)/3;
    for (int idx=0; idx<size; idx++) {
        VEC_ELEM(steps,idx) = 1.;
        VEC_ELEM(steps,idx+totalSize) = 1.;
        // VEC_ELEM(steps,idx+2*totalSize) = 1.;
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
	ctf.K=1; // get pure CTF with no envelope
	currentDefocusU=ctf.DeltafU=defocusU;
	currentDefocusV=ctf.DeltafV=defocusV;
	currentAngle=ctf.azimuthal_angle=angle;
	ctf.produceSideInfo();
}

void ProgForwardZernikeImages::rotateCoefficients() {
	int pos = 3*vecSize;
	size_t idxY0=(VEC_XSIZE(clnm)-8)/3;
	size_t idxZ0=2*idxY0;

	double rot = old_rot+p(pos+2);
	double tilt = old_tilt+p(pos+3);
	double psi = old_psi+p(pos+4);

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
    int l1,n,l2,m;
	l1 = 0;
	n = 0;
	l2 = 0;
	m = 0;
	size_t idxY0=(VEC_XSIZE(clnm)-8)/3;
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
    Matrix1D<double> pos, c, w;
    pos.initZeros(3);
	c.initZeros(3);
	w.initZeros(8);

	// TODO: Poner primero i y j en el loop, acumular suma y guardar al final
	for (int k=STARTINGZ(mV); k<=FINISHINGZ(mV); k+=loop_step)
	{
		for (int i=STARTINGY(mV); i<=FINISHINGY(mV); i+=loop_step)
		{
			for (int j=STARTINGX(mV); j<=FINISHINGX(mV); j+=loop_step)
			{
				if (A3D_ELEM(V_mask,k,i,j) == 1) {
					ZZ(pos) = k; YY(pos) = i; XX(pos) = j;
					// pos = R_inv * pos;
					double gx=0.0, gy=0.0, gz=0.0;
					// TODO: Sacar al bucle de z
					double k2=ZZ(pos)*ZZ(pos);
					double kr=ZZ(pos)*iRmaxF;
					double k2i2=k2+YY(pos)*YY(pos);
					double ir=YY(pos)*iRmaxF;
					double r2=k2i2+XX(pos)*XX(pos);
					double jr=XX(pos)*iRmaxF;
					double rr=sqrt(r2)*iRmaxF;
					for (size_t idx=0; idx<idxY0; idx++) {
						if (VEC_ELEM(steps_cp,idx) == 1) {
							double zsph=0.0;
							l1 = VEC_ELEM(vL1,idx);
							n = VEC_ELEM(vN,idx);
							l2 = VEC_ELEM(vL2,idx);
							m = VEC_ELEM(vM,idx);
							zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
							XX(c) = VEC_ELEM(clnm,idx); YY(c) = VEC_ELEM(clnm,idx+idxY0); ZZ(c) = VEC_ELEM(clnm,idx+idxZ0);
							c = R_inv * c;
							if (rr>0 || l2==0) {
								gx += XX(c)  *(zsph);
								gy += YY(c)  *(zsph);
								gz += ZZ(c)  *(zsph);
							}
						}
					}
					XX(pos) += gx; YY(pos) += gy; ZZ(pos) += gz;
					pos = R * pos;
					int x0 = FLOOR(XX(pos));
					int x1 = x0 + 1;
					int y0 = FLOOR(YY(pos));
					int y1 = y0 + 1;
					int z0 = FLOOR(ZZ(pos));
					int z1 = z0 + 1;
					double voxel_mV = A3D_ELEM(mV,k,i,j);
					w = weightsInterpolation3D(XX(pos),YY(pos),ZZ(pos));
					if (!mV.outside(z0, y0, x0)) {
						A2D_ELEM(mP,y0,x0) += VEC_ELEM(w,0) * voxel_mV;
						A3D_ELEM(Vdeformed(),z0,y0,x0) += VEC_ELEM(w,0) * voxel_mV;
					}
					if (!mV.outside(z1,y0,x0)) {
						A2D_ELEM(mP,y0,x0) += VEC_ELEM(w,1) * voxel_mV;
						A3D_ELEM(Vdeformed(),z1,y0,x0) += VEC_ELEM(w,1) * voxel_mV;
					}
					if (!mV.outside(z0,y1,x0)) {
						A2D_ELEM(mP,y1,x0) += VEC_ELEM(w,2) * voxel_mV;
						A3D_ELEM(Vdeformed(),z0,y1,x0) += VEC_ELEM(w,2) * voxel_mV;
					}
					if (!mV.outside(z1,y1,x0)) {
						A2D_ELEM(mP,y1,x0) += VEC_ELEM(w,3) * voxel_mV;
						A3D_ELEM(Vdeformed(),z1,y1,x0) += VEC_ELEM(w,3) * voxel_mV;
					}
					if (!mV.outside(z0,y0,x1)) {
						A2D_ELEM(mP,y0,x1) += VEC_ELEM(w,4) * voxel_mV;
						A3D_ELEM(Vdeformed(),z0,y0,x1) += VEC_ELEM(w,4) * voxel_mV;
					}
					if (!mV.outside(z1,y0,x1)) {
						A2D_ELEM(mP,y0,x1) += VEC_ELEM(w,5) * voxel_mV;
						A3D_ELEM(Vdeformed(),z1,y0,x1) += VEC_ELEM(w,5) * voxel_mV;
					}
					if (!mV.outside(z0,y1,x1)) {
						A2D_ELEM(mP,y1,x1) += VEC_ELEM(w,6) * voxel_mV;
						A3D_ELEM(Vdeformed(),z0,y1,x1) += VEC_ELEM(w,6) * voxel_mV;
					}
					if (!mV.outside(z1,y1,x1)) {
						A2D_ELEM(mP,y1,x1) += VEC_ELEM(w,7) * voxel_mV;
						A3D_ELEM(Vdeformed(),z1,y1,x1) += VEC_ELEM(w,7) * voxel_mV;
					}
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

void ProgForwardZernikeImages::removePixels() {
	for (int i=1+STARTINGY(I()); i<=FINISHINGY(I()); i+=loop_step) {
		for (int j=1+STARTINGX(I()); j<=FINISHINGX(I()); j+=loop_step) {
			A2D_ELEM(I(),i,j) = 0.0;
		}
	}
}

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