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

#include "angular_sph_alignment.h"
#include "core/transformations.h"
#include "core/xmipp_image_extension.h"
#include "core/xmipp_image_generic.h"
#include "data/projection.h"
#include "data/mask.h"

// Empty constructor =======================================================
ProgAngularSphAlignment::ProgAngularSphAlignment() : Rerunable("") {
  resume = false;
  produces_a_metadata = true;
  each_image_produces_an_output = false;
  showOptimization = false;
}

ProgAngularSphAlignment::~ProgAngularSphAlignment() = default;

// Read arguments ==========================================================
void ProgAngularSphAlignment::readParams()
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
    lambda = getDoubleParam("--regularization");
	resume = checkParam("--resume");
	Rerunable::setFileName(fnOutDir + "/sphDone.xmd");
}

// Show ====================================================================
void ProgAngularSphAlignment::show()
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
    << "Optimize alignment:  " << optimizeAlignment  << std::endl
    << "Optimize deformation:" << optimizeDeformation<< std::endl
	<< "Optimize defocus;    " << optimizeDefocus    << std::endl
    << "Phase flipped:       " << phaseFlipped       << std::endl
    << "Regularization:      " << lambda             << std::endl
    ;
}

// usage ===================================================================
void ProgAngularSphAlignment::defineParams()
{
    addUsageLine("Make a continuous angular assignment with deformations");
	defaultComments["-i"].clear();
	defaultComments["-i"].addComment("Metadata with initial alignment");
	defaultComments["-o"].clear();
	defaultComments["-o"].addComment("Metadata with the angular alignment and deformation parameters");
    XmippMetadataProgram::defineParams();
    addParamsLine("   --ref <volume>              : Reference volume");
	addParamsLine("  [--mask <m=\"\">]            : Reference volume mask");
	addParamsLine("  [--odir <outputDir=\".\">]   : Output directory");
    addParamsLine("  [--max_shift <s=-1>]         : Maximum shift allowed in pixels");
    addParamsLine("  [--max_angular_change <a=5>] : Maximum angular change allowed (in degrees)");
    addParamsLine("  [--max_resolution <f=4>]     : Maximum resolution (A)");
    addParamsLine("  [--sampling <Ts=1>]          : Sampling rate (A/pixel)");
    addParamsLine("  [--Rmax <R=-1>]              : Maximum radius (px). -1=Half of volume size");
    addParamsLine("  [--RDef <r=-1>]              : Maximum radius of the deformation (px). -1=Half of volume size");
    addParamsLine("  [--l1 <l1=3>]                : Degree Zernike Polynomials=1,2,3,...");
	addParamsLine("  [--l2 <l2=2>]                : Harmonical depth of the deformation=1,2,3,...");
    addParamsLine("  [--optimizeAlignment]        : Optimize alignment");
    addParamsLine("  [--optimizeDeformation]      : Optimize deformation");
	addParamsLine("  [--optimizeDefocus]          : Optimize defocus");
    addParamsLine("  [--phaseFlipped]             : Input images have been phase flipped");
    addParamsLine("  [--regularization <l=0.01>]  : Regularization weight");
	addParamsLine("  [--resume]                   : Resume processing");
    addExampleLine("A typical use is:",false);
    addExampleLine("xmipp_angular_sph_alignment -i anglesFromContinuousAssignment.xmd --ref reference.vol -o assigned_anglesAndDeformations.xmd --optimizeAlignment --optimizeDeformation --depth 1");
}

void ProgAngularSphAlignment::preProcess()
{
    V.read(fnVolR);
    V().setXmippOrigin();
    Xdim=XSIZE(V());
    Vdeformed().initZeros(V());

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
    fillVectorTerms(L1,L2);

    createWorkFiles();
}

void ProgAngularSphAlignment::finishProcessing() {
	XmippMetadataProgram::finishProcessing();
	rename(Rerunable::getFileName().c_str(), fn_out.c_str());
}

// #define DEBUG
double ProgAngularSphAlignment::tranformImageSph(double *pclnm, double rot, double tilt, double psi,
		double deltaDefocusU, double deltaDefocusV, double deltaDefocusAngle)
{
	const MultidimArray<double> &mV=V();
	FOR_ALL_ELEMENTS_IN_MATRIX1D(clnm)
		VEC_ELEM(clnm,i)=pclnm[i+1];
	double deformation=0.0;
	totalDeformation=0.0;
	P().initZeros((int)XSIZE(I()),(int)XSIZE(I()));
    P().setXmippOrigin();
	deformVol(P(), mV, deformation, rot, tilt, psi);
	if (hasCTF) {
		applyCTFImage(deltaDefocusU, deltaDefocusV, deltaDefocusAngle);
	}
    double cost=0;
	if (old_flip)
	{
		MAT_ELEM(A,0,0)*=-1;
		MAT_ELEM(A,0,1)*=-1;
		MAT_ELEM(A,0,2)*=-1;
	}

	applyGeometry(xmipp_transformation::LINEAR,Ifilteredp(),Ifiltered(),A,xmipp_transformation::IS_NOT_INV,xmipp_transformation::DONT_WRAP,0.);
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
		Image<double> save(P);
		save.write("PPPtheo.xmp");
		save()=Ifilteredp();
		save.write("PPPfilteredp.xmp");
		save()=Ifiltered();
		save.write("PPPfiltered.xmp");
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

double continuousSphCost(double *x, void *_prm)
{
	ProgAngularSphAlignment *prm=(ProgAngularSphAlignment *)_prm;
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
			deltaDefocusU, deltaDefocusV, deltaDefocusAngle);
}

// Predict =================================================================
//#define DEBUG
void ProgAngularSphAlignment::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
    Matrix1D<double> steps;
    int totalSize = 3*vecSize+8;
	p.initZeros(totalSize);
	clnm.initZeros(totalSize);

	rowOut=rowIn;

	flagEnabled=1;

	rowIn.getValue(MDL_ANGLE_ROT,old_rot);
	rowIn.getValue(MDL_ANGLE_TILT,old_tilt);
	rowIn.getValue(MDL_ANGLE_PSI,old_psi);
	rowIn.getValue(MDL_SHIFT_X,old_shiftX);
	rowIn.getValue(MDL_SHIFT_Y,old_shiftY);
	if (rowIn.containsLabel(MDL_FLIP))
    	rowIn.getValue(MDL_FLIP,old_flip);
	else
		old_flip = false;
	
	if (rowIn.containsLabel(MDL_CTF_DEFOCUSU) || rowIn.containsLabel(MDL_CTF_MODEL))
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
		        minimizepos(h,steps);
			}
			steps_cp = steps;
			powellOptimizer(p, 1, totalSize, &continuousSphCost, this, 0.01, cost, iter, steps, verbose>=2);

			if (verbose>=3)
			{
				showOptimization = true;
				continuousSphCost(p.adaptForNumericalRecipes(),this);
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
			std::cerr << XE << std::endl;
			std::cerr << "Warning: Cannot refine " << fnImg << std::endl;
			flagEnabled=-1;
		}
	}

		//AJ NEW
		writeImageParameters(fnImg);
		//END AJ

}
#undef DEBUG

void ProgAngularSphAlignment::writeImageParameters(const FileName &fnImg) {
	MetaDataVec md;
    int pos = 3*vecSize;
	size_t objId = md.addObject();
	md.setValue(MDL_IMAGE, fnImg, objId);
	if (flagEnabled==1) {
		md.setValue(MDL_ENABLED, 1, objId);
	}
	else {
		md.setValue(MDL_ENABLED, -1, objId);
	}
	md.setValue(MDL_ANGLE_ROT,   old_rot+p(pos+2), objId);
	md.setValue(MDL_ANGLE_TILT,  old_tilt+p(pos+3), objId);
	md.setValue(MDL_ANGLE_PSI,   old_psi+p(pos+4), objId);
	md.setValue(MDL_SHIFT_X,     old_shiftX+p(pos+0), objId);
	md.setValue(MDL_SHIFT_Y,     old_shiftY+p(pos+1), objId);
	md.setValue(MDL_FLIP,        old_flip, objId);
	md.setValue(MDL_SPH_DEFORMATION, totalDeformation, objId);
	std::vector<double> vectortemp;
	for (int j = 0; j < VEC_XSIZE(clnm); j++) {
		vectortemp.push_back(clnm(j));
	}
	md.setValue(MDL_SPH_COEFFICIENTS, vectortemp, objId);
	md.setValue(MDL_COST,        correlation, objId);
	md.append(Rerunable::getFileName());
}

void ProgAngularSphAlignment::numCoefficients(int l1, int l2, int &nc) const
{	
	// l1 -> Degree Zernike
	// l2 & h --> Degree SPH
    for (int h=0; h<=l2; h++)
    {
		// For the current SPH degree (h), determine the number of SPH components/equations
        int numSPH = 2*h+1;
		// Finf the total number of radial components with even degree for a given l1 and h
        int count=l1-h+1;
        int numEven=(count>>1)+(count&1 && !(h&1));
		// Total number of components is the number of SPH as many times as Zernike components
        if (h%2 == 0) {
            nc += numSPH*numEven;
		}
        else {
        	nc += numSPH*(count-numEven);
		}
    }
}

void ProgAngularSphAlignment::minimizepos(int l2, Matrix1D<double> &steps) const
{
    int size = 0;
	numCoefficients(L1,l2,size);
    auto totalSize = (int)((steps.size()-8)/3);
    for (int idx=0; idx<size; idx++) {
        VEC_ELEM(steps,idx) = 1.;
        VEC_ELEM(steps,idx+totalSize) = 1.;
        VEC_ELEM(steps,idx+2*totalSize) = 1.;
    }	
}

void ProgAngularSphAlignment::fillVectorTerms(int l1, int l2)
{
    int idx = 0;
	vL1.initZeros(vecSize);
	vN.initZeros(vecSize);
	vL2.initZeros(vecSize);
	vM.initZeros(vecSize);
    for (int h=0; h<=l2; h++) {
        int totalSPH = 2*h+1;
        auto aux = (int)(std::floor(totalSPH/2));
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

void ProgAngularSphAlignment::applyCTFImage(double const &deltaDefocusU, double const &deltaDefocusV, 
											double const &deltaDefocusAngle) {
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

void ProgAngularSphAlignment::updateCTFImage(double defocusU, double defocusV, double angle)
{
	ctf.K=1; // get pure CTF with no envelope
	currentDefocusU=ctf.DeltafU=defocusU;
	currentDefocusV=ctf.DeltafV=defocusV;
	currentAngle=ctf.azimuthal_angle=angle;
	ctf.produceSideInfo();
}

void ProgAngularSphAlignment::deformVol(MultidimArray<double> &mP, const MultidimArray<double> &mV, double &def,
                                        double rot, double tilt, double psi)
{
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
    Matrix2D<double> R;
    R.initIdentity(3);
    Euler_angles2matrix(rot, tilt, psi, R, false);
    R = R.inv();
    Matrix1D<double> pos;
    pos.initZeros(3);

	// TODO: Poner primero i y j en el loop, acumular suma y guardar al final
	for (int k=STARTINGZ(mV); k<=FINISHINGZ(mV); k++)
	{
		for (int i=STARTINGY(mV); i<=FINISHINGY(mV); i++)
		{
			for (int j=STARTINGX(mV); j<=FINISHINGX(mV); j++)
			{
                ZZ(pos) = k; YY(pos) = i; XX(pos) = j;
                pos = R * pos;
				double gx=0.0;
				double gy=0.0;
				double gz=0.0;
				// TODO: Sacar al bucle de z
				double k2=ZZ(pos)*ZZ(pos);
				double kr=ZZ(pos)*iRmaxF;
				double k2i2=k2+YY(pos)*YY(pos);
				double ir=YY(pos)*iRmaxF;
				double r2=k2i2+XX(pos)*XX(pos);
				double jr=XX(pos)*iRmaxF;
				double rr=sqrt(r2)*iRmaxF;
				if (r2<RmaxF2) {
					for (size_t idx=0; idx<idxY0; idx++) {
						if (VEC_ELEM(steps_cp,idx) == 1) {
							double zsph=0.0;
							int l1 = VEC_ELEM(vL1,idx);
							int n = VEC_ELEM(vN,idx);
							int l2 = VEC_ELEM(vL2,idx);
							int m = VEC_ELEM(vM,idx);
							zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
							if (rr>0 || l2==0) {
								gx += VEC_ELEM(clnm,idx)        *zsph;
								gy += VEC_ELEM(clnm,idx+idxY0)  *zsph;
								gz += VEC_ELEM(clnm,idx+idxZ0)  *zsph;
							}
						}
					}
					auto k_mask = (int)(ZZ(pos)+gz);
					auto i_mask = (int)(YY(pos)+gy);
					auto j_mask = (int)(XX(pos)+gx);
					int voxelI_mask = 0;
					if (!V_mask.outside(k_mask, i_mask, j_mask)) {
						voxelI_mask = A3D_ELEM(V_mask, k_mask, i_mask, j_mask);
					}
					if (voxelI_mask == 1) {
						double voxelI=mV.interpolatedElement3D(XX(pos)+gx,YY(pos)+gy,ZZ(pos)+gz);
						A2D_ELEM(mP,i,j) += voxelI;
						sumVd += voxelI;
						modg += gx*gx+gy*gy+gz*gz;
						Ncount++;
					}
				}
			}
		}
	}

	def = sqrt(modg/Ncount);
	totalDeformation = def;
}

