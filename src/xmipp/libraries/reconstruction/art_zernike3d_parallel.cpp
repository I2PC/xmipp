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

#include "art_zernike3d_parallel.h"
#include "core/transformations.h"
#include "core/xmipp_image_extension.h"
#include "core/xmipp_image_generic.h"
#include "data/projection.h"
#include "data/mask.h"
#include "data/cpu.h"

#define FORWARD_ART   1
#define BACKWARD_ART -1

// Empty constructor =======================================================
ProgArtZernike3DParallel::ProgArtZernike3DParallel()
{
	resume = false;
    produces_a_metadata = true;
    each_image_produces_an_output = false;
    ctfImage = NULL;
    showOptimization = false;
}

ProgArtZernike3DParallel::~ProgArtZernike3DParallel()
{
	delete ctfImage;
}

// Read arguments ==========================================================
void ProgArtZernike3DParallel::readParams()
{
	XmippMetadataProgram::readParams();
	fnVolR = getParam("--ref");
	fnOutDir = getParam("--odir");
    RmaxDef = getIntParam("--RDef");
    phaseFlipped = checkParam("--phaseFlipped");
	Ts = getDoubleParam("--sampling");
    L1 = getIntParam("--l1");
	L2 = getIntParam("--l2");
    lambda = getDoubleParam("--regularization");
	resume = checkParam("--resume");
	fnDone = fnOutDir + "/sphDone.xmd";
	fnVolO = fnOutDir + "/Refined.vol";
	int threads = getIntParam("--thr");
    if (0 >= threads) {
        threads = CPU::findCores();
    }
    m_threadPool.resize(threads);
	keep_input_columns = true;
}

// Show ====================================================================
void ProgArtZernike3DParallel::show()
{
    if (!verbose)
        return;
	XmippMetadataProgram::show();
    std::cout
    << "Output directory:    " << fnOutDir 			 << std::endl
    << "Reference volume:    " << fnVolR             << std::endl
	<< "Sampling:            " << Ts                 << std::endl
    << "Max. Radius Deform.  " << RmaxDef            << std::endl
    << "Zernike Degree:      " << L1                 << std::endl
    << "SH Degree:           " << L2                 << std::endl
    << "Phase flipped:       " << phaseFlipped       << std::endl
    << "Regularization:      " << lambda             << std::endl
    ;
}

// usage ===================================================================
void ProgArtZernike3DParallel::defineParams()
{
    addUsageLine("Template-based canonical volume reconstruction through Zernike3D coefficients");
	defaultComments["-i"].clear();
	defaultComments["-i"].addComment("Metadata with initial alignment");
	defaultComments["-o"].clear();
	defaultComments["-o"].addComment("Refined volume");
    XmippMetadataProgram::defineParams();
    addParamsLine("   --ref <volume>              : Reference volume");
	addParamsLine("  [--odir <outputDir=\".\">]   : Output directory");
	addParamsLine("  [--sampling <Ts=1>]          : Sampling rate (A/pixel)");
    addParamsLine("  [--RDef <r=-1>]              : Maximum radius of the deformation (px). -1=Half of volume size");
    addParamsLine("  [--l1 <l1=3>]                : Degree Zernike Polynomials=1,2,3,...");
	addParamsLine("  [--l2 <l2=2>]                : Harmonical depth of the deformation=1,2,3,...");
    addParamsLine("  [--phaseFlipped]             : Input images have been phase flipped");
    addParamsLine("  [--regularization <l=0.01>]  : ART regularization weight");
	addParamsLine("  [--resume]                   : Resume processing");
	addParamsLine("  [--thr <N=-1>]                      : Maximal number of the processing CPU threads");
    addExampleLine("A typical use is:",false);
    addExampleLine("xmipp_art_zernike3d -i anglesFromContinuousAssignment.xmd --ref reference.vol -o assigned_anglesAndDeformations.xmd --l1 3 --l2 2");
}

// Produce side information ================================================
void ProgArtZernike3DParallel::createWorkFiles() {
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

void ProgArtZernike3DParallel::preProcess()
{
    V.read(fnVolR);
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
	FilterCTF.FilterBand = CTF;
	FilterCTF.ctf.enable_CTFnoise = false;
	FilterCTF.ctf.produceSideInfo();

	vecSize = 0;
	numCoefficients(L1,L2,vecSize);
    fillVectorTerms(L1,L2,vL1,vN,vL2,vM);

    createWorkFiles();
}

void ProgArtZernike3DParallel::finishProcessing() {
	XmippMetadataProgram::finishProcessing();
	Vrefined.write(fnVolO);
}

// #define DEBUG
void ProgArtZernike3DParallel::forwardModel()
{
	// deformVol(P(), W(), mV, rot, tilt, psi, k);
	// if (hasCTF)
    // {
	// 	updateCTFImage(defocusU,defocusV,defocusAngle);
	// 	FilterCTF.ctf = ctf;
	// 	FilterCTF.generateMask(P());
	// 	if (phaseFlipped)
	// 		FilterCTF.correctPhase();
	// 	FilterCTF.applyMaskSpace(P());
	// }
	if (flip)
	{
		MAT_ELEM(A,0,0)*=-1;
		MAT_ELEM(A,0,1)*=-1;
		MAT_ELEM(A,0,2)*=-1;
	}

	applyGeometry(LINEAR,I_shifted(),I(),A,IS_NOT_INV,DONT_WRAP,0.);
	
	// Compute difference image and divide by weights
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(I()){
		if (DIRECT_A2D_ELEM(W(), i, j) != 0) {
			DIRECT_A2D_ELEM(Idiff(), i, j) = lambda * (DIRECT_A2D_ELEM(I_shifted(), i, j) - DIRECT_A2D_ELEM(P(), i, j)) / DIRECT_A2D_ELEM(W(), i, j);
		}
		// } else {
		// 	DIRECT_A2D_ELEM(Idiff(), i, j) = lambda * (DIRECT_A2D_ELEM(I_shifted(), i, j) - DIRECT_A2D_ELEM(P(), i, j));
		// }
	}
}

void ProgArtZernike3DParallel::updateART(int k) {
	int l1,n,l2,m;
	double r_x, r_y, r_z;
	const auto mV = V();
	r_x = 0.0;
	r_y = 0.0;
	r_z = 0.0;
	l1 = 0;
	n = 0;
	l2 = 0;
	m = 0;
	size_t idxY0=VEC_XSIZE(clnm)/3;
	size_t idxZ0=2*idxY0;
	double RmaxF=RmaxDef;
	double RmaxF2=RmaxF*RmaxF;
	double iRmaxF=1.0/RmaxF;
    // Rotation Matrix
    Matrix2D<double> R;
    R.initIdentity(3);
    Euler_angles2matrix(rot, tilt, psi, R, false);
    R = R.inv();
    Matrix1D<double> pos, w;
    pos.initZeros(3);
	w.initZeros(8);

	for (int i=STARTINGY(mV); i<=FINISHINGY(mV); i++)
	{
		for (int j=STARTINGX(mV); j<=FINISHINGX(mV); j++)
		{
			ZZ(pos) = k; YY(pos) = i; XX(pos) = j;
			pos = R * pos;
			double gx=0.0, gy=0.0, gz=0.0;
			double k2=ZZ(pos)*ZZ(pos);
			double kr=ZZ(pos)*iRmaxF;
			double k2i2=k2+YY(pos)*YY(pos);
			double ir=YY(pos)*iRmaxF;
			double r2=k2i2+XX(pos)*XX(pos);
			double jr=XX(pos)*iRmaxF;
			double rr=sqrt(r2)*iRmaxF;
			if (r2<RmaxF2) {
				for (size_t idx=0; idx<idxY0; idx++) {
					double zsph=0.0;
					l1 = VEC_ELEM(vL1,idx);
					n = VEC_ELEM(vN,idx);
					l2 = VEC_ELEM(vL2,idx);
					m = VEC_ELEM(vM,idx);
					zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
					if (rr>0 || l2==0) {
						gx += VEC_ELEM(clnm,idx)        *(zsph);
						gy += VEC_ELEM(clnm,idx+idxY0)  *(zsph);
						gz += VEC_ELEM(clnm,idx+idxZ0)  *(zsph);
					}
				}
			}
			r_x = XX(pos)+gx; r_y = YY(pos)+gy; r_z = ZZ(pos)+gz;
			int x0 = FLOOR(r_x);
			int x1 = x0 + 1;
			int y0 = FLOOR(r_y);
			int y1 = y0 + 1;
			int z0 = FLOOR(r_z);
			int z1 = z0 + 1;
			double Idiff_val = A2D_ELEM(Idiff(),i,j);
			w = weightsInterpolation3D(r_x,r_y,r_z);
			if (!Vrefined().outside(z0, y0, x0))
				A3D_ELEM(Vrefined(),z0,y0,x0) += Idiff_val * VEC_ELEM(w,0);
			if (!Vrefined().outside(z1,y0,x0))
				A3D_ELEM(Vrefined(),z1,y0,x0) += Idiff_val * VEC_ELEM(w,1);
			if (!Vrefined().outside(z0,y1,x0))
				A3D_ELEM(Vrefined(),z0,y1,x0) += Idiff_val * VEC_ELEM(w,2);
			if (!Vrefined().outside(z1,y1,x0))
				A3D_ELEM(Vrefined(),z1,y1,x0) += Idiff_val * VEC_ELEM(w,3);
			if (!Vrefined().outside(z0,y0,x1))
				A3D_ELEM(Vrefined(),z0,y0,x1) += Idiff_val * VEC_ELEM(w,4);
			if (!Vrefined().outside(z1,y0,x1))
				A3D_ELEM(Vrefined(),z1,y0,x1) += Idiff_val * VEC_ELEM(w,5);
			if (!Vrefined().outside(z0,y1,x1))
				A3D_ELEM(Vrefined(),z0,y1,x1) += Idiff_val * VEC_ELEM(w,6);
			if (!Vrefined().outside(z1,y1,x1))
				A3D_ELEM(Vrefined(),z1,y1,x1) += Idiff_val * VEC_ELEM(w,7);
		}
	}
}

void ProgArtZernike3DParallel::parallelART(int direction) {
    // parallelize at the level of Z slices
    const auto &mV = V();
    auto futures = std::vector<std::future<void>>();
    futures.reserve(mV.zdim);
    auto forward_routine = [this](int thrId, int k) {
		deformVol(k);
    };
	auto backward_routine = [this](int thrId, int k) {
		updateART(k);
    };
    for (int k=STARTINGZ(mV); k<=FINISHINGZ(mV); ++k) {
		if (direction == BACKWARD_ART)
			futures.emplace_back(m_threadPool.push(backward_routine, k));
		else
			futures.emplace_back(m_threadPool.push(forward_routine, k));
    }
    // wait till all slices are processed
    for (auto &f : futures) {
        f.get();
    }
}


// Predict =================================================================
//#define DEBUG
void ProgArtZernike3DParallel::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
	rowOut=rowIn;
	flagEnabled=1;

	rowIn.getValue(MDL_ANGLE_ROT,rot);
	rowIn.getValue(MDL_ANGLE_TILT,tilt);
	rowIn.getValue(MDL_ANGLE_PSI,psi);
	rowIn.getValue(MDL_SHIFT_X,shiftX);
	rowIn.getValue(MDL_SHIFT_Y,shiftY);
	std::vector<double> vectortemp;
	vectortemp.clear();
	rowIn.getValue(MDL_SPH_COEFFICIENTS,vectortemp);
	clnm.initZeros(vectortemp.size()-8);
	for(int i=0; i < vectortemp.size()-8; i++){
   		VEC_ELEM(clnm,i) = vectortemp[i];
	}
	removeOverdeformation();
	if (rowIn.containsLabel(MDL_FLIP))
    	rowIn.getValue(MDL_FLIP,flip);
	else
		flip = false;
	
	if ((rowIn.containsLabel(MDL_CTF_DEFOCUSU) || rowIn.containsLabel(MDL_CTF_MODEL)))
	{
		hasCTF=true;
		ctf.readFromMdRow(rowIn);
		ctf.Tm = Ts;
		ctf.produceSideInfo();
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
	P().initZeros((int)XSIZE(I()),(int)XSIZE(I()));
	P().setXmippOrigin();
	W().initZeros((int)XSIZE(I()),(int)XSIZE(I()));
	W().setXmippOrigin();
	Idiff().initZeros((int)XSIZE(I()),(int)XSIZE(I()));
	Idiff().setXmippOrigin();
	I_shifted().initZeros((int)XSIZE(I()),(int)XSIZE(I()));
	I_shifted().setXmippOrigin();

	// Forward Model
	parallelART(FORWARD_ART);
	forwardModel();

	// ART update
	parallelART(BACKWARD_ART);

}
#undef DEBUG

void ProgArtZernike3DParallel::checkPoint() {
	getOutputMd().write(fnDone);
	Vrefined.write(fnVolO);
}

void ProgArtZernike3DParallel::numCoefficients(int l1, int l2, int &vecSize)
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

void ProgArtZernike3DParallel::fillVectorTerms(int l1, int l2, Matrix1D<int> &vL1, Matrix1D<int> &vN, 
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

void ProgArtZernike3DParallel::updateCTFImage(double defocusU, double defocusV, double angle)
{
	ctf.K=1; // get pure CTF with no envelope
	ctf.produceSideInfo();
}

void ProgArtZernike3DParallel::deformVol(int k)
{
	const MultidimArray<double> &mV=V();
	MultidimArray<double> &mP=P();
	MultidimArray<double> &mW=W();
    int l1,n,l2,m;
	l1 = 0;
	n = 0;
	l2 = 0;
	m = 0;
	size_t idxY0=VEC_XSIZE(clnm)/3;
	size_t idxZ0=2*idxY0;
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
	for (int i=STARTINGY(mV); i<=FINISHINGY(mV); i++)
	{
		for (int j=STARTINGX(mV); j<=FINISHINGX(mV); j++)
		{
			ZZ(pos) = k; YY(pos) = i; XX(pos) = j;
			pos = R * pos;
			double gx=0.0, gy=0.0, gz=0.0;
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
					double zsph=0.0;
					l1 = VEC_ELEM(vL1,idx);
					n = VEC_ELEM(vN,idx);
					l2 = VEC_ELEM(vL2,idx);
					m = VEC_ELEM(vM,idx);
					zsph=ZernikeSphericalHarmonics(l1,n,l2,m,jr,ir,kr,rr);
					if (rr>0 || l2==0) {
						gx += VEC_ELEM(clnm,idx)        *(zsph);
						gy += VEC_ELEM(clnm,idx+idxY0)  *(zsph);
						gz += VEC_ELEM(clnm,idx+idxZ0)  *(zsph);
					}
				}
			}
			double voxelI=mV.interpolatedElement3D(XX(pos)+gx,YY(pos)+gy,ZZ(pos)+gz);
			if (voxelI != 0.0) {
				A2D_ELEM(mP,i,j) += voxelI;
				A2D_ELEM(mW,i,j) += 1;
			}
		}
	}
}

Matrix1D<double> ProgArtZernike3DParallel::weightsInterpolation3D(double x, double y, double z) {
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

void ProgArtZernike3DParallel::removeOverdeformation() {
	int pos = 3*vecSize;
	size_t idxY0=(VEC_XSIZE(clnm))/3;
	size_t idxZ0=2*idxY0;

	Matrix2D<double> R, R_inv;
	R.initIdentity(3);
	R_inv.initIdentity(3);
    Euler_angles2matrix(rot, tilt, psi, R, false);
    R_inv = R.inv();
	Matrix1D<double> c;
	c.initZeros(3);
	for (size_t idx=0; idx<idxY0; idx++) {
		XX(c) = VEC_ELEM(clnm,idx); YY(c) = VEC_ELEM(clnm,idx+idxY0); ZZ(c) = VEC_ELEM(clnm,idx+idxZ0);
		c = R * c;
		ZZ(c) = 0.0;
		c = R_inv * c;
		VEC_ELEM(clnm,idx) = XX(c); VEC_ELEM(clnm,idx+idxY0) = YY(c); VEC_ELEM(clnm,idx+idxZ0) = ZZ(c);
	}
}