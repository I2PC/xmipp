/***************************************************************************
 *
 * Authors:     Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
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

 #include "subtract_projection.h"
 #include "core/transformations.h"
 #include "core/metadata_extension.h"
 #include "core/multidim_array.h"
 #include "core/xmipp_image_extension.h"
 #include "core/xmipp_image_generic.h"
 #include "core/xmipp_image_base.h"
 #include "core/xmipp_fft.h"
 #include "core/xmipp_fftw.h"
 #include "core/linear_system_helper.h"
 #include "data/projection.h"
 #include "data/mask.h"
 #include "data/filters.h"
 #include "data/morphology.h"
 #include <iostream>
 #include <string>
 #include <sstream>
 #include "data/image_operate.h"
 #include <iostream>
 #include <cstdlib>
 #include <vector>
 #include <utility>


 // Empty constructor =======================================================
ProgSubtractProjection::ProgSubtractProjection()
{
	produces_a_metadata = true;
    each_image_produces_an_output = true;
    keep_input_columns = true;
	save_metadata_stack = true;
    remove_disabled = false;
	projector = nullptr;
	projectorMask = nullptr;
	rank = 0;
}

ProgSubtractProjection::~ProgSubtractProjection()
{
	delete projector;
	delete projectorMask;
}

 // Read arguments ==========================================================
 void ProgSubtractProjection::readParams()
 {
	XmippMetadataProgram::readParams();
 	fnVolR = getParam("--ref");
	fnMask=getParam("--mask");
	sigma=getIntParam("--sigma");
	sampling = getDoubleParam("--sampling");
	padFourier = getDoubleParam("--padding");
    maxResol = getDoubleParam("--max_resolution");
	limitfreq = getIntParam("--limit_freq");
	cirmaskrad = getDoubleParam("--cirmaskrad");
	fnProj = getParam("--save"); 
	nonNegative = checkParam("--nonNegative");
	boost = checkParam("--boost");
	subtract = checkParam("--subtract");
 }

 // Show ====================================================================
 void ProgSubtractProjection::show() const{
    if (!verbose)
        return;
	std::cout
	<< "Input particles:\t" << fnParticles << std::endl
	<< "Reference volume:\t" << fnVolR << std::endl
	<< "Mask of the region to keep:\t" << fnMask << std::endl
	<< "Sigma of low pass filter:\t" << sigma << std::endl
	<< "Sampling rate:\t" << sampling << std::endl
	<< "Padding factor:\t" << padFourier << std::endl
    << "Max. Resolution:\t" << maxResol << std::endl
	<< "Limit frequency:\t" << limitfreq << std::endl
	<< "Output particles:\t" << fnOut << std::endl;
 }

 // usage ===================================================================
 void ProgSubtractProjection::defineParams()
 {
	 //Usage
     addUsageLine("This program computes the subtraction between particles and a reference"); 
	 addUsageLine(" volume, by computing its projections with the same angles that input particles have."); 
	 addUsageLine(" Then, each particle and the correspondent projection of the reference volume are numerically");
	 addUsageLine(" adjusted and subtracted using a mask which denotes the region to keep or subtract.");
     //Parameters
	 XmippMetadataProgram::defineParams();
     addParamsLine("--ref <volume>\t: Reference volume to subtract");
     addParamsLine("[--mask <mask=\"\">]\t: 3D mask for region to keep, no mask implies subtraction of whole images");
	 addParamsLine("[--sampling <sampling=1>]\t: Sampling rate (A/pixel)");
	 addParamsLine("[--max_resolution <f=4>]\t: Maximum resolution (A)");
	 addParamsLine("[--fmask_width <w=40>]\t: Extra width of final mask (A). -1 means no masking."); 
	 addParamsLine("[--padding <p=2>]\t: Padding factor for Fourier projector");
	 addParamsLine("[--sigma <s=2>]\t: Decay of the filter (sigma) to smooth the mask transition");
	 addParamsLine("[--limit_freq <l=0>]\t: Limit frequency (= 1) or not (= 0) in adjustment process");
	 addParamsLine("[--nonNegative]\t: Ignore particles with negative beta0 or R2"); 
	 addParamsLine("[--boost]\t: Perform a boosting of original particles"); 
	 addParamsLine("[--cirmaskrad <c=-1.0>]\t: Radius of the circular mask");
	 addParamsLine("[--save <structure=\"\">]\t: Path for saving intermediate files"); 
	 addParamsLine("[--subtract]\t: The mask contains the region to SUBTRACT"); 
     addExampleLine("A typical use is:",false);
     addExampleLine("xmipp_subtract_projection -i input_particles.xmd --ref input_map.mrc --mask mask_vol.mrc "
    		 "-o output_particles --sampling 1 --fmask_width 40 --max_resolution 4");
 }

 void ProgSubtractProjection::readParticle(const MDRow &r) {
	r.getValueOrDefault(MDL_IMAGE, fnImgI, "no_filename");
	I.read(fnImgI);
	I().setXmippOrigin();
 }

 void ProgSubtractProjection::writeParticle(MDRow &rowOut, FileName fnImgOut, Image<double> &img, double R2a, double b0save, double b1save) {
	img.write(fnImgOut);
	rowOut.setValue(MDL_IMAGE, fnImgOut);
	rowOut.setValue(MDL_SUBTRACTION_R2, R2a); 
	rowOut.setValue(MDL_SUBTRACTION_BETA0, b0save); 
	rowOut.setValue(MDL_SUBTRACTION_BETA1, b1save); 
	if (nonNegative && (disable || R2a < 0)) 
	{
		rowOut.setValue(MDL_ENABLED, -1);
	}
 }

 void ProgSubtractProjection::createMask(const FileName &fnM, Image<double> &m, Image<double> &im) {
	if (fnM.isEmpty()) 
	{
		m().initZeros((int)XSIZE(V()),(int)YSIZE(V()),(int)ZSIZE(V()));
		im = m;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(im())
			DIRECT_MULTIDIM_ELEM(im(),n) += 1; 
	}
	else 
	{
		m.read(fnM);
		m().setXmippOrigin();
		im = m;
		if (!subtract)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(im())
				DIRECT_MULTIDIM_ELEM(im(),n) = (DIRECT_MULTIDIM_ELEM(m(),n)*(-1))+1;
		} 
	}
 }

 Image<double> ProgSubtractProjection::binarizeMask(Projection &m) const {
 	double maxMaskVol;
 	double minMaskVol;
	MultidimArray<double> &mm=m();
 	mm.computeDoubleMinMax(minMaskVol, maxMaskVol);

	// Binarization threshold = 10% of max value in projection
 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mm)
		DIRECT_MULTIDIM_ELEM(mm,n) = (DIRECT_MULTIDIM_ELEM(mm,n)>0.1*maxMaskVol) ? 1:0; 
 	return m;
 }

 Image<double> ProgSubtractProjection::invertMask(const Image<double> &m) {
	PmaskI = m;
	MultidimArray<double> &mPmaskI=PmaskI();
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mPmaskI)
		DIRECT_MULTIDIM_ELEM(mPmaskI,n) = (DIRECT_MULTIDIM_ELEM(mPmaskI,n)*(-1))+1;
	return PmaskI;
 }

 Image<double> ProgSubtractProjection::applyCTF(const MDRow &r, Projection &proj) {
	if (r.containsLabel(MDL_CTF_DEFOCUSU) || r.containsLabel(MDL_CTF_MODEL)){
		ctf.readFromMdRow(r);
		ctf.Tm = sampling;
		ctf.produceSideInfo();
	 	FilterCTF.FilterBand = CTF;
	 	FilterCTF.ctf.enable_CTFnoise = false;
		FilterCTF.ctf = ctf;
		// Padding before apply CTF
		MultidimArray <double> &mpad = padp();
		mpad.setXmippOrigin();
		MultidimArray<double> &mproj = proj();
		mproj.setXmippOrigin();
		mproj.window(mpad,STARTINGY(mproj)*(int)padFourier, STARTINGX(mproj)*(int)padFourier, FINISHINGY(mproj)*(int)padFourier, FINISHINGX(mproj)*(int)padFourier);
		FilterCTF.generateMask(mpad);
		FilterCTF.applyMaskSpace(mpad); 
		//Crop to restore original size
		mpad.window(mproj,STARTINGY(mproj), STARTINGX(mproj), FINISHINGY(mproj), FINISHINGX(mproj));
	}
	return proj;
 }

void ProgSubtractProjection::processParticle(const MDRow &rowprocess, int sizeImg) {
	
	// Read metadata information for projection
	readParticle(rowprocess);
	rowprocess.getValueOrDefault(MDL_ANGLE_ROT, part_angles.rot, 0);
	rowprocess.getValueOrDefault(MDL_ANGLE_TILT, part_angles.tilt, 0);
	rowprocess.getValueOrDefault(MDL_ANGLE_PSI, part_angles.psi, 0);
	roffset.initZeros(2);
	rowprocess.getValueOrDefault(MDL_SHIFT_X, roffset(0), 0);
	rowprocess.getValueOrDefault(MDL_SHIFT_Y, roffset(1), 0);
	roffset *= -1;
	
	// Project volume + apply translation, CTF and mask
	projectVolume(*projector, P, sizeImg, sizeImg, part_angles.rot, part_angles.tilt, part_angles.psi, ctfImage);
	selfTranslate(xmipp_transformation::LINEAR, P(), roffset, xmipp_transformation::WRAP);
	Pctf = applyCTF(rowprocess, P);
	MultidimArray<double> &mPctf = Pctf();
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mPctf)
		DIRECT_MULTIDIM_ELEM(mPctf,n) = DIRECT_MULTIDIM_ELEM(mPctf,n) * DIRECT_MULTIDIM_ELEM(cirmask(),n);

	// FT of projection and particle
	transformerP.FourierTransform(Pctf(), PFourier, false);
	transformerI.FourierTransform(I(), IFourier, false);
}

MultidimArray< std::complex<double> > ProgSubtractProjection::computeEstimationImage(const MultidimArray<double> &Img, 
const MultidimArray<double> &InvM, FourierTransformer &transformerImgiM) {
	ImgiM().initZeros(Img);
	ImgiM().setXmippOrigin();
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Img)
		DIRECT_MULTIDIM_ELEM(ImgiM(),n) = DIRECT_MULTIDIM_ELEM(Img,n) * DIRECT_MULTIDIM_ELEM(InvM,n);
	transformerImgiM.FourierTransform(ImgiM(),ImgiMFourier,false);
	return ImgiMFourier;
}

 double ProgSubtractProjection::evaluateFitting(const MultidimArray< std::complex<double> > &y,
                                                const MultidimArray< std::complex<double> > &yp) const{
	double sumY = 0;
	double sumY2 = 0;
	double sumE2 = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(y) {
		double realyn = real(DIRECT_MULTIDIM_ELEM(y, n)); 
		double imagyn = imag(DIRECT_MULTIDIM_ELEM(y, n)); 
		double ereal = realyn - real(DIRECT_MULTIDIM_ELEM(yp, n)); 
		double eimag = imagyn - imag(DIRECT_MULTIDIM_ELEM(yp, n));  
		sumE2 += ereal*ereal + eimag*eimag; 
		sumY += realyn + imagyn; 
		sumY2 += realyn*realyn + imagyn*imagyn; 
	}
	auto meanY = sumY / (2.0*(double)MULTIDIM_SIZE(y));
	auto varY = sumY2 / (2.0*(double)MULTIDIM_SIZE(y)) - meanY*meanY; 
	auto R2 = 1.0 - (sumE2/(2.0*(double)MULTIDIM_SIZE(y))) / varY;
	return R2;
 }

Matrix1D<double> ProgSubtractProjection::checkBestModel(MultidimArray< std::complex<double> > &PFourierf, const MultidimArray< std::complex<double> > &PFourierf0,
 const MultidimArray< std::complex<double> > &PFourierf1, const MultidimArray< std::complex<double> > &IFourierf) const { 
	// Compute R2 coefficient for order 0 model (R20) and order 1 model (R21)
	auto N = 2.0*(double)MULTIDIM_SIZE(PFourierf);
	double R20adj = evaluateFitting(IFourierf, PFourierf0); // adjusted R2 for an order 0 model = R2
	double R21 = evaluateFitting(IFourierf, PFourierf1); 
	double R21adj = 1.0 - (1.0 - R21) * (N - 1.0) / (N - 2.0); // adjusted R2 for an order 1 model -> p = 2
	//Decide best fitting
	Matrix1D<double> R2(2);
	if (R21adj > R20adj) { // Order 1: T(w) = b01 + b1*wi 
		PFourierf = PFourierf1;
		R2(0) = R21adj;
		R2(1) = 1;
	} 
	else { // Order 0: T(w) = b00 
		PFourierf = PFourierf0;
		R2(0) = R20adj;
		R2(1) = 0;		
	}
	return R2;
}


// ------------------------------------------------	MAIN METHODS
void ProgSubtractProjection::preProcess() {
	// Read input volume, mask and particles metadata
	show();
	V.read(fnVolR);
	V().setXmippOrigin();

	// Create 2D circular mask to avoid edge artifacts after wrapping
	size_t Xdim;
	size_t Ydim;
	size_t Zdim;
	size_t Ndim;
	V.getDimensions(Xdim, Ydim, Zdim, Ndim);
	cirmask().initZeros((int)Ydim, (int)Xdim);
	cirmask().setXmippOrigin();
	if (cirmaskrad == -1.0)
		cirmaskrad = (double)XSIZE(V())/2;
	RaisedCosineMask(cirmask(), cirmaskrad*0.8, cirmaskrad*0.9);
	cirmask.write(formatString("%s/cirmask.mrc", fnProj.c_str()));
	
	// Create mock image of same size as particles (and referencce volume) to get
	I().initZeros((int)Ydim, (int)Xdim);
	I().initConstant(1);
	transformerI.FourierTransform(I(), IFourier, false);

	// Initialize Gaussian LPF to smooth mask
	FilterG.FilterShape=REALGAUSSIAN;
	FilterG.FilterBand=LOWPASS;
	FilterG.w1=sigma;

	// Construct frequencies image
	wi.initZeros(IFourier);
	Matrix1D<double> w(2); 	
	double cutFreq = sampling/maxResol;
	for (int i=0; i<YSIZE(wi); i++) {
		FFT_IDX2DIGFREQ(i,YSIZE(IFourier),YY(w)) 
		for (int j=0; j<XSIZE(wi); j++)  {
			FFT_IDX2DIGFREQ(j,XSIZE(IFourier),XX(w))
			DIRECT_A2D_ELEM(wi,i,j) = (int)round((sqrt(YY(w)*YY(w) + XX(w)*XX(w))) * (int)XSIZE(IFourier)); // indexes
		}
	}
	if (limitfreq == 0)
		maxwiIdx = (int)XSIZE(wi); 
	else
		DIGFREQ2FFT_IDX(cutFreq, (int)YSIZE(IFourier), maxwiIdx)

	if (rank==0)
	{
		// Read or create mask keep and compute inverse of mask keep (mask subtract)
		createMask(fnMask, vM, ivM);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
			DIRECT_MULTIDIM_ELEM(V(),n) = DIRECT_MULTIDIM_ELEM(V(),n)*DIRECT_MULTIDIM_ELEM(ivM(),n); 
		// Initialize Fourier projectors
		std::cout << "-------Initializing projectors-------" << std::endl;
		projector = new FourierProjector(V(), padFourier, cutFreq, xmipp_transformation::BSPLINE3);
		projectorMask = new FourierProjector(vM(), padFourier, cutFreq, xmipp_transformation::BSPLINE3);
		std::cout << "-------Projectors initialized-------" << std::endl;
	}
	else
	{
		projector = new FourierProjector(padFourier,cutFreq,xmipp_transformation::BSPLINE3);
		projectorMask = new FourierProjector(padFourier,cutFreq,xmipp_transformation::BSPLINE3);
	}
 }

void ProgSubtractProjection::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
 { 
	// Initialize aux variable
	disable = false;

	// Project volume and process projections 
	const auto sizeI = (int)XSIZE(I());
	processParticle(rowIn, sizeI);

	// Build projected and final masks
	if (fnMask.isEmpty())  // If there is no provided mask
	{
		M().initZeros(P());
		// inverse mask (iM) is all 1s
		iM = invertMask(M);
	}
	else  // If a mask has been provided
	{
		projectVolume(*projectorMask, Pmask, sizeI, sizeI, part_angles.rot, part_angles.tilt, part_angles.psi, ctfImage);	

		// Apply binarization, shift and gaussian filter to the projected mask
		M = binarizeMask(Pmask);
		selfTranslate(xmipp_transformation::LINEAR, M(), roffset, xmipp_transformation::DONT_WRAP);
		FilterG.applyMaskSpace(M());

		if (subtract) // If the mask contains the part to SUBTRACT: iM = input mask
			iM = M;
		else // If the mask contains the part to KEEP: iM = INVERSE of original mask
			iM = invertMask(M);
	}
	
	// Compute estimation images: IiM = I*iM and PiM = P*iM	
	IiMFourier = computeEstimationImage(I(), iM(), transformerIiM);
	PiMFourier = computeEstimationImage(Pctf(), iM(), transformerPiM);	

	// Estimate transformation with model of order 0: T(w) = beta00 and model of order 1: T(w) = beta01 + beta1*w
	MultidimArray<double> num0;
	num0.initZeros(maxwiIdx+1); 
	MultidimArray<double> den0;
	den0.initZeros(maxwiIdx+1);
	Matrix2D<double> A1;
	A1.initZeros(2,2);
	Matrix1D<double> b1;
	b1.initZeros(2);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PiMFourier) {
		int win = DIRECT_MULTIDIM_ELEM(wi, n);
		if (win < maxwiIdx) 
		{
			double realPiMFourier = real(DIRECT_MULTIDIM_ELEM(PiMFourier,n));
			double imagPiMFourier = imag(DIRECT_MULTIDIM_ELEM(PiMFourier,n));
			DIRECT_MULTIDIM_ELEM(num0,win) += real(DIRECT_MULTIDIM_ELEM(IiMFourier,n)) * realPiMFourier
											+ imag(DIRECT_MULTIDIM_ELEM(IiMFourier,n)) * imagPiMFourier;
			DIRECT_MULTIDIM_ELEM(den0,win) += realPiMFourier*realPiMFourier + imagPiMFourier*imagPiMFourier;
			A1(0,0) += realPiMFourier*realPiMFourier + imagPiMFourier*imagPiMFourier;
			A1(0,1) += win*(realPiMFourier + imagPiMFourier);
			A1(1,1) += 2*win;
			b1(0) += real(DIRECT_MULTIDIM_ELEM(IiMFourier,n)) * realPiMFourier + imag(DIRECT_MULTIDIM_ELEM(IiMFourier,n)) * imagPiMFourier;
			b1(1) += win*(real(DIRECT_MULTIDIM_ELEM(IiMFourier,n))+imag(DIRECT_MULTIDIM_ELEM(IiMFourier,n)));
		}
	}
	A1(1,0) = A1(0,1);

	std::cout << "FEDE ESTUVO AQUI!!!!!!! <3<3<3" << std::endl;
	double beta0save = 1;
	double beta1save = 0;	
	double beta00 = 1;
	double beta01 = 0;	
	double beta1 = 0;	
	Matrix1D<double> R2adj(2);
	R2adj(0) = 0;
	R2adj(1) = 0;	

	// // Compute beta00 from order 0 model
	// double beta00 = num0.sum()/den0.sum();
	// if (nonNegative && beta00 < 0) 
	// {
	// 	disable = true;
	// }
	// // Apply adjustment order 0: PFourier0 = T(w) * PFourier = beta00 * PFourier
	// PFourier0 = PFourier;
	// FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PFourier0) 
	// 	DIRECT_MULTIDIM_ELEM(PFourier0,n) *= beta00; 
	// PFourier0(0,0) = IiMFourier(0,0); 

	// // Compute beta01 and beta1 from order 1 model
	// PseudoInverseHelper h;
	// h.A = A1;
	// h.b = b1;
	// Matrix1D<double> betas1;
	// solveLinearSystem(h,betas1); 
	// double beta01 = betas1(0);
	// double beta1 = betas1(1);

	// // Apply adjustment order 1: PFourier1 = T(w) * PFourier = (beta01 + beta1*w) * PFourier
	// PFourier1 = PFourier;
	// FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PFourier1)
	// 	DIRECT_MULTIDIM_ELEM(PFourier1,n) *= (beta01+beta1*DIRECT_MULTIDIM_ELEM(wi,n)); 
	// PFourier1(0,0) = IiMFourier(0,0); 

	// // Check best model
	// Matrix1D<double> R2adj = checkBestModel(PFourier, PFourier0, PFourier1, IFourier);
	// double beta0save;
	// double beta1save;		
	// if (R2adj(1) == 0)
	// {
	// 	beta0save = beta00;
	// 	beta1save = 0;
	// }
	// else
	// {
	// 	beta0save = beta01;
	// 	beta1save = beta1;
	// }

	// Create empty new image for output particle
	MultidimArray<double> &mIdiff=Idiff();
	mIdiff.initZeros(I());
	mIdiff.setXmippOrigin();

	if (boost) // Boosting of original particles
	{
		if (R2adj(1) == 0)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(IFourier) 
				DIRECT_MULTIDIM_ELEM(IFourier,n) /= beta00; 
		} 
		else if (R2adj(1) == 1)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(IFourier)
				DIRECT_MULTIDIM_ELEM(IFourier,n) /= (beta01+beta1*DIRECT_MULTIDIM_ELEM(wi,n)); 
		}
		transformerI.inverseFourierTransform(IFourier, Idiff());
	} 
	else  // Subtraction
	{
		// Recover adjusted projection (P) in real space
		transformerP.inverseFourierTransform(PFourier, P());
		mIdiff.initZeros(I());
		mIdiff.setXmippOrigin();
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mIdiff)
			DIRECT_MULTIDIM_ELEM(mIdiff,n) = DIRECT_MULTIDIM_ELEM(I(),n)-DIRECT_MULTIDIM_ELEM(P(),n);
	}
	writeParticle(rowOut, fnImgOut, Idiff, R2adj(0), beta0save, beta1save); 
}

void ProgSubtractProjection::postProcess()
{
	getOutputMd().write(fn_out);
}
