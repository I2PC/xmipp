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
 #include "core/multidim_array.h"
 #include "core/xmipp_image_extension.h"
 #include "core/xmipp_image_generic.h"
 #include "core/xmipp_fft.h"
 #include "core/xmipp_fftw.h"
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

 // Read arguments ==========================================================
 void ProgSubtractProjection::readParams()
 {
	fnParticles = getParam("-i");
 	fnVolR = getParam("--ref");
	fnOut=getParam("-o");
	if (fnOut=="")
		fnOut="output_particle_";
	// fnMaskVol=getParam("--maskVol");
	fnMask=getParam("--mask");
	sigma=getIntParam("--sigma");
	sampling = getDoubleParam("--sampling");
	padFourier = getDoubleParam("--padding");
    maxResol = getDoubleParam("--max_resolution");
	fmaskWidth = getIntParam("--fmask_width");
	fnProj = getParam("--save"); // JUST FOR SAVING INTERM FILES -> DELETE
 }

 // Show ====================================================================
 void ProgSubtractProjection::show() const{
    if (!verbose)
        return;
	std::cout<< "Input particles:\t" << fnParticles << std::endl
	<< "Reference volume:\t" << fnVolR << std::endl
	// << "Mask of the reference volume:\t" << fnMaskVol << std::endl
	<< "Mask of the region to keep:\t" << fnMask << std::endl
	<< "Sigma of low pass filter:\t" << sigma << std::endl
	<< "Sampling rate:\t" << sampling << std::endl
	<< "Padding factor:\t" << padFourier << std::endl
    << "Max. Resolution:\t" << maxResol << std::endl
	<< "Output particles:\t" << fnOut << std::endl
	<< "Path for saving:\t" << fnProj << std::endl; // JUST FOR SAVING INTERM FILES -> DELETE
 }

 // usage ===================================================================
 void ProgSubtractProjection::defineParams()
 {
     //Usage
     addUsageLine("");
     //Parameters
     addParamsLine("-i <particles>\t: Particles metadata (.xmd file)");
     addParamsLine("--ref <volume>\t: Reference volume to subtract");
	//  addParamsLine("[--maskVol <maskVol>]\t: 3D mask for the reference volume");
     addParamsLine("[--mask <mask=\"\">]\t: 3D mask for region to keep, no mask implies subtraction of whole images");
	 addParamsLine("[-o <structure=\"\">]\t: Output filename suffix for subtracted particles");
     addParamsLine("\t: If no name is given, then output_particles");
	 addParamsLine("[--sampling <sampling=1>]\t: Sampling rate (A/pixel)");
	 addParamsLine("[--max_resolution <f=4>]\t: Maximum resolution (A)");
	 addParamsLine("[--fmask_width <w=40>]\t: extra width of final mask (A)"); 
	 addParamsLine("[--padding <p=2>]\t: Padding factor for Fourier projector");
	 addParamsLine("[--sigma <s=2>]\t: Decay of the filter (sigma) to smooth the mask transition");
	 addParamsLine("[--save <structure=\"\">]\t: path for saving intermediate files"); // JUST FOR SAVING INTERM FILES -> DELETE
     addExampleLine("A typical use is:",false);
     addExampleLine("xmipp_subtract_projection -i input_particles.xmd --ref input_map.mrc --mask mask_vol.mrc "
    		 "-o output_particles --sampling 1 --max_resolution 4");
 }

 void ProgSubtractProjection::readParticle(const MDRowVec &r){
	r.getValueOrDefault(MDL_IMAGE, fnImage, "no_filename");
	I.read(fnImage);
	I().setXmippOrigin();
	I.write(formatString("%s0_I.mrc", fnProj.c_str())); 
 }

 void ProgSubtractProjection::writeParticle(const int &ix, Image<double> &img) {
	FileName out = formatString("%d@%s.mrcs", ix, fnOut.c_str());
	img.write(out);
	mdParticles.setValue(MDL_IMAGE, out, ix);
 }

 void ProgSubtractProjection::createMask(const FileName &fnM, Image<double> &m) {
	if (fnM.isEmpty()) 
		m().initZeros((int)XSIZE(V()),(int)YSIZE(V()));
	else {
		m.read(fnM);
		m().setXmippOrigin();
	}
 }

 Image<double> ProgSubtractProjection::binarizeMask(Projection &m) const{
 	double maxMaskVol;
 	double minMaskVol;
 	m().computeDoubleMinMax(minMaskVol, maxMaskVol);
 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(m())
		DIRECT_MULTIDIM_ELEM(m(),n) = (DIRECT_MULTIDIM_ELEM(m(),n)>0.1*maxMaskVol) ? 1:0; 
 	return m;
 }

 Image<double> ProgSubtractProjection::invertMask(const Image<double> &m) const{
	Image<double> PmaskI = m;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PmaskI())
		DIRECT_MULTIDIM_ELEM(PmaskI,n) = (DIRECT_MULTIDIM_ELEM(PmaskI,n)*(-1))+1;
	return PmaskI;
 }

 Image<double> ProgSubtractProjection::applyCTF(const MDRowVec &r, Projection &proj) {
	if (r.containsLabel(MDL_CTF_DEFOCUSU) || r.containsLabel(MDL_CTF_MODEL)){
	 	CTFDescription ctf;
		ctf.readFromMdRow(r);
		ctf.produceSideInfo();
	    FourierFilter FilterCTF;
	 	FilterCTF.FilterBand = CTF;
	 	FilterCTF.ctf.enable_CTFnoise = false;
		FilterCTF.ctf = ctf;
		// Padding before apply CTF
		Image<double> padp;
		int pad;
		pad = int(XSIZE(V())/2);
		padp().initZeros(pad*2, pad*2);
		MultidimArray <double> &mpad = padp();
		mpad.setXmippOrigin();
	    MultidimArray<double> &mproj = proj();
	    mproj.setXmippOrigin();
		mproj.window(mpad,STARTINGY(mproj)-pad, STARTINGX(mproj)-pad, FINISHINGY(mproj)+pad, FINISHINGX(mproj)+pad);
		FilterCTF.generateMask(mpad);
		FilterCTF.applyMaskSpace(mpad);
	    //Crop to restore original size
		mpad.window(mproj,STARTINGY(mproj), STARTINGX(mproj), FINISHINGY(mproj), FINISHINGX(mproj));
	}
	return proj;
 }

void ProgSubtractProjection::processParticle(size_t iparticle, int sizeImg, FourierTransformer &transformerImg) {
	row = mdParticles.getRowVec(iparticle);
	readParticle(row);
	row.getValueOrDefault(MDL_ANGLE_ROT, part_angles.rot, 0);
	row.getValueOrDefault(MDL_ANGLE_TILT, part_angles.tilt, 0);
	row.getValueOrDefault(MDL_ANGLE_PSI, part_angles.psi, 0);
	roffset.initZeros(2);
	row.getValueOrDefault(MDL_SHIFT_X, roffset(0), 0);
	row.getValueOrDefault(MDL_SHIFT_Y, roffset(1), 0);
	roffset *= -1;
	projectVolume(*projector, P, sizeImg, sizeImg, part_angles.rot, part_angles.tilt, part_angles.psi, ctfImage);	
	P.write(formatString("%s0_P.mrc", fnProj.c_str()));
	Pctf = applyCTF(row, P);
	transformerImg.FourierTransform(Pctf(), PFourier, false);
}

MultidimArray< std::complex<double> > ProgSubtractProjection::computeEstimationImage(const MultidimArray<double> &Img, 
const MultidimArray<double> &InvM, FourierTransformer &transformerImgiM) {
	Image<double> ImgiM;
	MultidimArray< std::complex<double> > ImgiMFourier;
	ImgiM().initZeros(Img);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Img)
		DIRECT_MULTIDIM_ELEM(ImgiM(),n) = DIRECT_MULTIDIM_ELEM(Img,n) * DIRECT_MULTIDIM_ELEM(InvM,n);
	transformerImgiM.FourierTransform(ImgiM(),ImgiMFourier,false);
	ImgiM.write(formatString("%s4_ImgiM.mrc", fnProj.c_str()));
	return ImgiMFourier;
}

 double ProgSubtractProjection::evaluateFitting(const MultidimArray<double> &y, const MultidimArray<double> &yp) const{
	double sumY = 0;
	double sumY2 = 0;
	double sumE2 = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(y) {
		double e = DIRECT_MULTIDIM_ELEM(y, n) - DIRECT_MULTIDIM_ELEM(yp, n);
		sumE2 += e * e;
		sumY += DIRECT_MULTIDIM_ELEM(y, n);
		sumY2 += DIRECT_MULTIDIM_ELEM(y, n) * DIRECT_MULTIDIM_ELEM(y, n);
	}
	auto meanY = sumY / (double)MULTIDIM_SIZE(y);
	auto varY = sumY2 / (double)MULTIDIM_SIZE(y) - meanY * meanY;
	auto R2 = 1 - sumE2 / varY;
	return R2;
 }

void ProgSubtractProjection::checkBestModel(const MultidimArray<double> &beta, MultidimArray<double> &betap, int deg) const{
	auto N = (int)MULTIDIM_SIZE(beta);
	// Fit order 0 beta=beta0
	double beta00 = beta.computeAvg();
	MultidimArray<double> betap0;
	betap0.initZeros(beta);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(beta)
		DIRECT_MULTIDIM_ELEM(betap0, n) = beta00;
	double R20 = evaluateFitting(beta, betap0);
	double R20adj = 1.0 - (1.0 - R20) * (N - 1.0) / (N - 1.0);
	// Fit order 1 beta=beta0+beta1*idx
	MultidimArray<double> idx(N);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(idx)
		DIRECT_MULTIDIM_ELEM(idx, n) = double(n);
	double sumX = 0;
	double sumX2 = 0;
	double sumY = 0;
	double sumXY = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(beta) {
		sumX += DIRECT_MULTIDIM_ELEM(idx, n);
		sumX2 += DIRECT_MULTIDIM_ELEM(idx, n) * DIRECT_MULTIDIM_ELEM(idx, n);
		sumY += DIRECT_MULTIDIM_ELEM(beta, n);
		sumXY += DIRECT_MULTIDIM_ELEM(idx, n) * DIRECT_MULTIDIM_ELEM(beta, n);
	}
	double beta11 = (N * sumXY - sumX * sumY) / (N * sumX2 - sumX * sumX);
	double beta10 = (sumY - beta11 * sumX) / N;
	MultidimArray<double> betap1;
	betap1.initZeros(beta);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(beta)
		DIRECT_MULTIDIM_ELEM(betap1, n) = beta10 + beta11 * DIRECT_MULTIDIM_ELEM(idx, n);
	double R21 = evaluateFitting(beta, betap1);
	double R21adj = 1.0 - (1.0 - R21) * (N - 1.0) / (N - 1.0 - 1.0);
	// Decide fitting
	if (R21adj > R20adj)
	{
		betap = betap1;
		deg = 1;
	}
	else
	{
		betap = betap0;
		deg = 0;
	}
}

 void ProgSubtractProjection::run() {
	show();
	// Read input volume, masks and particles metadata
	V.read(fnVolR);
	V().setXmippOrigin();
	// vrM.read(fnMaskVol);
	// vrM().setXmippOrigin();
	createMask(fnMask, vM);
	vM().setXmippOrigin();
	mdParticles.read(fnParticles);
	// Initialize Gaussian LPF to smooth mask
	FilterG.FilterShape=REALGAUSSIAN;
	FilterG.FilterBand=LOWPASS;
	FilterG.w1=sigma;
	// Initialize Fourier projectors
	double cutFreq = sampling/maxResol;
	projector = std::make_unique<FourierProjector>(V(), padFourier, cutFreq, xmipp_transformation::BSPLINE3);
	// std::unique_ptr<FourierProjector> projectorMaskVol;
	// projectorMaskVol = std::make_unique<FourierProjector>(vrM(), padFourier, cutFreq, xmipp_transformation::BSPLINE3);
	std::unique_ptr<FourierProjector> projectorMask;
	projectorMask = std::make_unique<FourierProjector>(vM(), padFourier, cutFreq, xmipp_transformation::BSPLINE3);
	// Read first particle
	const auto sizeI = (int)XSIZE(I());
	processParticle(1, sizeI, transformer);
	const MultidimArray<double> &mPctf = Pctf();
	// Construct frequencies image
	MultidimArray<int> wi;
	wi.initZeros(PFourier);
	Matrix1D<double> w(2); 	
	for (int i=0; i<YSIZE(wi); i++) {
		FFT_IDX2DIGFREQ(i,YSIZE(mPctf),YY(w)) 
		for (int j=0; j<XSIZE(wi); j++)  {
			FFT_IDX2DIGFREQ(j,XSIZE(mPctf),XX(w))
			DIRECT_A2D_ELEM(wi,i,j) = (int)round((sqrt(YY(w)*YY(w) + XX(w)*XX(w))) * (int)XSIZE(mPctf)); 
		}
	}
	auto maxwiIdx = (int)XSIZE(wi);
	Image<double> wi_img;
	typeCast(wi, wi_img());
	wi_img.write(formatString("%s1_wi.mrc", fnProj.c_str()));

    for (size_t i = 1; i <= mdParticles.size(); ++i) {  
     	// Project volume (for particle 1 it is already done before the loop)
		if (i != 1)
			processParticle(i, sizeI, transformer);
    	// Project and process masks

		// Volume mask
		// projectVolume(*projectorMaskVol, PmaskVol, sizeI, sizeI, part_angles.rot, part_angles.tilt, part_angles.psi, ctfImage);
		// Image<double> p_img = binarizeMask(PmaskVol);
		// p_img.write(formatString("%s1_maskVol.mrc", fnProj.c_str())); 		
		// FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(P())
		// 	DIRECT_MULTIDIM_ELEM(P(),n) *= DIRECT_MULTIDIM_ELEM(p_img(),n);	
		// typeCast(P(), p_img());
		// p_img.write(formatString("%s1_pmask.mrc", fnProj.c_str())); 

		// Keep mask
		projectVolume(*projectorMask, Pmask, sizeI, sizeI, part_angles.rot, part_angles.tilt, part_angles.psi, ctfImage);	
		Pmask.write(formatString("%s2_Mask.mrc", fnProj.c_str()));
		M = binarizeMask(Pmask);
		M.write(formatString("%s2_MaskBin.mrc", fnProj.c_str()));
		// Build final mask (projected mask + 2D dilation)
		auto fmaskWidth_px = fmaskWidth/(int)sampling;
		if (fnMask.isEmpty()) {
			iM = invertMask(M);
			iM.write(formatString("%s3_iM.mrc", fnProj.c_str()));
			Mfinal = iM;
			Mfinal.write(formatString("%s6_maskFinal.mrc", fnProj.c_str())); 
		}
		else {
			Mfinal().initZeros(M());
			dilate2D(M(), Mfinal(), 8, 0, fmaskWidth_px); 
			Mfinal.write(formatString("%s6_maskFinal.mrc", fnProj.c_str())); 
			FilterG.applyMaskSpace(M());
			M.write(formatString("%s2_MaskSmooth.mrc", fnProj.c_str()));
			iM = invertMask(M);
			iM.write(formatString("%s3_iM.mrc", fnProj.c_str()));
		}
		// Compute IiM = I*iM		
		FourierTransformer transformerIiM;
		MultidimArray< std::complex<double> > IiMFourier = computeEstimationImage(I(), iM(), transformerIiM);
		// Compute PiM = P*iM
		FourierTransformer transformerPiM;
		MultidimArray< std::complex<double> > PiMFourier = computeEstimationImage(P(), iM(), transformerPiM);
		// Estimate transformation T(w) 
		MultidimArray<double> num;
		num.initZeros(maxwiIdx); 
		MultidimArray<double> den;
		den.initZeros(maxwiIdx);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PiMFourier) {
			if (n <= maxwiIdx) {
				int win = DIRECT_MULTIDIM_ELEM(wi, n);
				double realPiMFourier = real(DIRECT_MULTIDIM_ELEM(PiMFourier,n));
				double imagPiMFourier = imag(DIRECT_MULTIDIM_ELEM(PiMFourier,n));
				DIRECT_MULTIDIM_ELEM(num,win) += real(DIRECT_MULTIDIM_ELEM(IiMFourier,n)) * realPiMFourier
												+ imag(DIRECT_MULTIDIM_ELEM(IiMFourier,n)) * imagPiMFourier;
				DIRECT_MULTIDIM_ELEM(den,win) += realPiMFourier*realPiMFourier + imagPiMFourier*imagPiMFourier;
			}
		}
		MultidimArray<double> beta;
		beta.initZeros(maxwiIdx); 
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(beta) {
			if (DIRECT_MULTIDIM_ELEM(den,n) == 0)
				DIRECT_MULTIDIM_ELEM(beta,n) = 0;
			else
				DIRECT_MULTIDIM_ELEM(beta,n) = DIRECT_MULTIDIM_ELEM(num,n) / DIRECT_MULTIDIM_ELEM(den,n);	
		} 
		MultidimArray<double> betap;	
		betap.initZeros(maxwiIdx); 
		int degree;
		checkBestModel(beta, betap, degree);  
		double beta1 = betap.computeAvg();
		std::complex<double> beta0;
		// Apply adjustment: PFourierAdjusted = T(w) * PFourier
		if (degree == 0) // Degree 0: T(w) = b1
		{
			beta0 = IiMFourier(0,0)-beta1*PiMFourier(0,0); 
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PFourier) {
				if (n <= maxwiIdx)
					DIRECT_MULTIDIM_ELEM(PFourier,n) *= beta1;
			}
			PFourier(0,0) = beta0 + beta1*PFourier(0,0); 
		}
		else if (degree == 1) // Degree 1: T(w) = b1 + b2*wi
		{
			beta0 = IiMFourier(0,0)-(beta1+betap(0)*wi(0,0))*PiMFourier(0,0); 
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PFourier) {
				if (n <= maxwiIdx)
					DIRECT_MULTIDIM_ELEM(PFourier,n) *= (beta1+betap((int)n)*DIRECT_MULTIDIM_ELEM(wi,n));
			}
			PFourier(0,0) = beta0 + (beta1+betap(0)*wi(0,0))*PFourier(0,0); 
		}
		// Recover adjusted projection (P) in real space
		transformer.inverseFourierTransform(PFourier, P());
		P.write(formatString("%s5_Padj.mrc", fnProj.c_str()));
		// Subtraction
		Idiff().initZeros(I());
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Idiff())
			DIRECT_MULTIDIM_ELEM(Idiff(),n) = (DIRECT_MULTIDIM_ELEM(I(),n)-DIRECT_MULTIDIM_ELEM(P(),n))*DIRECT_MULTIDIM_ELEM(Mfinal(),n);
		Idiff.write(formatString("%s7_subtraction.mrc", fnProj.c_str()));
		// Write particle
		writeParticle(int(i), Idiff);
    }
    mdParticles.write(fnParticles);
 }