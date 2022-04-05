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
	fnMask=getParam("--mask");
	fnMaskVol=getParam("--maskVol");
	iter=getIntParam("--iter");
	sigma=getIntParam("--sigma");
	lambda=getDoubleParam("--lambda");
	subtractAll = checkParam("--subAll");
	sampling = getDoubleParam("--sampling");
	padFourier = getDoubleParam("--padding");
    maxResol = getDoubleParam("--max_resolution");
	fmaskWidth = getDoubleParam("--fmask_width");
 }

 // Show ====================================================================
 void ProgSubtractProjection::show() const{
    if (!verbose)
        return;
	std::cout<< "Input particles:\t" << fnParticles << std::endl
	<< "Reference volume:\t" << fnVolR << std::endl
	<< "Volume mask:\t" << fnMaskVol << std::endl
	<< "Mask:\t" << fnMask << std::endl
	<< "Sigma:\t" << sigma << std::endl
	<< "Iterations:\t" << iter << std::endl
	<< "Relaxation factor:\t" << lambda << std::endl
	<< "Sampling:\t" << sampling << std::endl
	<< "Padding factor:\t" << padFourier << std::endl
    << "Max. Resolution:\t" << maxResol << std::endl
	<< "Output particles:\t" << fnOut << std::endl;
 }

 // usage ===================================================================
 void ProgSubtractProjection::defineParams()
 {
     //Usage
     addUsageLine("");
     //Parameters
     addParamsLine("-i <particles>\t: Particles metadata (.xmd file)");
     addParamsLine("--ref <volume>\t: Reference volume to subtract");
     addParamsLine("[-o <structure=\"\">]\t: Output filename suffix for subtracted particles");
     addParamsLine("\t: If no name is given, then output_particles");
     addParamsLine("[--maskVol <maskVol=\"\">]\t: 3D mask for region to keep");
     addParamsLine("[--mask <mask=\"\">]\t: final 3D mask for the region of subtraction");
     addParamsLine("[--sigma <s=2>]\t: Decay of the filter (sigma) to smooth the mask transition");
     addParamsLine("[--iter <n=1>]\t: Number of iterations");
     addParamsLine("[--lambda <l=0>]\t: Relaxation factor for Fourier Amplitude POCS (between 0 and 1)");
	 addParamsLine("[--subAll]\t: Perform the subtraction of the whole image");
	 addParamsLine("[--sampling <sampling=1>]\t: Sampling rate (A/pixel)");
	 addParamsLine("[--padding <p=2>]\t: Padding factor for Fourier projector");
	 addParamsLine("[--max_resolution <f=4>]\t: Maximum resolution (A)");
	 addParamsLine("[--fmask_width <w=40>]\t: extra width of final mask (A)"); 
     addExampleLine("A typical use is:",false);
     addExampleLine("xmipp_subtract_projection -i input_particles.xmd --ref input_map.mrc --maskVol mask_vol.vol --mask mask.vol "
    		 "-o output_particles --iter 5 --lambda 1 --sigma 3 --sampling 1 --padding 2 --max_resolution 4");
 }

 Image<double> ProgSubtractProjection::createMask(const FileName &fnM, Image<double> &m) {
	if (fnM.isEmpty()) {
		m().resizeNoCopy(I());
		m().initConstant(1.0);
	}
	else {
		m.read(fnM);
		m().setXmippOrigin();
	}
	return m;
 }

 Image<double> ProgSubtractProjection::binarizeMask(Projection &m) const{
 	double maxMaskVol;
 	double minMaskVol;
 	m().computeDoubleMinMax(minMaskVol, maxMaskVol);
 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(m())
		DIRECT_MULTIDIM_ELEM(m(),n) = (DIRECT_MULTIDIM_ELEM(m(),n)>0.05*maxMaskVol) ? 1:0; 
 	return m;
 }

 Image<double> invertMask(const Image<double> &m) {
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
		// Padding
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
	    //Crop
		mpad.window(mproj,STARTINGY(mproj), STARTINGX(mproj), FINISHINGY(mproj), FINISHINGX(mproj));
	}
	return proj;
 }

 double evaluateFitting(const MultidimArray<double> &y, const MultidimArray<double> &yp)
 {
	double sumY = 0;
	double sumY2 = 0;
	double sumE2 = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(y)
	{
		double e = DIRECT_MULTIDIM_ELEM(y, n) - DIRECT_MULTIDIM_ELEM(yp, n);
		sumE2 += e * e;
		sumY += DIRECT_MULTIDIM_ELEM(y, n);
		sumY2 += DIRECT_MULTIDIM_ELEM(y, n) * DIRECT_MULTIDIM_ELEM(y, n);
	}
	auto meanY = sumY / MULTIDIM_SIZE(y);
	auto varY = sumY2 / MULTIDIM_SIZE(y) - meanY * meanY;
	auto R2 = 1 - sumE2 / varY;
	return R2;
 }

void checkBestModel(const MultidimArray<double> &beta, MultidimArray<double> &betap)
{
	auto N = MULTIDIM_SIZE(beta);

	// Fit order 0 beta=beta0
	double beta00 = beta.computeAvg();
	MultidimArray<double> betap0;
	betap0.initZeros(beta);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(beta)
		DIRECT_MULTIDIM_ELEM(betap0, n) = beta00;
	double R20 = evaluateFitting(beta, betap0);
	double R20adj = 1 - (1 - R20) * (N - 1) / (N - 1);

	// Fit order 1 beta=beta0+beta1*idx
	MultidimArray<double> idx(N);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(idx)
		DIRECT_MULTIDIM_ELEM(idx, n) = double(n);
	double sumX = 0;
	double sumX2 = 0;
	double sumY = 0;
	double sumXY = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(beta)
	{
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
	double R21adj = 1 - (1 - R21) * (N - 1) / (N - 1 - 1);

	// Decide fitting
	if (R21adj > R20adj)
		betap = betap1;
	else
		betap = betap0;
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

 void ProgSubtractProjection::run() {
	show();
	// Read input volume and create masks
	V.read(fnVolR);
	V().setXmippOrigin();
 	mdParticles.read(fnParticles);
 	vM = createMask(fnMaskVol, vM); // Actually now this mask is mask keep and the other is the final one

	// Initialize Gaussian LPF to smooth mask
	FilterG.FilterShape=REALGAUSSIAN;
	FilterG.FilterBand=LOWPASS;
	FilterG.w1=sigma;

	// Initialize Fourier projectors
	double cutFreq = sampling/maxResol;
	FourierProjector *projector = new FourierProjector(V(), padFourier, cutFreq, xmipp_transformation::BSPLINE3);
	FourierProjector *projectorMask = new FourierProjector(vM(), padFourier, cutFreq, xmipp_transformation::BSPLINE3);

	// Construct image representing the frequency of each pixel 
	row = mdParticles.getRowVec(1);
	readParticle(row);
	struct Angles part_angles;
	row.getValueOrDefault(MDL_ANGLE_ROT, part_angles.rot, 0);
	row.getValueOrDefault(MDL_ANGLE_TILT, part_angles.tilt, 0);
	row.getValueOrDefault(MDL_ANGLE_PSI, part_angles.psi, 0);
	roffset.initZeros(2);
	row.getValueOrDefault(MDL_SHIFT_X, roffset(0), 0);
	row.getValueOrDefault(MDL_SHIFT_Y, roffset(1), 0);
	roffset *= -1;

	// Project volume
	const MultidimArray<double> *ctfImage = nullptr;
	projectVolume(*projector, P, (int)XSIZE(I()), (int)XSIZE(I()), part_angles.rot, part_angles.tilt, part_angles.psi, ctfImage);
	P.write(formatString("%s0_initialProjection.mrc", fnProj.c_str()));
	Pctf = applyCTF(row, P);
	const MultidimArray<double> &mPctf = Pctf();
	FourierTransformer transformerP;
	transformerP.FourierTransform(Pctf(),PFourier,false);

	// Construct frequencies image
	MultidimArray<int> wi;
	wi.initZeros(PFourier);
	Matrix1D<double> w(2); 	
	for (int i=0; i<YSIZE(wi); i++) 
	{
		FFT_IDX2DIGFREQ(i,YSIZE(mPctf),YY(w)) 
		for (int j=0; j<XSIZE(wi); j++)  
		{
			FFT_IDX2DIGFREQ(j,XSIZE(mPctf),XX(w))
			DIRECT_A2D_ELEM(wi,i,j) = (int)round((sqrt(YY(w)*YY(w) + XX(w)*XX(w))) * (int)XSIZE(mPctf)); 
		}
	}
	auto maxwiIdx = XSIZE(wi);
	// int maxwiIdx = YSIZE(mPctf);
	Image<double> wi_img;
	typeCast(wi, wi_img());
	wi_img.write(formatString("%s0_wi.mrc", fnProj.c_str()));

    for (size_t i = 1; i <= mdParticles.size(); ++i) {  // the begining is done twice for the first particle
    	// Read particle and metadata
    	row = mdParticles.getRowVec(i);
    	readParticle(row);
    	struct Angles part_angles;
     	row.getValueOrDefault(MDL_ANGLE_ROT, part_angles.rot, 0);
     	row.getValueOrDefault(MDL_ANGLE_TILT, part_angles.tilt, 0);
     	row.getValueOrDefault(MDL_ANGLE_PSI, part_angles.psi, 0);
     	roffset.initZeros(2);
     	row.getValueOrDefault(MDL_SHIFT_X, roffset(0), 0);
     	row.getValueOrDefault(MDL_SHIFT_Y, roffset(1), 0);
     	roffset *= -1;

     	// Project volume
     	const MultidimArray<double> *ctfImage2 = nullptr;
    	projectVolume(*projector, P, (int)XSIZE(I()), (int)XSIZE(I()), part_angles.rot, part_angles.tilt, part_angles.psi, ctfImage2);
    	P.write(formatString("%s0_initialProjection.mrc", fnProj.c_str()));

    	// Project and smooth big mask
    	projectVolume(*projectorMask, Pmask, (int)XSIZE(I()), (int)XSIZE(I()), part_angles.rot, part_angles.tilt, part_angles.psi, ctfImage2);
    	Pmask.write(formatString("%s1_initalMaskProjection.mrc", fnProj.c_str()));
    	M = binarizeMask(Pmask);
		M.write(formatString("%s1_MaskBin.mrc", fnProj.c_str()));
		FilterG.applyMaskSpace(M());
		M.write(formatString("%s1_MaskSmooth.mrc", fnProj.c_str()));

		// Apply CTF
		Pctf = applyCTF(row, P);
		Pctf.write(formatString("%s2_Pctf.mrc", fnProj.c_str()));
		const MultidimArray<double> &mPctf2 = Pctf();

		// Fourier Transform
		FourierTransformer transformerP2;
		transformerP2.FourierTransform(Pctf(),PFourier,false);

		// Compute inverse mask
		iM = invertMask(M);
		iM.write(formatString("%s3_iM.mrc", fnProj.c_str()));

		// Compute IiM = I*iM
		Image<double> IiM;
		const MultidimArray<double> &mI = I();
		IiM().initZeros((int)XSIZE(mI),(int)YSIZE(mI));
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mI)
		 	DIRECT_MULTIDIM_ELEM(IiM(),n) = DIRECT_MULTIDIM_ELEM(mI,n) * DIRECT_MULTIDIM_ELEM(iM,n);
		IiM.write(formatString("%s4_IiM.mrc", fnProj.c_str()));
		FourierTransformer transformerIiM;
		MultidimArray< std::complex<double> > IiMFourier;
		transformerIiM.FourierTransform(IiM(),IiMFourier,false);

		// Compute PiM = P*iM
		Image<double> PiM;
		PiM().initZeros((int)XSIZE(mI),(int)YSIZE(mI));
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mPctf2)
		 	DIRECT_MULTIDIM_ELEM(PiM(),n) = DIRECT_MULTIDIM_ELEM(mPctf2,n) * DIRECT_MULTIDIM_ELEM(iM,n);
		PiM.write(formatString("%s4_PiM.mrc", fnProj.c_str()));
		FourierTransformer transformerPiM;
		MultidimArray< std::complex<double> > PiMFourier;
		transformerPiM.FourierTransform(PiM(),PiMFourier,false);

		// Estimate transformation T(w) //
		MultidimArray<double> num;
		num.initZeros(maxwiIdx); 
		MultidimArray<double> den;
		den.initZeros(maxwiIdx);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PiMFourier) 
		{
			if (n <= maxwiIdx)
			{
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
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(beta)
		{
			if (DIRECT_MULTIDIM_ELEM(den,n) == 0)
				DIRECT_MULTIDIM_ELEM(beta,n) = 0;
			else
				DIRECT_MULTIDIM_ELEM(beta,n) = DIRECT_MULTIDIM_ELEM(num,n) / DIRECT_MULTIDIM_ELEM(den,n);	
		} 
		MultidimArray<double> betap;	
		betap.initZeros(maxwiIdx); 
		checkBestModel(beta, betap);
		double betaMean = betap.computeAvg();
		std::complex<double> beta0; 
		beta0 = IiMFourier(1,1)-betaMean*PiMFourier(1,1); // if betap = beta1 (order 1) -> betaMean is ok?

		// Apply adjustment: PFourierAdjusted = PFourier*betaMean
		std::complex<double> PFourier11 = PFourier(1,1);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PFourier) 
		{
			if (n <= maxwiIdx)
				DIRECT_MULTIDIM_ELEM(PFourier,n) *= DIRECT_MULTIDIM_ELEM(betap,n);
		}
		PFourier(1,1) = beta0 + betaMean*PFourier11; // if betap = beta1 (order 1) -> betaMean is ok?
		transformer.inverseFourierTransform(PFourier, P());
		P.write(formatString("%s5_Padj.mrc", fnProj.c_str()));

		// Build final mask 
		//already projected + 2D dilation 
		double fmaskWidth_px;
		fmaskWidth_px = fmaskWidth/sampling;
		Image<double> Mfinal;
		Mfinal = M; 
		dilate2D(Mfinal(), Mfinal(), 0, 0, fmaskWidth_px); 
		Mfinal.write(formatString("%s6_maskFinal.mrc", fnProj.c_str())); // It seems there is no change

		// Subtraction
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I())
			DIRECT_MULTIDIM_ELEM(I(),n) = DIRECT_MULTIDIM_ELEM(I(),n)-DIRECT_MULTIDIM_ELEM(P(),n);
		I.write(formatString("%s7_subtraction.mrc", fnProj.c_str()));
		// FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I()) 
		// 	DIRECT_MULTIDIM_ELEM(I(),n) *= DIRECT_MULTIDIM_ELEM(Mfinal(),i); // Join this to the loop above	?
		// I.write(formatString("%s8_subtractionMasked.mrc", fnProj.c_str())); // Same but with values ~1e-17

		// Write particle
		writeParticle(int(i), I);
    }
    mdParticles.write(fnParticles);
 }