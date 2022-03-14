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
 #include "data/fourier_projection.h"
 #include "data/projection.h"
 #include "data/mask.h"
 #include "data/filters.h"
 #include "data/morphology.h"
 #include <core/alglib/dataanalysis.h>
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
     addParamsLine("\t: If no name is given, then output_particles.xmd");
     addParamsLine("[--maskVol <maskVol=\"\">]\t: 3D mask for input volume");
     addParamsLine("[--mask <mask=\"\">]\t: 3D mask for the region of subtraction");
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

 void ProgSubtractProjection::readParticle(const MDRowVec &r){
	r.getValueOrDefault(MDL_IMAGE, fnImage, "no_filename");
	I.read(fnImage);
	I().setXmippOrigin();
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
 	vM = createMask(fnMaskVol, vM); // Actually now this mask is mask keep and the other seems to not be needed
	// Initialize Gaussian LPF to smooth mask
	FilterG.FilterShape=REALGAUSSIAN;
	FilterG.FilterBand=LOWPASS;
	FilterG.w1=sigma;
	// Initialize Gaussian LPF to threshold mask with ctf
	FilterG2.FilterShape=REALGAUSSIAN;
	FilterG2.FilterBand=LOWPASS;
	FilterG2.w1=2;
	// Initialize LPF to filter at desired resolution
	Filter2.FilterBand=LOWPASS;
	Filter2.FilterShape=RAISED_COSINE;
	Filter2.raised_w=0.02;
	double cutFreq = sampling/maxResol;
	Filter2.w1 = cutFreq;
	// Initialize Fourier projectors
	FourierProjector *projector = new FourierProjector(V(), padFourier, cutFreq, xmipp_transformation::BSPLINE3);
	FourierProjector *projectorMask = new FourierProjector(vM(), padFourier, cutFreq, xmipp_transformation::BSPLINE3);

    for (size_t i = 1; i <= mdParticles.size(); ++i) {
    	// Read particle and metadata
    	row = mdParticles.getRowVec(i);
    	readParticle(row);
    	struct Angles angles;
     	row.getValueOrDefault(MDL_ANGLE_ROT, angles.rot, 0);
     	row.getValueOrDefault(MDL_ANGLE_TILT, angles.tilt, 0);
     	row.getValueOrDefault(MDL_ANGLE_PSI, angles.psi, 0);
     	roffset.initZeros(2);
     	row.getValueOrDefault(MDL_SHIFT_X, roffset(0), 0);
     	row.getValueOrDefault(MDL_SHIFT_Y, roffset(1), 0);
     	roffset *= -1;

     	// Project volume
     	MultidimArray<double> *ctfImage = nullptr;
    	projectVolume(*projector, P, (int)XSIZE(I()), (int)XSIZE(I()), angles.rot, angles.tilt, angles.psi, ctfImage);
    	P.write(formatString("%s0_initialProjection.mrc", fnProj.c_str()));

    	// Project and smooth big mask
    	projectVolume(*projectorMask, Pmask, (int)XSIZE(I()), (int)XSIZE(I()), angles.rot, angles.tilt, angles.psi, ctfImage);
    	Pmask.write(formatString("%s0_initalMaskProjection.mrc", fnProj.c_str()));
    	M = binarizeMask(Pmask);
		M.write(formatString("%s1_MaskBin.mrc", fnProj.c_str()));
		FilterG.applyMaskSpace(M());
		M.write(formatString("%s1_MaskSmooth.mrc", fnProj.c_str()));

		// Fourier Transform
		FourierTransformer transformerP;
		transformerP.FourierTransform(P(),PFourier,false);

		// Construct image representing the frequency of each pixel
		auto deltaWx = 1/(double)XSIZE(P());
		auto deltaWy = 1/(double)YSIZE(P());
		MultidimArray<double> wx;
		wx.initZeros(XSIZE(P()), YSIZE(P()));
		wx += 1;
		MultidimArray<double> aux;
		aux.initZeros(XSIZE(P()), 1);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(aux)
			DIRECT_MULTIDIM_ELEM(aux,n) += n;
		for (int j=0; j<YSIZE(wx); ++j)
		{
		 	for (int i=0; i<XSIZE(wx); ++i)
		 	{
				DIRECT_A2D_ELEM(wx,i,j) *= DIRECT_A2D_ELEM(aux,i,0) * deltaWx - 0.5;
			}
		}
		MultidimArray<double> wy;
		wy.initZeros(XSIZE(P()), YSIZE(P()));
		wy += 1;
		aux.initZeros(1, YSIZE(P()));
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(aux)
			DIRECT_MULTIDIM_ELEM(aux,n) += n;
		for (int j=0; j<YSIZE(wy); ++j)
		{
		 	for (int i=0; i<XSIZE(wy); ++i)
		 	{
				DIRECT_A2D_ELEM(wy,i,j) *= DIRECT_A2D_ELEM(aux,i,0) * deltaWy- 0.5;
			}
		}
		MultidimArray<double> w;
		w.initZeros(XSIZE(wx),YSIZE(wx));
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(w)
			DIRECT_MULTIDIM_ELEM(w,n) = sqrt(DIRECT_MULTIDIM_ELEM(wx,n)*DIRECT_MULTIDIM_ELEM(wx,n) + DIRECT_MULTIDIM_ELEM(wy,n)*DIRECT_MULTIDIM_ELEM(wy,n));
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(w)
			DIRECT_MULTIDIM_ELEM(w,n) = round(DIRECT_MULTIDIM_ELEM(w,n)*XSIZE(P()));
		CenterFFT(w, false);
	 	double maxw;
	 	double minw;
	 	w.computeDoubleMinMax(minw, maxw); // compute just max?

		// Apply CTF
		Pctf = applyCTF(row, P);
		Pctf.write(formatString("%s3_Pctf.mrc", fnProj.c_str()));
		transformer.FourierTransform(Pctf(),PFourier,false);
		// Compute inverse mask
		iM = invertMask(M);
		iM.write(formatString("%s4_iM.mrc", fnProj.c_str()));
		// Compute IiM = I*iM
		Image<double> IiM;
		IiM().initZeros(XSIZE(wx),YSIZE(wx));
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I())
		 	DIRECT_MULTIDIM_ELEM(IiM(),n) = DIRECT_MULTIDIM_ELEM(I(),n) * DIRECT_MULTIDIM_ELEM(iM,n);
		IiM.write(formatString("%s4_IiM.mrc", fnProj.c_str()));
		FourierTransformer transformerIiM;
		MultidimArray< std::complex<double> > IiMFourier;
		transformerIiM.FourierTransform(IiM(),IiMFourier,false);
		// Compute PiM = P*iM
		Image<double> PiM;
		PiM().initZeros(XSIZE(wx),YSIZE(wx));
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Pctf())
		 	DIRECT_MULTIDIM_ELEM(PiM(),n) = DIRECT_MULTIDIM_ELEM(Pctf(),n) * DIRECT_MULTIDIM_ELEM(iM,n);
		PiM.write(formatString("%s4_PiM.mrc", fnProj.c_str()));
		FourierTransformer transformerPiM;
		MultidimArray< std::complex<double> > PiMFourier;
		transformerPiM.FourierTransform(PiM(),PiMFourier,false);

		// -.-.-.-.-.-.-.-.-

		// Estimate transformation T(w) //
		double nyquist = 2*sampling; 
		MultidimArray< std::complex<double> > num;
		num.initZeros(nyquist, 1); 
		MultidimArray< std::complex<double> > den;
		den.initZeros(nyquist, 1);

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PiMFourier) 
		{
			DIRECT_MULTIDIM_ELEM(num,n) += real(DIRECT_MULTIDIM_ELEM(IiMFourier,n)*DIRECT_MULTIDIM_ELEM(PiMFourier,n)) 
											+ imag(DIRECT_MULTIDIM_ELEM(IiMFourier,n)*DIRECT_MULTIDIM_ELEM(PiMFourier,n));
			DIRECT_MULTIDIM_ELEM(den,n) += real(DIRECT_MULTIDIM_ELEM(PiMFourier,n)*DIRECT_MULTIDIM_ELEM(PiMFourier,n)) 
											+ imag(DIRECT_MULTIDIM_ELEM(PiMFourier,n)*DIRECT_MULTIDIM_ELEM(PiMFourier,n));
		}
			
		alglib::linearmodel lm;
		alglib::lrreport ar;
		int info;
		MultidimArray< std::complex<double> > beta;
		beta.initZeros(nyquist, 1);
		MultidimArray<double> R2;
		R2.initZeros(nyquist, 1);

		for (int i=0; i<nyquist; ++i) 
		{
			for (int ix=0; ix<3; ix++)
			{
				alglib::real_2d_array quotient = real(num/den);
				lrbuild(quotient, XSIZE(IiMFourier), ix, info, lm, ar); 
				beta(ix,i) = lrunpack(lm, quotient, ix);
				R2(ix, i) = lrrmserror(lm, quotient, XSIZE(IiMFourier));
			}
		}

		std::complex<double> beta1 = beta(0).mean();
		std::complex<double> beta2 = beta(1).mean();
		std::complex<double> beta3 = beta(2).mean();
		// double Tw = beta1 + beta2*w + beta3*w*w;
		std::complex<double> Tw = beta1;
		MultidimArray< std::complex<double> > PFourierAdjusted;
		PFourierAdjusted = Tw * PFourier;
		std::complex<double> beta0;
		beta0 = IiMFourier(1,1)-beta1*PiMFourier(1,1);

		// Apply adjustment
		PFourierAdjusted(1,1) = beta0 + Tw*PFourier(1,1);
		Image<double> PAdjusted;
		transformer.inverseFourierTransform(PFourierAdjusted, PAdjusted);

		// Build final mask 
		double fmaskWidth_px;
		fmaskWidth_px = fmaskWidth/sampling;
		Image<double> Mfinal;
		Mfinal().resizeNoCopy(I());  // Change by a mask fmaskWidth_px bigger for each side than Idiff != 0
		Mfinal().initConstant(1.0);

		// Subtraction
		Image<double> Idiff;
		Idiff().resizeNoCopy(I());  
		Idiff().initConstant(1.0);

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Idiff())
			DIRECT_MULTIDIM_ELEM(Idiff(),i) = (DIRECT_MULTIDIM_ELEM(I(),i)-DIRECT_MULTIDIM_ELEM(PAdjusted(),i)) 
											* DIRECT_MULTIDIM_ELEM(Mfinal(),i); 

		// Write particle
		writeParticle(int(i), Idiff);
    }
    mdParticles.write(fnParticles);
 }
