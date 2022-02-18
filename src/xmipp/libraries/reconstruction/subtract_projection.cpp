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
	cutFreq=getDoubleParam("--cutFreq");
	lambda=getDoubleParam("--lambda");
	fnProj=getParam("--saveProj");
	subtractAll = checkParam("--subAll");
 }

 // Show ====================================================================
 void ProgSubtractProjection::show() const{
    if (!verbose)
        return;
	std::cout<< "Input particles:\t" << fnParticles << std::endl
	<< "Reference volume:\t" << fnVolR      << std::endl
	<< "Volume mask:\t" << fnMaskVol   << std::endl
	<< "Mask:\t" << fnMask      << std::endl
	<< "Sigma:\t" << sigma       << std::endl
	<< "Iterations:\t" << iter        << std::endl
	<< "Cutoff frequency:\t" << cutFreq     << std::endl
	<< "Relaxation factor:\t" << lambda      << std::endl
	<< "Output particles:\t" << fnOut 	     << std::endl;
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
     addParamsLine("[--sigma <s=3>]\t: Decay of the filter (sigma) to smooth the mask transition");
     addParamsLine("[--iter <n=1>]\t: Number of iterations");
     addParamsLine("[--cutFreq <f=0>]\t: Cutoff frequency (<0.5)");
     addParamsLine("[--lambda <l=0>]\t: Relaxation factor for Fourier Amplitude POCS (between 0 and 1)");
	 addParamsLine("[--saveProj <structure=\"\"> ]\t: Save subtraction intermediate files (projection adjusted)");
	 addParamsLine("[--subAll]\t: Perform the subtraction of the whole image");
     addExampleLine("A typical use is:",false);
     addExampleLine("xmipp_subtract_projection -i input_particles.xmd --ref input_map.mrc --maskVol mask_vol.vol --mask mask.vol -o output_particles --iter 5 --lambda 1 --cutFreq 0.44 --sigma 3");
 }

 void ProgSubtractProjection::POCSmaskProj(const MultidimArray<double> &maskpocs, MultidimArray<double> &Ipocs) const{
 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Ipocs)
 	DIRECT_MULTIDIM_ELEM(Ipocs,n)*=DIRECT_MULTIDIM_ELEM(maskpocs,n);
 }

 void ProgSubtractProjection::POCSFourierAmplitudeProj(const MultidimArray<double> &A, MultidimArray< std::complex<double> > &FI,
		 double lambdapocs, const MultidimArray<double> &rQ, int Isize) const{
 	int Isize2 = Isize/2;
 	double Isizei = 1.0/Isize;
 	double wx;
 	double wy;
 	for (int i=0; i<YSIZE(A); ++i) {
 		FFT_IDX2DIGFREQ_FAST(i,Isize,Isize2,Isizei,wy)
 		double wy2 = wy*wy;
 		for (int j=0; j<XSIZE(A); ++j) {
 			FFT_IDX2DIGFREQ_FAST(j,Isize,Isize2,Isizei,wx)
 			double w = sqrt(wx*wx + wy2);
 			auto iw = (int)round(w*Isize);
 			double mod = std::abs(DIRECT_A2D_ELEM(FI,i,j));
 			if (mod>1e-6)
 				DIRECT_A2D_ELEM(FI,i,j)*=(((1-lambdapocs)+lambdapocs*DIRECT_A2D_ELEM(A,i,j))/mod)*DIRECT_MULTIDIM_ELEM(rQ,iw);
 		}
 	}
 }

 void ProgSubtractProjection::POCSMinMaxProj(MultidimArray<double> &Ppocs, double Im, double IM) const{
 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Ppocs) {
 		double val = DIRECT_MULTIDIM_ELEM(Ppocs,n);
 		if (val<Im)
 			DIRECT_MULTIDIM_ELEM(Ppocs,n) = Im;
 		else if (val>IM)
 			DIRECT_MULTIDIM_ELEM(Ppocs,n) = IM;
 		}
 }

 void ProgSubtractProjection::extractPhaseProj(MultidimArray< std::complex<double> > &FI) const{
 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FI) {
 		const auto *ptr = (double *)&DIRECT_MULTIDIM_ELEM(FI,n);
 		double phi = atan2(*(ptr+1),*ptr);
 		DIRECT_MULTIDIM_ELEM(FI,n) = std::complex<double>(cos(phi),sin(phi));
 	}
 }

 void ProgSubtractProjection::POCSFourierPhaseProj(const MultidimArray< std::complex<double> > &phase, MultidimArray< std::complex<double> > &FI) const{
 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(phase)
 		DIRECT_MULTIDIM_ELEM(FI,n)=std::abs(DIRECT_MULTIDIM_ELEM(FI,n))*DIRECT_MULTIDIM_ELEM(phase,n);
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

 void ProgSubtractProjection::percentileMinMax(const MultidimArray<double> &img, double &m, double &M) const{
 	MultidimArray<double> sortedI;
 	long size;
 	size = img.xdim * img.ydim;
 	img.sort(sortedI);
 	auto p005 = static_cast<double>(size) * 0.005;
 	auto p995 = static_cast<double>(size) * 0.995;
 	m = DIRECT_MULTIDIM_ELEM(sortedI, int(p005));
 	M = DIRECT_MULTIDIM_ELEM(sortedI, int(p995));
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
		// padding
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
		// crop
		mpad.window(mproj,STARTINGY(mproj), STARTINGX(mproj), FINISHINGY(mproj), FINISHINGX(mproj));
	}
	return proj;
 }

 MultidimArray<double> computeRadialAvg(const Image<double> &img, MultidimArray<double> &radial_mean){
	Image<double> imgRad;
 	imgRad = img;
	FourierTransformer transformerRad;
	MultidimArray< std::complex<double> > IFourierRad;
	MultidimArray<double> IFourierMagRad;
	transformerRad.completeFourierTransform(imgRad(),IFourierRad);
	CenterFFT(IFourierRad, true);
	FFT_magnitude(IFourierRad,IFourierMagRad);
	// Compute radial average profile (1D)
	IFourierMagRad.setXmippOrigin();
	Matrix1D<int> center(2);
	center.initZeros();
	MultidimArray<int> radial_count;
	radialAverage(IFourierMagRad, center, radial_mean, radial_count);
	int my_rad;
	FOR_ALL_ELEMENTS_IN_ARRAY3D(IFourierMagRad) {
		my_rad = (int)floor(sqrt((double)(i * i + j * j + k * k)));
		imgRad(k, i, j) = radial_mean(my_rad);
	}
	return radial_mean;
 }

 MultidimArray<double> computeRadQuotient(MultidimArray<double> &rq, const MultidimArray<double> & rmI,
		 const MultidimArray<double> &rmP){
	rq = rmI/rmP;
	FOR_ALL_ELEMENTS_IN_ARRAY1D(rq) {
		rq(i) = std::min(rq(i), 1.0);
	}
	return rq;
 }

 void ProgSubtractProjection::runIteration() {
	transformer.FourierTransform(P(),PFourier,false);
	POCSFourierAmplitudeProj(IFourierMag,PFourier, lambda, radQuotient, (int)XSIZE(I()));
	transformer.inverseFourierTransform();
	POCSMinMaxProj(P(), Imin, Imax);
	transformer.FourierTransform();
	POCSFourierPhaseProj(PFourierPhase,PFourier);
	transformer.inverseFourierTransform();
 }

 Image<double> ProgSubtractProjection::thresholdMask(Image<double> &m){
 	double maxMaskVol;
 	double minMaskVol;
 	m().computeDoubleMinMax(minMaskVol, maxMaskVol);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(m())
		DIRECT_MULTIDIM_ELEM(m(),n)=(std::abs(DIRECT_MULTIDIM_ELEM(m(),n)>maxMaskVol/20)) ? 1:0;
	FilterG2.applyMaskSpace(m());
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(m())
		DIRECT_MULTIDIM_ELEM(m(),n)=(std::abs(DIRECT_MULTIDIM_ELEM(m(),n)>0.5)) ? 1:0;
	for (int n=0; n<10; ++n) {
		closing2D(m(), m(), 4, 0, 16);
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

 Image<double> subtraction(Image<double> &I1, const Image<double> &I2,
		const Image<double> &minv, const Image<double> &m, bool subAll){
	 if (subAll){
		 FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I1())
			DIRECT_MULTIDIM_ELEM(I1,n) = DIRECT_MULTIDIM_ELEM(I1,n)-(DIRECT_MULTIDIM_ELEM(I2,n)*DIRECT_MULTIDIM_ELEM(minv,n));
	 }
	 else{
		 FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I1())
			DIRECT_MULTIDIM_ELEM(I1,n) = (DIRECT_MULTIDIM_ELEM(I1,n)-(DIRECT_MULTIDIM_ELEM(I2,n)*DIRECT_MULTIDIM_ELEM(minv,n)))*DIRECT_MULTIDIM_ELEM(m,n);
	 }

	return I1;
 }

 void ProgSubtractProjection::writeParticle(const int &ix, Image<double> &img) {
	FileName out = formatString("%d@%s.mrcs", ix, fnOut.c_str());
	img.write(out);
	mdParticles.setValue(MDL_IMAGE, out, ix);
 }

 void ProgSubtractProjection::run() {
	show();
	V.read(fnVolR);
	V().setXmippOrigin();
 	mdParticles.read(fnParticles);
 	maskVol = createMask(fnMaskVol, maskVol);
 	mask = createMask(fnMask, mask);
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
	Filter2.w1=cutFreq;
    for (size_t i = 1; i <= mdParticles.size(); ++i) {
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
    	projectVolume(V(), P, (int)XSIZE(I()), (int)XSIZE(I()), angles.rot, angles.tilt, angles.psi, &roffset);
    	projectVolume(maskVol(), PmaskVol, (int)XSIZE(I()), (int)XSIZE(I()), angles.rot, angles.tilt, angles.psi, &roffset);
    	PmaskVolI = binarizeMask(PmaskVol);
		FilterG.applyMaskSpace(PmaskVolI());
		POCSmaskProj(PmaskVolI(), P());
		POCSmaskProj(PmaskVolI(), I());
		row.getValueOrDefault(MDL_IMAGE, fnPart, "no_filename");
		Pctf = applyCTF(row, P);
    	struct Radial radial;
    	radial.meanI = computeRadialAvg(I, radial.meanI);
    	radial.meanP = computeRadialAvg(Pctf, radial.meanP);
    	radQuotient = computeRadQuotient(radQuotient, radial.meanI, radial.meanP);
		percentileMinMax(I(), Imin, Imax);
		transformer.FourierTransform(I(),IFourier,false);
		FFT_magnitude(IFourier,IFourierMag);
		transformer.FourierTransform(Pctf(),PFourierPhase,true);
		extractPhaseProj(PFourierPhase);
		for (int n=0; n<iter; ++n) {
			runIteration();
			if (cutFreq!=0) {
				Filter2.generateMask(Pctf());
				Filter2.do_generate_3dmask=true;
				Filter2.applyMaskSpace(Pctf());
			}
		}
		Image<double> IFiltered;
    	I.read(fnImage);
		IFiltered() = I();
		if (cutFreq!=0)
			Filter2.applyMaskSpace(IFiltered());
    	projectVolume(mask(), Pmask, (int)XSIZE(I()), (int)XSIZE(I()), angles.rot, angles.tilt, angles.psi, &roffset);
    	Pmaskctf = applyCTF(row, Pmask);
    	Pmaskctf = thresholdMask(Pmaskctf);
		PmaskInv = invertMask(Pmaskctf);
    	FilterG.w1=sigma;
		FilterG.applyMaskSpace(Pmaskctf());

		I = subtraction(I, P, PmaskInv, Pmaskctf, subtractAll);
		writeParticle(int(i), I);
    }
    mdParticles.write(fnParticles);
 }
