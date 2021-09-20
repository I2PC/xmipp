// /***************************************************************************
//  *
//  * Authors:    Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
//  *
//  * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
//  *
//  * This program is free software; you can redistribute it and/or modify
//  * it under the terms of the GNU General Public License as published by
//  * the Free Software Foundation; either version 2 of the License, or
//  * (at your option) any later version.
//  *
//  * This program is distributed in the hope that it will be useful,
//  * but WITHOUT ANY WARRANTY; without even the implied warranty of
//  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  * GNU General Public License for more details.
//  *
//  * You should have received a copy of the GNU General Public License
//  * along with this program; if not, write to the Free Software
//  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
//  * 02111-1307  USA
//  *
//  *  All comments concerning this program package may be sent to the
//  *  e-mail address 'xmipp@cnb.csic.es'
//  ***************************************************************************/

 #include "subtract_projection.h"
 #include "core/transformations.h"
 #include "core/xmipp_image_extension.h"
 #include "core/xmipp_image_generic.h"
 #include "core/xmipp_fft.h"
 #include "core/xmipp_fftw.h"
 #include "data/projection.h"
 #include "data/mask.h"
 #include <iostream>
 #include <string>
 #include <sstream>
 #include "data/image_operate.h"

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
	fnPart=getParam("--savePart");
	if (fnPart=="")
		fnPart="particle_filtered.mrc";
	fnProj=getParam("--saveProj");
	if (fnProj=="")
		fnProj="projection_adjusted.mrc";
 }

 // Show ====================================================================
 void ProgSubtractProjection::show()
 {
    if (!verbose)
        return;
	std::cout
	<< "Input particles:   	" << fnParticles << std::endl
	<< "Reference volume:   " << fnVolR      << std::endl
	<< "Volume mask:  		" << fnMaskVol   << std::endl
	<< "Mask:    	   		" << fnMask      << std::endl
	<< "Sigma:				" << sigma       << std::endl
	<< "Iterations:    	   	" << iter        << std::endl
	<< "Cutoff frequency:  	" << cutFreq     << std::endl
	<< "Relaxation factor:  " << lambda      << std::endl
	<< "Output particles: 	" << fnOut 	     << std::endl
	;
 }

 // usage ===================================================================
 void ProgSubtractProjection::defineParams()
 {
     //Usage
     addUsageLine("");
     //Parameters
     addParamsLine("-i <particles>          	: Particles metadata (.xmd file)");
     addParamsLine("--ref <volume>      		: Reference volume to subtract");
     addParamsLine("[-o <structure=\"\">] 		: Output filename suffix for subtracted particles");
     addParamsLine("                      		: If no name is given, then output_particles.xmd");
     addParamsLine("[--maskVol <maskVol=\"\">]  : 3D mask for input volume");
     addParamsLine("[--mask <mask=\"\">]  		: 3D mask for the region of subtraction");
     addParamsLine("[--sigma <s=3>]    			: Decay of the filter (sigma) to smooth the mask transition");
     addParamsLine("[--iter <n=1>]        		: Number of iterations");
     addParamsLine("[--cutFreq <f=0>]       	: Cutoff frequency (<0.5)");
     addParamsLine("[--lambda <l=0>]       		: Relaxation factor for Fourier Amplitude POCS (between 0 and 1)");
	 addParamsLine("[--savePart <structure=\"\"> ]  : Save subtraction intermediate files (particle filtered)");
	 addParamsLine("[--saveProj <structure=\"\"> ]  : Save subtraction intermediate files (projection adjusted)");
     addExampleLine("A typical use is:",false);
     addExampleLine("xmipp_subtract_projection -i input_particles.xmd --ref input_map.mrc --maskVol mask_vol.vol --mask mask.vol -o output_particles --iter 5 --lambda 1 --cutFreq 0.44 --sigma 3");
 }

 void POCSmaskProj(const MultidimArray<double> &mask, MultidimArray<double> &I) {
 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I)
 	DIRECT_MULTIDIM_ELEM(I,n)*=DIRECT_MULTIDIM_ELEM(mask,n);
 }

 void POCSnonnegativeProj(MultidimArray<double> &I) {
 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I)
 	DIRECT_MULTIDIM_ELEM(I,n)=std::max(0.0,DIRECT_MULTIDIM_ELEM(I,n));
 }

 void POCSFourierAmplitudeProj(const MultidimArray<double> &A, MultidimArray< std::complex<double> > &FI, double lambda, MultidimArray<double> &rQ, int Isize) {
 	int Isize2 = Isize/2;
 	double Isizei = 1.0/Isize;
 	double wx, wy;
 	for (int i=0; i<YSIZE(A); ++i) {
 		FFT_IDX2DIGFREQ_FAST(i,Isize,Isize2,Isizei,wy);
 		double wy2 = wy*wy;
 		for (int j=0; j<XSIZE(A); ++j) {
 			FFT_IDX2DIGFREQ_FAST(j,Isize,Isize2,Isizei,wx);
 			double w = sqrt(wx*wx + wy2);
 			int iw = (int)round(w*Isize);
 			double mod = std::abs(DIRECT_A2D_ELEM(FI,i,j));
 			if (mod>1e-6)
 				DIRECT_A2D_ELEM(FI,i,j)*=(((1-lambda)+lambda*DIRECT_A2D_ELEM(A,i,j))/mod)*DIRECT_MULTIDIM_ELEM(rQ,iw);
 		}
 	}
 }

 void POCSFourierAmplitudeProj0(const MultidimArray<double> &A, MultidimArray< std::complex<double> > &FI, double lambda) {
 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(A) {
 		double mod = std::abs(DIRECT_MULTIDIM_ELEM(FI,n));
 		if (mod>1e-6)
 			DIRECT_MULTIDIM_ELEM(FI,n)*=((1-lambda)+lambda*DIRECT_MULTIDIM_ELEM(A,n))/mod;
 		}
 }

 void POCSMinMaxProj(MultidimArray<double> &P, double Im, double IM) {
 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(P) {
 		double val = DIRECT_MULTIDIM_ELEM(P,n);
 		if (val<Im)
 			DIRECT_MULTIDIM_ELEM(P,n) = Im;
 		else if (val>IM)
 			DIRECT_MULTIDIM_ELEM(P,n) = IM;
 		}
 }

 void extractPhaseProj(MultidimArray< std::complex<double> > &FI) {
 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FI) {
 		double *ptr = (double *)&DIRECT_MULTIDIM_ELEM(FI,n);
 		double phi = atan2(*(ptr+1),*ptr);
 		DIRECT_MULTIDIM_ELEM(FI,n) = std::complex<double>(cos(phi),sin(phi));
 	}
 }

 void POCSFourierPhaseProj(const MultidimArray< std::complex<double> > &phase, MultidimArray< std::complex<double> > &FI) {
 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(phase)
 	DIRECT_MULTIDIM_ELEM(FI,n)=std::abs(DIRECT_MULTIDIM_ELEM(FI,n))*DIRECT_MULTIDIM_ELEM(phase,n);
 }

 Image<double> ProgSubtractProjection::createMask(FileName &fnM, Image<double> &m) {
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
	r.getValue(MDL_IMAGE, fnImage);
	std::cout<< "Particle: " << fnImage << std::endl;
	I.read(fnImage);
	I().setXmippOrigin();
	I.write(formatString("%s/I.mrc", fnProj.c_str()));
 }

 void ProgSubtractProjection::percentileMinMax(const MultidimArray<double> &img, double &m, double &M) {
 	MultidimArray<double> sortedI;
 	int p0005, p99, size;
 	size = img.xdim * img.ydim;
 	p0005 = size * 0.005;
 	p99 = size * 0.995;
 	img.sort(sortedI);
 	m = sortedI(p0005);
 	M = sortedI(p99);
 }

 void ProgSubtractProjection::applyCTF(const MDRowVec &r) {
	if ((r.containsLabel(MDL_CTF_DEFOCUSU) || r.containsLabel(MDL_CTF_MODEL))){
		hasCTF=true;
		ctf.readFromMdRow(r);
		ctf.produceSideInfo();
		defocusU=ctf.DeltafU;
		defocusV=ctf.DeltafV;
		ctfAngle=ctf.azimuthal_angle;
	 	FilterCTF.FilterBand = CTF;
	 	FilterCTF.ctf.enable_CTFnoise = false;
		FilterCTF.ctf = ctf;
		FilterCTF.generateMask(P());
		FilterCTF.applyMaskSpace(P());
		P.write(formatString("%s/Pctf.mrc", fnProj.c_str()));
	}
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
	// Compute IradAvg profile (1D)
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
	P.write(formatString("%s/Pamp.mrc", fnProj.c_str()));
	POCSMinMaxProj(P(), Imin, Imax);
	P.write(formatString("%s/Pminmax.mrc", fnProj.c_str()));
	transformer.FourierTransform();
	POCSFourierPhaseProj(PFourierPhase,PFourier);
	transformer.inverseFourierTransform();
	P.write(formatString("%s/Pphase.mrc", fnProj.c_str()));
 }

 Image<double> ProgSubtractProjection::binarizeMask(Projection &m) {
 	double maxMaskVol, minMaskVol;
 	m().computeDoubleMinMax(minMaskVol, maxMaskVol);
 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(m())
 		DIRECT_MULTIDIM_ELEM(m(),n) =(DIRECT_MULTIDIM_ELEM(m(),n)>0.05*maxMaskVol) ? 1:0;
 	return m;
 }

 Image<double> ProgSubtractProjection::normMask(Image<double> &m) {
 	double maxMaskVol, minMaskVol;
 	m().computeDoubleMinMax(minMaskVol, maxMaskVol);
 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(m())
 		DIRECT_MULTIDIM_ELEM(m(),n) /= maxMaskVol;
 	return m;
 }

 Image<double> invertMask(const Image<double> &m) {
	Image<double> PmaskI = m;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PmaskI())
		DIRECT_MULTIDIM_ELEM(PmaskI,n) = (DIRECT_MULTIDIM_ELEM(PmaskI,n)*(-1))+1;
	return PmaskI;
 }

 Image<double> subtraction(Image<double> &I1, const Image<double> &I2,
		 const Image<double> &minv, const Image<double> &m){
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I1())
		DIRECT_MULTIDIM_ELEM(I1,n) = (DIRECT_MULTIDIM_ELEM(I1,n)-(DIRECT_MULTIDIM_ELEM(I2,n)*DIRECT_MULTIDIM_ELEM(minv,n))) * DIRECT_MULTIDIM_ELEM(m,n);
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
	// Gaussian LPF to smooth mask => after projection!
	FilterG.FilterShape=REALGAUSSIAN;
	FilterG.FilterBand=LOWPASS;
	FilterG.w1=sigma;
	// LPF to filter at desired resolution => just for volume projection?
	Filter2.FilterBand=LOWPASS;
	Filter2.FilterShape=RAISED_COSINE;
	Filter2.raised_w=0.02;
	Filter2.w1=cutFreq;
    for (size_t i = 1; i < mdParticles.size(); ++i) {
    	row = mdParticles.getRowVec(i);
    	readParticle(row);
     	row.getValue(MDL_ANGLE_ROT, rot);
     	row.getValue(MDL_ANGLE_TILT, tilt);
     	row.getValue(MDL_ANGLE_PSI, psi);
    	projectVolume(V(), P, (int)XSIZE(I()), (int)XSIZE(I()), rot, tilt, psi);
		P.write(formatString("%s/P.mrc", fnProj.c_str()));
    	projectVolume(maskVol(), PmaskVol, (int)XSIZE(I()), (int)XSIZE(I()), rot, tilt, psi);
    	PmaskVol.write(formatString("%s/mask.mrc", fnProj.c_str()));
    	PmaskVolI = binarizeMask(PmaskVol);
    	PmaskVolI.write(formatString("%s/maskBin.mrc", fnProj.c_str()));
		FilterG.applyMaskSpace(PmaskVolI());
		PmaskVolI.write(formatString("%s/maskBinGauss.mrc", fnProj.c_str()));
		POCSmaskProj(PmaskVolI(), P());
		POCSmaskProj(PmaskVolI(), I());
		I.write(formatString("%s/ImaskVol.mrc", fnProj.c_str()));
		P.write(formatString("%s/PmaskVol.mrc", fnProj.c_str()));
		applyCTF(row);
		radial_meanI = computeRadialAvg(I, radial_meanI);
		radial_meanI.write(formatString("%s/Irad.txt", fnProj.c_str()));
		radial_meanP = computeRadialAvg(P, radial_meanP);
		radial_meanP.write(formatString("%s/Prad.txt", fnProj.c_str()));
		radQuotient = computeRadQuotient(radQuotient, radial_meanI, radial_meanP);
		radQuotient.write(formatString("%s/radQuotient.txt", fnProj.c_str()));
		percentileMinMax(I(), Imin, Imax);
		transformer.FourierTransform(I(),IFourier,false);
		FFT_magnitude(IFourier,IFourierMag);
		transformer.FourierTransform(P(),PFourierPhase,true);
		extractPhaseProj(PFourierPhase);
		for (int n=0; n<iter; ++n) {
			runIteration();
			if (cutFreq!=0) {
				Filter2.generateMask(P());
				Filter2.do_generate_3dmask=true;
				Filter2.applyMaskSpace(P());
				P.write(formatString("%s/Pfilt.mrc", fnProj.c_str()));
			}
		}
		Image<double> IFiltered;
    	I.read(fnImage);
		IFiltered() = I();
		if (cutFreq!=0)
			Filter2.applyMaskSpace(IFiltered());
    	projectVolume(mask(), Pmask, (int)XSIZE(I()), (int)XSIZE(I()), rot, tilt, psi);
    	Pmask.write(formatString("%s/maskFocus.mrc", fnProj.c_str()));
    	PmaskFilt = binarizeMask(Pmask);
    	PmaskFilt = normMask(PmaskFilt);
    	PmaskFilt.write(formatString("%s/maskFocusBin.mrc", fnProj.c_str()));
		PmaskInv = invertMask(PmaskFilt);
		PmaskInv.write(formatString("%s/maskFocusInv.mrc", fnProj.c_str()));
		FilterG.applyMaskSpace(PmaskFilt());
		PmaskFilt.write(formatString("%s/maskFocusBinGauss.mrc", fnProj.c_str()));
		I.write(formatString("%s/Ifilt.mrc", fnProj.c_str()));
		P.write(formatString("%s/Padj.mrc", fnProj.c_str()));
		I = subtraction(I, P, PmaskInv, PmaskFilt);
		writeParticle(i, I);
    }
    mdParticles.write(fnParticles);
 }
