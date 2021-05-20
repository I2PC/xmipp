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


void POCSmaskProj(const MultidimArray<double> &mask, MultidimArray<double> &I)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I)
	DIRECT_MULTIDIM_ELEM(I,n)*=DIRECT_MULTIDIM_ELEM(mask,n);
}

//void POCSnonnegativeProj(MultidimArray<double> &I)
//{
//	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I)
//	DIRECT_MULTIDIM_ELEM(I,n)=std::max(0.0,DIRECT_MULTIDIM_ELEM(I,n));
//}

void POCSFourierAmplitudeProj(const MultidimArray<double> &A, MultidimArray< std::complex<double> > &FI, double lambda, MultidimArray<double> &rQ, int Isize)
{
	int Isize2 = Isize/2;
	double Isizei = 1.0/Isize;
	double wx, wy;
	for (int i=0; i<YSIZE(A); ++i)
	{
		FFT_IDX2DIGFREQ_FAST(i,Isize,Isize2,Isizei,wy);
		double wy2 = wy*wy;
		for (int j=0; j<XSIZE(A); ++j)
		{
			FFT_IDX2DIGFREQ_FAST(j,Isize,Isize2,Isizei,wx);
			double w = sqrt(wx*wx + wy2);
			int iw = (int)round(w*Isize);
			double mod = std::abs(DIRECT_A2D_ELEM(FI,i,j));
			if (mod>1e-6)
				DIRECT_A2D_ELEM(FI,i,j)*=(((1-lambda)+lambda*DIRECT_A2D_ELEM(A,i,j))/mod)*DIRECT_MULTIDIM_ELEM(rQ,iw);

		}
	}
}

void POCSFourierAmplitudeProj0(const MultidimArray<double> &A, MultidimArray< std::complex<double> > &FI, double lambda)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(A)
		{
		double mod = std::abs(DIRECT_MULTIDIM_ELEM(FI,n));
		if (mod>1e-6)
			DIRECT_MULTIDIM_ELEM(FI,n)*=((1-lambda)+lambda*DIRECT_MULTIDIM_ELEM(A,n))/mod;
		}
}

void POCSMinMaxProj(MultidimArray<double> &P, double Im, double IM)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(P)
		{
		double val = DIRECT_MULTIDIM_ELEM(P,n);
		if (val<Im)
			DIRECT_MULTIDIM_ELEM(P,n) = Im;
		else if (val>IM)
			DIRECT_MULTIDIM_ELEM(P,n) = IM;
		}
}

void extractPhaseProj(MultidimArray< std::complex<double> > &FI)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FI) {
		double *ptr = (double *)&DIRECT_MULTIDIM_ELEM(FI,n);
		double phi = atan2(*(ptr+1),*ptr);
		DIRECT_MULTIDIM_ELEM(FI,n) = std::complex<double>(cos(phi),sin(phi));
	}
}

void POCSFourierPhaseProj(const MultidimArray< std::complex<double> > &phase, MultidimArray< std::complex<double> > &FI)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(phase)
	DIRECT_MULTIDIM_ELEM(FI,n)=std::abs(DIRECT_MULTIDIM_ELEM(FI,n))*DIRECT_MULTIDIM_ELEM(phase,n);
}

void binarizeMask(MultidimArray<double> &Pmask)
{
	double maxMaskVol, minMaskVol;
	Pmask.computeDoubleMinMax(minMaskVol, maxMaskVol);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Pmask)
		DIRECT_MULTIDIM_ELEM(Pmask,n) =(DIRECT_MULTIDIM_ELEM(Pmask,n)>0.05*maxMaskVol) ? 1:0;
}

//void normMask(MultidimArray<double> &Pmask)
//{
//	double maxMaskVol, minMaskVol;
//	Pmask.computeDoubleMinMax(minMaskVol, maxMaskVol);
//	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Pmask)
//		DIRECT_MULTIDIM_ELEM(Pmask,n) /= maxMaskVol;
//}

void percentileMinMax(const MultidimArray<double> &I, double &min, double &max)
{
	MultidimArray<double> sortedI;
	int p0005, p99, size;
	size = I.xdim * I.ydim;
	p0005 = size * 0.005;
	p99 = size * 0.995;
	I.sort(sortedI);
	min = sortedI(p0005);
	max = sortedI(p99);
}


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

 void ProgSubtractProjection::run()
 {
	show();
	// Read input volume and particles.xmd
	V.read(fnVolR);
	V().setXmippOrigin();
 	MultidimArray<double> &mV=V();
 	mdParticles.read(fnParticles);

 	// Read or create input masks
	if (fnMaskVol!="")
	{
		maskVol.read(fnMaskVol);
		maskVol=maskVol();
		maskVol().setXmippOrigin();
	}
	else
	{
		maskVol().resizeNoCopy(I());
		maskVol().initConstant(1.0);
	}

	if (fnMask!="")
	{
		mask.read(fnMask);
		mask=mask();
		mask().setXmippOrigin();
	}
	else
	{
		mask().resizeNoCopy(I());
		mask().initConstant(1.0);
	}

	// Gaussian LPF to smooth mask => after projection!
	FourierFilter FilterG;
	FilterG.FilterShape=REALGAUSSIAN;
	FilterG.FilterBand=LOWPASS;
	FilterG.w1=sigma;

	// LPF to filter at desired resolution => just for volume projection?
	FourierFilter Filter2;
	Filter2.FilterBand=LOWPASS;
	Filter2.FilterShape=RAISED_COSINE;
	Filter2.raised_w=0.02;
	Filter2.w1=cutFreq;

 	int ix_particle = 0;

    FOR_ALL_OBJECTS_IN_METADATA(mdParticles)
    {
    	// Read particle image
    	mdParticles.getRow(row,__iter.objId);
    	row.getValue(MDL_IMAGE, fnImage);
		std::cout<< "Particle: " << fnImage << std::endl;
    	I.read(fnImage);
    	I().setXmippOrigin();
		I.write("I.mrc");
		// Read particle metadata
     	row.getValue(MDL_ANGLE_ROT, rot);
     	row.getValue(MDL_ANGLE_TILT, tilt);
     	row.getValue(MDL_ANGLE_PSI, psi);

    	// Compute projection of the volume
    	projectVolume(mV, P, (int)XSIZE(I()), (int)XSIZE(I()), rot, tilt, psi);
		P.write("P.mrc");
    	// Compute projection of the volume mask
		MultidimArray<double> &mMaskVol=maskVol();
    	projectVolume(mMaskVol, PmaskVol, (int)XSIZE(I()), (int)XSIZE(I()), rot, tilt, psi);
    	PmaskVol.write("mask.mrc");
    	// Binarize volume mask
    	binarizeMask(PmaskVol());
    	PmaskVol.write("maskBin.mrc");
    	// Filter mask with Gaussian
		FilterG.applyMaskSpace(PmaskVol());
    	PmaskVol.write("maskBinGaus.mrc");

		// Apply bin volume mask to particle and volume projection
		POCSmaskProj(PmaskVol(), P());
		POCSmaskProj(PmaskVol(), I());
		I.write("ImaskVol.mrc");  //
		P.write("PmaskVol.mrc");

    	// Check if particle has CTF and apply it
		//
     	if ((row.containsLabel(MDL_CTF_DEFOCUSU) || row.containsLabel(MDL_CTF_MODEL)))
     	{
     		hasCTF=true;
     		ctf.readFromMdRow(row);
     		ctf.produceSideInfo();
     		defocusU=ctf.DeltafU;
     		defocusV=ctf.DeltafV;
     		ctfAngle=ctf.azimuthal_angle;

 	 	 	FilterCTF.FilterBand = CTF;
 	 	 	FilterCTF.ctf.enable_CTFnoise = false;
 	 		FilterCTF.ctf = ctf;
 	 		FilterCTF.generateMask(P());
 	 		FilterCTF.applyMaskSpace(P());
 			P.write("Pctf.mrc");
 	 	}

     	// Compute radial averages
    	Image<double> Irad, Prad;
    	Irad = I;
    	Prad = P;

		// Compute |FT(radial averages)|
		FourierTransformer transformerRad;
		MultidimArray< std::complex<double> > IFourierRad, PFourierRad;
		MultidimArray<double> IFourierMagRad, PFourierMagRad, radQuotient;
		transformerRad.completeFourierTransform(Irad(),IFourierRad);  //Irad for 2D
		CenterFFT(IFourierRad, true);
		FFT_magnitude(IFourierRad,IFourierMagRad);
		transformerRad.completeFourierTransform(Prad(),PFourierRad);  //Prad for 2D
		CenterFFT(PFourierRad, true);
		FFT_magnitude(PFourierRad,PFourierMagRad);

    	// Compute IradAvg profile (1D)
        IFourierMagRad.setXmippOrigin();
        Matrix1D<int> center(2);
        center.initZeros();
        MultidimArray<double> radial_meanI;
        MultidimArray<int> radial_count;
        radialAverage(IFourierMagRad, center, radial_meanI, radial_count);
        radial_meanI.write("Irad.txt");
        int my_rad;
        FOR_ALL_ELEMENTS_IN_ARRAY3D(IFourierMagRad)
        {
            my_rad = (int)floor(sqrt((double)(i * i + j * j + k * k)));
            Irad(k, i, j) = radial_meanI(my_rad);
        }
		Irad.write("Irad.mrc");

    	// Compute PradAvg profile (1D)
		PFourierMagRad.setXmippOrigin();
        center.initZeros();
        MultidimArray<double> radial_meanP;
        radialAverage(PFourierMagRad, center, radial_meanP, radial_count);
        radial_meanP.write("Prad.txt");
        FOR_ALL_ELEMENTS_IN_ARRAY3D(PFourierMagRad)
        {
            my_rad = (int)floor(sqrt((double)(i * i + j * j + k * k)));
            Prad(k, i, j) = radial_meanP(my_rad);
        }
		Prad.write("Prad.mrc");

		// Compute adjustment quotient for POCS amplitude (and POCS phase?)
		radQuotient = radial_meanI/radial_meanP;
		FOR_ALL_ELEMENTS_IN_ARRAY1D(radQuotient)
		{
			radQuotient(i) = std::min(radQuotient(i), 1.0);
		}
		radQuotient.write("radQuotient.txt");

     	// Compute what need for the loop of POCS
		FourierTransformer transformer;
		MultidimArray< std::complex<double> > IFourier, PFourier;
		MultidimArray<double> IFourierMag;
		double Imin, Imax;
		percentileMinMax(I(), Imin, Imax);
		transformer.FourierTransform(I(),IFourier,false);
		FFT_magnitude(IFourier,IFourierMag);
		MultidimArray<std::complex<double> > PFourierPhase;
		transformer.FourierTransform(P(),PFourierPhase,true);
		extractPhaseProj(PFourierPhase);

		// POCS loop
		for (int n=0; n<iter; ++n)
		{
			transformer.FourierTransform(P(),PFourier,false);
			POCSFourierAmplitudeProj(IFourierMag,PFourier, lambda, radQuotient, (int)XSIZE(I()));
			transformer.inverseFourierTransform();
			P.write("Pamp.mrc");
//			POCSMinMaxProj(P(), Imin, Imax);
			P.write("Pminmax.mrc");
			transformer.FourierTransform();
			POCSFourierPhaseProj(PFourierPhase,PFourier);
			transformer.inverseFourierTransform();
			P.write("Pphase.mrc");
			if (cutFreq!=0)
			{
				Filter2.generateMask(P());
				Filter2.do_generate_3dmask=true;
				Filter2.applyMaskSpace(P());
				P.write("Pfilt.mrc");
			}
		}

//		Image<double> IFiltered;
//    	I.read(fnImage);
//		IFiltered() = I();
//		if (cutFreq!=0)
//			Filter2.applyMaskSpace(IFiltered());

		// Project subtraction mask
		MultidimArray<double> &mMask=mask();
    	projectVolume(mMask, Pmask, (int)XSIZE(I()), (int)XSIZE(I()), rot, tilt, psi);
    	Pmask.write("maskfocus.mrc");
    	// Binarize subtraction mask
    	binarizeMask(Pmask());
//    	normMask(Pmask());
    	Pmask.write("maskfocusbin.mrc");

    	// Invert projected mask
		PmaskInv = Pmask;
    	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PmaskInv())
		DIRECT_MULTIDIM_ELEM(PmaskInv,n) = (DIRECT_MULTIDIM_ELEM(PmaskInv,n)*(-1))+1;
    	PmaskInv.write("maskfocusInv.mrc");

    	// Filter mask with Gaussian
		FilterG.applyMaskSpace(Pmask());
    	Pmask.write("maskfocusbingaus.mrc");

		// Save particle and projection adjusted
//		if (fnPart!="" && fnProj!="")
//		{
		I.write("Ifilt.mrc");
		P.write("Padj.mrc");
//		}

    	// SUBTRACTION
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I())
		DIRECT_MULTIDIM_ELEM(I,n) = (DIRECT_MULTIDIM_ELEM(I,n)-(DIRECT_MULTIDIM_ELEM(P,n)*DIRECT_MULTIDIM_ELEM(PmaskInv,n))) * DIRECT_MULTIDIM_ELEM(Pmask,n);


//		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I())
//		DIRECT_MULTIDIM_ELEM(I,n) = (DIRECT_MULTIDIM_ELEM(I,n)-DIRECT_MULTIDIM_ELEM(P,n))* DIRECT_MULTIDIM_ELEM(Pmask,n);

		// Save subtracted particles in metadata
		ix_particle++;
		FileName out = formatString("%d@%s.mrcs", ix_particle, fnOut.c_str());
		I.write(out);
		mdParticles.setValue(MDL_IMAGE, out, ix_particle);
    }
    mdParticles.write(fnParticles);
 }



