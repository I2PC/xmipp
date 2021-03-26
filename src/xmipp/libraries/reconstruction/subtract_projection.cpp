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
 #include "data/projection.h"
 #include "data/mask.h"
 #include <iostream>
 #include <string>
 #include <sstream>


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

void POCSFourierAmplitudeProj(const MultidimArray<double> &A, MultidimArray< std::complex<double> > &FI, double lambda)
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

//void computeEnergyProj(MultidimArray<double> &Idiff, MultidimArray<double> &Iact, double energy)
//{
//	Idiff = Idiff - Iact;
//	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Idiff)
//	energy+=DIRECT_MULTIDIM_ELEM(Idiff,n)*DIRECT_MULTIDIM_ELEM(Idiff,n);
//	energy = sqrt(energy/MULTIDIM_SIZE(Idiff));
//	std::cout<< "Energy: " << energy << std::endl;
//}

void BinarizeMask(const Image<double> &mask, MultidimArray<double> &Pmask)
{
	double maxMaskVol, minMaskVol;
	mask().computeDoubleMinMax(minMaskVol, maxMaskVol);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Pmask)
		DIRECT_MULTIDIM_ELEM(Pmask,n) =(DIRECT_MULTIDIM_ELEM(Pmask,n)>0.1*maxMaskVol) ? 1:0;
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

	// Gaussian LPF to smooth mask => after project!
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
    	// Read particle image and metadata
    	mdParticles.getRow(row,__iter.objId);
    	row.getValue(MDL_IMAGE, fnImage);
		std::cout<< "Particle: " << fnImage << std::endl;
    	I.read(fnImage);
    	I().setXmippOrigin();
     	row.getValue(MDL_ANGLE_ROT, rot);
     	row.getValue(MDL_ANGLE_TILT, tilt);
     	row.getValue(MDL_ANGLE_PSI, psi);
    	// Compute projection of the volume
    	projectVolume(mV, P, (int)XSIZE(I()), (int)XSIZE(I()), rot, tilt, psi);
    	// Compute projection of the volume mask
		MultidimArray<double> &mMaskVol=maskVol();
    	projectVolume(mMaskVol, PmaskVol, (int)XSIZE(I()), (int)XSIZE(I()), rot, tilt, psi);
    	// Binarize volume mask
    	BinarizeMask(maskVol(), PmaskVol());
    	// Filter mask with Gaussian
		FilterG.applyMaskSpace(PmaskVol());

		// Apply bin volume mask to particle and volume projection
		POCSmaskProj(PmaskVol(), P());
		POCSmaskProj(PmaskVol(), I());

    	// Check if particle has CTF and apply it
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
 	 	}

		FourierTransformer transformer;
		MultidimArray< std::complex<double> > IFourier, PFourier;
		MultidimArray<double> IFourierMag;

		double Imin, Imax;
		I().computeDoubleMinMax(Imin, Imax);

		transformer.FourierTransform(I(),IFourier,false);
		FFT_magnitude(IFourier,IFourierMag);
		double std1 = I().computeStddev();

		MultidimArray<std::complex<double> > PFourierPhase;
		transformer.FourierTransform(P(),PFourierPhase,true);
		extractPhaseProj(PFourierPhase);

		for (int n=0; n<iter; ++n)
		{
			transformer.FourierTransform(P(),PFourier,false);
			POCSFourierAmplitudeProj(IFourierMag,PFourier, lambda);
			transformer.inverseFourierTransform();
			POCSMinMaxProj(P(), Imin, Imax);
			transformer.FourierTransform();
			POCSFourierPhaseProj(PFourierPhase,PFourier);
			transformer.inverseFourierTransform();
			double std2 = P().computeStddev();
			P()*=std1/std2;
			if (cutFreq!=0)
			{
				Filter2.generateMask(P());
				Filter2.do_generate_3dmask=true;
				Filter2.applyMaskSpace(P());
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
    	// Binarize subtraction mask
    	BinarizeMask(mask(), Pmask());

    	// Invert projected mask
		PmaskInv = Pmask;
    	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PmaskInv())
		DIRECT_MULTIDIM_ELEM(PmaskInv,n) = (DIRECT_MULTIDIM_ELEM(PmaskInv,n)*(-1))+1;

    	// Filter mask with Gaussian
		FilterG.applyMaskSpace(Pmask());

		// Save binarized mask and projection adjusted
		if (fnPart!="" && fnProj!="")
		{
			Pmask.write(fnPart);
			P.write(fnProj);
		}

    	// SUBTRACTION
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I())
		DIRECT_MULTIDIM_ELEM(I,n) = (DIRECT_MULTIDIM_ELEM(I,n)-(DIRECT_MULTIDIM_ELEM(P,n)*DIRECT_MULTIDIM_ELEM(PmaskInv,n))) * (DIRECT_MULTIDIM_ELEM(Pmask,n));

		// Save subtracted particles in metadata
		ix_particle++;
		FileName out = formatString("%d@%s.mrcs", ix_particle, fnOut.c_str());
		I.write(out);
		mdParticles.setValue(MDL_IMAGE, out, ix_particle);
    }
    mdParticles.write(fnParticles);
 }



