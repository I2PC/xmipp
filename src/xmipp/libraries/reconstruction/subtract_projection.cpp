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


// #include "volume_subtraction.cpp"


void POCSmaskProj(const MultidimArray<double> &mask, MultidimArray<double> &I)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I)
	DIRECT_MULTIDIM_ELEM(I,n)*=DIRECT_MULTIDIM_ELEM(mask,n);
}

void POCSnonnegativeProj(MultidimArray<double> &I)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I)
	DIRECT_MULTIDIM_ELEM(I,n)=std::max(0.0,DIRECT_MULTIDIM_ELEM(I,n));
}

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

void computeEnergyProj(MultidimArray<double> &Idiff, MultidimArray<double> &Iact, double energy)
{
	Idiff = Idiff - Iact;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Idiff)
	energy+=DIRECT_MULTIDIM_ELEM(Idiff,n)*DIRECT_MULTIDIM_ELEM(Idiff,n);
	energy = sqrt(energy/MULTIDIM_SIZE(Idiff));
	std::cout<< "Energy: " << energy << std::endl;
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
	iter=getIntParam("--iter");
	sigma=getIntParam("--sigma");
	cutFreq=getDoubleParam("--cutFreq");
	lambda=getDoubleParam("--lambda");
//	fnVol1F=getParam("--saveV1");
//	if (fnVol1F=="")
//		fnVol1F="volume1_filtered.mrc";
//	fnVol2A=getParam("--saveV2");
//	if (fnVol2A=="")
//		fnVol2A="volume2_adjusted.mrc";
 }

 // Show ====================================================================
 void ProgSubtractProjection::show()
 {
    if (!verbose)
        return;
	std::cout
	<< "Input particles:   	" << fnParticles << std::endl
	<< "Reference volume:   " << fnVolR      << std::endl
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
     addParamsLine("-i <particles>          : Particles metadata (.xmd file)");
     addParamsLine("--ref <volume>      	: Reference volume to subtract");
     addParamsLine("[-o <structure=\"\">] 	: Output filename suffix for subtracted particles");
     addParamsLine("                      	: If no name is given, then output_particles.xmd");
     addParamsLine("[--mask <mask=\"\">]  	: Mask for the region of subtraction");
     addParamsLine("[--sigma <s=3>]    		: Decay of the filter (sigma) to smooth the mask transition");
     addParamsLine("[--iter <n=1>]        	: Number of iterations");
     addParamsLine("[--cutFreq <f=0>]       : Cutoff frequency (<0.5)");
     addParamsLine("[--lambda <l=0>]       	: Relaxation factor for Fourier Amplitude POCS (between 0 and 1)");
//        addParamsLine("[--saveV1 <structure=\"\"> ]  : Save subtraction intermediate files (vol1 filtered)");
//        addParamsLine("[--saveV2 <structure=\"\"> ]  : Save subtraction intermediate files (vol2 adjusted)");
    addExampleLine("A typical use is:",false);
    addExampleLine("");
 }

 void ProgSubtractProjection::run()
 {
	show();
	V.read(fnVolR);
 	MultidimArray<double> &mV=P();
 	mdParticles.read(fnParticles);
 	MDRow row;
 	double rot, tilt, psi;
 	FileName fnImage;
 	String ix;

    FOR_ALL_OBJECTS_IN_METADATA(mdParticles)
    {
    	// Compute projection of the volume
    	mdParticles.getRow(row,__iter.objId);
    	row.getValue(MDL_IMAGE, fnImage);
    	I.read(fnImage);
     	row.getValue(MDL_ANGLE_ROT, rot);
     	row.getValue(MDL_ANGLE_TILT, tilt);
     	row.getValue(MDL_ANGLE_PSI, psi);
    	projectVolume(mV, P, (int)XSIZE(I()), (int)XSIZE(I()), rot, tilt, psi);

    	// Check if particle has CTF
     	if ((row.containsLabel(MDL_CTF_DEFOCUSU) || row.containsLabel(MDL_CTF_MODEL)))
     	{
     		hasCTF=true;
     		ctf.readFromMdRow(row);
     		ctf.produceSideInfo();
     		defocusU=ctf.DeltafU;
     		defocusV=ctf.DeltafV;
     		ctfAngle=ctf.azimuthal_angle;
     	}
     	else
     		hasCTF=false;

     	// If it has CTF, apply it
 	 	if (hasCTF)
 	 	{
 	 	 	FilterCTF.FilterBand = CTF;
 	 	 	FilterCTF.ctf.enable_CTFnoise = false;
 	 		FilterCTF.ctf = ctf;
 	 		FilterCTF.generateMask(P());
 	 		FilterCTF.applyMaskSpace(P());
 	 	}

 	 	// Projection subtraction
 	 	Image<double> Idiff;  // I = V1, P = V
		FourierTransformer transformer;
		MultidimArray< std::complex<double> > IFourier, PFourier;
		MultidimArray<double> IFourierMag;
//		MultidimArray<double> mask1;
		Image<double> mask;
		if (fnMask!="")
		{
			mask.read(fnMask);
			mask=mask();
		}
		else
		{
			mask().resizeNoCopy(I());
			mask().initConstant(1.0);
		}
		mask.clear();
		POCSmaskProj(mask(),I());
		POCSnonnegativeProj(I());

		double Imin, Imax;
		I().computeDoubleMinMax(Imin, Imax);

		transformer.FourierTransform(I(),IFourier,false);
		FFT_magnitude(IFourier,IFourierMag);
		double std1 = I().computeStddev();

		POCSmaskProj(mask(),P());

		MultidimArray<std::complex<double> > PFourierPhase;
		transformer.FourierTransform(P(),PFourierPhase,true);
		extractPhaseProj(PFourierPhase);

		FourierFilter Filter2;
		double energy, std2;
		energy = 0;
		Idiff = P;

		Filter2.FilterBand=LOWPASS;
		Filter2.FilterShape=RAISED_COSINE;
		Filter2.raised_w=0.02;
		Filter2.w1=cutFreq;

		for (int n=0; n<iter; ++n)
		{
			std::cout<< "---Iter " << n << std::endl;
			transformer.FourierTransform(P(),PFourier,false);
			POCSFourierAmplitudeProj(IFourierMag,PFourier, lambda);
			transformer.inverseFourierTransform();
			computeEnergyProj(Idiff(), P(), energy);
			Idiff = P;

			POCSMinMaxProj(P(), Imin, Imax);
			computeEnergyProj(Idiff(), P(), energy);
			Idiff = P;

			POCSmaskProj(mask(),P());
			computeEnergyProj(Idiff(), P(), energy);
			Idiff = P;
			transformer.FourierTransform();
			POCSFourierPhaseProj(PFourierPhase,PFourier);
			transformer.inverseFourierTransform();
			computeEnergyProj(Idiff(), P(), energy);
			Idiff = P;
			POCSnonnegativeProj(P());
			computeEnergyProj(Idiff(), P(), energy);
			Idiff = P;
			std2 = P().computeStddev();
			P()*=std1/std2;
			computeEnergyProj(Idiff(), P(), energy);
			Idiff = P;
			if (cutFreq!=0)
			{
				Filter2.generateMask(P());
				Filter2.do_generate_3dmask=true;
				Filter2.applyMaskSpace(P());
				computeEnergyProj(Idiff(), P(), energy);
				Idiff = P;
			}
		}

		FourierFilter Filter;
		Filter.FilterShape=REALGAUSSIAN;
		Filter.FilterBand=LOWPASS;
		Filter.w1=sigma;
		Filter.applyMaskSpace(mask());
		Image<double> IFiltered;
    	I.read(fnImage);
		IFiltered() = I();
		if (cutFreq!=0)
			Filter2.applyMaskSpace(IFiltered());

//		if (fnVol1F!="" && fnVol2A!="")
//		{
//			IFiltered.write(fnVol1F);
//			V.write(fnVol2A);
//		}

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I())
		DIRECT_MULTIDIM_ELEM(I,n) = DIRECT_MULTIDIM_ELEM(I,n)*(1-DIRECT_MULTIDIM_ELEM(mask,n)) + (DIRECT_MULTIDIM_ELEM(IFiltered, n) -
				std::min(DIRECT_MULTIDIM_ELEM(P,n), DIRECT_MULTIDIM_ELEM(IFiltered, n)))*DIRECT_MULTIDIM_ELEM(mask,n);
		row.getValue(MDL_ITEM_ID, ix);
		I.write(fnOut + ix);
    }

 }



