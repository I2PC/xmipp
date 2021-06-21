/***************************************************************************
 *
 * Authors:    Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
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

#include <core/xmipp_program.h>
#include <core/xmipp_fftw.h>
#include <core/histogram.h>
#include <data/fourier_filter.h>
#include "core/transformations.h"


void POCSmask(const MultidimArray<double> &mask, MultidimArray<double> &I)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I)
	DIRECT_MULTIDIM_ELEM(I,n)*=DIRECT_MULTIDIM_ELEM(mask,n);
}

void POCSnonnegative(MultidimArray<double> &I)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I)
	DIRECT_MULTIDIM_ELEM(I,n)=std::max(0.0,DIRECT_MULTIDIM_ELEM(I,n));
}

void POCSFourierAmplitude(const MultidimArray<double> &A, MultidimArray< std::complex<double> > &FI, double lambda)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(A)
		{
		double mod = std::abs(DIRECT_MULTIDIM_ELEM(FI,n));
		if (mod>1e-10)  // Condition to avoid divide by zero, values smaller than this threshold are considered zero
			DIRECT_MULTIDIM_ELEM(FI,n)*=((1-lambda)+lambda*DIRECT_MULTIDIM_ELEM(A,n))/mod;
		}
}

void POCSFourierAmplitudeRadAvg(MultidimArray< std::complex<double> > &V, double lambda, MultidimArray<double> &rQ, int V1size_x, int V1size_y, int V1size_z)
{
	int V1size2_x = V1size_x/2;
	double V1sizei_x = 1.0/V1size_x;
	int V1size2_y = V1size_y/2;
	double V1sizei_y = 1.0/V1size_y;
	int V1size2_z = V1size_z/2;
	double V1sizei_z = 1.0/V1size_z;
	double wx, wy, wz;
	for (int k=0; k<V1size_z; ++k)
		{
			FFT_IDX2DIGFREQ_FAST(k,V1size_z,V1size2_z,V1sizei_z,wz);
			double wz2 = wz*wz;
			for (int i=0; i<V1size_y; ++i)
			{
				FFT_IDX2DIGFREQ_FAST(i,V1size_y,V1size2_y,V1sizei_y,wy);
				double wy2 = wy*wy;
				for (int j=0; j<V1size_x; ++j)
				{
					FFT_IDX2DIGFREQ_FAST(j,V1size_x,V1size2_x,V1sizei_x,wx);
					double w = sqrt(wx*wx + wy2 + wz2);
					int iw = (int)round(w*V1size_x); // size_x??
					DIRECT_A3D_ELEM(V,k,i,j)*=(1-lambda)+lambda*DIRECT_MULTIDIM_ELEM(rQ,iw);
				}
			}
		}
}

void POCSMinMax(MultidimArray<double> &V, double v1m, double v1M)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V)
		{
		double val = DIRECT_MULTIDIM_ELEM(V,n);
		if (val<v1m)
			DIRECT_MULTIDIM_ELEM(V,n) = v1m;
		else if (val>v1M)
			DIRECT_MULTIDIM_ELEM(V,n) = v1M;
		}
}

void extractPhase(MultidimArray< std::complex<double> > &FI)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FI) {
		double *ptr = (double *)&DIRECT_MULTIDIM_ELEM(FI,n);
		double phi = atan2(*(ptr+1),*ptr);
		DIRECT_MULTIDIM_ELEM(FI,n) = std::complex<double>(cos(phi),sin(phi));
	}
}

void POCSFourierPhase(const MultidimArray< std::complex<double> > &phase, MultidimArray< std::complex<double> > &FI)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(phase)
	DIRECT_MULTIDIM_ELEM(FI,n)=std::abs(DIRECT_MULTIDIM_ELEM(FI,n))*DIRECT_MULTIDIM_ELEM(phase,n);
}

void computeEnergy(MultidimArray<double> &Vdiff, MultidimArray<double> &Vact, double energy)
{
		Vdiff = Vdiff - Vact;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Vdiff)
		energy+=DIRECT_MULTIDIM_ELEM(Vdiff,n)*DIRECT_MULTIDIM_ELEM(Vdiff,n);
		energy = sqrt(energy/MULTIDIM_SIZE(Vdiff));
		std::cout<< "Energy: " << energy << std::endl;
}

class ProgVolumeSubtraction: public XmippProgram
{
private:
	FileName fnVol1, fnVol2, fnOut, fnMask1, fnMask2, fnVol1F, fnVol2A, fnMaskSub;
	bool sub; bool eq; bool computeE; bool radavg;
	int iter; int sigma;
	double cutFreq; double lambda;

    void defineParams()
    {
        //Usage
        addUsageLine("This program scales a volume in order to assimilate it to another one. Then, it can calculate the subtraction of the two volumes.");
        //Parameters
        addParamsLine("--i1 <volume>          	: Reference volume");
        addParamsLine("--i2 <volume>          	: Volume to modify");
        addParamsLine("[-o <structure=\"\">] 	: Volume 2 modified or volume difference");
        addParamsLine("                      	: If no name is given, then output_volume.mrc");
        addParamsLine("[--sub] 			        : Perform the subtraction of the volumes. Output will be the difference");
        addParamsLine("[--sigma <s=3>]    		: Decay of the filter (sigma) to smooth the mask transition");
        addParamsLine("[--iter <n=1>]			: Number of iterations");
        addParamsLine("[--mask1 <mask=\"\">]	: Mask for volume 1");
        addParamsLine("[--mask2 <mask=\"\">]	: Mask for volume 2");
        addParamsLine("[--maskSub <mask=\"\">]	: Mask for subtraction region");
        addParamsLine("[--cutFreq <f=0>]       	: Cutoff frequency (<0.5)");
        addParamsLine("[--lambda <l=0>]       	: Relaxation factor for Fourier Amplitude POCS (between 0 and 1)");
        addParamsLine("[--radavg]				: Match the rotationally averaged Fourier amplitudes when adjusting the amplitudes instead of taking directly them from the reference volume");
        addParamsLine("[--computeEnergy]		: Do not compute the energy difference between each step");
        addParamsLine("[--saveV1 <structure=\"\"> ]	: Save subtraction intermediate files (vol1 filtered)");
        addParamsLine("[--saveV2 <structure=\"\"> ]	: Save subtraction intermediate files (vol2 adjusted)");
    }

    void readParams()
    {
    	fnVol1=getParam("--i1");
    	fnVol2=getParam("--i2");
    	fnOut=getParam("-o");
    	if (fnOut=="")
    		fnOut="output_volume.mrc";
    	sub=checkParam("--sub");
    	iter=getIntParam("--iter");
    	sigma=getIntParam("--sigma");
    	fnMask1=getParam("--mask1");
    	fnMask2=getParam("--mask2");
    	fnMaskSub=getParam("--maskSub");
    	cutFreq=getDoubleParam("--cutFreq");
    	lambda=getDoubleParam("--lambda");
    	fnVol1F=getParam("--saveV1");
    	if (fnVol1F=="")
    		fnVol1F="volume1_filtered.mrc";
    	fnVol2A=getParam("--saveV2");
    	if (fnVol2A=="")
    		fnVol2A="volume2_adjusted.mrc";
    	radavg=checkParam("--radavg");
    	computeE=checkParam("--computeEnergy");
    }

    void show()
    {
    	std::cout
    	<< "Input volume 1:    		" << fnVol1      << std::endl
    	<< "Input volume 2:    	   	" << fnVol2      << std::endl
    	<< "Input mask 1:    	   	" << fnMask1     << std::endl
    	<< "Input mask 2:    	   	" << fnMask2     << std::endl
    	<< "Input mask sub:			" << fnMaskSub   << std::endl
    	<< "Sigma:					" << sigma       << std::endl
    	<< "Iterations:				" << iter        << std::endl
    	<< "Cutoff frequency:		" << cutFreq     << std::endl
    	<< "Relaxation factor:		" << lambda      << std::endl
		<< "Match radial averages	" << radavg		 << std::endl
    	<< "Output:					" << fnOut 	     << std::endl
    	;
    }

    void run()
    {
    	show();
    	Image<double> V, Vdiff, V1;
    	V1.read(fnVol1);
    	MultidimArray<double> mask1;
		Image<double> mask;
    	if (fnMask1!="" && fnMask2!="")
    	{
			mask.read(fnMask1);
    		mask1=mask();
			mask.read(fnMask2);
			mask()*=mask1;
		}
		else
		{
            mask().resizeNoCopy(V1());
            mask().initConstant(1.0);
		}
    	mask1.clear();
		POCSmask(mask(),V1());
    	POCSnonnegative(V1());
    	double v1min, v1max;
		V1().computeDoubleMinMax(v1min, v1max);

		V.read(fnVol2);
		POCSmask(mask(),V());
    	POCSnonnegative(V());

		// Compute |FT(radial averages)|
    	MultidimArray<double> V1rad, Vrad;
    	V1rad = V1();
    	Vrad = V();
    	V1rad.setXmippOrigin();
    	Vrad.setXmippOrigin();
		FourierTransformer transformerRad;
		MultidimArray< std::complex<double> > V1FourierRad, VFourierRad;
		MultidimArray<double> V1FourierMagRad, VFourierMagRad, radQuotient;
		transformerRad.completeFourierTransform(V1rad,V1FourierRad);
		CenterFFT(V1FourierRad, true);
		FFT_magnitude(V1FourierRad,V1FourierMagRad);
		transformerRad.completeFourierTransform(Vrad,VFourierRad);
		CenterFFT(VFourierRad, true);
		FFT_magnitude(VFourierRad,VFourierMagRad);

    	// Compute V1radAvg and VradAvg profile (1D)
        V1FourierMagRad.setXmippOrigin();
        MultidimArray<double> radial_meanV1;
        MultidimArray<int> radial_count;
        Matrix1D<int> center(2);
        center.initZeros();
        radialAverageNonCubic(V1FourierMagRad, center, radial_meanV1, radial_count);
        radial_meanV1.write("V1rad.txt");
        FOR_ALL_ELEMENTS_IN_ARRAY1D(V1FourierRad)
            V1rad(i) = radial_meanV1(i);
		V1rad.write("V1rad.mrc");
		VFourierMagRad.setXmippOrigin();
        center.initZeros();
        MultidimArray<double> radial_meanV;
        radialAverageNonCubic(VFourierMagRad, center, radial_meanV, radial_count);
        radial_meanV.write("Vrad.txt");
        FOR_ALL_ELEMENTS_IN_ARRAY1D(VFourierMagRad)
        	Vrad(i) = radial_meanV(i);
		Vrad.write("Vrad.mrc");

		// Compute adjustment quotient for POCS amplitude
		radQuotient = radial_meanV1/radial_meanV;
		FOR_ALL_ELEMENTS_IN_ARRAY1D(radQuotient)
			radQuotient(i) = std::min(radQuotient(i), 1.0);
		radQuotient.write("radQuotient.txt");

     	// Compute what need for the loop of POCS
    	FourierTransformer transformer1; FourierTransformer transformer2;
    	MultidimArray< std::complex<double> > V1Fourier, V2Fourier;
    	MultidimArray<double> V1FourierMag;
    	transformer1.FourierTransform(V1(),V1Fourier,false);
    	FFT_magnitude(V1Fourier,V1FourierMag);
		double std1 = V1().computeStddev();
    	MultidimArray<std::complex<double> > V2FourierPhase;
    	transformer2.FourierTransform(V(),V2FourierPhase,true);
    	extractPhase(V2FourierPhase);
    	int V1size_x = (int)XSIZE(V1());
    	int V1size_y = (int)YSIZE(V1());
    	int V1size_z = (int)ZSIZE(V1());

		FourierFilter Filter2;
		double energy, std2;
		if (computeE)
		{
			energy = 0;
			Vdiff = V;
		}

		Filter2.FilterBand=LOWPASS;
		Filter2.FilterShape=RAISED_COSINE;
		Filter2.raised_w=0.02;
		Filter2.w1=cutFreq;

    	for (int n=0; n<iter; ++n)
    	{
        	if (computeE)
        		std::cout<< "---Iter " << n << std::endl;
    		transformer2.FourierTransform(V(),V2Fourier,false);
    		if (radavg)
    			POCSFourierAmplitudeRadAvg(V2Fourier, lambda, radQuotient, V1size_x, V1size_y, V1size_z);
    		else
    			POCSFourierAmplitude(V1FourierMag,V2Fourier, lambda);
        	transformer2.inverseFourierTransform();
    		V.write("VPOCSAmp.mrc");
        	if (computeE)
        	{
        		computeEnergy(Vdiff(), V(), energy);
        		Vdiff = V;
        	}
    		POCSMinMax(V(), v1min, v1max);
        	if (computeE)
        	{
        		computeEnergy(Vdiff(), V(), energy);
        		Vdiff = V;
        	}
			POCSmask(mask(),V());
        	if (computeE)
        	{
        		computeEnergy(Vdiff(), V(), energy);
        		Vdiff = V;
        	}
    		transformer2.FourierTransform();
    		POCSFourierPhase(V2FourierPhase,V2Fourier);
        	transformer2.inverseFourierTransform();
        	if (computeE)
        	{
        		computeEnergy(Vdiff(), V(), energy);
        		Vdiff = V;
        	}
        	POCSnonnegative(V());
        	if (computeE)
        	{
        		computeEnergy(Vdiff(), V(), energy);
        		Vdiff = V;
        	}
			std2 = V().computeStddev();
			V()*=std1/std2;
        	if (computeE)
        	{
        		computeEnergy(Vdiff(), V(), energy);
        		Vdiff = V;
        	}

    		if (cutFreq!=0)
    		{
    			Filter2.generateMask(V());
				Filter2.do_generate_3dmask=true;
				Filter2.applyMaskSpace(V());
	        	if (computeE)
	        	{
	        		computeEnergy(Vdiff(), V(), energy);
	        		Vdiff = V;
	        	}
    		}
    	}

		FourierFilter Filter;
		Filter.FilterShape=REALGAUSSIAN;
		Filter.FilterBand=LOWPASS;
		Filter.w1=sigma;
		Filter.applyMaskSpace(mask());
		Image<double> V1Filtered;
		V1.read(fnVol1);
		V1Filtered() = V1();
		if (cutFreq!=0)
			Filter2.applyMaskSpace(V1Filtered());

    	if (sub==true)
    	{
        	if (!fnVol1F.isEmpty() && !fnVol2A.isEmpty())
    		{
    			V1Filtered.write(fnVol1F);
    			V.write(fnVol2A);
    		}

        	if (!fnMaskSub.isEmpty())
        		mask.read(fnMaskSub);
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V1())
			DIRECT_MULTIDIM_ELEM(V1,n) = DIRECT_MULTIDIM_ELEM(V1,n)*(1-DIRECT_MULTIDIM_ELEM(mask,n)) + (DIRECT_MULTIDIM_ELEM(V1Filtered, n) -
					std::min(DIRECT_MULTIDIM_ELEM(V,n), DIRECT_MULTIDIM_ELEM(V1Filtered, n)))*DIRECT_MULTIDIM_ELEM(mask,n);
			V1.write(fnOut);
    	}

    	else
    		V.write(fnOut);
    }
};
