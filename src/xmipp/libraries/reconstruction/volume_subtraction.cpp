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

void POCSFourierAmplitudeRadAvg(MultidimArray< std::complex<double> > &V, double lambda, const MultidimArray<double> &rQ, int V1size_x, int V1size_y, int V1size_z)
{
	int V1size2_x = V1size_x/2;
	double V1sizei_x = 1.0/V1size_x;
	int V1size2_y = V1size_y/2;
	double V1sizei_y = 1.0/V1size_y;
	int V1size2_z = V1size_z/2;
	double V1sizei_z = 1.0/V1size_z;
	double wx;
	double wy;
	double wz;
	for (int k=0; k<V1size_z; ++k)
	{
		FFT_IDX2DIGFREQ_FAST(k,V1size_z,V1size2_z,V1sizei_z,wz)
		double wz2 = wz*wz;
		for (int i=0; i<V1size_y; ++i)
		{
			FFT_IDX2DIGFREQ_FAST(i,V1size_y,V1size2_y,V1sizei_y,wy)
			double wy2 = wy*wy;
			for (int j=0; j<V1size_x; ++j)
			{
				FFT_IDX2DIGFREQ_FAST(j,V1size_x,V1size2_x,V1sizei_x,wx)
				double w = sqrt(wx*wx + wy2 + wz2);
				auto iw = (int)round(w*V1size_x);
				std::cout<< "26" << std::endl;
				std::cout<< "sizeX V" << XSIZE(V) << std::endl;
				std::cout<< "sizeY V" << YSIZE(V) << std::endl;
				std::cout<< "sizeZ V" << ZSIZE(V) << std::endl;
				std::cout<< "size rQ" << XSIZE(rQ )<< std::endl;
				DIRECT_A3D_ELEM(V,k,i,j)*=(1-lambda)+lambda*DIRECT_MULTIDIM_ELEM(rQ,iw);
				std::cout<< "27" << std::endl;
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
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FI) 
			{
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

void centerFFTMagnitude(MultidimArray<double> &VolRad, MultidimArray< std::complex<double> > &VolFourierRad, MultidimArray<double> &VolFourierMagRad)
{
	FourierTransformer transformerRad;
	transformerRad.completeFourierTransform(VolRad,VolFourierRad);
	CenterFFT(VolFourierRad, true);
	FFT_magnitude(VolFourierRad,VolFourierMagRad);
	VolFourierMagRad.setXmippOrigin();
}

void radialAverage(const MultidimArray<double> &VolFourierMagRad, const MultidimArray< std::complex<double> > &VolFourierRad, MultidimArray<double> const & Volrad, MultidimArray<double> radial_mean)
{
	Matrix1D<int> center(2);
	center.initZeros();
	MultidimArray<int> radial_count;
	radialAverageNonCubic(VolFourierMagRad, center, radial_mean, radial_count);
	FOR_ALL_ELEMENTS_IN_ARRAY1D(VolFourierRad)
	Volrad(i) = radial_mean(i);
}

void subtraction(MultidimArray<double> &V1, const MultidimArray<double> &V1Filtered, const MultidimArray<double> &V, const MultidimArray<double> &mask)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V1)
			DIRECT_MULTIDIM_ELEM(V1,n) = DIRECT_MULTIDIM_ELEM(V1,n)*(1-DIRECT_MULTIDIM_ELEM(mask,n)) + (DIRECT_MULTIDIM_ELEM(V1Filtered, n) -
					std::min(DIRECT_MULTIDIM_ELEM(V,n), DIRECT_MULTIDIM_ELEM(V1Filtered, n)))*DIRECT_MULTIDIM_ELEM(mask,n);
}

class ProgVolumeSubtraction: public XmippProgram
{
private:
	FileName fnVol1, fnVol2, fnOut, fnMask1, fnMask2, fnVol1F, fnVol2A, fnMaskSub;
	bool sub; bool computeE; bool radavg; bool saveVol1Filt; bool saveVol2Adj;
	int iter; int sigma;
	double cutFreq; double lambda;

	void defineParams()
	{
		//Usage
		addUsageLine("This program modifies a volume as much as possible in order to assimilate it to another one, "
				"without loosing the important information in it ('adjustment process'). Then, the subtraction of "
				"the two volumes can be optionally calculated. Sharpening: reference volume must be an atomic "
				"structure previously converted into a density map of the same specimen than in input volume 2.");
		//Parameters
		addParamsLine("--i1 <volume>			: Reference volume");
		addParamsLine("--i2 <volume>			: Volume to modify");
		addParamsLine("[-o <structure=\"\">]		: Volume 2 modified or volume difference");
		addParamsLine("					: If no name is given, then output_volume.mrc");
		addParamsLine("[--sub]				: Perform the subtraction of the volumes. Output will be the difference");
		addParamsLine("[--sigma <s=3>]			: Decay of the filter (sigma) to smooth the mask transition");
		addParamsLine("[--iter <n=5>]\t: Number of iterations for the adjustment process");
		addParamsLine("[--mask1 <mask=\"\">]		: Mask for volume 1");
		addParamsLine("[--mask2 <mask=\"\">]		: Mask for volume 2");
		addParamsLine("[--maskSub <mask=\"\">]		: Mask for subtraction region");
		addParamsLine("[--cutFreq <f=0>]\t: Filter both volumes with a filter which specified cutoff frequency (i.e. resolution inverse, <0.5)");
		addParamsLine("[--lambda <l=1>]\t: Relaxation factor for Fourier Amplitude POCS, i.e. 'how much modification of volume Fourier amplitudes', between 1 (full modification, recommended) and 0 (no modification)");
		addParamsLine("[--radavg]\t: Match the radially averaged Fourier amplitudes when adjusting the amplitudes instead of taking directly them from the reference volume");
		addParamsLine("[--computeEnergy]\t: Compute the energy difference between each step (energy difference gives information about the convergence of the adjustment process, while it can slightly slow the performance)");
		addParamsLine("[--saveV1 <structure=\"\"> ]\t: Save subtraction intermediate file (vol1 filtered) just when option --sub is passed, if not passed the input reference volume is not modified");
		addParamsLine("[--saveV2 <structure=\"\"> ]\t: Save subtraction intermediate file (vol2 adjusted) just when option --sub is passed, if not passed the output of the program is this file");
	}

	void readParams()
	{
		fnVol1=getParam("--i1");
		fnVol2=getParam("--i2");
		fnOut=getParam("-o");
		if (fnOut.isEmpty())
			fnOut="output_volume.mrc";
		sub=checkParam("--sub");
		iter=getIntParam("--iter");
		sigma=getIntParam("--sigma");
		fnMask1=getParam("--mask1");
		fnMask2=getParam("--mask2");
		fnMaskSub=getParam("--maskSub");
		cutFreq=getDoubleParam("--cutFreq");
		lambda=getDoubleParam("--lambda");
		saveVol1Filt=checkParam("--saveV1");
		saveVol2Adj=checkParam("--saveV2");
		fnVol1F=getParam("--saveV1");
		fnVol2A=getParam("--saveV2");
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
		<< "Input mask sub:		" << fnMaskSub   << std::endl
		<< "Sigma:			" << sigma       << std::endl
		<< "Iterations:			" << iter        << std::endl
		<< "Cutoff frequency:		" << cutFreq     << std::endl
		<< "Relaxation factor:		" << lambda      << std::endl
		<< "Match radial averages:\t" << radavg	 << std::endl
		<< "Output:			" << fnOut 	 << std::endl
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
		MultidimArray<double> V1rad;
		V1rad = V1();
		V1rad.setXmippOrigin();
		MultidimArray< std::complex<double> > V1FourierRad;
		MultidimArray<double> V1FourierMagRad;
		centerFFTMagnitude(V1rad, V1FourierRad, V1FourierMagRad);

		MultidimArray<double> Vrad;
		Vrad = V();
		Vrad.setXmippOrigin();
		MultidimArray< std::complex<double> > VFourierRad;
		MultidimArray<double> VFourierMagRad;
		centerFFTMagnitude(Vrad, VFourierRad, VFourierMagRad);

		// Compute V1radAvg and VradAvg profile (1D)
		MultidimArray<double> radial_meanV1;
		radialAverage(V1FourierMagRad, V1FourierRad, V1rad, radial_meanV1);
		MultidimArray<double> radial_meanV;
		radialAverage(VFourierMagRad, VFourierRad, Vrad, radial_meanV);

		// Compute adjustment quotient for POCS amplitude
		MultidimArray<double> radQuotient;
		std::cout<< "radial_meanV1" << XSIZE(radial_meanV1) << std::endl;
		std::cout<< "radial_meanV" << XSIZE(radial_meanV) << std::endl;
		radQuotient = radial_meanV1/radial_meanV;
		std::cout<< "radQuotient" << XSIZE(radQuotient) << std::endl;
		FOR_ALL_ELEMENTS_IN_ARRAY1D(radQuotient)
			radQuotient(i) = std::min(radQuotient(i), 1.0);
		std::cout<< "radQuotientMin" << XSIZE(radQuotient) << std::endl;

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
			if (radavg)
			{
				auto V1size_x = (int)XSIZE(V1());
				auto V1size_y = (int)YSIZE(V1());
				auto V1size_z = (int)ZSIZE(V1());
				transformer2.completeFourierTransform(V(),V2Fourier);
				CenterFFT(V2Fourier, true);
				POCSFourierAmplitudeRadAvg(V2Fourier, lambda, radQuotient, V1size_x, V1size_y, V1size_z);
				std::cout<< "3" << std::endl;
			}
			else
			{
				transformer2.FourierTransform(V(),V2Fourier,false);
				POCSFourierAmplitude(V1FourierMag,V2Fourier, lambda);
			}
			transformer2.inverseFourierTransform();

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
			if (saveVol1Filt)
			{
				if (fnVol1F.isEmpty())
					fnVol1F="volume1_filtered.mrc";
				V1Filtered.write(fnVol1F);
			}
			if (saveVol2Adj)
			{
				if (fnVol2A.isEmpty())
					fnVol2A="volume2_adjusted.mrc";
				V.write(fnVol2A);
			}

			if (!fnMaskSub.isEmpty())
				mask.read(fnMaskSub);

			subtraction(V1(), V1Filtered(), V(), mask());
			V1.write(fnOut);
		}

		else
			V.write(fnOut);
	}
};
