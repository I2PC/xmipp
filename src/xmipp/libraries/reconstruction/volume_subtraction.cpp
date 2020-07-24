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
#include <data/fourier_filter.h>


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

void POCSFourierAmplitude(const MultidimArray<double> &A, MultidimArray< std::complex<double> > &FI)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(A)
	DIRECT_MULTIDIM_ELEM(FI,n)*=DIRECT_MULTIDIM_ELEM(A,n)/std::abs(DIRECT_MULTIDIM_ELEM(FI,n));
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

class ProgVolumeSubtraction: public XmippProgram
{
protected:
	FileName fnVol1, fnVol2, fnDiff, fnMask1, fnMask2;
//	bool pdb;
	bool sub;
	int iter, sigma;
	double cutFreq;

    void defineParams()
    {
        //Usage
        addUsageLine("This program scales a volume in order to assimilate it to another one. Then, it can calculate the subtraction of the two volumes.");
        //Parameters
        addParamsLine("--i1 <volume>          	: Reference volume");
        addParamsLine("--i2 <volume>          	: Volume to modify");
        addParamsLine("[-o <structure=\"\">] 	: Volume 2 modified or volume difference");
        addParamsLine("                      	: If no name is given, then output_volume.mrc");
//        addParamsLine("[--pdb]    			 	: Second volume come from a pdb");
        addParamsLine("[--sub] 			        : Perform the subtraction of the volumes. Output will be the difference");
        addParamsLine("[--sigma <s=3>]    		: Decay of the filter (sigma) to smooth the mask transition");
        addParamsLine("[--iter <n=1>]        	: Number of iterations");
        addParamsLine("[--mask1 <mask=\"\">]  	: Mask for volume 1");
        addParamsLine("[--mask2 <mask=\"\">]  	: Mask for volume 2");
        addParamsLine("[--cutFreq <f=0>]       	: Cutoff frequency (<0.5)");
    }

    void readParams()
    {
    	fnVol1=getParam("--i1");
    	fnVol2=getParam("--i2");
    	fnDiff=getParam("-o");
    	if (fnDiff=="")
    		fnDiff="output_volume.mrc";
//    	pdb=checkParam("--pdb");
    	sub=checkParam("--sub");
    	iter=getIntParam("--iter");
    	sigma=getIntParam("--sigma");
    	fnMask1=getParam("--mask1");
    	fnMask2=getParam("--mask2");
    	cutFreq=getDoubleParam("--cutFreq");
    }

    void show()
    {
    	std::cout
    	<< "Input volume 1:    		" << fnVol1      << std::endl
    	<< "Input volume 2:    	   	" << fnVol2      << std::endl
    	<< "Input mask 1:    	   	" << fnMask1     << std::endl
    	<< "Input mask 2:    	   	" << fnMask2     << std::endl
    	<< "Sigma:    	   			" << sigma       << std::endl
    	<< "Iterations:    	   		" << iter        << std::endl
    	<< "Cutoff frequency:   	" << cutFreq     << std::endl
    	<< "Output difference: 		" << fnDiff 	 << std::endl
    	;
    }

    void run()
    {
    	show();

    	Image<double> V;
    	FourierTransformer transformer;
    	MultidimArray< std::complex<double> > V1Fourier, V2Fourier;
    	MultidimArray<double> V1FourierMag;

    	V.read(fnVol1);
    	MultidimArray<double> mask1;
		Image<double> mask;
		// if masks => compute common mask
    	if (fnMask1!="" && fnMask2!="")
    	{
			mask.read(fnMask1);
    		mask1=mask();
			mask.read(fnMask2);
			mask()*=mask1;
		}
		else  // mask all 1s of size V1
		{
            mask().resizeNoCopy(V());
            mask().initConstant(1.0);
		}
    	mask1.clear();
//		mask.write("commonmask.mrc");
		POCSmask(mask(),V());
    	POCSnonnegative(V());
    	transformer.FourierTransform(V(),V1Fourier,false);
    	FFT_magnitude(V1Fourier,V1FourierMag);
		double std1 = V().computeStddev();
		std::cout << "std1 " << std1 << std::endl;
		double mean1 = V().computeMean();
		std::cout << "mean1 " << mean1 << std::endl;
//		V.write("V1masked.mrc");
		V.read(fnVol2);
		POCSmask(mask(),V());
//		V.write("V2masked.mrc");

    	MultidimArray<std::complex<double> > V2FourierPhase;
    	transformer.FourierTransform(V(),V2FourierPhase,true);
    	extractPhase(V2FourierPhase);

    	for (int n=0; n<iter; ++n)
    	{
    		transformer.FourierTransform(V(),V2Fourier,false);
    		POCSFourierAmplitude(V1FourierMag,V2Fourier);
        	transformer.inverseFourierTransform();
			POCSmask(mask(),V());
//    		V.write(formatString("V2masked_Amp1_%d.mrc", n));
    		transformer.FourierTransform();
    		//V2FourierPhase = V2Fourier;
        	//extractPhase(V2FourierPhase);
    		POCSFourierPhase(V2FourierPhase,V2Fourier);
        	transformer.inverseFourierTransform();
//    		V.write(formatString("V2masked_Amp1_ph2_%d.mrc", n));
        	POCSnonnegative(V());
			double  std2 = V().computeStddev();
			std::cout << "std2 " << std2 << std::endl;
			double mean2 = V().computeMean();
			std::cout << "mean2 " << mean2 << std::endl;
			V()*=std1/std2;
//    		V.write(formatString("V2masked_Amp1_ph2_nonneg_%d.mrc", n));
    	}

//		Image<double> V1, V1Filtered;
//    	V1.read(fnVol1);
//    	V1() -= V();
//    	V1.write("subtraction.mrc");
		// Filter common mask with gaussian for smoothing
		FourierFilter Filter;
		Filter.FilterShape=REALGAUSSIAN;
		Filter.FilterBand=LOWPASS;
		Filter.w1=sigma;
		Filter.applyMaskSpace(mask());
//		mask.write("maskfilter.mrc");
		Image<double> V1, V1Filtered;
		V1.read(fnVol1);
		V1Filtered() = V1();
		// If cutoff freq param is passed, filter V1 and V2
		if (cutFreq!=0)
		{
			FourierFilter Filter2;
			Filter2.FilterBand=LOWPASS;
			Filter2.FilterShape=RAISED_COSINE;
			Filter2.raised_w=0.02;
			Filter2.w1=cutFreq;
			Filter2.generateMask(V());
			Filter2.do_generate_3dmask=true;
			Filter2.applyMaskSpace(V());
//			V.write("V2filter.mrc");
			Filter2.applyMaskSpace(V1Filtered());
//			V1Filtered.write("V1filter.mrc");
		}

    	if (sub==true)
    	{
    		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V1())
    		DIRECT_MULTIDIM_ELEM(V1,n) = DIRECT_MULTIDIM_ELEM(V1,n)*(1-DIRECT_MULTIDIM_ELEM(mask,n)) + (DIRECT_MULTIDIM_ELEM(V1Filtered, n) -
    				std::min(DIRECT_MULTIDIM_ELEM(V,n), DIRECT_MULTIDIM_ELEM(V1Filtered, n)))*DIRECT_MULTIDIM_ELEM(mask,n);
    		V1.write(fnDiff);
    	}

    	else
    		V.write(fnDiff);
    }
};
