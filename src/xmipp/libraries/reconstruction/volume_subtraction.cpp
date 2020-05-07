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

void POCSmask(const MultidimArray<int> &mask, MultidimArray<double> &I)
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

void extractPhase(MultidimArray< std::complex<double> > &FI, double tol=1e-4)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FI) {
		double A = std::abs(DIRECT_MULTIDIM_ELEM(FI,n));
		if (A>tol)
		DIRECT_MULTIDIM_ELEM(FI,n)/=A;
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
	bool pdb;
	int iter;

    void defineParams()
    {
        //Usage
        addUsageLine("Calculate the subtraction of two volumes");
        //Parameters
        addParamsLine("-i1 <volume>          	: First volume to subtract");
        addParamsLine("-i2 <volume>          	: Second volume to subtract");
        addParamsLine("[-o <structure=\"\">] 	: Volume difference");
        addParamsLine("                      	: If no name is given, then volume_diff.mrc");
        addParamsLine("[--pdb]    			 	: Second volume come from a pdb");
        addParamsLine("[--iter <n=1>]        	: Number of iterations");
        addParamsLine("[--mask1 <mask=\"\">]  	: Mask for volume 1");
        addParamsLine("[--mask2 <mask=\"\">]  	: Mask for volume 2");
    }

    void readParams()
    {
    	fnVol1=getParam("-i1");
    	fnVol2=getParam("-i2");
    	fnDiff=getParam("-o");
    	if (fnDiff=="")
    		fnDiff="volume_diff.mrc";
    	pdb=checkParam("--pdb");
    	iter=getIntParam("--iter");
    	fnMask1=getParam("--mask1");
    	fnMask2=getParam("--mask2");
    }

    void show()
    {
    	std::cout
    	<< "Input volume 1:    		" << fnVol1      << std::endl
    	<< "Input volume 2:    	   	" << fnVol2      << std::endl
    	<< "Input mask 1:    	   	" << fnMask1     << std::endl
    	<< "Input mask 2:    	   	" << fnMask2     << std::endl
    	<< "Iterations:    	   		" << iter        << std::endl
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

    	// read vol1
    	V.read(fnVol1);
    	// FT vol1
    	//transformer.FourierTransform(V(),V1Fourier,false);
    	//FFT_magnitude(V1Fourier,V1FourierMag);

    	MultidimArray<int> mask1;
		Image<int> mask;
    	if (fnMask1!="" && fnMask2!="")
    	{
    		//read mask1
			mask.read(fnMask1);
    		mask1=mask();
			// read mask2
			mask.read(fnMask2);
			// mask intersection
			mask()*=mask1;
		}
		else
		{
            typeCast(V(), mask1);
            FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mask1)
			DIRECT_MULTIDIM_ELEM(mask1,n)=1;
            mask()=mask1;
		}

		mask.write("commonmask.mrc");
		// mask vol1 with common mask
		POCSmask(mask(),V());
    	POCSnonnegative(V());
    	transformer.FourierTransform(V(),V1Fourier,false);
    	FFT_magnitude(V1Fourier,V1FourierMag);
		double std1 = V().computeStddev();
		V.write("V1masked.mrc");
		//read vol2
		V.read(fnVol2);
		// mask vol2 with common mask
		POCSmask(mask(),V());
		V.write("V2masked.mrc");

    	// Get original phase vol2
    	MultidimArray<std::complex<double> > V2FourierPhase;
    	transformer.FourierTransform(V(),V2FourierPhase,true);
    	extractPhase(V2FourierPhase);

    	for (int n=0; n<iter; ++n)
    	{
    		// Apply POCS to modify iteratively vol2
    		transformer.FourierTransform(V(),V2Fourier,false);
    		POCSFourierAmplitude(V1FourierMag,V2Fourier);
        	transformer.inverseFourierTransform();
			POCSmask(mask(),V());
    		V.write(formatString("V2masked_Amp1_%d.mrc", n));
    		transformer.FourierTransform();
    		//V2FourierPhase = V2Fourier;
        	//extractPhase(V2FourierPhase);
    		POCSFourierPhase(V2FourierPhase,V2Fourier);
        	transformer.inverseFourierTransform();
    		V.write(formatString("V2masked_Amp1_ph2_%d.mrc", n));
        	POCSnonnegative(V());
			double  std2 = V().computeStddev();
			V()*=std1/std2;
    		V.write(formatString("V2masked_Amp1_ph2_nonneg_%d.mrc", n));
    	}
    	// FT final vol2
    	//transformer.FourierTransform(V(),V2Fourier,false);

    	// Define m depending on if vol2 is a pdb
    	MultidimArray<double> m;
    	MultidimArray<double> V2FourierMag;
		FFT_magnitude(V2Fourier,V2FourierMag);
    	if (pdb==true)
    		m=V1FourierMag;
    	else
			MultidimArrayMIN(V1FourierMag,V2FourierMag,m);

    	//transformer.inverseFourierTransform();
    	// Subtraction: m*(vol1-vol2modif)
    	Image<double> V1;
    	V1.read(fnVol1);
    	V1()-= V();

    	//FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V2Fourier)
    	//DIRECT_MULTIDIM_ELEM(V2Fourier,n)=DIRECT_MULTIDIM_ELEM(m,n)*(DIRECT_MULTIDIM_ELEM(V1Fourier,n)/DIRECT_MULTIDIM_ELEM(V1FourierMag,n)-
    	//		                                                     DIRECT_MULTIDIM_ELEM(V2Fourier,n)/DIRECT_MULTIDIM_ELEM(V2FourierMag,n));
		V1.write(fnDiff);
    }
};
