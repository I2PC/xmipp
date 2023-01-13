/***************************************************************************
 *
 * Authors:     Jose Luis Vilas (joseluis.vilas-prieto@yale.edu)
 *                             or (jlvilas@cnb.csic.es)
 *
 * Yale University, New Haven, Connecticut, United States of America
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

#include "tomo_sta_deblurring.h"
#include <core/metadata_extension.h>
#include <data/monogenic_signal.h>

#include <random>
#include <limits>
#include "fftwT.h"
#include <CTPL/ctpl_stl.h>
#include <type_traits>
#include <chrono>

void ProgSTADeblurring::defineParams()
{
	addUsageLine("Calculate Fourier Shell Occupancy - FSO curve - via directional FSC measurements.");
	addUsageLine("Following outputs are generated:");
	addUsageLine("  1) FSO curve");
	addUsageLine("  2) Global resolution from FSO and FSC");
	addUsageLine("Reference: J.L. Vilas, H.D. Tagare, XXXXX (2021)");
	addUsageLine("+* Fourier Shell Occupancy (FSO)", true);
	addUsageLine("+ The Fourier Shell Occupancy Curve can be obtained from a set of directional FSC (see below).");
	addUsageLine("+ In contrast, when the OFSC shows a slope the map will be anisotropic. The lesser slope the higher resolution isotropy.");
	addUsageLine("+ ");
	addUsageLine("+* Directional Fourier Shell Correlation (dFSC)", true);
	addUsageLine("+ This program estimates the directional FSC between two half maps along all posible directions on the projection sphere.");
	addUsageLine(" ");
	addUsageLine("+* Resolution Distribution and 3DFSC", true);
	addUsageLine("+ The directional-FSC, dFSC is estimated along 321 directions on the projection sphere. For each direction the corresponding");
	addUsageLine(" ");
	addUsageLine(" ");
	addSeeAlsoLine("resolution_fsc");

	addParamsLine("   --subtomos <input_file>                  : Metadata with the list of subtomograms");
	addParamsLine("   [--reference <input_file=\"\">]               : Reference, this map use to be the STA result that will be refined by mean s of the smart STA of this method. If a map is not provided, then the algorithm will average all provided subtomos.");
	addParamsLine("   [--mask <input_file=\"\">]               : Mask for the reference. It must be smooth.");

	addParamsLine("   [-o <output_folder=\"\">]          : Folder where the results will be stored.");
	addParamsLine("   [--stack]                          : Folder where the results will be stored.");

	addParamsLine("   [--sampling <Ts=1>]                : (Optional) Pixel size (Angstrom). If it is not provided by default will be 1 A/px.");
	addParamsLine("   [--precon <pRecon=0.9>]            : (Optional) Pixel size (Angstrom). If it is not provided by default will be 1 A/px.");
	addParamsLine("   [--niters <niters=1>]            	 : (Optional) Pixel size (Angstrom). If it is not provided by default will be 1 A/px.");
	addParamsLine("   [--spectral ] 	             	 : (Optional) Spectral approach.");
	addParamsLine("   [--threads <Nthreads=1>]           : (Optional) Number of threads to be used");

	addExampleLine("Resolution of two half maps half1.mrc and half2.mrc with a sampling rate of 2 A/px", false);
	addExampleLine("xmipp_reconstruction_sta_deblurring --half1 half1.mrc --half2 half2.mrc --sampling_rate 2 ");
	addExampleLine("Resolution of two half maps half1.mrc and half2.mrc with a sampling rate of 2 A/px and a mask mask.mrc", false);
	addExampleLine("xmipp_reconstruction_sta_deblurring --half1 half1.mrc --half2 half2.mrc --mask mask.mrc --sampling_rate 2 ");
}

void ProgSTADeblurring::readParams()
{
	fnSubtomos = getParam("--subtomos");
	fnRef = getParam("--reference");
	fnMask = getParam("--mask");
	fnOut = getParam("-o");
	isStack = checkParam("--stack");
	pRecon = getDoubleParam("--precon");
	sampling = getDoubleParam("--sampling");
	niters = getIntParam("--niters");
	Nthreads = getIntParam("--threads");
}

void ProgSTADeblurring::generateProjections(FileName &fnVol, double &sampling_rate)
{
	FileName fnGallery, fnGalleryMetaData;

	// Generate projections
	fnGallery=formatString("%s/gallery.stk", fnOut.c_str());

	String args=formatString("-i %s -o %s --sampling_rate %f", fnVol.c_str(), fnGallery.c_str(), sampling_rate);
	//We are considering the psi sampling = angular sampling rate

	std::cout << args << std::endl;
	String cmd=(String)"xmipp_angular_project_library " + args;
	system(cmd.c_str());
}


template<typename T>
void ProgSTADeblurring::createReference(MultidimArray<T> &refMap)
{
	// Read subtomos
	MetaDataVec md;
	md.read(fnSubtomos);

	FileName fnVol;
	Image<T> subtomo;
	MultidimArray<T> &ptrsubtomo = subtomo();
	MultidimArray<T> ptrsubtomoAligned;
	

	bool wrap =false;
	
	Image<double> saveImg;
	saveImg() = subtomoRotated;
	subtomoRotated.write(fnImgOut);
	mdAligned.resetGeo(false);


	
	size_t nsubtomos = 0;

	// for (size_t  k=0; k<md.size(); k++)
	for (const auto& row : md)
    {

		row.getValue(MDL_IMAGE, fnVol);

		Matrix2D<double> eulerMat;
		eulerMat.initIdentity(4);

		subtomo.read(fnVol);

		ptrsubtomoAligned.resizeNoCopy(ptrsubtomo);
		subtomo.setXmippOrigin();
		ptrsubtomoAligned.setXmippOrigin();

		geo2TransformationMatrix(row, eulerMat);
    	applyGeometry(xmipp_transformation::LINEAR, ptrsubtomoAligned, ptrsubtomo, eulerMat, xmipp_transformation::IS_NOT_INV, xmipp_transformation::DONT_WRAP);


		FileName fn = "caca.xmd";
		Image<double> saveImg;
		saveImg() = ptrsubtomoAligned;
		subtomoRotated.write(fn);

		// TODO: it should possible to avoid reading again and again by reserving memory for the map and loading the data in that memory direction?
		

		if (refMap.getDim() < 1)
		{
			refMap.initZeros(ptrsubtomo);
		}
		std::cout << "count = " << nsubtomos << std::endl;

		refMap += ptrsubtomo;

		nsubtomos++;

	}
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(refMap)
	{
		DIRECT_MULTIDIM_ELEM(refMap,n) /= nsubtomos;
	}


	Image<T> svImg;
	svImg() = refMap;
	FileName fn;
	svImg.write(fnOut+"/sta.mrc");
}


void ProgSTADeblurring::defineReference(MultidimArray<double> &ptrRef)
{

	// Reading the reference or creating one
	if (fnRef != "")
	{
		Image<double> refImg;
		auto &ptrRef = refImg();
		refImg.read(fnRef);
	}
	else
	{
		std::cout << " Creating a reference map ... " << std::endl;
		using std::chrono::high_resolution_clock;
		using std::chrono::duration_cast;
		using std::chrono::duration;
		using std::chrono::milliseconds;

		auto t1 = high_resolution_clock::now();

		createReference(ptrRef);

		auto t2 = high_resolution_clock::now();
		/* Getting number of milliseconds as an integer. */
    	auto ms_int = duration_cast<milliseconds>(t2 - t1);

    	/* Getting number of milliseconds as a double. */
    	duration<double, std::milli> ms_double = t2 - t1;

    	std::cout << ms_int.count() << "ms\n";
    	std::cout << ms_double.count() << "ms\n";
	}
}

template<typename T>
void ProgSTADeblurring::createMask(MultidimArray<T> &vol, MultidimArray<T> &mask)
{
	Image<double> maskImg;

	if (fnMask=="")
	{
		std::cout << "Creating the mask " << std::endl;
		mask.resizeNoCopy(vol);

		int limitRadius = XSIZE(vol)*0.5;
		limitRadius *= limitRadius;

		double radius = 0;
		
		mask.initZeros(vol);


		std::vector<double> maskElems;
		//TODO: change FOR_ALL to enhancer the performance
		// Here we look at the background (outside the sphere)
		FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
		{
			if ((k*k + i*i + j*j)>=limitRadius)
			{
				double val = A3D_ELEM(vol, k, i, j);
				double val2 = val*val;
				maskElems.push_back(val2);
			}
		}

		int NE2 = maskElems.size();
		int sumIn = 0;

		std::cout << "NE2 = " << NE2 << std::endl;

		FOR_ALL_ELEMENTS_IN_ARRAY3D(vol)
		{
			double volValue = A3D_ELEM(vol, k, i, j);
			volValue *= volValue;

			for (int idx = 0; idx<NE2; idx++)
			{
				double aux = maskElems[idx];
				
				if (aux<(volValue))
				{
					sumIn += aux;
				}
			}
			if (sumIn/NE2>0.995)
			{
				A3D_ELEM(mask, k, i, j) *= 1;
			}
			else{
				A3D_ELEM(mask, k, i, j) = 0;
			}
		}

		double lastVal = -1e38;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(vol)
		{
			double val = DIRECT_MULTIDIM_ELEM(vol,n);
			if (val>lastVal)
			{
				lastVal = val;
			}
		}

		lastVal = lastVal/10;

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(vol)
		{
			double val = DIRECT_MULTIDIM_ELEM(vol,n);
			if (val>lastVal)
			{
				DIRECT_MULTIDIM_ELEM(mask,n) = 1;
			}
			else{
				DIRECT_MULTIDIM_ELEM(mask,n) = 0;
			}
		}
	}else
	{
		std::cout << "Loading mask " << std::endl;
		maskImg.read(fnMask);
		mask = maskImg();
	}

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(vol)
	{
		DIRECT_MULTIDIM_ELEM(vol,n) *= DIRECT_MULTIDIM_ELEM(mask,n);
	}
	
}


void ProgSTADeblurring::normalizeReference(MultidimArray<double> &map, MultidimArray<std::complex<double>> &FTmap)
{
	FourierTransformer transformerRef(FFTW_BACKWARD);
	transformerRef.setThreadsNumber(Nthreads);

	transformerRef.FourierTransform(map, FTmap);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FTmap)
	{
		DIRECT_MULTIDIM_ELEM(FTmap, n) /= abs(DIRECT_MULTIDIM_ELEM(FTmap, n) + 1e-38);
	}
}


void ProgSTADeblurring::weightsRealAndFourier(MultidimArray<double> &map, MetaDataVec &mdSubtomos, std::vector<double> &meanPhaseCorr,
											std::vector<double> &allCorrReal, size_t &Nsubtomos)
{
	long s = 0;
	FileName fnVol;

	Image<double> volImg;
	auto &ptrvolImg  = volImg();
	FourierTransformer transformerSubtomo(FFTW_BACKWARD);
	transformerSubtomo.setThreadsNumber(Nthreads);

	for (const auto& row : mdSubtomos)
	{
		std::cout << "s = " << s << std::endl;
		row.getValue(MDL_IMAGE, fnVol);
		
		volImg.read(fnVol);

		double corrValreal;

		corr3(map, ptrvolImg, corrValreal);

		//Computing the FFT
		transformerSubtomo.FourierTransform(ptrvolImg, FTsubtomo, false);

		double cumW = 0;

		MultidimArray<double> thresholdMissingWedge;
		missingWedgeDetection(FTsubtomo, thresholdMissingWedge);


		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FTsubtomo)
		{
			std::complex<double> z = DIRECT_MULTIDIM_ELEM(FTsubtomo, n);
			// std::cout << " " << z << std::endl;
			std::complex<double> zz;

			double f = DIRECT_MULTIDIM_ELEM(freqMap, n);

			if (f>0.5)
				continue;

			auto idx = (int) round(f * xvoldim);

			double critivalValue = dAi(thresholdMissingWedge, idx);

			if (critivalValue>z)
				continue;


			zz = conj(z/abs(z+1e-38));

			//std::cout << " " << zz  << " " << cumW << std::endl;
			// TODO:it is much more efficient to conjungate the reference instead of conjugate the subtomo
			//std::cout << " " << DIRECT_MULTIDIM_ELEM(FTref, n) << std::endl;
			
			// std::cout << "--------------" << std::endl;
			double auxVal = real(zz*DIRECT_MULTIDIM_ELEM(FTref, n));
			// std::cout << " " << auxVal  << " " << cumW << std::endl;
			double w = std::max(0.0, auxVal);
			// std::cout << " " << w  << std::endl;
			// std::cout << "....................." << std::endl;
			// cumW += w;
			// exit(0);

			cumW += w;

			//DIRECT_MULTIDIM_ELEM(phaseCorr, n) = w;
			//DIRECT_MULTIDIM_ELEM(cumphaseCorr, n) += w;

		}

		allCorrReal.push_back(corrValreal);
		meanPhaseCorr.push_back(cumW/FTsubtomo.zyxdim);


		// FileName fnAux = formatString("phaseCorr_%i.mrc", s);

		// Image<double> imgAux;
		// imgAux()= phaseCorr;
		// imgAux.write(fnAux);

		s++;
	}
	Nsubtomos = s;
}


// void ProgSTADeblurring::normalizeSubtomos(MultidimArray<double> &phaseCorr, std::vector<double> &phaseCorrPersubtomo)
// {
// 	long s = 0;
// 	// Now the subtomos are normalized and the weight per subtomogram is stored
	
// 	Image<double> volImg;
// 	auto &ptrvolImg  = volImg();

// 	size_t siz;
// 	siz = phaseCorr.zyxdim;

// 	std::cout << "number of elements = " << siz << std::endl;

// 	for (size_t tom = 0; tom<Nsubtomos; tom++)
// 	{
// 		std::cout << "normalizing weights subtomo = " << tom << std::endl;
		
// 		FileName fnAux = formatString("phaseCorr_%i.mrc", tom);
		
// 		volImg.read(fnAux);

// 		double sumW = 0;

// 		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FTsubtomo)
// 		{
// 			DIRECT_MULTIDIM_ELEM(ptrvolImg, n) /= DIRECT_MULTIDIM_ELEM(cumWeights, n);
			
// 			sumW += DIRECT_MULTIDIM_ELEM(ptrvolImg, n);
// 		}
// 		// The average per image
// 		phaseCorrPersubtomo.push_back(sumW/siz);

// 		tom++;
// 	}
// }


void ProgSTADeblurring::rankingSubtomograms(std::vector<double> &meanPhaseCorr, 
											std::vector<double> &allCorrReal, std::vector<double> &wSubtomo)
{
	long ss=0;

	size_t Nsubtomos;
	Nsubtomos = meanPhaseCorr.size();

	for (size_t idxtomo = 0; idxtomo<Nsubtomos; idxtomo++)
	{
		std::cout << "estimating final weights s = " << ss << std::endl;
		
		double meanphaseCorrValue = meanPhaseCorr[idxtomo]; 
		long count  = 0;

		for (size_t idxtomo2= 0; idxtomo2<Nsubtomos; idxtomo2++)
		{
			if (meanPhaseCorr[idxtomo2] < meanphaseCorrValue)
			{
				count++;
			}
		}

		double aux = ((double) count/((double) meanPhaseCorr.size()));

		//be carefull s is reused variable and it is the number of subtomos
		if (aux < (1-pRecon))
		{
			aux = 0.0;
		}
		else
		{
			aux = 1.0;
		}
		
		wSubtomo.push_back(aux*meanphaseCorrValue*allCorrReal[idxtomo]);

		ss++;
	}
}

void ProgSTADeblurring::weightedAverage(MetaDataVec &mdSubtomos, std::vector<double> &weightsVector, 
										std::vector<double> &allCorrReal, std::vector<double> &meanPhaseCorr, MultidimArray<double> &wAvg)
{
	std::cout << "estimating the weighted average ..." << std::endl;

	FileName fnVol;

	Image<double> volImg;
	auto &ptrvolImg  = volImg();

	MetaDataVec mdWeights;
	MDRowVec rowOut;

	size_t s = 0;

	for (const auto& row : mdSubtomos)
	{
		std::cout << "s = " << s << std::endl;
		row.getValue(MDL_IMAGE, fnVol);

		volImg.read(fnVol);

		double w= weightsVector[s];

		rowOut.setValue(MDL_IMAGE, fnVol);
		rowOut.setValue(MDL_WEIGHT, w);
		rowOut.setValue(MDL_WEIGHT_REALCORR, allCorrReal[s]);
		rowOut.setValue(MDL_WEIGHT_PHASECORR, meanPhaseCorr[s]);
		mdWeights.addRow(rowOut);


		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(ptrvolImg)
		{
			DIRECT_MULTIDIM_ELEM(wAvg, n) += w*DIRECT_MULTIDIM_ELEM(ptrvolImg, n);
		}

		s++;
	}

	Image<double> saveImg;
	saveImg()= wAvg;
	saveImg.write(fnOut+"/weightedsta.mrc");
	mdWeights.write(fnOut+"/weights.xmd");
}

// void ProgSTADeblurring::2dImages(MultidimArray<double> &refMap, MultidimArray<std::complex<double>> &FTref, MultidimArray<double> &mask)
// {

// }


void ProgSTADeblurring::phaseCorrelationStep(MultidimArray<double> &map,  MultidimArray<std::complex<double>> &FTref, MultidimArray<double> &mask)
{
	MetaDataVec mdSubtomos;
	mdSubtomos.read(fnSubtomos);

	// Image<double> volImg;
	// auto &ptrvolImg  = volImg();

	// MultidimArray<double> phaseCorr, cumphaseCorr;
	// phaseCorr.resizeNoCopy(FTref);
	// cumphaseCorr.resizeNoCopy(FTref);
	// cumphaseCorr.initZeros();

	std::vector<double> allCorrReal, meanPhaseCorr, wSubtomo;
	size_t Nsubtomos=0;

	MultidimArray<double> wAvg;
	wAvg.initZeros(map);

	weightsRealAndFourier(map, mdSubtomos, meanPhaseCorr, allCorrReal, Nsubtomos);
	

	//std::vector<double> phaseCorrPersubtomo;
	//normalizeSubtomos(phaseCorrPersubtomo);

	// Ranking the subtomos
	rankingSubtomograms(meanPhaseCorr, allCorrReal, wSubtomo);


	weightedAverage(mdSubtomos, wSubtomo, allCorrReal, meanPhaseCorr, wAvg);

	map = wAvg;

	// FileName fnVol;

	// for (const auto& row : mdSubtomos)
	// {
	// 	std::cout << "s = " << s << std::endl;
	// 	row.getValue(MDL_IMAGE, fnVol);
		
	// 	fnAux = formatString("fourierWeight_%i.mrc", s);
	// 	volImg.read(fnVol);

	// 	//Computing the FFT
	// 	transformerSubtomo.FourierTransform(ptrvolImg, FTsubtomo, false);

	// 	volImg.read(fnAux);

	// 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FTsubtomo)
	// 	{
	// 		DIRECT_MULTIDIM_ELEM(FTsubtomo, n) *= DIRECT_MULTIDIM_ELEM(ptrvolImg, n)*probSubtomo[s];
	// 	}

	// 	corr3(refMap, ptrvolImg, corrVal);

	// 	allCorr.push_back(corrVal);

	// 	s++;
	// }


}

//TODO: optimize performance
void ProgSTADeblurring::corr3(MultidimArray<double> &vol1, MultidimArray<double> &vol2, double &corrVal)
{
	double sumX = 0;
	double sumY = 0;
	double N = 0;
	
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(vol1)
	{
		sumX += DIRECT_MULTIDIM_ELEM(vol1, n);
		sumY += DIRECT_MULTIDIM_ELEM(vol2, n);
		N += 1.0;
	}

	double meanX = sumX/N;
	double meanY = sumY/N;


	double sumValTot = 0;

	double num = 0;
	double den1 = 0;
	double den2 = 0;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(vol1)
	{
		double xVol = DIRECT_MULTIDIM_ELEM(vol1, n) - meanX;
		double yVol = DIRECT_MULTIDIM_ELEM(vol2, n) - meanY;

		num += (xVol) * (yVol);
		den1 += (xVol) * (xVol);
		den2 += (yVol) * (yVol);
	}

	corrVal = std::max(0.0, num/sqrt(den1*den2));

}

// void ProgSTADeblurring::missingWedgeExclusion()
// {

// }

// void ProgSTADeblurring::normalizeGallery()
// {
// 	for (size_t i = 1; i<Nimgs; ++i)
// 	{
		
// 		normalizeReference();
// 	}
	
// }


void ProgSTADeblurring::defineFrequencies(const MultidimArray< std::complex<double> > &mapfftV,
		const MultidimArray<double> &inputVol, MultidimArray<double> &freqMapIdx)
{
	// Initializing the frequency vectors
	freq_fourier_z.initZeros(ZSIZE(mapfftV));
	freq_fourier_x.initZeros(XSIZE(mapfftV));
	freq_fourier_y.initZeros(YSIZE(mapfftV));

	// u is the frequency
	double u;

	// Defining frequency components. First element should be 0, it is set as the smallest number to avoid singularities

	VEC_ELEM(freq_fourier_z,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<ZSIZE(mapfftV); ++k){
		FFT_IDX2DIGFREQ(k,ZSIZE(inputVol), u);
		VEC_ELEM(freq_fourier_z, k) = u;
	}

	VEC_ELEM(freq_fourier_y,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<YSIZE(mapfftV); ++k){
		FFT_IDX2DIGFREQ(k,YSIZE(inputVol), u);
		VEC_ELEM(freq_fourier_y, k) = u;
	}

	VEC_ELEM(freq_fourier_x,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<XSIZE(mapfftV); ++k){
		FFT_IDX2DIGFREQ(k,XSIZE(inputVol), u);
		VEC_ELEM(freq_fourier_x, k) = u;
	}

	//Initializing map with frequencies
	freqMap.resizeNoCopy(mapfftV);
	freqMap.initConstant(1.0);  //Nyquist is 2, we take 1.9 greater than Nyquist

	xvoldim = XSIZE(inputVol);
	yvoldim = YSIZE(inputVol);
	zvoldim = ZSIZE(inputVol);
	freqElems.initZeros(xvoldim/2+1);

	// Directional frequencies along each direction
	double uz, uy, ux, uz2, uz2y2;
	long n=0;
	int idx = 0;

	// Ncomps is the number of frequencies lesser than Nyquist
	long Ncomps = 0;
	std::vector<long> fourierIndices;
		
	for(size_t k=0; k<ZSIZE(mapfftV); ++k)
	{
		uz = VEC_ELEM(freq_fourier_z, k);
		uz2 = uz*uz;
		for(size_t i=0; i<YSIZE(mapfftV); ++i)
		{
			uy = VEC_ELEM(freq_fourier_y, i);
			uz2y2 = uz2 + uy*uy;

			for(size_t j=0; j<XSIZE(mapfftV); ++j)
			{
				ux = VEC_ELEM(freq_fourier_x, j);
				ux = sqrt(uz2y2 + ux*ux);

				if	(ux<=0.5)
				{
					idx = (int) round(ux * xvoldim);
					DIRECT_MULTIDIM_ELEM(freqMap,n) = ux;
					DIRECT_MULTIDIM_ELEM(freqElems, idx) += 1
				}				
				++n;
			}
		}
	}
}


void ProgSTADeblurring::missingWedgeDetection(MultidimArray<double> &myfft, MultidimArray<double> &thresholdMissingWedge)
{
	double criticalZ=icdf_gauss(0.2);
	// num and den are de numerator and denominator of the fsc

	auto ZdimFT1=(int)ZSIZE(myfft);
	auto YdimFT1=(int)YSIZE(myfft);
	auto XdimFT1=(int)XSIZE(myfft);

	// meanShell and stdShell will be the mean and std of the shel values.
	MultidimArray<double> meanShell, stdShell;
	size_t Nelems = xvoldim/2+1;
	meanShell.initZeros(Nelems);
	stdShell.initZeros(Nelems);
	thresholdMissingWedge.initZeros(Nelems);

	long n = 0;
	for (int k=0; k<ZdimFT1; k++)
	{
		for (int i=0; i<YdimFT1; i++)
		{
			for (int j=0; j<XdimFT1; j++)
			{
				double f = DIRECT_MULTIDIM_ELEM(freqMap,n);
				++n;

				// Only reachable frequencies
				// To speed up the algorithm, only are considered those voxels with frequency lesser than Nyquist, 0.5. The vector idx_count
				// stores all frequencies lesser than Nyquist. This vector determines the frequency of each component of
				// real_z1z2, absz1_vec, absz2_vec.
				if (f>0.5)
					continue;
				
				// Index of each frequency
				auto idx = (int) round(f * xvoldim);

				// Fourier coefficients of both halves
				std::complex<double> &fourierValue = dAkij(myfft, k, i, j);

				auto absVal = abs(fourierValue);
				dAi(meanShell, idx)  += absVal;
				dAi(stdShell, idx)  += absVal*absVal;
			}
		}
	}

	FOR_ALL_ELEMENTS_IN_ARRAY1D(meanShell)
	{
		long Nel = dAi(freqElems,i);
		double m = dAi(meanShell,i)/Nel;
		double m2 = dAi(stdShell,i)/Nel;

		double stdVal = m2/Nel - m*m;

		dAi(thresholdmissingWedge, i) = m + criticalZ * sqrt(stdVal);
	}

}



void ProgSTADeblurring::resetGeo(MetaDataVecRow &row, bool addLabels = true)
{
	row.setValue(MDL_ORIGIN_X,  0., addLabels);
	row.setValue(MDL_ORIGIN_Y,  0., addLabels);
	row.setValue(MDL_ORIGIN_Z,  0., addLabels);
	row.setValue(MDL_SHIFT_X,   0., addLabels);
	row.setValue(MDL_SHIFT_Y,   0., addLabels);
	row.setValue(MDL_SHIFT_Z,   0., addLabels);
	row.setValue(MDL_ANGLE_ROT, 0., addLabels);
	row.setValue(MDL_ANGLE_TILT,0., addLabels);
	row.setValue(MDL_ANGLE_PSI, 0., addLabels);
	row.setValue(MDL_WEIGHT,   1., addLabels);
	row.setValue(MDL_FLIP,     false, addLabels);
	row.setValue(MDL_SCALE,    1., addLabels);
}


void ProgSTADeblurring::applyTransformationMatrix(MultidimArray<double> &subtomoOrig, MultidimArray<double> &subtomoRotated, Matrix2D<double> &eulermatrix,
												  MetaDataVecRow &mdAligned)
{
	
	bool wrap =false;
	
	subtomoOrig.setXmippOrigin();
	subtomoRotated.setDatatype(subtomoOrig.getDatatype());
	subtomoRotated.resize(1, ZSIZE(subtomoOrig), YSIZE(subtomoOrig), XSIZE(subtomoOrig), false);
	subtomoRotated.setXmippOrigin();
	applyGeometry(xmipp_transformation::BSPLINE3, subtomoRotated, subtomoOrig, eulermatrix, xmipp_transformation::IS_NOT_INV, wrap, 0.);
	Image<double> saveImg;
	saveImg() = subtomoRotated;
	subtomoRotated.write(fnImgOut);
	mdAligned.resetGeo(false);
}


void ProgSTADeblurring::run()
{
	std::cout << "Starting ... " << std::endl;

	MultidimArray<double> ptrRef;

	defineReference(ptrRef);

	if (isStack)
	{
		FileName fnVol;
		fnVol = fnOut+"/sta.mrc";
		generateProjections(fnVol, sampling);
	}

	//MultidimArray<std::complex<double>> FTref;

	for (size_t iter = 0; iter<niters; ++iter)
	{
		// FourierTransformer transformerRef(FFTW_BACKWARD);
		// transformerRef.setThreadsNumber(Nthreads);
		// transformerRef.FourierTransform(ptrRef, FTref, false);

		//The mask is created and applied to the reference
		createMask(ptrRef, mask);

		std::cout << "after loading mask" << std::endl;

		if (isStack)
		{
			// normalizeGallery(ptrRef, FTref);
			phaseCorrelationStep(ptrRef, FTref, mask);
		}
		else
		{
			normalizeReference(ptrRef, FTref);
			phaseCorrelationStep(ptrRef, FTref, mask);
		}
	}



	
	// FourierTransformer transformerSubtomo(FFTW_BACKWARD);
	// transformerSubtomo.setThreadsNumber(Nthreads);

	// FT_sta.resizeNoCopy(FTref);
	// FT_sta.initZeros();
	// FTsubtomo.resizeNoCopy(FTref);




	// std::cout << "Finished!" << std::endl;	

	// Image<double> STA_smart;
	// STA_smart() = ptrSTA;
	// STA_smart.write(fnOut+"/STA_smart.mrc");
}


