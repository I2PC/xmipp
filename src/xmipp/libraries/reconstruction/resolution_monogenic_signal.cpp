/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
 * 			   Carlos Oscar S. Sorzano            coss@cnb.csic.es (2016)
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

#include "resolution_monogenic_signal.h"
//#define DEBUG
//#define DEBUG_MASK

void ProgMonogenicSignalRes::readParams()
{
	fnVol = getParam("--vol");
	fnVol2 = getParam("--vol2");
	fnMeanVol = getParam("--meanVol");
	fnOut = getParam("-o");
	fnMask = getParam("--mask");
	fnMaskExl = getParam("--maskExclusion");
	fnMaskOut = getParam("--mask_out");
	fnchim = getParam("--chimera_volume");
	sampling = getDoubleParam("--sampling_rate");
	R = getDoubleParam("--volumeRadius");
	minRes = getDoubleParam("--minRes");
	maxRes = getDoubleParam("--maxRes");
	freq_step = getDoubleParam("--step");
	exactres = checkParam("--exact");
	noiseOnlyInHalves = checkParam("--noiseonlyinhalves");
	fnSpatial = getParam("--filtered_volume");
	significance = getDoubleParam("--significance");
	fnMd = getParam("--md_outputdata");
	automaticMode = checkParam("--automatic");
	nthrs = getIntParam("--threads");
}


void ProgMonogenicSignalRes::defineParams()
{
	addUsageLine("This function determines the local resolution of a map");
	addParamsLine("  --vol <vol_file=\"\">   : Input volume");
	addParamsLine("  [--mask <vol_file=\"\">]  : Mask defining the macromolecule");
	addParamsLine("                          :+ If two half volume are given, the noise is estimated from them");
	addParamsLine("                          :+ Otherwise the noise is estimated outside the mask");
	addParamsLine("  [--maskExclusion <vol_file=\"\">]  : Mask defining voxels to exlude analysis of resolution");
	addParamsLine("  [--mask_out <vol_file=\"\">]  : sometimes the provided mask is not perfect, and contains voxels out of the particle");
	addParamsLine("                          :+ Thus the algorithm calculated a tight mask to the volume");
	addParamsLine("  [--vol2 <vol_file=\"\">]: Half volume 2");
	addParamsLine("  [-o <output=\"MGresolution.vol\">]: Local resolution volume (in Angstroms)");
	addParamsLine("  [--meanVol <vol_file=\"\">]: Mean volume of half1 and half2 (only it is neccesary the two haves are used)");
	addParamsLine("  [--chimera_volume <output=\"Chimera_resolution_volume.vol\">]: Local resolution volume for chimera viewer (in Angstroms)");
	addParamsLine("  [--sampling_rate <s=1>]   : Sampling rate (A/px)");
	addParamsLine("                            : Use -1 to disable this option");
	addParamsLine("  [--volumeRadius <s=100>]   : This parameter determines the radius of a sphere where the volume is");
	addParamsLine("  [--step <s=0.25>]       : The resolution is computed at a number of frequencies between mininum and");
	addParamsLine("                            : maximum resolution px/A. This parameter determines that number");
	addParamsLine("  [--minRes <s=30>]         : Minimum resolution (A)");
	addParamsLine("  [--maxRes <s=1>]          : Maximum resolution (A)");
	addParamsLine("  [--exact]                 : The search for resolution will be exact (slower) of approximated (fast).");
	addParamsLine("                            : Usually there are no difference between both in the resolution map.");
	addParamsLine("  [--noiseonlyinhalves]     : The noise estimation is only performed inside the mask");
	addParamsLine("  [--filtered_volume <vol_file=\"\">]       : The input volume is locally filtered at local resolutions.");
	addParamsLine("  [--significance <s=0.95>]    : The level of confidence for the hypothesis test.");
	addParamsLine("  [--md_outputdata <file=\".\">]  : It is a control file. The provided mask can contain voxels of noise.");
	addParamsLine("                                  : Moreover, voxels inside the mask cannot be measured due to an unsignificant");
	addParamsLine("                                  : SNR. Thus, a new mask is created. This metadata file, shows, the number of");
	addParamsLine("                                  : voxels of the original mask, and the created mask");
	addParamsLine("  [--automatic]                   : Resolution range is not neccesary provided");
	addParamsLine("  [--threads <s=4>]               : Number of threads");
}


void ProgMonogenicSignalRes::produceSideInfo()
{
	std::cout << "Starting..." << std::endl;
	Image<double> V;
	if ((fnVol !="") && (fnVol2 !=""))
	{
		Image<double> V1, V2;
		fftN=new MultidimArray< std::complex<double> >;
		V1.read(fnVol);
		V2.read(fnVol2);
		V()=0.5*(V1()+V2());
		V.write(fnMeanVol);

		V1()-=V2();
		V1()/=2;
		FourierTransformer transformer2;
		#ifdef DEBUG
		  V1.write("diff_volume.vol");
		#endif

		transformer2.FourierTransform(V1(), *fftN);
		halfMapsGiven = true;
	}
	else{
	    V.read(fnVol);
	    halfMapsGiven = false;
	    fftN=&fftV;
	}
	V().setXmippOrigin();

	// Prepare mask
	MultidimArray<int> &pMask=mask(), &pMaskExl=maskExcl();
	MultidimArray<double> &inputVol = V();

	if (fnMask != ""){
		mask.read(fnMask);
		mask().setXmippOrigin();}
	else{
		std::cout << "Error: a mask ought to be provided" << std::endl;
		exit(0);}

	double radius, radiuslimit;
	Monogenic mono;
	//The mask changes!! all voxels out of the inscribed sphere are set to -1
	MultidimArray<double> radMap;
	mono.proteinRadiusVolumeAndShellStatistics(pMask, radius, NVoxelsOriginalMask, radMap);
	double smoothparam = 0;
	mono.findCliffValue(radMap, inputVol, radius, radiuslimit, pMask, smoothparam);

	if (fnMaskExl != ""){
		maskExcl.read(fnMaskExl);
		maskExcl().setXmippOrigin();}
	else
		maskExcl = mask;


	transformer_inv.setThreadsNumber(nthrs);

	FourierTransformer transformer;

	VRiesz.resizeNoCopy(inputVol);

	if (fnSpatial!="")
		VresolutionFiltered().initZeros(V());

	transformer.FourierTransform(inputVol, fftV);

	// Frequency volume
	iu = mono.fourierFreqs_3D(fftV, inputVol, freq_fourier_x, freq_fourier_y, freq_fourier_z);

//	#ifdef DEBUG_MASK
	mask.write("mask.vol");
//	#endif


	if (freq_step < 0.25)
		freq_step = 0.25;


	V.clear();
}


void ProgMonogenicSignalRes::firstMonoResEstimation(MultidimArray< std::complex<double> > &myfftV,
		FourierTransformer &transformer_inv, double freq, double freqH, double freqL, MultidimArray<double> &amplitude,
		int count, FileName fnDebug, double &mean_Signal, double &mean_noise, double &thresholdFirstEstimation)
{
	Monogenic mono;
	std::cout << "freq " << freq << "    freqL " << freqL << "    freqH " << freqH <<  std::endl;
//	amplitudeMonogenicSignal3D(myfftV, freq, freqL, freqH, amplitude, count, fnDebug);
	size_t xdim, ydim, zdim, ndim;
	mask().getDimensions(xdim, ydim, zdim, ndim);

	amplitude.initZeros(zdim, ydim, xdim);
	amplitude.resizeNoCopy(VRiesz);
	mono.amplitudeMonoSig3D_LPF(myfftV, transformer_inv,
					fftVRiesz, fftVRiesz_aux, VRiesz,
					freq, freqH, freqL, iu,
					freq_fourier_x, freq_fourier_y, freq_fourier_z,
					amplitude, count, fnDebug);


	Image<double> saveImg;
	saveImg = amplitude;
	FileName iternumber;
	iternumber = formatString("_Amplitude_dif_%i.vol", 999);
	saveImg.write(iternumber);
	saveImg.clear();

	double sumS=0, sumN=0, NN = 0, NS = 0;
	MultidimArray<int> &pMask = mask();
	std::vector<double> noiseValues;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
	{
		double amplitudeValue=DIRECT_MULTIDIM_ELEM(amplitude, n);
		if (DIRECT_MULTIDIM_ELEM(pMask, n)>=1)
		{
//			std::cout << "means_signal = " << amplitudeValue<< std::endl;
			sumS  += amplitudeValue;
			NS += 1.0;
		}
		else if (DIRECT_MULTIDIM_ELEM(pMask, n)==0)
		{
			noiseValues.push_back(amplitudeValue);
			sumN  += amplitudeValue;
			NN += 1.0;
		}
	}
	std::sort(noiseValues.begin(),noiseValues.end());

	mean_Signal = sumS/NS;
	mean_noise = sumN/NN;
	std::cout << "sumS = " << sumS << "  sumN = " << sumN << std::endl;
	std::cout << "NS = " << NS << "  NN = " << NN << std::endl;

	std::cout << "means_signal = " << mean_Signal << "  mean_noise = " << mean_noise << std::endl;

	thresholdFirstEstimation = noiseValues[size_t(noiseValues.size()*significance)];
}


void ProgMonogenicSignalRes::postProcessingLocalResolutions(MultidimArray<double> &resolutionVol,
		std::vector<double> &list, MultidimArray<double> &resolutionChimera, double &cut_value, MultidimArray<int> &pMask)
{
	MultidimArray<double> resolutionVol_aux = resolutionVol;
	double last_resolution_2 = list[(list.size()-1)];

	double Nyquist;
	Nyquist = 2*sampling;

	// Count number of voxels with resolution
	size_t N=0;
	if (automaticMode)
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resolutionVol)
			if (DIRECT_MULTIDIM_ELEM(resolutionVol, n)>=(Nyquist)) //the value 0.001 is a tolerance
				++N;
	}
	else
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resolutionVol)
			if (DIRECT_MULTIDIM_ELEM(resolutionVol, n)>=(last_resolution_2-0.001)) //the value 0.001 is a tolerance
				++N;
	}


	// Get all resolution values
	MultidimArray<double> resolutions(N);
	size_t N_iter=0;
	if (automaticMode)
	{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resolutionVol)
		if (DIRECT_MULTIDIM_ELEM(resolutionVol, n)>=Nyquist)
			DIRECT_MULTIDIM_ELEM(resolutions,N_iter++)=DIRECT_MULTIDIM_ELEM(resolutionVol, n);
	}
	else
	{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resolutionVol)
		if (DIRECT_MULTIDIM_ELEM(resolutionVol, n)>(last_resolution_2-0.001))
			DIRECT_MULTIDIM_ELEM(resolutions,N_iter++)=DIRECT_MULTIDIM_ELEM(resolutionVol, n);
	}
	// Sort value and get threshold
	std::sort(&A1D_ELEM(resolutions,0),&A1D_ELEM(resolutions,N));
	double filling_value = A1D_ELEM(resolutions, (int)(0.5*N)); //median value
	double trimming_value = A1D_ELEM(resolutions, (int)((1-cut_value)*N));

	double init_res, last_res;

	init_res = list[0];
	last_res = list[(list.size()-1)];
	
	std::cout << "--------------------------" << std::endl;
	std::cout << "last_res = " << last_res << std::endl;

	resolutionChimera = resolutionVol;

	if (automaticMode)
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resolutionVol)
		{
			if (DIRECT_MULTIDIM_ELEM(pMask,n) == 0)
			{
			  DIRECT_MULTIDIM_ELEM(resolutionChimera, n) = filling_value;
			}
		}
	}
	else
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resolutionVol)
		{
			if (DIRECT_MULTIDIM_ELEM(resolutionVol, n) < last_res)
			{
				if (DIRECT_MULTIDIM_ELEM(pMask, n) >=1)
				{
					DIRECT_MULTIDIM_ELEM(resolutionChimera, n) = filling_value;
					DIRECT_MULTIDIM_ELEM(resolutionVol, n) = filling_value;
				}
				else
				{
					DIRECT_MULTIDIM_ELEM(resolutionChimera, n) = filling_value;
					DIRECT_MULTIDIM_ELEM(pMask,n) = 0;
				}
			}
			if (DIRECT_MULTIDIM_ELEM(resolutionVol, n) > trimming_value)
			{
			  DIRECT_MULTIDIM_ELEM(pMask,n) = 0;
			  DIRECT_MULTIDIM_ELEM(resolutionVol, n) = filling_value;
			  DIRECT_MULTIDIM_ELEM(resolutionChimera, n) = filling_value;
			}
		}
	}
	//#ifdef DEBUG_MASK
	Image<int> imgMask;
	imgMask = pMask;
	imgMask.write(fnMaskOut);
	//#endif
}

void ProgMonogenicSignalRes::refiningMask(const MultidimArray< std::complex<double> > &myfftV,
								MultidimArray<double> iu, int thrs, MultidimArray<int> &pMask)
{
	Monogenic mono;
	MultidimArray<double> amplitude;

	amplitude.initZeros(pMask);
	mono.monogenicAmplitude_3D_Fourier(myfftV, iu, amplitude, thrs);
	Image<double> amp;
	amp() = amplitude;
	amp.write("beforefilter.vol");
	realGaussianFilter(amplitude, 4);

	amp() = amplitude;
	amp.write("aftefilter.vol");

	double sumS=0, sumN=0, NN = 0, NS = 0;
	std::vector<double> noiseValues;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
	{
		double amplitudeValue=DIRECT_MULTIDIM_ELEM(amplitude, n);
		if (DIRECT_MULTIDIM_ELEM(pMask, n)>=1)
		{
//			std::cout << "means_signal = " << amplitudeValue<< std::endl;
			sumS  += amplitudeValue;
			NS += 1.0;
		}
		else if (DIRECT_MULTIDIM_ELEM(pMask, n)==0)
		{
			noiseValues.push_back(amplitudeValue);
			sumN  += amplitudeValue;
			NN += 1.0;
		}
	}
	std::sort(noiseValues.begin(),noiseValues.end());

	double mean_Signal = sumS/NS;
	double mean_noise = sumN/NN;
	std::cout << "sumS = " << sumS << "  sumN = " << sumN << std::endl;
	std::cout << "NS = " << NS << "  NN = " << NN << std::endl;

	std::cout << "means_signal = " << mean_Signal << "  mean_noise = " << mean_noise << std::endl;

	double thresholdFirstEstimation = noiseValues[size_t(noiseValues.size()*0.99)];

	NVoxelsOriginalMask = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
	{
		if (DIRECT_MULTIDIM_ELEM(pMask, n) >=1)
		{
			if (DIRECT_MULTIDIM_ELEM(amplitude, n)<thresholdFirstEstimation)
			{
				DIRECT_MULTIDIM_ELEM(pMask, n) = 0;
			}
			else
			{
				++NVoxelsOriginalMask;
			}
		}
	}
	Image<int> imgMask;
	imgMask = pMask;
	imgMask.write("refinedMask.vol");
}


void ProgMonogenicSignalRes::run()
{
	produceSideInfo();

	Image<double> outputResolution;
	outputResolution().resizeNoCopy(VRiesz);


	MultidimArray<int> &pMask = mask(), &pMaskExcl = maskExcl();
	MultidimArray<double> &pOutputResolution = outputResolution();
	MultidimArray<double> &pVfiltered = Vfiltered();
	MultidimArray<double> &pVresolutionFiltered = VresolutionFiltered();
	MultidimArray<double> amplitudeMS, amplitudeMN;

	std::cout << "Looking for maximum frequency ..." << std::endl;
	double criticalZ=icdf_gauss(significance);
	double criticalW=-1;
	double resolution, resolution_2, last_resolution = 10000;  //A huge value for achieving
												//last_resolution < resolution
	double meanS, sdS2, meanN, sdN2, thr95;
	double freq, freqH, freqL;
	double max_meanS = -1e38;
	double cut_value = 0.025;


//	Image<int> imgMask2;
//	imgMask2 = mask;
//	imgMask2.write("mascara_original.vol");



	double w0, wF;
	double Nyquist = 2*sampling;
	if (minRes<2*sampling)
		minRes = Nyquist;
	if (automaticMode)
		minRes = Nyquist;
	else
	{
		w0 = sampling/maxRes;
		wF = sampling/minRes;
	}
	bool doNextIteration=true;
	bool lefttrimming = false;
	int fourier_idx, last_fourier_idx = -1;

	//A first MonoRes estimation to get an accurate mask

	double mean_Signal, mean_noise, thresholdFirstEstimation;

	DIGFREQ2FFT_IDX((maxRes+3)/sampling, ZSIZE(VRiesz), fourier_idx);

	FFT_IDX2DIGFREQ(fourier_idx, ZSIZE(VRiesz), freq);
	FFT_IDX2DIGFREQ(fourier_idx + 2, ZSIZE(VRiesz), freqH);
	FFT_IDX2DIGFREQ(fourier_idx - 2, ZSIZE(VRiesz), freqL);

	//std::cout << " freq = " << freq << " freqH = " << freqH << " freqL= " << freq <<std::endl;

	int count_res = 0;
	FileName fnDebug;

	refiningMask(fftV, iu, 2, pMask);

	exit(0);



	amplitudeMS.resizeNoCopy(pOutputResolution);
	firstMonoResEstimation(fftV, transformer_inv, freq, freqH, freqL, amplitudeMS,
			count_res, fnDebug, mean_Signal, mean_noise, thresholdFirstEstimation);

	//refining the mask
//	std::cout << "mean_Signal = " << mean_Signal << std::endl;
	NVoxelsOriginalMask = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitudeMS)
	{
		if (DIRECT_MULTIDIM_ELEM(pMask, n) >=1)
		{
			if (DIRECT_MULTIDIM_ELEM(amplitudeMS, n)<thresholdFirstEstimation)
			{
				std::cout << "cacaaaaaaaaaaaaa" <<std::endl;
				DIRECT_MULTIDIM_ELEM(pMask, n) = 0;
			}
			else
			{
				++NVoxelsOriginalMask;
			}
		}
	}
	std::cout << "thresholdFirstEstimation = " << thresholdFirstEstimation << std::endl;

//	imgMask2 = mask;
//	imgMask2.write("mascara_refinada.vol");

	int iter=0, volsize;
	//TODO: take minimum size
	volsize = XSIZE(pMask);

	std::vector<double> list;

	std::cout << "Analyzing frequencies" << std::endl;
	std::vector<double> noiseValues;
	Monogenic mono;

	amplitudeMN.resizeNoCopy(pOutputResolution);

	pOutputResolution.initZeros(amplitudeMN);

	do
	{
		bool continueIter = false;
		bool breakIter = false;

		mono.resolution2eval(count_res, freq_step,
						resolution, last_resolution,
						freq, freqH,
						last_fourier_idx, volsize,
						continueIter, breakIter,
						sampling, minRes, maxRes,
						doNextIteration, automaticMode);

		if (continueIter)
			continue;

		if (breakIter)
			break;

		std::cout << "resolution = " << resolution << std::endl;

		list.push_back(resolution);

		if (iter <2)
			resolution_2 = list[0];
		else
			resolution_2 = list[iter - 2];

		fnDebug = "Signal";

		freqL = freq + 0.01;
//		if (freqL>=0.5)
//			freqL = 0.5;

		mono.amplitudeMonoSig3D_LPF(fftV, transformer_inv,
				fftVRiesz, fftVRiesz_aux, VRiesz,
				freq, freqH, freqL, iu,
				freq_fourier_x, freq_fourier_y, freq_fourier_z,
				amplitudeMS, iter, fnDebug);

		if (halfMapsGiven){
			fnDebug = "Noise";
			mono.amplitudeMonoSig3D_LPF(*fftN, transformer_inv,
							fftVRiesz, fftVRiesz_aux, VRiesz,
							freq, freqH, freqL, iu,
							freq_fourier_x, freq_fourier_y, freq_fourier_z,
							amplitudeMN, iter, fnDebug);}

		double sumS=0, sumS2=0, sumN=0, sumN2=0, NN = 0, NS = 0;
		noiseValues.clear();

		if (exactres)
		{
			if (halfMapsGiven)
			{
				mono.statisticsInBinaryMask(amplitudeMS, amplitudeMN,
						pMask, pMaskExcl, meanS, sdS2, meanN, sdN2, significance, thr95, NS, NN);
			}
			else
			{
				std::cout << " Entro " << std::endl;
				mono.statisticsInOutBinaryMask(amplitudeMS,
						pMask, pMaskExcl, meanS, sdS2, meanN, sdN2, significance, thr95, NS, NN);
			}
		}
		else
		{
			if (halfMapsGiven)
			{
				if (noiseOnlyInHalves)
				{
					mono.statisticsInBinaryMask(amplitudeMS, amplitudeMN,
											pMask, pMaskExcl, meanS, sdS2, meanN, sdN2, significance, thr95, NS, NN);
				}
				else
				{
					std::cout << "entroooo0" << std::endl;
					FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitudeMS)
					{
						double amplitudeValue=DIRECT_MULTIDIM_ELEM(amplitudeMS, n);
						double amplitudeValueN=DIRECT_MULTIDIM_ELEM(amplitudeMN, n);
						if (DIRECT_MULTIDIM_ELEM(pMaskExcl, n)>=1)
						{
							sumS  += amplitudeValue;
							sumS2 += amplitudeValue*amplitudeValue;
							++NS;
						}
						if (DIRECT_MULTIDIM_ELEM(pMask, n)>=0)
						{
							sumN  += amplitudeValueN;
							sumN2 += amplitudeValueN*amplitudeValueN;
							++NN;
						}
					}
				}
			}
			else
			{
				mono.statisticsInOutBinaryMask(amplitudeMS,
										pMask, pMaskExcl, meanS, sdS2, meanN, sdN2, significance, thr95, NS, NN);
			}
		}
	
//		#ifdef DEBUG
//		std::cout << "NS" << NS << std::endl;
//		std::cout << "NVoxelsOriginalMask" << NVoxelsOriginalMask << std::endl;
//		std::cout << "NS/NVoxelsOriginalMask = " << NS/NVoxelsOriginalMask << std::endl;
//		#endif
		
		if ( (NS/NVoxelsOriginalMask)<cut_value ) //when the 2.5% is reached then the iterative process stops
		{
			std::cout << "Search of resolutions stopped due to mask has been completed" << std::endl;
			doNextIteration =false;
			Nvoxels = 0;

			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitudeMS)
			{
				if (DIRECT_MULTIDIM_ELEM(pOutputResolution, n) == 0)
					DIRECT_MULTIDIM_ELEM(pMask, n) = 0;
				else
				{
					Nvoxels++;
					DIRECT_MULTIDIM_ELEM(pMask, n) = 1;
				}
			}

			#ifdef DEBUG_MASK
			mask.write("partial_mask.vol");
			#endif

			lefttrimming = true;
		}
		else
		{
			if (NS == 0)
			{
				std::cout << "There are no points to compute inside the mask" << std::endl;
				std::cout << "If the number of computed frequencies is low, perhaps the provided"
						"mask is not enough tight to the volume, in that case please try another mask" << std::endl;
				break;
			}

			// Check local resolution
			double thresholdNoise;
			if (exactres)
				thresholdNoise = thr95;
			else
				thresholdNoise = meanN+criticalZ*sqrt(sdN2);

//			#ifdef DEBUG
			  std::cout << "Iteration = " << iter << ",   Resolution= " << resolution <<
					  ",   Signal = " << meanS << ",   Noise = " << meanN << ",  Threshold = "
					  << thresholdNoise <<std::endl;
//			#endif

			double z=(meanS-meanN)/sqrt(sdS2/NS+sdN2/NN);

//			std::cout << "z = " << z << "  zcritical = " << criticalZ << std::endl;

//			if (automaticMode)
//			{
//				if (z>criticalZ)
//				{
//					// Check local resolution
//					double thresholdNoise;
//					if (exactres)
//					{
//						std::sort(noiseValues.begin(),noiseValues.end());
//						thresholdNoise = noiseValues[size_t(noiseValues.size()*significance)];
//					}
//					else
//						thresholdNoise = meanN+criticalZ*sqrt(sdN2);
//
//					#ifdef DEBUG
//					  std::cout << "Iteration = " << iter << ",   Resolution= " << resolution <<
//							  ",   Signal = " << meanS << ",   Noise = " << meanN << ",  Threshold = "
//							  << thresholdNoise << std::endl;
//					#endif
//					double NRES=0;
//					FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitudeMS)
//					{
//						if (DIRECT_MULTIDIM_ELEM(pMask, n)>=1)
//							if (DIRECT_MULTIDIM_ELEM(amplitudeMS, n)<thresholdNoise)
//							{
//								DIRECT_MULTIDIM_ELEM(pMask, n) = 1;
//								DIRECT_MULTIDIM_ELEM(pOutputResolution, n) = resolution;//sampling/freq;
//								if (fnSpatial!="")
//									DIRECT_MULTIDIM_ELEM(pVresolutionFiltered,n)=DIRECT_MULTIDIM_ELEM(pVfiltered,n);
//							}
//							else{
//								++NRES;
//								DIRECT_MULTIDIM_ELEM(pMask, n) += 1;
//								if (DIRECT_MULTIDIM_ELEM(pMask, n) >2)
//								{
//									DIRECT_MULTIDIM_ELEM(pMask, n) = -1;
//									DIRECT_MULTIDIM_ELEM(pOutputResolution, n) = resolution_2;//maxRes - counter*R_;
//								}
//							}
//					}
////					std::cout << " NRES" << NRES << std::endl;
//					if ( ( NRES/((double)NVoxelsOriginalMask) ) > 0.8 )
//					{
//						mask.read(fnMask);
//					}
//				}
//			}
//			else
//			{
				if (meanS>max_meanS)
					max_meanS = meanS;

				if (meanS<0.001*max_meanS){
					std::cout << "Search of resolutions stopped due to too low signal" << std::endl;
					break;}

				if (fnSpatial!=""){
					mono.setLocalResolutionMapAndFilter(amplitudeMS, pMask, pOutputResolution,
							pVfiltered, pVresolutionFiltered, thresholdNoise,  resolution, resolution_2);}
				else{
					Image<double> saveImg;
					saveImg() = amplitudeMS;

					FileName iternumber;
					iternumber = formatString("_Amplitude_new_%i.vol", iter);
					saveImg.write(iternumber);
					saveImg.clear();

					mono.setLocalResolutionMap(amplitudeMS, pMaskExcl, pOutputResolution,
							 thresholdNoise,  resolution, resolution_2);}

				// Is the mean inside the signal significantly different from the noise?
				z=(meanS-meanN)/sqrt(sdS2/NS+sdN2/NN);

				#ifdef DEBUG
					std::cout << "thresholdNoise = " << thresholdNoise << std::endl;
					std::cout << "  meanS= " << meanS << " sigma2S= " << sdS2 << " NS= " << NS << std::endl;
					std::cout << "  meanN= " << meanN << " sigma2N= " << sdN2 << " NN= " << NN << std::endl;
					std::cout << "  z=" << z << " (" << criticalZ << ")" << std::endl;
				#endif

					if (z<criticalZ){
					criticalW = freq;
					std::cout << "Search stopped due to z>Z (hypothesis test)" << std::endl;
					doNextIteration=false;}

				if (doNextIteration){
					if (resolution <= (minRes-0.001))
						doNextIteration = false;}
			}
//		}
		iter++;
		last_resolution = resolution;
	} while (doNextIteration);

	if (lefttrimming == false)
	{
	  Nvoxels = 0;
	  FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitudeMS)
	  {
	    if (DIRECT_MULTIDIM_ELEM(pOutputResolution, n) == 0)
	      DIRECT_MULTIDIM_ELEM(pMask, n) = 0;
	    else
	    {
	      Nvoxels++;
	      DIRECT_MULTIDIM_ELEM(pMask, n) = 1;
	    }
	  }
	#ifdef DEBUG_MASK
	  //mask.write(fnMaskOut);
	#endif
	}
	amplitudeMN.clear();
	amplitudeMS.clear();

	MultidimArray<double> resolutionFiltered, resolutionChimera;
	postProcessingLocalResolutions(pOutputResolution, list, resolutionChimera, cut_value, pMask);


	Image<double> outputResolutionImage;
	outputResolutionImage() = pOutputResolution;//resolutionFiltered;
	outputResolutionImage.write(fnOut);
	outputResolutionImage() = resolutionChimera;
	outputResolutionImage.write(fnchim);


	#ifdef DEBUG
		outputResolution.write("resolution_simple.vol");
	#endif

	if (fnSpatial!="")
	{
		mask.read(fnMask);
		mask().setXmippOrigin();
		Vfiltered.read(fnVol);
		pVfiltered=Vfiltered();
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(pVfiltered)
		if (DIRECT_MULTIDIM_ELEM(pMask,n)==1)
			DIRECT_MULTIDIM_ELEM(pVfiltered,n)-=DIRECT_MULTIDIM_ELEM(pVresolutionFiltered,n);
		Vfiltered.write(fnSpatial);

		VresolutionFiltered().clear();
		Vfiltered().clear();
	}

	MetaData md;
	size_t objId;
	objId = md.addObject();
	md.setValue(MDL_IMAGE, fnOut, objId);
	md.setValue(MDL_COUNT, (size_t) NVoxelsOriginalMask, objId);
	md.setValue(MDL_SCALE, R, objId);
	md.setValue(MDL_COUNT2, (size_t) Nvoxels, objId);

	md.write(fnMd);
}
