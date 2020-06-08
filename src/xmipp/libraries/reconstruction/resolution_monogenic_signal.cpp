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
	minRes = getDoubleParam("--minRes");
	maxRes = getDoubleParam("--maxRes");
	freq_step = getDoubleParam("--step");

	fnMask = getParam("--mask");
	fnMaskExl = getParam("--maskExcl");

	sampling = getDoubleParam("--sampling_rate");
	significance = getDoubleParam("--significance");
	fnOut = getParam("-o");
	gaussian = checkParam("--gaussian");
	noiseOnlyInHalves = checkParam("--noiseonlyinhalves");
	nthrs = getIntParam("--threads");
}


void ProgMonogenicSignalRes::defineParams()
{
	addUsageLine("MONORES: This algorithm estimate the local resolution map from a single reconstruction");
	addUsageLine("or two half maps.");
	addUsageLine("Reference: J.L. Vilas et al, MonoRes: Automatic and Accurate Estimation of ");
	addUsageLine("Local Resolution for Electron Microscopy Maps, Structure, 26, 337-344, (2018).");
	addUsageLine("  ");
	addUsageLine("  ");
	addParamsLine("  --vol <vol_file=\"\">         : Input map to estimate its local resolution map.");
	addParamsLine("                                : If two half maps are used, it will be the first half map.");
	addParamsLine("  --minRes <s=30>               : Lowest resolution in (A) for the resolution range");
	addParamsLine("                                : to be analyzed.");
	addParamsLine("  --maxRes <s=1>                : Highest resolution in (A) for the resolution range");
	addParamsLine("                                : to be analyzed.");
	addParamsLine("  --sampling_rate <s=1>         : Sampling rate (A/px)");
	addParamsLine("  -o <output_file=\"\">         : Folder where the results will be stored.");
	addParamsLine("                                : If two half maps are used this should be the first one.");
	addParamsLine("  [--vol2 <vol_file=\"\">]      : (Optional but recommended) Second half map to estimate its");
	addParamsLine("                                : local resolution map. The first one will be the --vol label.");
	addParamsLine("  [--mask <vol_file=\"\">]      : (Optional but recommended) A mask defining the region where ");
	addParamsLine("                                : the protein is.");
	addParamsLine("  [--maskExcl <vol_file=\"\">]  : (Optional) This mask excludes the masked region in the ");
	addParamsLine("                          	   : estimation of the local resolution.");

	addParamsLine("  [--step <s=0.25>]             : (Optional) The resolution is computed from low to high frequency");
	addParamsLine("  		                       : in steps of this parameter in (A).");
	addParamsLine("  [--significance <s=0.95>]     : (Optional) The level of confidence for the hypothesis test between");
	addParamsLine("                                : signal and noise.");
	addParamsLine("  [--threads <s=4>]             : (Optional) Number of threads to parallelize the algorithm.");
	addParamsLine("  [--noiseonlyinhalves]         : (Optional) The noise estimation is only performed inside the mask.");
	addParamsLine("                                : This feature only works when two half maps are provided as input.");
	addParamsLine("  [--gaussian]                  : (Optional) This flag assumes than the noise is gaussian.");
	addParamsLine("                                : Usually there are no difference between this assumption and the ");
	addParamsLine("                                : exact noise distribution. If this flag is not provided, the exact");
	addParamsLine("                                : distribution is estimated.");
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
		V.write(fnOut+"/meanMap.mrc");

		V1()-=V2();
		V1()/=2;
		FourierTransformer transformer2;

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

	double radius, radiuslimit, smoothparam = 0;
	Monogenic mono;
	//The mask changes!! all voxels out of the inscribed sphere are set to -1
	MultidimArray<double> radMap;
	mono.proteinRadiusVolumeAndShellStatistics(pMask, radius, NVoxelsOriginalMask, radMap);
	mono.findCliffValue(radMap, inputVol, radius, radiuslimit, pMask, smoothparam);

	if (fnMaskExl != ""){
		maskExcl.read(fnMaskExl);
		maskExcl().setXmippOrigin();
		MultidimArray<int> &pMaskExcl = maskExcl();
		excludeArea(pMask, pMaskExcl);
	}else
	{
		if (halfMapsGiven && noiseOnlyInHalves)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(pMask)
			{
				if (DIRECT_MULTIDIM_ELEM(pMask, n) <1)
					DIRECT_MULTIDIM_ELEM(pMask, n) = -1;
			}
		}
	}


	transformer_inv.setThreadsNumber(nthrs);
	FourierTransformer transformer;
	VRiesz.resizeNoCopy(inputVol);
	transformer.FourierTransform(inputVol, fftV);

	// Frequency volume
	iu = mono.fourierFreqs_3D(fftV, inputVol, freq_fourier_x, freq_fourier_y, freq_fourier_z);

	if (freq_step < 0.25)
		freq_step = 0.25;

	V.clear();
}


void ProgMonogenicSignalRes::excludeArea(MultidimArray<int> &pMask, MultidimArray<int> &pMaskExcl)
{
	if (halfMapsGiven)
	{
		if (noiseOnlyInHalves)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(pMask)
			{
				if (DIRECT_MULTIDIM_ELEM(pMask, n) ==1)
				{
					if (DIRECT_MULTIDIM_ELEM(pMaskExcl, n) == 1)
						DIRECT_MULTIDIM_ELEM(pMask, n) = -1;
				}
				else
				{
					DIRECT_MULTIDIM_ELEM(pMask, n) = -1;
				}
			}
		}
	}
	else
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(pMask)
		{
			if (DIRECT_MULTIDIM_ELEM(pMask, n) ==1)
			{
				if (DIRECT_MULTIDIM_ELEM(pMaskExcl, n) == 1)
					DIRECT_MULTIDIM_ELEM(pMask, n) = -1;
			}
		}
	}
}


void ProgMonogenicSignalRes::refiningMask(const MultidimArray< std::complex<double> > &myfftV,
								MultidimArray<double> iu, int thrs, MultidimArray<int> &pMask)
{
	Monogenic mono;
	MultidimArray<double> amplitude;

	amplitude.initZeros(pMask);
	mono.monogenicAmplitude_3D_Fourier(myfftV, iu, amplitude, thrs);

	realGaussianFilter(amplitude, 4);

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

	double thresholdFirstEstimation = noiseValues[size_t(noiseValues.size()*0.95)];

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
}


void ProgMonogenicSignalRes::postProcessingLocalResolutions(MultidimArray<double> &FilteredMap,
		MultidimArray<double> &resolutionVol,
		std::vector<double> &list, double &cut_value, MultidimArray<int> &pMask)
{
	MultidimArray<double> resolutionVol_aux = FilteredMap;
	double last_res = list[(list.size()-1)];
	last_res = last_res - 0.001; //the value 0.001 is a tolerance

	double Nyquist;
	Nyquist = 2*sampling;

	// Count number of voxels with resolution
	size_t N=0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FilteredMap)
		if (DIRECT_MULTIDIM_ELEM(FilteredMap, n)>=last_res)
			++N;


	// Get all resolution values
	MultidimArray<double> resolutions(N);
	size_t N_iter=0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FilteredMap)
		if (DIRECT_MULTIDIM_ELEM(FilteredMap, n)>last_res)
			DIRECT_MULTIDIM_ELEM(resolutions,N_iter++)=DIRECT_MULTIDIM_ELEM(FilteredMap, n);

	// Sort value and get threshold
	std::sort(&A1D_ELEM(resolutions,0),&A1D_ELEM(resolutions,N));
	double filling_value = A1D_ELEM(resolutions, (int)(0.5*N)); //median value

	last_res = list[(list.size()-1)];

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FilteredMap)
	{
		if (DIRECT_MULTIDIM_ELEM(FilteredMap, n) < last_res)
		{
			DIRECT_MULTIDIM_ELEM(FilteredMap, n) = filling_value;
			DIRECT_MULTIDIM_ELEM(pMask,n) = 0;
		}
		else
			DIRECT_MULTIDIM_ELEM(pMask,n) = 1;
	}

	//#ifdef DEBUG_MASK
	Image<int> imgMask;
	imgMask = pMask;
	imgMask.write(fnOut+"/refinedMask.mrc");
	//#endif

	double sigma = 3;
	realGaussianFilter(FilteredMap, sigma);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FilteredMap)
	{
		if (DIRECT_MULTIDIM_ELEM(pMask, n) > 0)
		{
			double valFilt = DIRECT_MULTIDIM_ELEM(FilteredMap, n);
			double valRes  = DIRECT_MULTIDIM_ELEM(resolutionVol, n);

			if (valFilt>valRes)
				DIRECT_MULTIDIM_ELEM(FilteredMap, n) = valRes;

			if (valFilt<Nyquist)
				DIRECT_MULTIDIM_ELEM(FilteredMap, n) = valRes;
		}
	}

	Image<double> outputResolutionImage;
	outputResolutionImage() = FilteredMap;
	outputResolutionImage.write(fnOut+"/monoresResolutionChimera.mrc");

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FilteredMap)
	{
		if (DIRECT_MULTIDIM_ELEM(pMask, n) == 0)
			DIRECT_MULTIDIM_ELEM(FilteredMap, n) = 0;
	}

	outputResolutionImage() = FilteredMap;//pOutputResolution;//resolutionFiltered;
	outputResolutionImage.write(fnOut+"/monoresResolutionMap.mrc");
}


void ProgMonogenicSignalRes::run()
{
	produceSideInfo();

	Image<double> outputResolution;
	outputResolution().resizeNoCopy(VRiesz);

	MultidimArray<int> &pMask = mask(), &pMaskExcl = maskExcl();
	MultidimArray<double> &pOutputResolution = outputResolution();
	MultidimArray<double> amplitudeMS, amplitudeMN;

	double criticalZ=icdf_gauss(significance);
	double criticalW=-1;
	double resolution, resolution_2, last_resolution = maxRes;  //A huge value for achieving
	double meanS, sdS2, meanN, sdN2, thr95;
	double freq, freqH, freqL;
	double max_meanS = -1e38, cut_value = 0.025;
	double mean_Signal, mean_noise, thresholdFirstEstimation;

	bool doNextIteration=true, lefttrimming = false;

	int fourier_idx, last_fourier_idx = -1;

	//Defining the resolution range:
	minRes = 2*sampling;
	DIGFREQ2FFT_IDX((maxRes+3)/sampling, ZSIZE(VRiesz), fourier_idx);
	FFT_IDX2DIGFREQ(fourier_idx, ZSIZE(VRiesz), freq);
	FFT_IDX2DIGFREQ(fourier_idx + 2, ZSIZE(VRiesz), freqH);
	FFT_IDX2DIGFREQ(fourier_idx - 2, ZSIZE(VRiesz), freqL);

	int count_res = 0;
	FileName fnDebug;

	//TODO: Set as advanced option
	if (noiseOnlyInHalves == false)
		refiningMask(fftV, iu, 2, pMask);


	amplitudeMS.resizeNoCopy(pOutputResolution);

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
						doNextIteration);

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

		freqL = freq + 0.02;
		freqH = freq - 0.02;
		if (freqL>=0.5)
			freqL = 0.5;
		if (freqH<=0.0)
			freqH = 0.0;

//		std::cout << resolution << " " << sampling/freqL << " " << sampling/freq << " " << sampling/freqH << std::endl;

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
							amplitudeMN, iter, fnDebug);
		}

		double sumS=0, sumS2=0, sumN=0, sumN2=0, NN = 0, NS = 0;
		noiseValues.clear();

		if (halfMapsGiven)
		{
			mono.statisticsInBinaryMask2(amplitudeMS, amplitudeMN,
										pMask, meanS, sdS2, meanN, sdN2, significance, thr95, NS, NN);
		}
		else
		{
			mono.statisticsInOutBinaryMask2(amplitudeMS,
										pMask, meanS, sdS2, meanN, sdN2, significance, thr95, NS, NN);
		}
		
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
			if (gaussian)
				thresholdNoise = meanN+criticalZ*sqrt(sdN2);
			else
				thresholdNoise = thr95;


			if (meanS>max_meanS)
				max_meanS = meanS;

			if (meanS<0.001*max_meanS){
				std::cout << "Search of resolutions stopped due to too low signal" << std::endl;
				break;}

			//TODO noise in half
			if (halfMapsGiven)
			{
				mono.setLocalResolutionHalfMaps(amplitudeMS, pMask, pOutputResolution,
						 thresholdNoise,  resolution, resolution_2);
			}
			else
			{
				mono.setLocalResolutionMap(amplitudeMS, pMask, pOutputResolution,
											 thresholdNoise,  resolution, resolution_2);
			}
			//}

			// Is the mean inside the signal significantly different from the noise?
			double z=(meanS-meanN)/sqrt(sdS2/NS+sdN2/NN);

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
	}
	amplitudeMN.clear();
	amplitudeMS.clear();

	MultidimArray<double> FilteredResolution = pOutputResolution;
	postProcessingLocalResolutions(FilteredResolution, pOutputResolution, list, cut_value, pMask);;
}
