/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
 *             Oier Lauzirika                         olauzirika@cnb.csic.es
 * 			   Carlos Oscar S. Sorzano                coss@cnb.csic.es (2019)
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

#include "resolution_monotomo.h"

#include <core/bilib/kernel.h>
#include "core/linear_system_helper.h"
#include "data/fftwT.h"
#include <cmath>

#include <data/aft.h>
#include <data/fft_settings.h>
#include <cmath>

void ProgMonoTomo::readParams()
{
	fnVol = getParam("--vol");
	fnVol2 = getParam("--vol2");
	fnMeanVol = getParam("--meanVol");
	fnOut = getParam("-o");
	sampling = getDoubleParam("--sampling_rate");
	minRes = getDoubleParam("--minRes");
	maxRes = getDoubleParam("--maxRes");
	resStep = getDoubleParam("--step");
	significance = getDoubleParam("--significance");
	nthrs = getIntParam("--threads");
}


void ProgMonoTomo::defineParams()
{
	addUsageLine("This function determines the local resolution of a tomogram. It makes use of two reconstructions, odd and even. The difference between them"
			"gives a noise reconstruction. Thus, by computing the local amplitude of the signal at different frequencies and establishing a comparison with"
			"the noise, the local resolution is computed");
	addParamsLine("  --vol <vol_file=\"\">   			: Half volume 1");
	addParamsLine("  --vol2 <vol_file=\"\">				: Half volume 2");
	addParamsLine("  -o <output=\"MGresolution.vol\">	: Local resolution volume (in Angstroms)");
	addParamsLine("  --meanVol <vol_file=\"\">			: Mean volume of half1 and half2 (only it is necessary the two haves are used)");
	addParamsLine("  [--sampling_rate <s=1>]   			: Sampling rate (A/px)");
	addParamsLine("  [--step <s=0.25>]       			: The resolution is computed at a number of frequencies between minimum and");
	addParamsLine("                            			: maximum resolution px/A. This parameter determines that number");
	addParamsLine("  [--minRes <s=30>]         			: Minimum resolution (A)");
	addParamsLine("  [--maxRes <s=1>]         			: Maximum resolution (A)");
	addParamsLine("  [--significance <s=0.95>]       	: The level of confidence for the hypothesis test.");
	addParamsLine("  [--threads <s=4>]               	: Number of threads");
}


void ProgMonoTomo::produceSideInfo()
{
	std::cout << "Starting..." << std::endl;
	std::cout << "           " << std::endl;
	std::cout << "IMPORTANT: If the angular step of the tilt series is higher than 3 degrees"<< std::endl;
	std::cout << "           then, the tomogram is not properly for MonoTomo. Despite this is not "<< std::endl;
	std::cout << "           optimal, MonoTomo will try to compute the local resolution." << std::endl;
	std::cout << "           " << std::endl;

	Image<float> signalTomo;
	Image<float> noiseTomo;
	signalTomo.read(fnVol);
	noiseTomo.read(fnVol2);

	auto &tomo = signalTomo();
	auto &noise = noiseTomo();

	// Compute the average and difference
	tomo += noise;
	tomo *= 0.5;
	noise -= tomo;

	tomo.setXmippOrigin();
	noise.setXmippOrigin();

	mask().resizeNoCopy(tomo);
	mask().initConstant(1);

	xdim = XSIZE(tomo);
	ydim = YSIZE(tomo);
	zdim = ZSIZE(tomo);


	smoothBorders(tomo, mask());
	smoothBorders(noise, mask());

	float normConst = xdim * ydim * zdim;

	const FFTSettings<float> fftSettingForward(xdim, ydim, zdim);
	const auto fftSettingBackward = fftSettingForward.createInverse();

	auto hw = CPU(nthrs);
	forward_transformer = std::make_unique<FFTwT<float>>();
	backward_transformer = std::make_unique<FFTwT<float>>();
	forward_transformer->init(hw, fftSettingForward);
	backward_transformer->init(hw, fftSettingBackward);

	const auto &fdim = fftSettingForward.fDim();
	fourierSignal.resize(fdim.size());
	fourierNoise.resize(fdim.size());
	forward_transformer->fft(MULTIDIM_ARRAY(tomo), fourierSignal.data());
	forward_transformer->fft(MULTIDIM_ARRAY(noise), fourierNoise.data());

	// Clear images in real space as they are not longer needed
	signalTomo.clear();
	noiseTomo.clear();

	xdimFT = fdim.x();
	ydimFT = fdim.y();
	zdimFT = fdim.z();


	VRiesz.resizeNoCopy(1, zdim, ydim, xdim);

}

void ProgMonoTomo::smoothBorders(MultidimArray<float> &vol, MultidimArray<int> &pMask)
{
	int N_smoothing = 10;

	int siz_z = zdim*0.5;
	int siz_y = ydim*0.5;
	int siz_x = xdim*0.5;


	int limit_distance_x = (siz_x-N_smoothing);
	int limit_distance_y = (siz_y-N_smoothing);
	int limit_distance_z = (siz_z-N_smoothing);

	long n=0;
	for(int k=0; k<zdim; ++k)
	{
		auto uz = (k - siz_z);
		for(int i=0; i<ydim; ++i)
		{
			auto uy = (i - siz_y);
			for(int j=0; j<xdim; ++j)
			{
				auto ux = (j - siz_x);

				if (abs(ux)>=limit_distance_x)
				{
					DIRECT_MULTIDIM_ELEM(vol, n) *= 0.5*(1+std::cos(PI*(limit_distance_x - std::abs(ux))/N_smoothing));
					DIRECT_MULTIDIM_ELEM(pMask, n) = 0;
				}
				if (abs(uy)>=limit_distance_y)
				{
					DIRECT_MULTIDIM_ELEM(vol, n) *= 0.5*(1+std::cos(PI*(limit_distance_y - std::abs(uy))/N_smoothing));
					DIRECT_MULTIDIM_ELEM(pMask, n) = 0;
				}
				if (abs(uz)>=limit_distance_z)
				{
					DIRECT_MULTIDIM_ELEM(vol, n) *= 0.5*(1+std::cos(PI*(limit_distance_z - std::abs(uz))/N_smoothing));
					DIRECT_MULTIDIM_ELEM(pMask, n) = 0;
				}
				++n;
			}
		}
	}
}


void ProgMonoTomo::amplitudeMonogenicSignal3D(const std::vector<std::complex<float>> &myfftV, float freq, float freqH,
												float freqL, MultidimArray<float> &amplitude, int count, FileName fnDebug)
{
	fftVRiesz.resize(fourierSignal.size());
	fftVRiesz_aux.resize(fourierSignal.size());
	std::complex<float> J(0,1);

	// Filter the input volume and add it to amplitude
	size_t n=0;
	float ideltal=PI/(freq-freqH);
	for(size_t k=0; k<zdimFT; ++k)
	{
		double uz;// = VEC_ELEM(freq_fourier_z, k);
		FFT_IDX2DIGFREQ(k, zdim, uz);
		auto uz2 = uz*uz;
		for(size_t i=0; i<ydimFT; ++i)
		{
			//const auto uy = VEC_ELEM(freq_fourier_y, i);
			double uy;
			FFT_IDX2DIGFREQ(i, ydim, uy);
			auto uy2uz2 = uz2 +uy*uy;
			for(size_t j=0; j<xdimFT; ++j)
			{
				double ux;
				FFT_IDX2DIGFREQ(j, xdim, ux);
				//const auto ux = VEC_ELEM(freq_fourier_x, j);
				const float u = std::sqrt(uy2uz2 + ux*ux);

				if (freqH<=u && u<=freq)
				{
					fftVRiesz[n] = myfftV[n]*0.5f*(1+std::cos((u-freq)*ideltal));
					fftVRiesz_aux[n] = -J*fftVRiesz[n]/u;
				}
				else
				{
					if (u>freq)
					{
						fftVRiesz[n] = myfftV[n];
						fftVRiesz_aux[n] = -J*fftVRiesz[n]/u;
					}
					else
					{
						fftVRiesz[n] = 0.0f;
						fftVRiesz_aux[n] = 0.0f;
					}
				}
				n++;

			}
		}
	}

	backward_transformer->ifft(fftVRiesz.data(), MULTIDIM_ARRAY(VRiesz));


	amplitude = VRiesz;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(VRiesz)
	{
		DIRECT_MULTIDIM_ELEM(amplitude, n) *= DIRECT_MULTIDIM_ELEM(VRiesz, n);
	}


	// Calculate first component of Riesz vector
	float uz, uy, ux;
	n=0;
	for(size_t k=0; k<zdimFT; ++k)
	{
		for(size_t i=0; i<ydimFT; ++i)
		{
			for(size_t j=0; j<xdimFT; ++j)
			{
				//ux = VEC_ELEM(freq_fourier_x, j);
				FFT_IDX2DIGFREQ(j, xdim, ux);
				fftVRiesz[n] = ux*fftVRiesz_aux[n];
				++n;
			}
		}
	}
	backward_transformer->ifft(fftVRiesz.data(), MULTIDIM_ARRAY(VRiesz));

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(VRiesz)
	{
		DIRECT_MULTIDIM_ELEM(amplitude, n) += DIRECT_MULTIDIM_ELEM(VRiesz, n)*DIRECT_MULTIDIM_ELEM(VRiesz, n);
	}

	// Calculate second and third components of Riesz vector
	n=0;
	for(size_t k=0; k<zdimFT; ++k)
	{
		//uz = VEC_ELEM(freq_fourier_z, k);
		// = VEC_ELEM(freq_fourier_z, k);
		FFT_IDX2DIGFREQ(k, zdim, uz);
		for(size_t i=0; i<ydimFT; ++i)
		{
			//uy = VEC_ELEM(freq_fourier_y, i);
			FFT_IDX2DIGFREQ(i, ydim, uy);
			for(size_t j=0; j<xdimFT; ++j)
			{
				fftVRiesz[n] = ux*fftVRiesz_aux[n];
				fftVRiesz_aux[n] = uz*fftVRiesz_aux[n];
				++n;
			}
		}
	}
	backward_transformer->ifft(fftVRiesz.data(), MULTIDIM_ARRAY(VRiesz));

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(VRiesz)
	{
		DIRECT_MULTIDIM_ELEM(amplitude, n) += DIRECT_MULTIDIM_ELEM(VRiesz, n)*DIRECT_MULTIDIM_ELEM(VRiesz, n);
	}


	backward_transformer->ifft(fftVRiesz_aux.data(), MULTIDIM_ARRAY(VRiesz));

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(VRiesz)
	{
		DIRECT_MULTIDIM_ELEM(amplitude, n) += DIRECT_MULTIDIM_ELEM(VRiesz, n)*DIRECT_MULTIDIM_ELEM(VRiesz, n);
		DIRECT_MULTIDIM_ELEM(amplitude, n) = std::sqrt(DIRECT_MULTIDIM_ELEM(amplitude, n));
	}

	// Low pass filter the monogenic amplitude
	forward_transformer->fft(MULTIDIM_ARRAY(amplitude), fftVRiesz.data());
	float raised_w = PI/(freqL-freq);

	n = 0;
	for(size_t k=0; k<zdimFT; ++k)
	{
		//uz = VEC_ELEM(freq_fourier_z, k);
		double uz;// = VEC_ELEM(freq_fourier_z, k);
		FFT_IDX2DIGFREQ(k, zdim, uz);
		auto uz2 = uz*uz;
		for(size_t i=0; i<ydimFT; ++i)
		{
			//uy = VEC_ELEM(freq_fourier_y, i);
			double uy;
			FFT_IDX2DIGFREQ(i, ydim, uy);
			auto uy2uz2 = uz2 +uy*uy;
			for(size_t j=0; j<xdimFT; ++j)
			{
				//ux = VEC_ELEM(freq_fourier_x, j);
				double ux;
				FFT_IDX2DIGFREQ(j, xdim, ux);
				auto u = std::sqrt(uy2uz2 + ux*ux);

				if (u>freqL)
				{
					fftVRiesz[n] = 0.0f;
				}
				else
				{
					if (freqL>=u && u>=freq)
					{
						fftVRiesz[n] *= 0.5f*(1 + std::cos(raised_w*(u-freq)));
					}
				}
				n++;

			}
		}
	}

	backward_transformer->ifft(fftVRiesz.data(), MULTIDIM_ARRAY(amplitude));
}


void ProgMonoTomo::localNoise(MultidimArray<float> &noiseMap, Matrix2D<double> &noiseMatrix, int boxsize, Matrix2D<double> &thresholdMatrix)
{
//	std::cout << "Analyzing local noise" << std::endl;

	int xdim = XSIZE(noiseMap);
	int ydim = YSIZE(noiseMap);

	int Nx = xdim/boxsize;
	int Ny = ydim/boxsize;



	// For the spline regression
	int lX=std::min(8,Nx-2), lY=std::min(8,Ny-2);
    WeightedLeastSquaresHelper helper;
    helper.A.initZeros(Nx*Ny,lX*lY);
    helper.b.initZeros(Nx*Ny);
    helper.w.initZeros(Nx*Ny);
    helper.w.initConstant(1);
    double hX = xdim / (double)(lX-3);
    double hY = ydim / (double)(lY-3);

	if ( (xdim<boxsize) || (ydim<boxsize) )
		std::cout << "Error: The tomogram in x-direction or y-direction is too small" << std::endl;

	std::vector<double> noiseVector(1);
	std::vector<double> x,y,t;

	int xLimit, yLimit, xStart, yStart;

	long counter;
    int idxBox=0;

	for (int X_boxIdx=0; X_boxIdx<Nx; ++X_boxIdx)
	{
		if (X_boxIdx==Nx-1)
		{
			xStart = STARTINGX(noiseMap) + X_boxIdx*boxsize;
			xLimit = FINISHINGX(noiseMap);
		}
		else
		{
			xStart = STARTINGX(noiseMap) + X_boxIdx*boxsize;
			xLimit = STARTINGX(noiseMap) + (X_boxIdx+1)*boxsize;
		}

		for (int Y_boxIdx=0; Y_boxIdx<Ny; ++Y_boxIdx)
		{
			if (Y_boxIdx==Ny-1)
			{
				yStart = STARTINGY(noiseMap) + Y_boxIdx*boxsize;
				yLimit =  FINISHINGY(noiseMap);
			}
			else
			{
				yStart = STARTINGY(noiseMap) + Y_boxIdx*boxsize;
				yLimit = STARTINGY(noiseMap) + (Y_boxIdx+1)*boxsize;
			}



			counter = 0;
			for (int i = yStart; i<yLimit; i++)
			{
				for (int j = xStart; j<xLimit; j++)
				{
					for (int k = STARTINGZ(noiseMap); k<FINISHINGZ(noiseMap); k++)
					{
						if (counter%257 == 0) //we take one voxel each 257 (prime number) points to reduce noise data
							noiseVector.push_back( A3D_ELEM(noiseMap, k, i, j) );
						++counter;
					}
				}
			}

			std::sort(noiseVector.begin(),noiseVector.end());
			noiseMatrix.initZeros(Ny, Nx);

			MAT_ELEM(noiseMatrix, Y_boxIdx, X_boxIdx) = noiseVector[size_t(noiseVector.size()*significance)];

			double tileCenterY=0.5*(yLimit+yStart)-STARTINGY(noiseMap); // Translated to physical coordinates
			double tileCenterX=0.5*(xLimit+xStart)-STARTINGX(noiseMap);
			// Construction of the spline equation system
			long idxSpline=0;
			for(int controlIdxY = -1; controlIdxY < (lY - 1); ++controlIdxY)
			{
				double tmpY = Bspline03((tileCenterY / hY) - controlIdxY);
				VEC_ELEM(helper.b,idxBox)=MAT_ELEM(noiseMatrix, Y_boxIdx, X_boxIdx);
				if (tmpY == 0.0)
				{
					idxSpline+=lX;
					continue;
				}

				for(int controlIdxX = -1; controlIdxX < (lX - 1); ++controlIdxX)
				{
					double tmpX = Bspline03((tileCenterX / hX) - controlIdxX);
					MAT_ELEM(helper.A,idxBox,idxSpline) = tmpY * tmpX;
					idxSpline+=1;
				}
			}
			x.push_back(tileCenterX);
			y.push_back(tileCenterY);
			t.push_back(MAT_ELEM(noiseMatrix, Y_boxIdx, X_boxIdx));
			noiseVector.clear();
			idxBox+=1;
		}
	}


	// Spline coefficients
	Matrix1D<double> cij;
	weightedLeastSquares(helper, cij);

	thresholdMatrix.initZeros(ydim, xdim);

	for (int i=0; i<ydim; ++i)
	{
		for (int j=0; j<xdim; ++j)
		{
			long idxSpline=0;

			for(int controlIdxY = -1; controlIdxY < (lY - 1); ++controlIdxY)
			{
				double tmpY = Bspline03((i / hY) - controlIdxY);

				if (tmpY == 0.0)
				{
					idxSpline+=lX;
					continue;
				}

				double xContrib=0.0;
				for(int controlIdxX = -1; controlIdxX < (lX - 1); ++controlIdxX)
				{
					double tmpX = Bspline03((j / hX) - controlIdxX);
					xContrib+=VEC_ELEM(cij,idxSpline) * tmpX;// *tmpY;
					idxSpline+=1;
				}
				MAT_ELEM(thresholdMatrix,i,j)+=xContrib*tmpY;
			}
		}
	}
}




void ProgMonoTomo::postProcessingLocalResolutions(MultidimArray<float> &resolutionVol,
		std::vector<float> &list)
{
	MultidimArray<float> resolutionVol_aux = resolutionVol;
	float init_res, last_res;

	init_res = list[0];
	last_res = list[(list.size()-1)];
	
	auto last_resolution_2 = list[last_res];

	double lowest_res;
	lowest_res = list[1]; //Example resolutions between 10-300, list(0)=300, list(1)=290, it is used list(1) due to background
	//is at 300 and the smoothing cast values of 299 and they must be removed.

	// Count number of voxels with resolution
	std::vector<float> resolVec(0);
	float rVol;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resolutionVol)
	{
		rVol = DIRECT_MULTIDIM_ELEM(resolutionVol, n);
		if ( (rVol>=(last_resolution_2-0.001)) && (rVol<=lowest_res) ) //the value 0.001 is a tolerance
		{
			resolVec.push_back(rVol);
		}
	}

	size_t N;
	N = resolVec.size();
	std::sort(resolVec.begin(), resolVec.end());

	std::cout << "median Resolution = " << resolVec[(int)(0.5*N)] << std::endl;
}



void ProgMonoTomo::lowestResolutionbyPercentile(MultidimArray<float> &resolutionVol,
		std::vector<float> &list, float &cut_value, float &resolutionThreshold)
{
	double last_resolution_2 = list[(list.size()-1)];

	double lowest_res;
	lowest_res = list[0];

	// Count number of voxels with resolution

	double rVol;
	std::vector<double> resolVec(0);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resolutionVol)
	{
		rVol = DIRECT_MULTIDIM_ELEM(resolutionVol, n);
		if (rVol>=(last_resolution_2-0.001))//&& (DIRECT_MULTIDIM_ELEM(resolutionVol, n)<lowest_res) ) //the value 0.001 is a tolerance
		{
			resolVec.push_back(rVol);
		}

	}
	size_t N;
	N = resolVec.size();

	std::sort(resolVec.begin(), resolVec.end());

	resolutionThreshold = resolVec[(int)((0.95)*N)];

	std::cout << "resolutionThreshold = " << resolutionThreshold <<  std::endl;
}


void ProgMonoTomo::gaussFilter(const MultidimArray<float> &vol, const float sigma, MultidimArray<float> &VRiesz)
{
	float isigma2 = (sigma*sigma);

	forward_transformer->fft(MULTIDIM_ARRAY(vol), fftVRiesz.data());
	size_t n=0;
	for(size_t k=0; k<zdimFT; ++k)
	{
		//const auto uz = VEC_ELEM(freq_fourier_z, k);
		double uz;
		FFT_IDX2DIGFREQ(k, zdim, uz);
		auto uz2 = uz*uz;

		for(size_t i=0; i<ydimFT; ++i)
		{
			//const auto uy = VEC_ELEM(freq_fourier_y, i);
			double uy;
			FFT_IDX2DIGFREQ(i, ydim, uy);
			double uy2uz2 = uz2 +uy*uy;
			for(size_t j=0; j<xdimFT; ++j)
			{
				//const auto ux = VEC_ELEM(freq_fourier_x, j);
				double ux;
				FFT_IDX2DIGFREQ(j, xdim, ux);
				const float u2 = (float) (uy2uz2 + ux*ux);

				fftVRiesz[n] *= std::exp(-PI*PI*u2*isigma2);

				n++;

			}
		}
	}

	backward_transformer->ifft(fftVRiesz.data(), MULTIDIM_ARRAY(VRiesz));
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(VRiesz)
	{
		DIRECT_MULTIDIM_ELEM(VRiesz, n) /= xdim*ydim*zdim;
	}
}

void ProgMonoTomo::run()
{
	produceSideInfo();

	MultidimArray<int> &pMask = mask();
	MultidimArray<float> outputResolution;

	outputResolution.initZeros(1, zdim, ydim, xdim);
	outputResolution.initConstant(maxRes);

	MultidimArray<float> amplitudeMS, amplitudeMN;

	float criticalZ = (float) icdf_gauss(significance);
	double criticalW=-1;
	double  resolution_2;
	float resolution, last_resolution = 10000;
	double freq, freqH, freqL;
	double max_meanS = -1e38;
	float cut_value = 0.025;
	int boxsize = 50;

	float Nyquist = 2*sampling;
	if (minRes<2*sampling)
		minRes = Nyquist;

	bool doNextIteration=true;

	bool lefttrimming = false;
	int last_fourier_idx = -1;

	int count_res = 0;
	FileName fnDebug;

	int iter=0;
	std::vector<float> list;

	std::cout << "Analyzing frequencies" << std::endl;
	std::cout << "                     " << std::endl;
	std::vector<float> noiseValues;
	float lastResolution = 1e38;

	size_t maxIdx = std::max(xdim, ydim);

	for (size_t idx = 0; idx<maxIdx; idx++)
	{
		float candidateResolution = maxRes - idx*resStep;

		if (candidateResolution<=minRes)
		{
			freq = 0.5;
		}
		else
		{
			freq = sampling/candidateResolution;
		}

		int fourier_idx;
		DIGFREQ2FFT_IDX(freq, zdim, fourier_idx);
		FFT_IDX2DIGFREQ(fourier_idx, zdim, freq);

		resolution = sampling/freq;

		if (lastResolution-resolution<resStep)
		{
			continue;
		}else
		{
			lastResolution = resolution;
		}

		freqL = sampling/(resolution + resStep);

		int fourier_idx_freqL;
		DIGFREQ2FFT_IDX(freqL, zdim, fourier_idx_freqL);

		if (fourier_idx_freqL == fourier_idx)
		{
			if (fourier_idx > 0){
				FFT_IDX2DIGFREQ(fourier_idx - 1, zdim, freqL);
			}
			else{
				freqL = sampling/(resolution + resStep);
			}
		}

		list.push_back(resolution);

		if (iter <2)
			resolution_2 = maxRes;
		else
			resolution_2 = list[iter - 2];

		freqL = freq + 0.01;
		freqH = freq - 0.01;

		fnDebug = formatString("Signal_%i.mrc", idx);
		amplitudeMonogenicSignal3D(fourierSignal, freq, freqH, freqL, amplitudeMS, idx, fnDebug);

		fnDebug = formatString("Noise_%i.mrc", idx);
		amplitudeMonogenicSignal3D(fourierNoise, freq, freqH, freqL, amplitudeMN, idx, fnDebug);

		Matrix2D<double> noiseMatrix;
		Matrix2D<double> thresholdMatrix;
		localNoise(amplitudeMN, noiseMatrix, boxsize, thresholdMatrix);


		float sumS=0, sumS2=0, sumN=0, sumN2=0, NN = 0, NS = 0;
		noiseValues.clear();

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitudeMS)
		{
			if (DIRECT_MULTIDIM_ELEM(pMask, n)>=1)
			{
				float amplitudeValue=DIRECT_MULTIDIM_ELEM(amplitudeMS, n);
				float amplitudeValueN=DIRECT_MULTIDIM_ELEM(amplitudeMN, n);
				sumS  += amplitudeValue;
				noiseValues.push_back(amplitudeValueN);
				sumN  += amplitudeValueN;
				++NS;
				++NN;
			}
		}

		if (NS == 0)
		{
			std::cout << "There are no points to compute inside the mask" << std::endl;
			std::cout << "If the number of computed frequencies is low, perhaps the provided"
					"mask is not enough tight to the volume, in that case please try another mask" << std::endl;
			break;
		}

		double meanS=sumS/NS;

		if (meanS>max_meanS)
			max_meanS = meanS;

		if (meanS<0.001*max_meanS)
		{
			std::cout << "Search of resolutions stopped due to too low signal" << std::endl;
			break;
		}

		std::cout << "resolution = " << resolution << "    resolution_2 = " << resolution_2 << std::endl;

		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(amplitudeMS)
		{
			if (DIRECT_A3D_ELEM(pMask, k,i,j)>=1)
			{
				if ( DIRECT_A3D_ELEM(amplitudeMS, k,i,j)>(float) MAT_ELEM(thresholdMatrix, i, j) )
				{

					DIRECT_A3D_ELEM(pMask,  k,i,j) = 1;
					DIRECT_A3D_ELEM(outputResolution, k,i,j) = resolution;
				}
				else{
					DIRECT_A3D_ELEM(pMask,  k,i,j) += 1;
					if (DIRECT_A3D_ELEM(pMask,  k,i,j) >2)
					{
						DIRECT_A3D_ELEM(pMask,  k,i,j) = -1;
						DIRECT_A3D_ELEM(outputResolution,  k,i,j) = resolution_2;
					}
				}
			}
		}
		iter++;
//		*/

	}

	Image<float> outputResolutionImage2;
	outputResolutionImage2() = outputResolution;
	outputResolutionImage2.write("local.mrc");

	amplitudeMN.clear();
	amplitudeMS.clear();

	//Convolution with a real gaussian to get a smooth map
	float sigma = 3.0f;
	gaussFilter(outputResolution, 3.0f, VRiesz);
	float resolutionThreshold;

	lowestResolutionbyPercentile(VRiesz, list, cut_value, resolutionThreshold);


	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(VRiesz)
	{
		auto resVal = DIRECT_MULTIDIM_ELEM(outputResolution, n);
		auto value = DIRECT_MULTIDIM_ELEM(VRiesz, n);
		if ( (value<resolutionThreshold) && (value>resVal) )
			value = resVal;
		if ( value<Nyquist)
			value = Nyquist;
		DIRECT_MULTIDIM_ELEM(VRiesz, n) = value;
	}

	gaussFilter(VRiesz, 3.0f, VRiesz);


	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(VRiesz)
	{
		auto resVal = DIRECT_MULTIDIM_ELEM(outputResolution, n);
		auto value = DIRECT_MULTIDIM_ELEM(VRiesz, n);
		if ( (value<resolutionThreshold) && (value>resVal) )
			value = resVal;
		if ( value<Nyquist)
			value = Nyquist;
		DIRECT_MULTIDIM_ELEM(VRiesz, n) = value;
	}
	Image<double> outputResolutionImage;
	MultidimArray<double> resolutionFiltered, resolutionChimera;

	postProcessingLocalResolutions(VRiesz, list);
	outputResolutionImage2() = VRiesz;
	outputResolutionImage2.write(fnOut);



}
