/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
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

#include "tomo_confidence_map.h"
#include <core/bilib/kernel.h>
#include "core/linear_system_helper.h"
#include <numeric>
//#define DEBUG
//#define DEBUG_MASK
//#define TEST_FRINGES



void ProgTomoConfidecenceMap::readParams()
{
	fnVol = getParam("--odd");
	fnVol2 = getParam("--even");
	fnOut = getParam("--odir");
	fnMask = getParam("--mask");
	medianFilterBool = checkParam("--medianFilter");
	applySmoothingBeforeConfidence = checkParam("--applySmoothingBeforeConfidence");
	applySmoothingAfterConfidence = checkParam("--applySmoothingAfterConfidence");
	sigmaGauss = (float)  getDoubleParam("--sigmaGauss");
	locRes = checkParam("--localResolution");
	lowRes = (float)  getDoubleParam("--lowRes");
	highRes = (float) getDoubleParam("--highRes");
	sigVal = (float)  getDoubleParam("--significance");
	sampling = (float) getDoubleParam("--sampling_rate");
	fdr = (float) getDoubleParam("--fdr");
	step = (float) getDoubleParam("--step");
	nthrs = getIntParam("--threads");
}


void ProgTomoConfidecenceMap::defineParams()
{
	addUsageLine("This function determines the local resolution of a tomogram. It makes use of two reconstructions, odd and even. The difference between them"
			"gives a noise reconstruction. Thus, by computing the local amplitude of the signal at different frequencies and establishing a comparison with"
			"the noise, the local resolution is computed");
	addParamsLine("  --odd <vol_file=\"\">              : Half volume 1");
	addParamsLine("  --even <vol_file=\"\">	            : Half volume 2");
	addParamsLine("  [--mask <vol_file=\"\">]           : Mask defining the signal. ");
	addParamsLine("  [--medianFilter]                   : Set true if a median filter should be applied.");
	addParamsLine("  [--applySmoothingBeforeConfidence] : Set true if a median filter should be applied.");
	addParamsLine("  [--applySmoothingAfterConfidence]  : Set true if a median filter should be applied.");
	addParamsLine("  [--sigmaGauss <s=2>]               : Set true if a median filter should be applied.");
	addParamsLine("  [--fdr <s=0.05>]                   : False discovery rate");
	addParamsLine("  [--localResolution]                : Put this fag to estimate the local resolution of the tomogram");
	addParamsLine("  [--lowRes <s=30>]                  : Minimum resolution (A)");
	addParamsLine("  [--highRes <s=1>]                  : Maximum resolution (A)");
	addParamsLine("  [--step <s=1>]                     : Step");
	addParamsLine("  [--significance <s=0.95>]          : The level of confidence for the hypothesis test.");
	addParamsLine("  --odir <output=\"resmap.mrc\">     : Local resolution volume (in Angstroms)");
	addParamsLine("  [--sampling_rate <s=1>]            : Sampling rate (A/px)");
	addParamsLine("  [--threads <s=4>]                  : Number of threads");
}

void ProgTomoConfidecenceMap::run()
{
	MultidimArray<float> significanceMap;

	readAndPrepareData();

	if (locRes)
	{
		estimateLocalResolution(significanceMap);
	}

	exit(0);

	bool normalize = true;
	
	confidenceMap(significanceMap, normalize, fullMap, noiseMap);

	MultidimArray<float> significanceMapFiltered;
	significanceMapFiltered = significanceMap;

	if (medianFilterBool)
	{
		std::cout << "applying a median filter" << std::endl;
		medianFilter(significanceMap, significanceMapFiltered);
	}
		

	Image<float> significanceImg;
	significanceImg() = significanceMapFiltered;
	// significanceImg() = significanceMap;
	significanceImg.write(fnOut);
}

void ProgTomoConfidecenceMap::estimateLocalResolution(MultidimArray<float> &significanceMap)
{
	std::cout << "Estimating local resolution ... " << std::endl;

	float nyquist = 2*sampling;
	float resolution = lowRes;
	resMap.resizeNoCopy(fullMap);
	resMap.initConstant(lowRes);

	MultidimArray<double> fm, nm;

	MultidimArray<int> mask;
	mask.initZeros(1, Zdim, Ydim, Xdim);
	mask.initConstant(1);

	bool normalize = true;
	size_t iter = 0;

	float freq, tail, lastResolution;
	lastResolution = 1e38;

	// idx = (freq/freqnyquist)*Ndim;  =>   idx = (sampling/res)/(sampling/niquist)*Ndim = (niquist/res)*Ndim
	size_t idx = (nyquist/lowRes)*ZSIZE(fullMap);

	//while (resolution>nyquist)
	for (size_t k = idx; k<ZSIZE(fullMap); k++)
	{
		frequencyToAnalyze(freq, tail, k);

		resolution = sampling/freq;
		if (lastResolution - resolution < step)
		{
			continue;
		}
		else
		{
			lastResolution = resolution;
		}
		std::cout << "resolution = " << sampling/freq << "  tail = " << sampling/tail << std::endl;

		filterNoiseAndMap(freq, tail, fm, nm, iter);

		MultidimArray<float> fm_float, nm_float;
		fm_float.resizeNoCopy(fullMap);
		nm_float.resizeNoCopy(fullMap);

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fm_float)
		{
			DIRECT_MULTIDIM_ELEM(fm_float, n) = (float) DIRECT_MULTIDIM_ELEM(fm, n);
			DIRECT_MULTIDIM_ELEM(nm_float, n) = (float) DIRECT_MULTIDIM_ELEM(nm, n);
		}

		confidenceMap(significanceMap, normalize, fm_float, nm_float);

		Image<float> saveImg;
		saveImg() = significanceMap;
		FileName fn = formatString("confidence_%i.mrc", iter);
		saveImg.write(fn);

	std::cout << "filtering ended " << std::endl;
		//float auxRes = sampling/freq;
		//updateResMap(resMap, significanceMap, mask, auxRes, iter);

		iter += 1;
	}

	Image<float> resmapImg;
	resmapImg() = resMap;
	resmapImg.write("resmap.mrc");

}


void ProgTomoConfidecenceMap::frequencyToAnalyze(float &freq, float &tail, int idx)
{
	freq = (float) idx/(2*ZSIZE(fullMap));
	
	//if idx > (ZSIZE(fullMap)-10)

	//TODO: check tail range
	tail = ((float) (idx - 3))/(2*ZSIZE(fullMap));
}


void ProgTomoConfidecenceMap::filterNoiseAndMap(float &freq, float &tail, MultidimArray<double> &fm, MultidimArray<double> &nm, size_t iter)
{
	// Image<double> saveImg;
	// Image<float> saveImg_float;
	// saveImg_float() = fullMap;
	// FileName fn2 = formatString("ffm_%i.mrc", iter);
	// saveImg_float.write(fn2);

	FourierTransformer transformer;
	transformer.setThreadsNumber(nthrs);
	
	fm.resizeNoCopy(fullMap);
	nm.resizeNoCopy(fullMap);
	
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fullMap)
	{
			DIRECT_MULTIDIM_ELEM(fm, n) = (double) DIRECT_MULTIDIM_ELEM(fullMap, n);
			DIRECT_MULTIDIM_ELEM(nm, n) = (double) DIRECT_MULTIDIM_ELEM(noiseMap, n);
	}	

	MultidimArray< std::complex<double> > fftfullmap, fftnoisemap;
	transformer.FourierTransform(fm, fftfullmap);
	transformer.FourierTransform(nm, fftnoisemap);

	//Computing the fft
	size_t ZdimFT = ZSIZE(fullMap);
	size_t YdimFT = YSIZE(fullMap);
	size_t XdimFT = XSIZE(fullMap)/2 + 1;


	double u;

	freq_fourier_z.initZeros(ZdimFT);
	freq_fourier_x.initZeros(XdimFT);
	freq_fourier_y.initZeros(YdimFT);

	VEC_ELEM(freq_fourier_z,0) = 1e-38;
	for(size_t k=0; k<ZdimFT; ++k)
	{
		FFT_IDX2DIGFREQ(k,Zdim, u);
		VEC_ELEM(freq_fourier_z,k) = (float) u;
	}

	VEC_ELEM(freq_fourier_y,0) = 1e-38;
	for(size_t k=0; k<YdimFT; ++k)
	{
		FFT_IDX2DIGFREQ(k,Ydim, u);
		VEC_ELEM(freq_fourier_y,k) = (float) u;
	}

	VEC_ELEM(freq_fourier_x,0) = 1e-38;
	for(size_t k=0; k<XdimFT; ++k)
	{
		FFT_IDX2DIGFREQ(k,Xdim, u);
		VEC_ELEM(freq_fourier_x,k) = (float) u;
	}


	// Filtering in Fourier Space
	float uz, uy, ux, uz2, u2, uz2y2;
	float idelta =PI/(freq-tail);
	// std::complex<float> J(0,1);
	float un;
	long n=0;
	for(size_t k=0; k<ZdimFT; ++k)
	{
		uz = VEC_ELEM(freq_fourier_z, k);
		uz2=uz*uz;

		for(size_t i=0; i<YdimFT; ++i)
		{
			uy = VEC_ELEM(freq_fourier_y, i);
			uz2y2=uz2+uy*uy;

			for(size_t j=0; j<XdimFT; ++j)
			{
				ux = VEC_ELEM(freq_fourier_x, j);
				un=sqrtf(uz2y2+ux*ux);
				if (un<=tail)
				{
					DIRECT_MULTIDIM_ELEM(fftfullmap, n) = 0;
					DIRECT_MULTIDIM_ELEM(fftnoisemap, n) = 0;
				}
				else
				{
					if (un<=freq)
					{
						double H;
						H = 0.5*(1+cosf((un-freq)*idelta));
						DIRECT_MULTIDIM_ELEM(fftfullmap, n) *= H;
						DIRECT_MULTIDIM_ELEM(fftnoisemap, n) *= H;
					}
				}
				++n;
			}
		}
	}

	transformer.inverseFourierTransform(fftfullmap, fm);
	transformer.inverseFourierTransform(fftnoisemap, nm);

	Image<double> saveImg;
	saveImg() = fm;
	FileName fn = formatString("fm_%i.mrc", iter);
	saveImg.write(fn);

	std::cout << "filtering ended " << std::endl;
}

void ProgTomoConfidecenceMap::updateResMap(MultidimArray<float> &resMap, MultidimArray<float> &significanceMap, MultidimArray<int> &mask, float &resolution, size_t iter)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resMap)
	{
		if (DIRECT_MULTIDIM_ELEM(mask, n)>=1)
		{
			if (DIRECT_MULTIDIM_ELEM(significanceMap, n)<sigVal)
			{
				DIRECT_MULTIDIM_ELEM(resMap, n) = resolution;
				DIRECT_MULTIDIM_ELEM(mask, n) = 1;
			}
			else
			{
				DIRECT_MULTIDIM_ELEM(mask, n) += 1;
				if (DIRECT_MULTIDIM_ELEM(mask, n) >2)
				{
					DIRECT_MULTIDIM_ELEM(mask, n) = -1;
//					DIRECT_A3D_ELEM(pOutputResolution,  k,i,j) = resolution_2;
				}
			}
		}
	}	
	Image<int> saveImg;
	saveImg() = mask;
	FileName fn = formatString("fmask_%i.mrc", iter);
	saveImg.write(fn);

	std::cout << "filtering ended " << std::endl;
}

// void ProgTomoConfidecenceMap::LPF(MultidimArray<float> &significanceMap, bool normalize, MultidimArray<float> &fullMap, MultidimArray<float> &noiseMap)
// {

// }

void ProgTomoConfidecenceMap::confidenceMap(MultidimArray<float> &significanceMap, bool normalize, MultidimArray<float> &fullMap, MultidimArray<float> &noiseMap)
{
	MultidimArray<float> noiseVarianceMap, noiseMeanMap;
	Matrix2D<float> thresholdMatrix_mean, thresholdMatrix_std;

	int boxsize = 50;

	MultidimArray<double> fullMap_double;
	MultidimArray<double> noiseMap_double;

	if (applySmoothingBeforeConfidence)
	{
		std::cout << "applying a gaussian bluring before confidence" << std::endl;

		//convertToDouble(fullMap, fullMap_double);
		convertToDouble(noiseMap, noiseMap_double);

		//realGaussianFilter(fullMap_double, sigmaGauss);

		// Image<double> significanceImgs;
		// significanceImgs() = fullMap_double;
		// // significanceImg() = significanceMap;
		
		// significanceImgs.write("fullmap.mrc");
		
		realGaussianFilter(noiseMap_double, sigmaGauss);

		//convertToFloat(fullMap_double, fullMap);
		convertToFloat(noiseMap_double, noiseMap);
	}
		


	//TODO: estimateNoiseStatistics and normalizeTomogram in the same step
	estimateNoiseStatistics(noiseMap, noiseVarianceMap, noiseMeanMap, boxsize, thresholdMatrix_mean, thresholdMatrix_std);

	normalizeTomogram(fullMap, noiseVarianceMap, noiseMeanMap);

	//MultidimArray<float> significanceMap;
	significanceMap.initZeros(fullMap);

	if (applySmoothingAfterConfidence)
	{
		std::cout << "applying a gaussian bluring after confidence" << std::endl;

		convertToDouble(fullMap, fullMap_double);

		realGaussianFilter(fullMap_double, sigmaGauss);

		convertToFloat(fullMap_double, fullMap);
	}

	computeSignificanceMap(fullMap, significanceMap, thresholdMatrix_mean, thresholdMatrix_std);



	// FDRcontrol(significanceMap);
}

void ProgTomoConfidecenceMap::convertToDouble(MultidimArray<float> &inTomo,
												MultidimArray<double> &outTomo)
{
	outTomo.resizeNoCopy(inTomo);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(inTomo)
	{
		DIRECT_MULTIDIM_ELEM(outTomo, n) = (double) DIRECT_MULTIDIM_ELEM(inTomo, n);
	}
}

void ProgTomoConfidecenceMap::convertToFloat(MultidimArray<double> &inTomo,
												MultidimArray<float> &outTomo)
{
	outTomo.resizeNoCopy(inTomo);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(inTomo)
	{
		DIRECT_MULTIDIM_ELEM(outTomo, n) = (float) DIRECT_MULTIDIM_ELEM(inTomo, n);
	}
}

void ProgTomoConfidecenceMap::readAndPrepareData()
{
	std::cout << "Starting ..." << std::endl;
	std::cout << "           " << std::endl;
	std::cout << "           " << std::endl;

	Image<float> V;

	Image<float> oddMap, evenMap;
	oddMap.read(fnVol);
	evenMap.read(fnVol2);
	auto &odd = oddMap();
	auto &even = evenMap();
	fullMap = 0.5*(odd+even);
	noiseMap = 0.5*(odd-even);

	size_t Ndim;
	fullMap.getDimensions(Xdim, Ydim, Zdim, Ndim);

}

void ProgTomoConfidecenceMap::medianFilter(MultidimArray<float> &input_tomogram,
									       MultidimArray<float> &output_tomogram)
{
	std::vector<float> sortedVector;

	for (int k=1; k<(Zdim-1); ++k)
	{
    	for (int i=1; i<(Ydim-1); ++i)
		{
            for (int j=1; j<(Xdim-1); ++j)
			{
				std::vector<float> sortedVector;
				sortedVector.push_back(DIRECT_A3D_ELEM(input_tomogram, k, i, j));
				sortedVector.push_back(DIRECT_A3D_ELEM(input_tomogram, k+1, i, j));
				sortedVector.push_back(DIRECT_A3D_ELEM(input_tomogram, k-1, i, j));
				sortedVector.push_back(DIRECT_A3D_ELEM(input_tomogram, k, i+1, j));
				sortedVector.push_back(DIRECT_A3D_ELEM(input_tomogram, k, i-1, j));
				sortedVector.push_back(DIRECT_A3D_ELEM(input_tomogram, k, i, j+1));
				sortedVector.push_back(DIRECT_A3D_ELEM(input_tomogram, k, i, j-1));

				std::sort(sortedVector.begin(),sortedVector.end());

				DIRECT_A3D_ELEM(output_tomogram, k, i, j) = sortedVector[3];
				sortedVector.clear();
			}
		}
	}
}


void ProgTomoConfidecenceMap::normalizeTomogram(MultidimArray<float> &fullMap, MultidimArray<float> &noiseVarianceMap, MultidimArray<float> &noiseMeanMap)
{
	std::cout << "normalize Tomogram ...." <<  std::endl;
	//calculate the test statistic
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fullMap)
	{
		DIRECT_MULTIDIM_ELEM(fullMap, n) -= DIRECT_MULTIDIM_ELEM(noiseMeanMap, n);
		DIRECT_MULTIDIM_ELEM(fullMap, n) /= sqrtf(DIRECT_MULTIDIM_ELEM(noiseVarianceMap, n));
	}
}

void ProgTomoConfidecenceMap::estimateNoiseStatistics(MultidimArray<float> &noiseMap, 
													 MultidimArray<float> &noiseVarianceMap, MultidimArray<float> &noiseMeanMap,
													 int boxsize, Matrix2D<float> &thresholdMatrix_mean, Matrix2D<float> &thresholdMatrix_std)
{
	std::cout << "Estimating noise statistics ...." <<  std::endl;
	noiseVarianceMap.initZeros(noiseMap);
	noiseMeanMap.initZeros(noiseMap);

//	std::cout << "Analyzing local noise" << std::endl;

	int Nx = Xdim/boxsize;
	int Ny = Ydim/boxsize;

	Matrix2D<double> noiseStdMatrix, noiseMeanMatrix;
	noiseStdMatrix.initZeros(Ny, Nx);
	noiseMeanMatrix = noiseStdMatrix;

	// For the spline regression. A minimum of 8 points are considered
	int lX=std::min(8,Nx-2), lY=std::min(8,Ny-2);
    WeightedLeastSquaresHelper helperStd, helperMean;
    helperStd.A.initZeros(Nx*Ny,lX*lY);
    helperStd.b.initZeros(Nx*Ny);
    helperStd.w.initZeros(Nx*Ny);
    helperStd.w.initConstant(1);
	helperMean = helperStd;


    double hX = Xdim / (double)(lX-3);
    double hY = Ydim / (double)(lY-3);

	if ( (Xdim<boxsize) || (Ydim<boxsize) )
		std::cout << "Error: The tomogram in x-direction or y-direction is too small" << std::endl;

	std::vector<float> noiseVector(1);
	// std::vector<double> x,y,t;

	int xLimit, yLimit, xStart, yStart;
	int startX, startY, startZ, finishX, finishY, finishZ;
	startX = 0;
	startY = 0;
	startZ = 0;
	finishX = Xdim;
	finishY = Ydim;
	finishZ = Zdim;

	long N;
    int idxBox=0;

	for (int X_boxIdx=0; X_boxIdx<Nx; ++X_boxIdx)
	{
		if (X_boxIdx==Nx-1)
		{
			xStart = startX + X_boxIdx*boxsize;
			xLimit = finishX;
		}
		else
		{
			xStart = startX + X_boxIdx*boxsize;
			xLimit = startX + (X_boxIdx+1)*boxsize;
		}

		for (int Y_boxIdx=0; Y_boxIdx<Ny; ++Y_boxIdx)
		{
			if (Y_boxIdx==Ny-1)
			{
				yStart = startY + Y_boxIdx*boxsize;
				yLimit =  finishY;
			}
			else
			{
				yStart = startY + Y_boxIdx*boxsize;
				yLimit = startY + (Y_boxIdx+1)*boxsize;
			}

			N = 0;
			long n = 0;
			float sum2=0, sum=0;

			for (int k = startZ; k<finishZ; k++)
			{
				for (int i = yStart; i<yLimit; i++)
				{
					for (int j = xStart; j<xLimit; j++)
					{
						if (n%257 == 0)
						{
							float aux = A3D_ELEM(noiseMap, k, i, j);
							sum += aux;
							sum2 += aux*aux;
							noiseVector.push_back( aux );
							N++;
						} //we take one voxel each 257 (prime number) points to reduce noise data
						n++;
					}
				}
			}
			double std2, meanValue;
			meanValue = (double ) noiseVector[size_t(noiseVector.size()*0.5)];
			std2 = sqrt( (double) (sum2/N - (sum/N)*(sum/N)));

			std::sort(noiseVector.begin(),noiseVector.end());
			MAT_ELEM(noiseMeanMatrix, Y_boxIdx, X_boxIdx) = meanValue;
			MAT_ELEM(noiseStdMatrix, Y_boxIdx, X_boxIdx) = std2;

			double tileCenterY=0.5*(yLimit+yStart)-startY; // Translated to physical coordinates
			double tileCenterX=0.5*(xLimit+xStart)-startX;
			// Construction of the spline equation system
			long idxSpline=0;
			for(int controlIdxY = -1; controlIdxY < (lY - 1); ++controlIdxY)
			{
				double tmpY = Bspline03((tileCenterY / hY) - controlIdxY);

				VEC_ELEM(helperMean.b,idxBox)=MAT_ELEM(noiseMeanMatrix, Y_boxIdx, X_boxIdx);
				VEC_ELEM(helperStd.b,idxBox)=MAT_ELEM(noiseStdMatrix, Y_boxIdx, X_boxIdx);

				if (tmpY == 0.0)
				{
					idxSpline+=lX;
					continue;
				}

				for(int controlIdxX = -1; controlIdxX < (lX - 1); ++controlIdxX)
				{
					double tmpX = Bspline03((tileCenterX / hX) - controlIdxX);
					MAT_ELEM(helperStd.A, idxBox,idxSpline) = tmpY * tmpX;
					MAT_ELEM(helperMean.A, idxBox,idxSpline) = tmpY * tmpX;
					idxSpline+=1;
				}
			}
			// x.push_back(tileCenterX);
			// y.push_back(tileCenterY);
			// t.push_back(MAT_ELEM(noiseMeanMatrix, Y_boxIdx, X_boxIdx));
			// t.push_back(MAT_ELEM(noiseStdMatrix, Y_boxIdx, X_boxIdx));
			noiseVector.clear();
			idxBox+=1;
		}
	}


	// Spline coefficients
	Matrix1D<double> cij_std, cij_mean;
	weightedLeastSquares(helperStd, cij_std);
	weightedLeastSquares(helperMean, cij_mean);

	thresholdMatrix_mean.initZeros(Ydim, Xdim);
	thresholdMatrix_std.initZeros(Ydim, Xdim);

	for (int i=0; i<Ydim; ++i)
	{
		for (int j=0; j<Xdim; ++j)
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

				double xContrib_mean=0.0;
				double xContrib_std=0.0;
				for(int controlIdxX = -1; controlIdxX < (lX - 1); ++controlIdxX)
				{
					double tmpX = Bspline03((j / hX) - controlIdxX);
					xContrib_mean += VEC_ELEM(cij_mean, idxSpline) * tmpX;// *tmpY;
					xContrib_std += VEC_ELEM(cij_std, idxSpline) * tmpX;// *tmpY;
					idxSpline+=1;
				}
				MAT_ELEM(thresholdMatrix_mean, i,j)+= (float) xContrib_mean*tmpY;
				MAT_ELEM(thresholdMatrix_std, i,j)+= (float) xContrib_std*tmpY;
			}
		}
	}
}



void ProgTomoConfidecenceMap::FDRcontrol(MultidimArray<float> &significanceMap)
{
	int nelems = Xdim*Ydim;

	long n=0;
	for (int k=0; k<Zdim; ++k)
	{
		std::vector<float> pValVector;
		for (int i=0; i<Ydim; ++i)
		{
			for (int j=0; j<Xdim; ++j)
			{
				pValVector.push_back( DIRECT_MULTIDIM_ELEM(significanceMap, n) );
				++n;
			}
		}

		// float prevPVal = 1.0;
		//TODO: check performance indexsort

		long nn = 0;
		for (auto idx: sort_indexes(pValVector)) 
		{
			if (  pValVector[idx] <= fdr *(nn+1)/nelems)
			{
				
				DIRECT_MULTIDIM_ELEM(significanceMap, k + idx) = 0;
			}
			++nn;
		}

		pValVector.clear();
	}
}


template <typename T>
std::vector<size_t> ProgTomoConfidecenceMap::sort_indexes(const std::vector<T> &v)
{
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}



void ProgTomoConfidecenceMap::computeSignificanceMap(MultidimArray<float> &fullMap, MultidimArray<float> &significanceMap,
													 Matrix2D<float> &thresholdMatrix_mean, Matrix2D<float> &thresholdMatrix_std)
{
	long n = 0;
	for (int k=0; k<Zdim; ++k)
	{
		for (int i=0; i<Ydim; ++i)
		{
			for (int j=0; j<Xdim; ++j)
			{
				float fm = DIRECT_MULTIDIM_ELEM(fullMap, n);
				
				float x = (fm - MAT_ELEM(thresholdMatrix_mean, i, j))/MAT_ELEM(thresholdMatrix_std, i, j);

				DIRECT_MULTIDIM_ELEM(significanceMap, n) = 0.5 * (1. + erf(x/sqrt(2.)));
				n++;
			}
		}
	}
}
