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
	fnOut = getParam("-o");
	fnMask = getParam("--mask");
	locRes = checkParam("--localResolution");
	lowRes = (float)  getDoubleParam("--lowRes");
	highRes = (float) getDoubleParam("--highRes");
	sigVal = (float)  getDoubleParam("--significance");
	sampling = getDoubleParam("--sampling_rate");
	fdr = (float) getDoubleParam("--fdr");
	step = (float) getDoubleParam("--step");
	nthrs = getIntParam("--threads");
}


void ProgTomoConfidecenceMap::defineParams()
{
	addUsageLine("This function determines the local resolution of a tomogram. It makes use of two reconstructions, odd and even. The difference between them"
			"gives a noise reconstruction. Thus, by computing the local amplitude of the signal at different frequencies and establishing a comparison with"
			"the noise, the local resolution is computed");
	addParamsLine("  --odd <vol_file=\"\">   			: Half volume 1");
	addParamsLine("  --even <vol_file=\"\">				: Half volume 2");
	addParamsLine("  [--mask <vol_file=\"\">]  			: Mask defining the signal. ");
	addParamsLine("  [--fdr <s=0.05>]         			: False discovery rate");
	addParamsLine("  [--localResolution]  			    : Put this fag to estimate the local resolution of the tomogram");
	addParamsLine("  [--lowRes <s=30>]         			: Minimum resolution (A)");
	addParamsLine("  [--highRes <s=1>]         			: Maximum resolution (A)");
	addParamsLine("  [--step <s=1>]         			: Step");
	addParamsLine("  [--significance <s=0.95>]       	: The level of confidence for the hypothesis test.");
	addParamsLine("  -o <output=\"MGresolution.vol\">	: Local resolution volume (in Angstroms)");
	addParamsLine("  [--sampling_rate <s=1>]   			: Sampling rate (A/px)");
	addParamsLine("  [--threads <s=4>]               	: Number of threads");
}

void ProgTomoConfidecenceMap::run()
{
	readAndPrepareData();

	MultidimArray<float> significanceMap;
	bool normalize = true;
	
	confidenceMap(significanceMap, normalize);

	Image<float> significanceImg;
	significanceImg() = significanceMap;
	significanceImg.write(fnOut);

}



void ProgTomoConfidecenceMap::confidenceMap(MultidimArray<float> &significanceMap, bool normalize)
{
	MultidimArray<float> noiseVarianceMap, noiseMeanMap;
	Matrix2D<float> thresholdMatrix_mean, thresholdMatrix_std;

	int boxsize = 50;

	//TODO: estimateNoiseStatistics and normalizeTomogram in the same step
	estimateNoiseStatistics(noiseMap, noiseVarianceMap, noiseMeanMap, boxsize, thresholdMatrix_mean, thresholdMatrix_std);

	normalizeTomogram(noiseMap, noiseVarianceMap, noiseMeanMap);

	//MultidimArray<float> significanceMap;
	significanceMap.initZeros(fullMap);

	computeSignificanceMap(fullMap, significanceMap, thresholdMatrix_mean, thresholdMatrix_std);

	//FDRcontrol(significanceMap);
}


void ProgTomoConfidecenceMap::readAndPrepareData()
{
	std::cout << "Starting ..." << std::endl;
	std::cout << "           " << std::endl;
	std::cout << "           " << std::endl;

	Image<float> fMap;
	fMap.read(fnVol);
	MultidimArray<float> oddMap, evenMap, fm;
	fm = fMap();
	
	size_t Ndim;
	fm.getDimensions(Xdim, Ydim, Zdim, Ndim);

	std::cout << " 1 ..." << Xdim << " " << Ydim << " " << Zdim << std::endl;
	
	oddMap.initZeros(1, Zdim, Ydim, Xdim);
	evenMap = oddMap;
	
	std::cout << " 2 ..." << XSIZE(evenMap)*YSIZE(evenMap)*ZSIZE(evenMap)  << std::endl;
	std::cout << " 3 ..." << XSIZE(oddMap)*YSIZE(oddMap)*ZSIZE(oddMap)  << std::endl;


	size_t n_even = 0;
	size_t n_odd = 0;
	long n =0;
	for (size_t k = 0; k < Zdim; k++)
	{	
		for (size_t i = 0; i < Ydim; i++)
		{
			for (size_t j = 0; j < Xdim; j++)
			{
				if ((i+j+k) % 2 == 1)
				{
					DIRECT_MULTIDIM_ELEM(oddMap, n) = DIRECT_MULTIDIM_ELEM(fm, n);
					n_odd += 1;
				}
				else
				{
					DIRECT_MULTIDIM_ELEM(evenMap, n) = DIRECT_MULTIDIM_ELEM(fm, n); 
					n_even += 1;
				}
				++n;
			}
		}
	}
	std::cout << " 4 ..." << n_odd  << std::endl;

	// Image<float> img;
	// img() = oddMap;
	// img.write("oddprefilter.mrc");

	chessBoardInterpolation(oddMap);
	chessBoardInterpolation(evenMap);

	// img() = oddMap;
	// img.write("oddafterfilter.mrc");


	std::cout << " 5 ..." << n_odd  << std::endl;
	std::cout << " 6 ..." << n_even  << std::endl;

	// Xdim = (size_t) Xdim/2;
	// Ydim = (size_t) Ydim/2;
	// Zdim = (size_t) Zdim/2;

	// std::cout << " 7 ..." << XSIZE(evenMap) << " " << ZSIZE(evenMap) << std::endl;

	fullMap = oddMap+evenMap;
	noiseMap = oddMap-evenMap;
	// noiseMap=fullMap;
	// std::cout << " 8 ..." << std::endl;

	// FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(evenMap)
	// {
	// 	DIRECT_MULTIDIM_ELEM(fullMap, n) += DIRECT_MULTIDIM_ELEM(evenMap, n);
	// 	DIRECT_MULTIDIM_ELEM(noiseMap, n) -= DIRECT_MULTIDIM_ELEM(evenMap, n);
	// }
	std::cout << " 9 ..." << std::endl;
}	

void ProgTomoConfidecenceMap::chessBoardInterpolation(MultidimArray<float> &tomo)
{
	for (size_t k = 1; k < (Zdim-1); k++)
	{	
		for (size_t i = 1; i < (Ydim-1); i++)
		{
			for (size_t j = 1; j < (Xdim-1); j++)
			{
				if (DIRECT_A3D_ELEM(tomo, k, i, j) == 0)
				{
					std::vector<float> medVec(6);
					medVec[0] = DIRECT_A3D_ELEM(tomo, k, i, j-1);
					medVec[1] = DIRECT_A3D_ELEM(tomo, k, i, j+1);
					medVec[2] = DIRECT_A3D_ELEM(tomo, k, i-1, j);
					medVec[3] = DIRECT_A3D_ELEM(tomo, k, i+1, j);
					medVec[4] = DIRECT_A3D_ELEM(tomo, k-1, i, j);
					medVec[5] = DIRECT_A3D_ELEM(tomo, k+1, i, j);
					sort(medVec.begin(), medVec.end());
					DIRECT_A3D_ELEM(tomo, k, i, j) = 0.5*(medVec[2] + medVec[3]);
				}
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


void ProgTomoConfidecenceMap::estimateNoiseStatistics(const MultidimArray<float> &noiseMap, 
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
