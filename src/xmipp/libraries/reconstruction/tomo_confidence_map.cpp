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
#include <numeric>
//#define DEBUG
//#define DEBUG_MASK
//#define TEST_FRINGES



void ProgTomoConfidecenceMap::readParams()
{
	fnOdd = getParam("--odd");
	fnEven = getParam("--even");
	fnOut = getParam("--odir");
	fnMask = getParam("--mask");
	medianFilterBool = checkParam("--medianFilter");
	applySmoothingBeforeConfidence = checkParam("--applySmoothingBeforeConfidence");
	applySmoothingAfterConfidence = checkParam("--applySmoothingAfterConfidence");
	sigmaGauss = (float)  getDoubleParam("--sigmaGauss");
	sigVal = (float)  getDoubleParam("--significance");
	sampling = (float) getDoubleParam("--sampling_rate");
	fdr = (float) getDoubleParam("--fdr");
	nthrs = getIntParam("--threads");
}


void ProgTomoConfidecenceMap::defineParams()
{
	addUsageLine("This program determines the confidence map of a tilt series or a tomogram. The result is a value between 0 and 1,");
	addUsageLine("being zero a fully reliable point and 0 absolutely unreliable. It means the values is the confidence of the voxels/pixel");
	addParamsLine("  --odd <vol_file=\"\">              : Half volume 1");
	addParamsLine("  --even <vol_file=\"\">	            : Half volume 2");
	addParamsLine("  [--mask <vol_file=\"\">]           : Mask defining the signal. ");
	addParamsLine("  [--medianFilter]                   : Set true if a median filter should be applied.");
	addParamsLine("  [--applySmoothingBeforeConfidence] : Set true if a smoothing before computing the confidence should be applied.");
	addParamsLine("  [--applySmoothingAfterConfidence]  : Set true if a smoothing after computing the confidence  should be applied.");
	addParamsLine("  [--sigmaGauss <s=2>]               : This is the std for the smoothing.");
	addParamsLine("  [--fdr <s=0.05>]                   : False discovery rate");
	addParamsLine("  [--significance <s=0.95>]          : The level of confidence for the hypothesis test.");
	addParamsLine("  --odir <output=\"resmap.mrc\">     : Local resolution volume (in Angstroms)");
	addParamsLine("  [--sampling_rate <s=1>]            : Sampling rate (A/px)");
	addParamsLine("  [--threads <s=4>]                  : Number of threads");
}

void ProgTomoConfidecenceMap::readHalfMaps(FileName &fnOdd, FileName &fnEven)
{
	Image<float> oddMap, evenMap;

	oddMap.read(fnOdd);
	evenMap.read(fnEven);
	auto &odd = oddMap();
	auto &even = evenMap();
	fullMap = 0.5*(odd+even);
	noiseMap = 0.5*(odd-even);

	size_t Ndim;
	fullMap.getDimensions(Xdim, Ydim, Zdim, Ndim);
}

void ProgTomoConfidecenceMap::run()
{
	std::cout << "Starting ..." << std::endl;
	std::cout << "           " << std::endl;
	std::cout << "           " << std::endl;

	MetaDataVec mdOdd, mdEven;
	std::vector<FileName> vecfnOdd(0), vecfnEven(0);
	std::vector<double> vectiltOdd(0), vectiltEven(0);
	

	bool oddismd, evenismd;
	oddismd = fnOdd.isMetaData();
	evenismd = fnEven.isMetaData();

	Image<float> oddMap, evenMap;
	MultidimArray<float> significanceMap;

	// Checking if it is a metadata or if it is a tomogram
	if (oddismd || evenismd)
	{
		if (oddismd && evenismd)
		{
			mdOdd.read(fnOdd);
			mdEven.read(fnEven);

			sortImages(mdOdd, vecfnOdd, vectiltOdd);
			sortImages(mdEven, vecfnEven, vectiltEven);

			if (vectiltEven.size() != vectiltOdd.size())
			{
				REPORT_ERROR(ERR_ARG_INCORRECT, "Metadata files have different number of tilt angles");
			}
			else
			{
				for (size_t idx=0; idx<vecfnOdd.size(); idx++)
				{
					FileName fnImgOdd, fnImgEven;
					fnImgOdd = vecfnOdd[idx];
					fnImgEven = vecfnEven[idx];
					readHalfMaps(fnImgOdd, fnImgEven);

					confidenceMap(significanceMap, true, fullMap, noiseMap);

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
					FileName fn;
					fn = fnImgOdd.getBaseName()  + formatString("-%i.mrc", idx);
					significanceImg.write(fnOut+"/"+fn);
				}
			}

		}
		else
		{
			REPORT_ERROR(ERR_ARG_INCORRECT, "At least one of the input files is not a metadata file");
		}
	}
	else
	{
		
		readHalfMaps(fnOdd, fnEven);

		MultidimArray<float> significanceMapFiltered;
		significanceMapFiltered = significanceMap;

		if (medianFilterBool)
		{
			std::cout << "applying a median filter" << std::endl;
			medianFilter(significanceMap, significanceMapFiltered);
		}
		

		Image<float> significanceImg;
		significanceImg() = significanceMapFiltered;
		significanceImg.write(fnOut);


	}


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


void ProgTomoConfidecenceMap::frequencyToAnalyze(float &freq, float &tail, int idx)
{
	freq = (float) idx/(2*ZSIZE(fullMap));
	
	//if idx > (ZSIZE(fullMap)-10)

	//TODO: check tail range
	tail = ((float) (idx - 3))/(2*ZSIZE(fullMap));
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


void ProgTomoConfidecenceMap::confidenceMap(MultidimArray<float> &significanceMap, bool normalize, MultidimArray<float> &fullMap, MultidimArray<float> &noiseMap)
{
	MultidimArray<float> noiseVarianceMap, noiseMeanMap;
	Matrix2D<float> thresholdMatrix_mean, thresholdMatrix_std;

	int boxsize = 50;

	MultidimArray<double> fullMap_double;
	MultidimArray<double> noiseMap_double;

	// if (applySmoothingBeforeConfidence)
	// {
	// 	std::cout << "applying a gaussian bluring before confidence" << std::endl;

	// 	//convertToDouble(fullMap, fullMap_double);
	// 	convertToDouble(noiseMap, noiseMap_double);

	// 	//realGaussianFilter(fullMap_double, sigmaGauss);

	// 	// Image<double> significanceImgs;
	// 	// significanceImgs() = fullMap_double;
	// 	// // significanceImg() = significanceMap;
		
	// 	// significanceImgs.write("fullmap.mrc");
		
	// 	realGaussianFilter(noiseMap_double, sigmaGauss);

	// 	//convertToFloat(fullMap_double, fullMap);
	// 	convertToFloat(noiseMap_double, noiseMap);
	// }

	//TODO: estimateNoiseStatistics and normalizeTomogram in the same step
	estimateNoiseStatistics(noiseMap, noiseVarianceMap, noiseMeanMap, boxsize, thresholdMatrix_mean, thresholdMatrix_std);

	normalizeTomogram(fullMap, noiseVarianceMap, noiseMeanMap);

	significanceMap.initZeros(fullMap);

	// if (applySmoothingAfterConfidence)
	// {
	// 	std::cout << "applying a gaussian bluring after confidence" << std::endl;

	// 	convertToDouble(fullMap, fullMap_double);

	// 	realGaussianFilter(fullMap_double, sigmaGauss);

	// 	convertToFloat(fullMap_double, fullMap);
	// }

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

void ProgTomoConfidecenceMap::sortImages(MetaDataVec &md, std::vector<FileName> &vecfn, std::vector<double> &vectilt)
{

		FileName fn;
		double tilt;
		
		for (const auto& row: md)
		{
			row.getValue(MDL_ANGLE_TILT, tilt);
			row.getValue(MDL_IMAGE, fn);
			vecfn.push_back(fn);
			vectilt.push_back(tilt);
		}
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

void ProgTomoConfidecenceMap::nosiseEstimation(WeightedLeastSquaresHelper &helperStd, WeightedLeastSquaresHelper &helperMean, 
												int lX, int lY, double hX, double hY, int Nx, int Ny, int boxsize, Matrix2D<double> &noiseStdMatrix, Matrix2D<double> &noiseMeanMatrix)
{
	helperStd.A.initZeros(Nx*Ny,lX*lY);
    helperStd.b.initZeros(Nx*Ny);
    helperStd.w.initZeros(Nx*Ny);
    helperStd.w.initConstant(1);
	helperMean = helperStd;


	if ( (Xdim<boxsize) || (Ydim<boxsize) )
		std::cout << "Error: The tomogram/tiltseries in x-direction or y-direction is too small" << std::endl;

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

			if (Zdim == 1)
			{
				for (int k = startZ; k<finishZ; k++)
				{
					for (int i = yStart; i<yLimit; i++)
					{
						for (int j = xStart; j<xLimit; j++)
						{
							float aux = A3D_ELEM(noiseMap, k, i, j);
							sum += aux;
							sum2 += aux*aux;
							noiseVector.push_back( aux );
							N++;
						}
					}
				}
			}
			else
			{
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

			noiseVector.clear();
			idxBox+=1;
		}
	}
}


void ProgTomoConfidecenceMap::estimateNoiseStatistics(MultidimArray<float> &noiseMap, 
													 MultidimArray<float> &noiseVarianceMap, MultidimArray<float> &noiseMeanMap,
													 int boxsize, Matrix2D<float> &thresholdMatrix_mean, Matrix2D<float> &thresholdMatrix_std)
{
	std::cout << "Estimating noise statistics ...." <<  std::endl;
	noiseVarianceMap.initZeros(noiseMap);
	noiseMeanMap.initZeros(noiseMap);

	int Nx = Xdim/boxsize;
	int Ny = Ydim/boxsize;

	// For the spline regression. A minimum of 8 points are considered
	int lX=std::min(8,Nx-2), lY=std::min(8,Ny-2);
	double hX = Xdim / (double)(lX-3);
    double hY = Ydim / (double)(lY-3);

	Matrix2D<double> noiseStdMatrix, noiseMeanMatrix;
	noiseStdMatrix.initZeros(Ny, Nx);
	noiseMeanMatrix = noiseStdMatrix;

    WeightedLeastSquaresHelper helperStd, helperMean;

	nosiseEstimation(helperStd, helperMean, lX, lY, hX, hY, Nx, Ny, boxsize, noiseStdMatrix, noiseMeanMatrix);

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
