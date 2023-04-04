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

void ProgTomoConfidecenceMap::readParams()
{
	fnTs = getParam("--tiltseries");
	fnOut = getParam("--odir");
	medianFilterBool = checkParam("--medianFilter");
	applySmoothingBeforeConfidence = checkParam("--applySmoothingBeforeConfidence");
	applySmoothingAfterConfidence = checkParam("--applySmoothingAfterConfidence");
	sigmaGauss = (float)  getDoubleParam("--sigmaGauss");
	sampling = (float) getDoubleParam("--sampling_rate");
	boxsize = getIntParam("--locality");
	nthrs = getIntParam("--threads");
}


void ProgTomoConfidecenceMap::defineParams()
{
	addUsageLine("This program determines the confidence map of a tilt series or a tomogram. The result is a value between 0 and 1,");
	addUsageLine("being zero a fully reliable point and 0 absolutely unreliable. It means the values is the confidence of the voxels/pixel");
	addParamsLine("  --tiltseries <xmd_file=\"\">       : Input metadata with the two half maps (odd-even)");
	addParamsLine("  [--medianFilter]                   : Set true if a median filter should be applied.");
	addParamsLine("  [--applySmoothingBeforeConfidence] : Set true if a smoothing before computing the confidence should be applied.");
	addParamsLine("  [--applySmoothingAfterConfidence]  : Set true if a smoothing after computing the confidence  should be applied.");
	addParamsLine("  [--sigmaGauss <s=2>]               : This is the std for the smoothing.");
	addParamsLine("  --odir <output=\"resmap.mrc\">     : Local resolution volume (in Angstroms)");
	addParamsLine("  [--sampling_rate <s=1>]            : Sampling rate (A/px)");
	addParamsLine("  [--locality <s=40>]                : Edge of the square local windows where local distribution of noise will be measured");
	addParamsLine("  [--threads <s=4>]                  : Number of threads");
}

void ProgTomoConfidecenceMap::defineFourierFilter(MultidimArray<std::complex<double>> &mapfftV)
{
	size_t XdimFT = XSIZE(mapfftV);
	size_t YdimFT = YSIZE(mapfftV);
	//size_t ZdimFT = ZSIZE(mapfftV);

	// Initializing the frequency vectors
	freq_fourier_x.initZeros(XdimFT);
	freq_fourier_y.initZeros(YdimFT);
	//freq_fourier_z.initZeros(ZdimFT);

	// u is the frequency
	double u;

	VEC_ELEM(freq_fourier_y, 0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<YdimFT; ++k){
		FFT_IDX2DIGFREQ(k, Ydim, u);
		VEC_ELEM(freq_fourier_y, k) = u;
	}

	VEC_ELEM(freq_fourier_x,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<XdimFT; ++k){
		FFT_IDX2DIGFREQ(k, Xdim, u);
		VEC_ELEM(freq_fourier_x, k) = u;
	}

	//Initializing map with frequencies
	fourierFilterShape.initZeros(mapfftV);  //Nyquist is 2, we take 1.9 greater than Nyquist

	// Directional frequencies along each direction
	double uz, uy, ux, uz2, uz2y2;
	long n=0;
	int idx = 0;


	uz2 = 0;
	for(size_t i=0; i<YdimFT; ++i)
	{
		uy = VEC_ELEM(freq_fourier_y, i);
		uz2y2 = uz2 + uy*uy;

		for(size_t j=0; j<XdimFT; ++j)
		{
			ux = VEC_ELEM(freq_fourier_x, j);
			ux = sqrt(uz2y2 + ux*ux);

			if	(ux<=0.5)
			{
				DIRECT_MULTIDIM_ELEM(fourierFilterShape, n) = 1.0;						
			}				
			++n;
		}
	}

}

void ProgTomoConfidecenceMap::nyquistFilter(MultidimArray<std::complex<double>> &fftImg)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fftImg)
	{
		DIRECT_MULTIDIM_ELEM(fftImg, n) *= DIRECT_MULTIDIM_ELEM(fourierFilterShape, n);
	}
}

void ProgTomoConfidecenceMap::run()
{
	std::cout << "Starting ..." << std::endl;
	std::cout << "           " << std::endl;
	std::cout << "           " << std::endl;

	MetaDataVec mdOdd, mdEven;
	std::vector<FileName> vecfnOdd(0), vecfnEven(0);
	std::vector<double> vectiltOdd(0), vectiltEven(0);
	
	FileName fnH1, fnH2;
	Image<float> oddMap, evenMap;
	MultidimArray<float> significanceMap;

	MetaDataVec mdIn, mdConf, mdDenoised;
	mdIn.read(fnTs);

	size_t idx = 0;
	bool createFilter = true;

	FourierTransformer transformer1(FFTW_BACKWARD);
	transformer1.setThreadsNumber(nthrs);

	FourierTransformer transformer2(FFTW_BACKWARD);
	transformer2.setThreadsNumber(nthrs);

	for (const auto& row: mdIn)
	{
		double tilt;
		row.getValue(MDL_ANGLE_TILT, tilt);
		row.getValue(MDL_HALF1, fnH1);
		row.getValue(MDL_HALF2, fnH2);
		
		oddMap.read(fnH1);
		evenMap.read(fnH2);
		auto &odd = oddMap();
		auto &even = evenMap();
		fullMap = 0.5*(odd+even);
		noiseMap = 0.5*(odd-even);

		FileName fn;

		size_t Ndim;
		fullMap.getDimensions(Xdim, Ydim, Zdim, Ndim);

		MultidimArray<double> fullMapDouble, noiseMapDouble;

		if (medianFilterBool)
		{
			std::cout << "applying a median filter" << std::endl;
			MultidimArray<float> fullMapfloatFiltered;
			fullMapfloatFiltered.resizeNoCopy(fullMap);
			medianFilter2D(fullMap, fullMapfloatFiltered);
			fullMap = fullMapfloatFiltered;
		}

		convertToDouble(fullMap, fullMapDouble);

		MultidimArray<std::complex<double>> fftImg, fftNoise;
        transformer1.FourierTransform(fullMapDouble, fftImg, false);
				
		if (createFilter)
		{
			defineFourierFilter(fftImg);
			createFilter = false;
		}

		nyquistFilter(fftImg);

		transformer1.inverseFourierTransform();

		convertToFloat(fullMapDouble, fullMap);
		
		convertToDouble(noiseMap, noiseMapDouble);
		
		transformer2.FourierTransform(noiseMapDouble, fftNoise, false);
		nyquistFilter(fftNoise);
		transformer2.inverseFourierTransform();
		
		convertToFloat(noiseMapDouble, noiseMap);

		confidenceMap(significanceMap, true, fullMap, noiseMap);

		MDRowVec rowOut;

		fn = formatString("confidence-%i.mrc", idx);

		//auto rowOut = row;
		rowOut.setValue(MDL_IMAGE, fn);
		rowOut.setValue(MDL_ANGLE_TILT, tilt);
		mdConf.addRow(rowOut);
		
		Image<float> significanceImg;
		significanceImg() = significanceMap;
		significanceImg.write(fnOut + "/" + fn);
		

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(significanceMap)
		{
			DIRECT_MULTIDIM_ELEM(fullMap, n) *= DIRECT_MULTIDIM_ELEM(significanceMap, n);
		}

		fn = formatString("denoised-%i.mrc", idx);
		significanceImg() = fullMap;
		significanceImg.write(fnOut + "/" + fn);

		rowOut.setValue(MDL_IMAGE, fn);
		mdDenoised.addRow(rowOut);
		idx++;
	}
	mdConf.write(fnOut+"/ts_confidence.xmd");
	mdDenoised.write(fnOut+"/ts_denoised.xmd");
}


void ProgTomoConfidecenceMap::confidenceMap(MultidimArray<float> &significanceMap, bool normalize, MultidimArray<float> &fullMap, MultidimArray<float> &noiseMap)
{
	MultidimArray<float> noiseVarianceMap, noiseMeanMap;
	Matrix2D<float> thresholdMatrix_mean, thresholdMatrix_std;

	MultidimArray<double> fullMap_double;
	MultidimArray<double> noiseMap_double;

	//TODO: estimateNoiseStatistics and normalizeTomogram in the same step
	estimateNoiseStatistics(noiseMap, noiseVarianceMap, noiseMeanMap, boxsize, thresholdMatrix_mean, thresholdMatrix_std);

	//normalizeTomogram(fullMap, noiseVarianceMap, noiseMeanMap);

	significanceMap.initZeros(fullMap);

	computeSignificanceMap(fullMap, significanceMap, thresholdMatrix_mean, thresholdMatrix_std);
}


void ProgTomoConfidecenceMap::sortImages(MetaDataVec &md, std::vector<FileName> &vecfn, std::vector<double> &vectilt)
{
		//TODO: apply the sorting
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

void ProgTomoConfidecenceMap::medianFilter3D(MultidimArray<float> &input_tomogram,
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

void ProgTomoConfidecenceMap::medianFilter2D(MultidimArray<float> &input_img,
											MultidimArray<float> &output_img)
{
	std::vector<float> sortedVector;

   	for (int i=1; i<(Ydim-1); ++i)
	{
		for (int j=1; j<(Xdim-1); ++j)
		{
			std::vector<float> sortedVector;
			sortedVector.push_back(DIRECT_A2D_ELEM(input_img, i, j));
			sortedVector.push_back(DIRECT_A2D_ELEM(input_img, i+1, j));
			sortedVector.push_back(DIRECT_A2D_ELEM(input_img, i-1, j));
			sortedVector.push_back(DIRECT_A2D_ELEM(input_img, i, j+1));
			sortedVector.push_back(DIRECT_A2D_ELEM(input_img, i, j-1));

			std::sort(sortedVector.begin(),sortedVector.end());

			DIRECT_A2D_ELEM(output_img, i, j) = sortedVector[2];
			sortedVector.clear();
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
			meanValue = (double) noiseVector[size_t(noiseVector.size()*0.5)];
			meanValue = (double) sum/N;
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


void ProgTomoConfidecenceMap::computeSignificanceMap(MultidimArray<float> &fullMap, MultidimArray<float> &significanceMap,
													 Matrix2D<float> &thresholdMatrix_mean, Matrix2D<float> &thresholdMatrix_std)
{
	MultidimArray<double> fullMapDouble;

	convertToDouble(fullMap, fullMapDouble);

	realGaussianFilter(fullMapDouble, (double) sigmaGauss);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fullMap)
	{
		if (DIRECT_MULTIDIM_ELEM(fullMap, n)>DIRECT_MULTIDIM_ELEM(fullMapDouble, n))
			DIRECT_MULTIDIM_ELEM(fullMap, n) = (float) DIRECT_MULTIDIM_ELEM(fullMapDouble, n);
	}

	//convertToFloat(fullMapDouble, fullMap);
	float invsigma2 = 1/(0.2636*0.2636);

	long n = 0;
	for (int k=0; k<Zdim; ++k)
	{
		for (int i=0; i<Ydim; ++i)
		{
			for (int j=0; j<Xdim; ++j)
			{
				float fm = DIRECT_MULTIDIM_ELEM(fullMap, n);
				
				float x = (fm - MAT_ELEM(thresholdMatrix_mean, i, j))/MAT_ELEM(thresholdMatrix_std, i, j);
				// The significanse is 0.5 * (1. + erf(x/sqrt(2.))) But we write 1-significante because de theimages are
				// black over white
				float val =  0.5 * (1. + erf(x/sqrt(2.)));

				if (val<0.5)
				{
					val = 0.0;
				}
				else
				{
					val = 1-expf(-invsigma2*(val-0.5)*(val-0.5));
				}
				DIRECT_MULTIDIM_ELEM(significanceMap, n) = val;//1 - 0.5 * (1. + erf(x/sqrt(2.)));
				//DIRECT_MULTIDIM_ELEM(significanceMap, n) = 1 - 0.5 * (1. + erf(x/sqrt(2.)));
				n++;
			}
		}
	}
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


