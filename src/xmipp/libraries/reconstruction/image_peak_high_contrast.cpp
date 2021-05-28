/***************************************************************************
 *
 * Authors:    Federico P. de Isidro Gomez			  fp.deisidro@cnb.csic.es
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

#include "image_peak_high_contrast.h"
#include <chrono>

void ProgImagePeakHighContrast::readParams()
{
	fnVol = getParam("--vol");
	fnOut = getParam("-o");
	boxSize = getIntParam("--boxSize");
	fiducialSize = getDoubleParam("--fiducialSize");
	sdThreshold = getIntParam("--sdThreshold");
    numberSampSlices = getIntParam("--numberSampSlices");
	numberCenterOfMass = getIntParam("--numberCenterOfMass");
	distanceThr = getIntParam("--distanceThr");
	numberOfCoordinatesThr = getIntParam("--numberOfCoordinatesThr");
	samplingRate = getDoubleParam("--samplingRate");

}


void ProgImagePeakHighContrast::defineParams()
{
	addUsageLine("This function determines the location of the outliers points in a volume");
	addParamsLine("  --vol <vol_file=\"\">                   				: Input volume.");
	addParamsLine("  [-o <output=\"coordinates3D.xmd\">]       				: Output file containing the 3D coodinates.");
  	addParamsLine("  [--boxSize <boxSize=32>]								: Box size of the peaked coordinates.");
	addParamsLine("  [--sdThreshold <sdThreshold=5>]      					: Number of SD a coordinate value must be over the mean to conisder that it belongs to a high contrast feature.");
  	addParamsLine("  [--numberSampSlices <numberSampSlices=10>]     		: Number of slices to use to determin the threshold value.");
  	addParamsLine("  [--numberCenterOfMass <numberCenterOfMass=10>]			: Number of initial center of mass to trim coordinates.");
  	addParamsLine("  [--distanceThr <distanceThr=10>]						: Minimum distance to consider two coordinates belong to the same center of mass.");
  	addParamsLine("  [--numberOfCoordinatesThr <numberOfCoordinatesThr=10>]	: Minimum number of coordinates attracted to a center of mass to consider it.");
  	addParamsLine("  [--fiducialSize <fiducialSize=100>]					: Fiducial size in Angstroms (A)");
  	addParamsLine("  [--samplingRate <samplingRate=1>]						: Sampling rate of the input tomogram (A/px)");

}


MultidimArray<double> ProgImagePeakHighContrast::preprocessVolume(MultidimArray<double> &inputTomo,
																  size_t xSize,
																  size_t ySize,
																  size_t zSize)
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Preprocessing volume..." << std::endl;
	#endif

	// Smoothing
	
	int siz_x = xSize*0.5;
	int siz_y = ySize*0.5;
	int siz_z = zSize*0.5;
	int N_smoothing = 10;

	size_t ux, uy, uz, uz2, uz2y2;

	int limit_distance_x = (siz_x-N_smoothing);
	int limit_distance_y = (siz_y-N_smoothing);
	int limit_distance_z = (siz_z-N_smoothing);

	long n=0;
	for(int k=0; k<zSize; ++k)
	{
		uz = (k - siz_z);
		for(int i=0; i<ySize; ++i)
		{
			uy = (i - siz_y);
			for(int j=0; j<xSize; ++j)
			{
				ux = (j - siz_x);

				if (abs(ux)>=limit_distance_x)
				{
					DIRECT_MULTIDIM_ELEM(inputTomo, n) *= 0.5*(1+cos(PI*(limit_distance_x - abs(ux))/(N_smoothing)));
				}
				if (abs(uy)>=limit_distance_y)
				{
					DIRECT_MULTIDIM_ELEM(inputTomo, n) *= 0.5*(1+cos(PI*(limit_distance_y - abs(uy))/(N_smoothing)));
				}
				if (abs(uz)>=limit_distance_z)
				{
					DIRECT_MULTIDIM_ELEM(inputTomo, n) *= 0.5*(1+cos(PI*(limit_distance_z - abs(uz))/(N_smoothing)));
				}
				++n;
			}
		}
	}


	// Band-pass filtering

	MultidimArray< std::complex<double> > fftV;

	FourierTransformer transformer;
	transformer.FourierTransform(inputTomo, fftV, false);

	n=0;

	MultidimArray< std::complex<double> >  fftFiltered;

	fftFiltered = fftV;
	double freqLow = samplingRate / (fiducialSize*1.1);
	double freqHigh = samplingRate/(fiducialSize*0.9);
	
	double w = 0.02; 
	double cutoffFreqHigh = freqHigh + w;
	double cutoffFreqLow = freqLow - w;
	double delta = PI / w;

	#ifdef DEBUG_FILTERPARAMS
	std::cout << samplingRate << std::endl;
	std::cout << fiducialSize << std::endl;
	std::cout << freqLow << std::endl;
	std::cout << freqHigh << std::endl;
	std::cout << cutoffFreqLow << std::endl;
	std::cout << cutoffFreqHigh << std::endl;
	#endif

	for(size_t k=0; k<ZSIZE(fftV); ++k)
	{
		double uz, uy, ux, uz2y2, uz2;

		FFT_IDX2DIGFREQ(k,ZSIZE(inputTomo),uz);
		uz2=uz*uz;

		for(size_t i=0; i<YSIZE(fftV); ++i)
		{
			FFT_IDX2DIGFREQ(i,YSIZE(inputTomo),uy);
			uz2y2=uz2+uy*uy;

			for(size_t j=0; j<XSIZE(fftV); ++j)
			{
				FFT_IDX2DIGFREQ(j,XSIZE(inputTomo),ux);
				double u=sqrt(uz2y2+ux*ux);

				if(u > cutoffFreqHigh || u < cutoffFreqLow)
				{
					DIRECT_MULTIDIM_ELEM(fftFiltered, n) = 0;
				} 
				else
				{
					if(u >= freqHigh && u < cutoffFreqHigh)
					{
						DIRECT_MULTIDIM_ELEM(fftFiltered, n) *= 0.5*(1+cos((u-freqHigh)*delta));
					}
				
					if (u <= freqLow && u > cutoffFreqLow)
					{
						DIRECT_MULTIDIM_ELEM(fftFiltered, n) *= 0.5*(1+cos((u-freqLow)*delta));
					}
				}
				
				++n;
			}
		}
	}

	MultidimArray<double>  volFiltered;
	volFiltered.resizeNoCopy(inputTomo);
	transformer.inverseFourierTransform(fftFiltered, volFiltered);

	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindex = fnOut.find_last_of(".");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameFilteredVolume;
    outputFileNameFilteredVolume = rawname + "_filter.mrc";

	Image<double> saveImage;
	saveImage() = volFiltered; 
	saveImage.write(outputFileNameFilteredVolume);
	#endif

	// *** make output volFiltered as the input volume (override)
	return volFiltered;
}


	void ProgImagePeakHighContrast::getHighContrastCoordinates(MultidimArray<double> volFiltered,
															   size_t xSize,
															   size_t ySize,
															   size_t zSize)
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Picking coordinates..." << std::endl;
	#endif

	size_t centralSlice = zSize/2;
	size_t minSamplingSlice = centralSlice - (numberSampSlices / 2);
	size_t maxSamplingSlice = centralSlice + (numberSampSlices / 2);

	#ifdef DEBUG
	std::cout << "Number of sampling slices: " << numberSampSlices << std::endl;
	std::cout << "Sampling region from slice " << minSamplingSlice << " to " << maxSamplingSlice << std::endl;
	#endif

	std::vector<double> sliceThresholdValue;
	double maxThreshold;
	
	for(size_t k = 0; k < zSize; ++k)
	{
		std::vector<int> sliceVector;
		
		// Calculate threshols value for the central slices of the volume
		if(k < maxSamplingSlice && k > minSamplingSlice)
		{
			for(size_t j = 0; j < ySize; ++j)
			{
				for(size_t i = 0; i < xSize; ++i)
				{
					#ifdef DEBUG_DIM
					std::cout << "i: " << i << " j: " << j << " k:" << k << std::endl;
					#endif

					sliceVector.push_back(DIRECT_ZYX_ELEM(volFiltered, k, i ,j));
				}
			}

			double sum = 0, sum2 = 0;
			int Nelems = 0;
			double average = 0;
			double standardDeviation = 0;
			double sliceVectorSize = sliceVector.size();

			for(size_t e = 0; e < sliceVectorSize; e++)
			{
				int value = sliceVector[e];
				sum += value;
				sum2 += value*value;
				++Nelems;
			}

			average = sum / sliceVectorSize;
			standardDeviation = sqrt(sum2/Nelems - average*average);

			double threshold = average-sdThreshold*standardDeviation;

			if (maxThreshold < threshold)
				maxThreshold = threshold;

			sliceThresholdValue.push_back(threshold);

			#ifdef DEBUG
			std::cout<< "Slice: " << k <<  " Threshold: " << threshold << std::endl;
			#endif
		}
	}

	std::cout << "Threshold value = " << maxThreshold << std::endl;
	maxThreshold = *std::min_element(sliceThresholdValue.begin(), sliceThresholdValue.end());

	std::cout << "Threshold value = " << maxThreshold << std::endl;

	#ifdef DEBUG
	std::cout << "Threshold value = " << maxThreshold << std::endl;
	#endif


	MultidimArray<double> binaryCoordinatesMapSlice;
	MultidimArray<double> labelCoordiantesMapSlice;
	MultidimArray<double> labelCoordiantesMap;

	labelCoordiantesMap.initZeros(zSize, ySize, xSize);

	
	for(size_t k = 0; k < zSize; k++)
	{	
		binaryCoordinatesMapSlice.initZeros(ySize, xSize);

		for(size_t j = 0; j < xSize; j++)
		{
			for(size_t i = 0; i < ySize; i++)
			{
				double value = DIRECT_A3D_ELEM(volFiltered, k, i, j);

				if (value<maxThreshold)
				{
					DIRECT_A2D_ELEM(binaryCoordinatesMapSlice, i, j) = 1.0;
				}
			}
		}
		#ifdef DEBUG
		std::cout << "Labeling slice " << k << std::endl;
		#endif

		// The value 8 is the neighbourhood
		int colour = labelImage2D(binaryCoordinatesMapSlice, labelCoordiantesMapSlice, 8);

		#ifdef DEBUG
		std::cout << "Colour: " << colour << std::endl;
		#endif

		// Remove coordinates thresholding the number of elements per label

		// These vectors will hold the list of labels and the nuber of coordinates associated to each of them
		std::vector<int> label;
		std::vector<int> numberCoordsPerLabel;
		
		for(size_t j = 0; j < xSize; j++)
		{
			for(size_t i = 0; i < ySize; i++)
			{
				int value = DIRECT_A2D_ELEM(labelCoordiantesMapSlice, i, j);

				if(value!=0)
				{
					bool labelExists = false;
					
					for(size_t n=0; n<label.size(); n++)
					{
						if(label[n]==value)
						{
						 	numberCoordsPerLabel[n] += 1;
							labelExists = true;
						}
					}

					if(labelExists==false)
					{
						label.push_back(value);
						numberCoordsPerLabel.push_back(1);
					}
				}
			}
		}

		for(size_t j = 0; j < xSize; j++)
		{
			for(size_t i = 0; i < ySize; i++)
			{
				for(size_t n=0; n<label.size(); n++)
				{
					if(label[n]==DIRECT_A2D_ELEM(labelCoordiantesMapSlice, i, j))
					{
						if(numberCoordsPerLabel[n]>numberOfCoordinatesThr)
						{						
							coordinates3Dx.push_back(j);
							coordinates3Dy.push_back(i);
							coordinates3Dz.push_back(k);

							DIRECT_A3D_ELEM(labelCoordiantesMap, k, i, j) = DIRECT_A2D_ELEM(labelCoordiantesMapSlice, i, j);
						}

						break;
					}
				}
			}
		}
    }

	#ifdef VERBOSE_OUTPUT
	std::cout << "Number of peaked coordinates: " << coordinates3Dx.size() << std::endl;
	#endif

	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindex = fnOut.find_last_of(".");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameFilteredVolume;
    outputFileNameFilteredVolume = rawname + "_label.mrc";

	Image<double> saveImage;
	saveImage() = labelCoordiantesMap; 
	saveImage.write(outputFileNameFilteredVolume);
	#endif
}


	void ProgImagePeakHighContrast::clusterHighContrastCoordinates(size_t xSize, size_t ySize, size_t zSize)
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Clustering coordinates..." << std::endl;
	#endif


	// These vectors keep track of the position of the center of mass
	std::vector<int> centerOfMassX(0);
    std::vector<int> centerOfMassY(0);
    std::vector<int> centerOfMassZ(0);

	// These vectors accumulate each coordinate attracted by every center of mass of calculate its mean at the end
	std::vector<std::vector<int>> centerOfMassXAcc;
	std::vector<std::vector<int>> centerOfMassYAcc;
	std::vector<std::vector<int>> centerOfMassZAcc;

	for(int i=0;i<numberCenterOfMass;i++)
	{
		int randomIndex = rand() % coordinates3Dx.size();

		int cx = coordinates3Dx[randomIndex];
		int cy = coordinates3Dy[randomIndex];
		int cz = coordinates3Dz[randomIndex];
		centerOfMassX.push_back(cx);
		centerOfMassY.push_back(cy);
		centerOfMassZ.push_back(cz);

		std::vector<int> newCenterOfMassX;
		std::vector<int> newCenterOfMassY;
		std::vector<int> newCenterOfMassZ;

		newCenterOfMassX.push_back(cx);
		newCenterOfMassY.push_back(cy);
		newCenterOfMassZ.push_back(cz);

		centerOfMassXAcc.push_back(newCenterOfMassX);
		centerOfMassYAcc.push_back(newCenterOfMassY);
		centerOfMassZAcc.push_back(newCenterOfMassZ);
	}

	int squareDistanceThr = distanceThr*distanceThr;
	bool attractedToMassCenter = false;

	for(size_t i = 0; i < coordinates3Dx.size(); i++)
	{
		// Check if the coordinate is attracted to any centre of mass
		attractedToMassCenter = false; 

		int xCoor = coordinates3Dx[i];
		int yCoor = coordinates3Dy[i];
		int zCoor = coordinates3Dz[i];

		for(size_t j = 0; j < centerOfMassX.size(); j++)
		{
			int xCM = centerOfMassX[j];
			int yCM = centerOfMassY[j];
			int zCM = centerOfMassZ[j];

			int squareDistance = (xCoor-xCM)*(xCoor-xCM)+(yCoor-yCM)*(yCoor-yCM)+(zCoor-zCM)*(zCoor-zCM);
			
			#ifdef DEBUG_DIST
			std::cout << "-----------------------------------------------------------------------" << std::endl;
			std::cout << "distance: " << squareDistance<< std::endl;
			std::cout << "threshold: " << squareDistanceThr<< std::endl;
			#endif

			if(squareDistance < squareDistanceThr)
			{
				// Update center of mass with new coordinate
				centerOfMassX[j]=xCM+(xCoor-xCM)/2;
				centerOfMassY[j]=yCM+(yCoor-yCM)/2;
				centerOfMassZ[j]=zCM+(zCoor-zCM)/2;

				// Add all the coordinate vectors to each center of mass
				centerOfMassXAcc[j].push_back(xCoor);
				centerOfMassYAcc[j].push_back(yCoor);
				centerOfMassZAcc[j].push_back(zCoor);

				attractedToMassCenter = true;
				break;
			}
		}

		if (attractedToMassCenter == false)
		{
			centerOfMassX.push_back(xCoor);
			centerOfMassY.push_back(yCoor);
			centerOfMassZ.push_back(zCoor);

			std::vector<int> newCenterOfMassX;
			std::vector<int> newCenterOfMassY;
			std::vector<int> newCenterOfMassZ;

			newCenterOfMassX.push_back(xCoor);
			newCenterOfMassY.push_back(yCoor);
			newCenterOfMassZ.push_back(zCoor);

			centerOfMassXAcc.push_back(newCenterOfMassX);
			centerOfMassYAcc.push_back(newCenterOfMassY);
			centerOfMassZAcc.push_back(newCenterOfMassZ);
		}
	}

	// Update the center of mass coordinates as the average of the accumulated vectors
	for(size_t i = 0; i < centerOfMassX.size(); i++)
	{
		int sumX = 0;
		int sumY = 0;
		int sumZ = 0;
		size_t centerOfMassAccSize = centerOfMassXAcc[i].size();

		for( size_t j = 0; j < centerOfMassAccSize; j++)
		{
			sumX += centerOfMassXAcc[i][j];
			sumY += centerOfMassYAcc[i][j];
			sumZ += centerOfMassZAcc[i][j];
		}

		centerOfMassX[i] = sumX / centerOfMassAccSize;
		centerOfMassY[i] = sumY / centerOfMassAccSize;
		centerOfMassZ[i] = sumZ / centerOfMassAccSize;
	}

	#ifdef VERBOSE_OUTPUT
	std::cout << "Prunning coordinates..." << std::endl;
	#endif

	for(size_t i=0;i<centerOfMassX.size();i++)
	{
		// Check that coordinates at the border of the volume are not outside when considering the box size
		if(centerOfMassX[i]<boxSize/2 or xSize-centerOfMassX[i]<boxSize/2 or
		   centerOfMassY[i]<boxSize/2 or ySize-centerOfMassY[i]<boxSize/2 or
		   centerOfMassZ[i]<boxSize/2 or zSize-centerOfMassZ[i]<boxSize/2)
		{
			centerOfMassX.erase(centerOfMassX.begin()+i);
			centerOfMassY.erase(centerOfMassY.begin()+i);
			centerOfMassZ.erase(centerOfMassZ.begin()+i);
			centerOfMassXAcc.erase(centerOfMassXAcc.begin()+i);
			centerOfMassYAcc.erase(centerOfMassYAcc.begin()+i);
			centerOfMassZAcc.erase(centerOfMassZAcc.begin()+i);
			i--;
		}

		// Check that number of coordinates per center of mass is higher than numberOfCoordinatesThr threshold
		if(centerOfMassXAcc[i].size() < numberOfCoordinatesThr)
		{
			centerOfMassX.erase(centerOfMassX.begin()+i);
			centerOfMassY.erase(centerOfMassY.begin()+i);
			centerOfMassZ.erase(centerOfMassZ.begin()+i);
			centerOfMassXAcc.erase(centerOfMassXAcc.begin()+i);
			centerOfMassYAcc.erase(centerOfMassYAcc.begin()+i);
			centerOfMassZAcc.erase(centerOfMassZAcc.begin()+i);
			i--;
		}
	}

	#ifdef VERBOSE_OUTPUT
	std::cout << "Number of centers of mass after trimming: " << centerOfMassX.size() << std::endl;
	#endif

	writeOutputCoordinates(centerOfMassX,
						   centerOfMassY,
						   centerOfMassZ);
}


	void ProgImagePeakHighContrast::writeOutputCoordinates(std::vector<int> centerOfMassX,
														   std::vector<int> centerOfMassY,
														   std::vector<int> centerOfMassZ)
{
	MetaData md;
	size_t id;

	for(size_t i=0;i<centerOfMassX.size();i++)
	{
		id = md.addObject();
		md.setValue(MDL_XCOOR, centerOfMassX[i], id);
		md.setValue(MDL_YCOOR, centerOfMassY[i], id);
		md.setValue(MDL_ZCOOR, centerOfMassZ[i], id);
	}

	md.write(fnOut);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Coordinates metadata saved at: " << fnOut << std::endl;
	#endif

}

void ProgImagePeakHighContrast::run()
{
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

	auto t1 = high_resolution_clock::now();
	
	Image<double> inputVolume;
	inputVolume.read(fnVol);

	auto &inputTomo=inputVolume();

	size_t xSize = XSIZE(inputTomo);
	size_t ySize = YSIZE(inputTomo);
	size_t zSize = ZSIZE(inputTomo);

	#ifdef DEBUG_DIM
	std::cout << "------------------ Input tomogram dimensions:" << std::endl;
	std::cout << "x " << XSIZE(inputTomo) << std::endl;
	std::cout << "y " << YSIZE(inputTomo) << std::endl;
	std::cout << "z " << ZSIZE(inputTomo) << std::endl;
	std::cout << "n " << NSIZE(inputTomo) << std::endl;
	#endif

	MultidimArray<double> volFiltered;

 	volFiltered = preprocessVolume(inputTomo, xSize, ySize, zSize);

	#ifdef DEBUG_DIM
	std::cout << "------------------ Filtered tomogram dimensions:" << std::endl;
	std::cout << "x " << XSIZE(volFiltered) << std::endl;
	std::cout << "y " << YSIZE(volFiltered) << std::endl;
	std::cout << "z " << ZSIZE(volFiltered) << std::endl;
	std::cout << "n " << NSIZE(volFiltered) << std::endl;
	#endif
	
	getHighContrastCoordinates(volFiltered, xSize, ySize, zSize);

	clusterHighContrastCoordinates(xSize, ySize, zSize);

	
	auto t2 = high_resolution_clock::now();
	/* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
 	std::cout << ms_int.count() << "ms\n";

}
