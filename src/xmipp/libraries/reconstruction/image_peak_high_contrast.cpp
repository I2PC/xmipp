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


void ProgImagePeakHighContrast::readParams()
{
	fnVol = getParam("--vol");
	fnOut = getParam("-o");
	boxSize = getIntParam("--boxSize");
	fiducialSize = getDoubleParam("--fiducialSize");
	ratioOfInitialCoordinates = getIntParam("--ratioInitialCoor");
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
	addParamsLine("  [--ratioInitialCoor <ratioInitialCoor=1>]      		: Ratio of initial coordinates to be considered as outliers. Ratio expressed as one coordinate per million.");
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
	std::cout << "Preprocessing volume..." << std::endl;

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
	std::cout << samplingRate << std::endl; //6.86
	std::cout << fiducialSize << std::endl; // 100
	std::cout << freqLow << std::endl; //0.062
	std::cout << freqHigh << std::endl; //0.076
	std::cout << cutoffFreqLow << std::endl; //0.042
	std::cout << cutoffFreqHigh << std::endl; //0.096
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

				// if(u < cutoffFreqLow)
				// {
				// 	DIRECT_MULTIDIM_ELEM(fftFiltered, n) = 0;
				// } 

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

	size_t lastindex = fnOut.find_last_of(".");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameFilteredVolume;
    outputFileNameFilteredVolume = rawname + "_filter.mrc";

	#ifdef DEBUG_OUTPUT_FILES
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
	std::cout << "Picking coordinates..." << std::endl;

	size_t centralSlice = zSize/2;
	size_t minSamplingSlice = centralSlice - (numberSampSlices / 2);
	size_t maxSamplingSlice = centralSlice + (numberSampSlices / 2);

	// std::vector<double> tomoVector(0);

	// double numberOfInitialCoordinates = xSize * ySize * numberSampSlices * (ratioOfInitialCoordinates / 1000000);

	#ifdef DEBUG
	std::cout << "Number of sampling slices: " << numberSampSlices << std::endl;
	// std::cout << "Number of initial coordinates: " << numberOfInitialCoordinates << std::endl;
	std::cout << "Sampling region from slice " << minSamplingSlice << " to " << maxSamplingSlice << std::endl;
	#endif

	// for(size_t k = centralSlice - (numberSampSlices/2); k <= centralSlice + (numberSampSlices / 2); ++k)
	// {
	// 	for(size_t j = 0; j < ySize; ++j)
	// 	{
	// 		for(size_t i = 0; i < xSize; ++i)
	// 		{
	// 			#ifdef DEBUG_DIM
	// 			std::cout << "i: " << i << " j: " << j << " k:" << k << std::endl;
	// 			#endif

	// 			tomoVector.push_back(DIRECT_ZYX_ELEM(volFiltered, k, i ,j));
	// 		}
	// 	}
	// }

	std::vector<double> sliceThresholdValue;
	
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

			double sum = 0;
			double average = 0;
			double standardDeviation = 0;
			double sliceVectorSize = sliceVector.size();

			for(size_t e = 0; e < sliceVectorSize; e++)
			{
				sum += sliceVector[e];
			}

			average = sum / sliceVectorSize;

			for(size_t f = 0; f < sliceVectorSize; f++)
			{
				int sliceVectorValue = sliceVector[f];
				standardDeviation += (sliceVectorValue-average)*(sliceVectorValue-average);
			}

			standardDeviation = sqrt(standardDeviation/sliceVectorSize);
			double threshold = average-ratioOfInitialCoordinates*standardDeviation;

			sliceThresholdValue.push_back(threshold);

			#ifdef DEBUG
			std::cout<< "Slice: " << k <<  " Threshold: " << threshold << std::endl;
			#endif
		}
	}

	double maxThreshold = *std::min_element(sliceThresholdValue.begin(), sliceThresholdValue.end());

	#ifdef DEBUG
	std::cout << "Threshold value = " << maxThreshold << std::endl;
	#endif

    std::vector<int> coordinates3Dx(0);
    std::vector<int> coordinates3Dy(0);
    std::vector<int> coordinates3Dz(0);

    // FOR_ALL_ELEMENTS_IN_ARRAY3D(volFiltered)
    // {
    //     double value = A3D_ELEM(volFiltered, k, i, j);

    //     if (value>=thresholdValue)
    //     {
    //         coordinates3Dx.push_back(j);
    //         coordinates3Dy.push_back(i);
    //         coordinates3Dz.push_back(k);
    //     }
    // }

	MultidimArray<double> binaryCoordinatesMapSlice;
	MultidimArray<double> labelCoordiantesMapSlice;
	MultidimArray<double> labelCoordiantesMap;

	labelCoordiantesMap.initZeros(zSize, ySize, xSize);

	
	for(size_t k = 0; k < zSize; k++)
	{	
		std::cout << "k=" << k << std::endl;


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
		std::cout << "labeling slice " << k << std::endl;

		int colour = labelImage2D(binaryCoordinatesMapSlice, labelCoordiantesMapSlice, 8);

		std::cout << "Colour: " << colour << std::endl;

		// OLD
		// FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(labelCoordiantesMapSlice){
		// 	DIRECT_A3D_ELEM(labelCoordiantesMap, k, i, j) = DIRECT_A2D_ELEM(labelCoordiantesMapSlice, i, j);
		// }

		// NEW
		/////////////////////////////////////////////////////////////////////////////////// REMOVE COORDINATES BY NUMBER OF ELEMENTS

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

		std::cout << "numberCoordsPerLabel generated!!" << std::endl;
		std::cout << label.size() << std::endl;
		std::cout << numberCoordsPerLabel.size() << std::endl;

		// for(int i=0; i<label.size(); i++)
		// {
		// 		std::cout << label[i] << std::endl;
		// 		std::cout << numberCoordsPerLabel[i] << std::endl;
		// 		std::cout << "---" << std::endl;
		// }

		for(size_t j = 0; j < xSize; j++)
		{
			for(size_t i = 0; i < ySize; i++)
			{
				// std::cout << "+++++++++" << std::endl;
				// std::cout << DIRECT_A2D_ELEM(labelCoordiantesMapSlice, i, j) << std::endl;
				// std::cout << "k=" << k << " j=" << j << " i=" << i << std::endl;

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

	#ifdef DEBUG
	std::cout << "Number of peaked coordinates: " << coordinates3Dx.size() << std::endl;
	#endif

	size_t lastindex = fnOut.find_last_of(".");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameFilteredVolume;
    outputFileNameFilteredVolume = rawname + "_label.mrc";

	Image<double> saveImage;
	saveImage() = labelCoordiantesMap; 
	saveImage.write(outputFileNameFilteredVolume);

	//////////////////////////////////////////////////////////////////////////////// ERODE OUTPUT LABELED MAP

	// MultidimArray<double> maskCoorinateMap;
	// maskCoorinateMap.initZeros(zSize, ySize, xSize);

	// FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(labelCoordiantesMap)
	// {
	// 	if(DIRECT_A3D_ELEM(labelCoordiantesMap, k, i, j)>0)
	// 	{
	// 		DIRECT_A3D_ELEM(maskCoorinateMap, k, i, j)=1;
	// 	}
	// }

	// MultidimArray<double> erodedMaskCoordinatesMap;

	// erodedMaskCoordinatesMap.initZeros(zSize, ySize, xSize);

	// erode3D(maskCoorinateMap, erodedMaskCoordinatesMap, 18, 1, 1);

	// size_t lastIndex = fnOut.find_last_of(".");
	// std::string rawName = fnOut.substr(0, lastIndex);
	// std::string outputFileNameErodedVolume;
    // // outputFileNameErodedVolume = rawName + "_eroded.mrc";

	// Image<double> saveImageBis;
	// saveImageBis() = erodedMaskCoordinatesMap; 
	// saveImageBis.write(outputFileNameErodedVolume);



	clusterHighContrastCoordinates(coordinates3Dx,
								   coordinates3Dy,
								   coordinates3Dz,
								   xSize,
								   ySize,
								   zSize);

}


	void ProgImagePeakHighContrast::clusterHighContrastCoordinates(std::vector<int> coordinates3Dx,
																   std::vector<int> coordinates3Dy,
																   std::vector<int> coordinates3Dz,
																   size_t xSize,
															   	   size_t ySize,
															       size_t zSize)
{
	std::cout << "Clustering coordinates..." << std::endl;


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

		centerOfMassX.push_back(coordinates3Dx[randomIndex]);
		centerOfMassY.push_back(coordinates3Dy[randomIndex]);
		centerOfMassZ.push_back(coordinates3Dz[randomIndex]);

		std::vector<int> newCenterOfMassX;
		std::vector<int> newCenterOfMassY;
		std::vector<int> newCenterOfMassZ;

		newCenterOfMassX.push_back(coordinates3Dx[randomIndex]);
		newCenterOfMassY.push_back(coordinates3Dy[randomIndex]);
		newCenterOfMassZ.push_back(coordinates3Dz[randomIndex]);

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
				centerOfMassX[j]=centerOfMassX[j]+(coordinates3Dx[i]-centerOfMassX[j])/2;
				centerOfMassY[j]=centerOfMassY[j]+(coordinates3Dy[i]-centerOfMassY[j])/2;
				centerOfMassZ[j]=centerOfMassZ[j]+(coordinates3Dz[i]-centerOfMassZ[j])/2;

				// Add all the coordinate vectors to each center of mass
				centerOfMassXAcc[j].push_back(coordinates3Dx[i]);
				centerOfMassYAcc[j].push_back(coordinates3Dy[i]);
				centerOfMassZAcc[j].push_back(coordinates3Dz[i]);

				attractedToMassCenter = true;
				break;
			}
		}

		if (attractedToMassCenter == false)
		{
			centerOfMassX.push_back(coordinates3Dx[i]);
			centerOfMassY.push_back(coordinates3Dy[i]);
			centerOfMassZ.push_back(coordinates3Dz[i]);

			std::vector<int> newCenterOfMassX;
			std::vector<int> newCenterOfMassY;
			std::vector<int> newCenterOfMassZ;

			newCenterOfMassX.push_back(coordinates3Dx[i]);
			newCenterOfMassY.push_back(coordinates3Dy[i]);
			newCenterOfMassZ.push_back(coordinates3Dz[i]);

			centerOfMassXAcc.push_back(newCenterOfMassX);
			centerOfMassYAcc.push_back(newCenterOfMassY);
			centerOfMassZAcc.push_back(newCenterOfMassZ);
		}
	}

	// Complete the coordinates associated to each center of mass finding coordinates within distanceThr distance 
	// to any of the coordinates of the set.

	// std::cout << "centerOfMassX.size()="<<centerOfMassX.size()<<std::endl;
	// std::cout << "centerOfMassXAcc.size()="<<centerOfMassXAcc.size()<<std::endl;
	// std::cout << "coordinates3Dx.size()="<<coordinates3Dx.size()<<std::endl;

	// for(size_t i = 0; i < centerOfMassXAcc.size(); i++)
	// {
	// 	std::cout << "centerOfMassXAcc["<<i<<"].size()="<<centerOfMassXAcc[i].size()<<std::endl;
	// }



	// for(size_t i = 0; i < centerOfMassXAcc.size(); i++)
	// {
	// 	std::cout << "centerOfMassXAcc["<<i<<"].size()="<<centerOfMassXAcc[i].size()<<std::endl;

	// 	for(size_t j = 0; j < centerOfMassXAcc[i].size(); j++)
	// 	{
	// 		int xCMA = centerOfMassXAcc[i][j];
	// 		int yCMA = centerOfMassYAcc[i][j];
	// 		int zCMA = centerOfMassZAcc[i][j];

	// 		for(size_t k = 0; k < coordinates3Dx.size(); ++k)
	// 		{

	// 			std::cout<<"i="<<i<<", j="<<j<<", k="<<k<<std::endl;
	// 			int xCoor = coordinates3Dx[k];
	// 			int yCoor = coordinates3Dy[k];
	// 			int zCoor = coordinates3Dz[k];
				
	// 			int squareDistance = (xCoor-xCMA)*(xCoor-xCMA)+(yCoor-yCMA)*(yCoor-yCMA)+(zCoor-zCMA)*(zCoor-zCMA);

	// 			if(squareDistance < squareDistanceThr)
	// 			{
	// 				std::cout << "squareDistance < squareDistanceThr!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<< std::endl;
	// 				// Add all the coordinate vectors to each center of mass
	// 				centerOfMassXAcc[j].push_back(coordinates3Dx[k]);
	// 				centerOfMassYAcc[j].push_back(coordinates3Dy[k]);
	// 				centerOfMassZAcc[j].push_back(coordinates3Dz[k]);
	// 			}
	// 		}
	// 	}
	// }

	std::cout << "Prunning coordinates..." << std::endl;


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

	#ifdef DEBUG
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
	
	#ifdef DEBUG
	std::cout << "Coordinates metadata saved at: " << fnOut << std::endl;
	#endif

}

void ProgImagePeakHighContrast::run()
{
	Image<double> inputVolume;
	inputVolume.read(fnVol);

	MultidimArray<double> &inputTomo=inputVolume();

	size_t xSize = XSIZE(inputTomo);
	size_t ySize = YSIZE(inputTomo);
	size_t zSize = ZSIZE(inputTomo);

	#ifdef DEBUG_DIM
	std::cout << "x " << XSIZE(inputTomo) << std::endl;
	std::cout << "y " << YSIZE(inputTomo) << std::endl;
	std::cout << "z " << ZSIZE(inputTomo) << std::endl;
	std::cout << "n " << NSIZE(inputTomo) << std::endl;
	#endif

	MultidimArray<double> volFiltered;

 	volFiltered = preprocessVolume(inputTomo, xSize, ySize, zSize);

	size_t xSizeFilter = XSIZE(volFiltered);
	size_t ySizeFilter = YSIZE(volFiltered);
	size_t zSizeFilter = ZSIZE(volFiltered);

	#ifdef DEBUG_DIM
	std::cout << "------------------ after Filtering:" << std::endl;
	std::cout << "x " << XSIZE(volFiltered) << std::endl;
	std::cout << "y " << YSIZE(volFiltered) << std::endl;
	std::cout << "z " << ZSIZE(volFiltered) << std::endl;
	std::cout << "n " << NSIZE(volFiltered) << std::endl;
	#endif
	
	getHighContrastCoordinates(volFiltered, xSizeFilter, ySizeFilter, zSizeFilter);
}
