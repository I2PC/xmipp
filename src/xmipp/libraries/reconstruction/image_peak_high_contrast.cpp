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
	centerFeatures = checkParam("--centerFeatures");
}


void ProgImagePeakHighContrast::defineParams()
{
	addUsageLine("This function determines the location of high contrast features in a volume.");
	addParamsLine("  --vol <vol_file=\"\">                   				: Input volume.");
	addParamsLine("  [-o <output=\"coordinates3D.xmd\">]       				: Output file containing the 3D coodinates.");
  	addParamsLine("  [--boxSize <boxSize=32>]								: Box size of the peaked coordinates.");
	addParamsLine("  [--sdThreshold <sdThreshold=5>]      					: Number of SD a coordinate value must be over the mean to conisder that it belongs to a high contrast feature.");
  	addParamsLine("  [--numberSampSlices <numberSampSlices=10>]     		: Number of slices to use to determin the threshold value.");
  	addParamsLine("  [--numberCenterOfMass <numberCenterOfMass=10>]			: Number of initial center of mass to trim coordinates.");
  	addParamsLine("  [--distanceThr <distanceThr=10>]						: Minimum distance to consider two coordinates belong to the same center of mass.");
  	addParamsLine("  [--numberOfCoordinatesThr <numberOfCoordinatesThr=10>]	: Minimum number of coordinates attracted to a center of mass to consider it.");
  	addParamsLine("  [--fiducialSize <fiducialSize=100>]					: Fiducial size in Angstroms (A).");
  	addParamsLine("  [--samplingRate <samplingRate=1>]						: Sampling rate of the input tomogram (A/px).");
	addParamsLine("  [--centerFeatures]										: Center peaked features in box.");
}


void ProgImagePeakHighContrast::preprocessVolume(MultidimArray<double> &inputTomo)
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Preprocessing volume..." << std::endl;
	#endif

	// Smoothing
	
	// int siz_x = xSize*0.5;
	// int siz_y = ySize*0.5;
	// int siz_z = zSize*0.5;
	// int N_smoothing = 10;

	// int ux, uy, uz, uz2, uz2y2;

	// int limit_distance_x = (siz_x-N_smoothing);
	// int limit_distance_y = (siz_y-N_smoothing);
	// int limit_distance_z = (siz_z-N_smoothing);

	// long n=0;
	// for(int k=0; k<zSize; ++k)
	// {
	// 	uz = (k - siz_z);
	// 	for(int i=0; i<ySize; ++i)
	// 	{
	// 		uy = (i - siz_y);
	// 		for(int j=0; j<xSize; ++j)
	// 		{
	// 			ux = (j - siz_x);

	// 			if (abs(ux)>=limit_distance_x)
	// 			{
	// 				DIRECT_MULTIDIM_ELEM(inputTomo, n) *= 0.5*(1+cos(PI*(limit_distance_x - abs(ux))/(N_smoothing)));
	// 			}
	// 			if (abs(uy)>=limit_distance_y)
	// 			{
	// 				DIRECT_MULTIDIM_ELEM(inputTomo, n) *= 0.5*(1+cos(PI*(limit_distance_y - abs(uy))/(N_smoothing)));
	// 			}
	// 			if (abs(uz)>=limit_distance_z)
	// 			{
	// 				DIRECT_MULTIDIM_ELEM(inputTomo, n) *= 0.5*(1+cos(PI*(limit_distance_z - abs(uz))/(N_smoothing)));
	// 			}
	// 			++n;
	// 		}
	// 	}
	// }


	// Band-pass filtering

	MultidimArray< std::complex<double> > fftV;

	FourierTransformer transformer;
	transformer.FourierTransform(inputTomo, fftV, false);

	int ux, uy, uz, uz2, uz2y2;

	int n=0;

	MultidimArray< std::complex<double> >  fftFiltered;

	fftFiltered = fftV;
	double freqLow = samplingRate / (fiducialSize*1.1);
	double freqHigh = samplingRate/(fiducialSize*0.9);
	
	double w = 0.02; 
	double cutoffFreqHigh = freqHigh + w;
	double cutoffFreqLow = freqLow - w;
	double delta = PI / w;

	#ifdef DEBUG_FILTERPARAMS
	std::cout << "samplingRate " << samplingRate << std::endl;
	std::cout << "fiducialSize " << fiducialSize << std::endl;
	std::cout << "freqLow " << freqLow << std::endl;
	std::cout << "freqHigh " << freqHigh << std::endl;
	std::cout << "cutoffFreqLow " << cutoffFreqLow << std::endl;
	std::cout << "cutoffFreqHigh " << cutoffFreqHigh << std::endl;
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

	transformer.inverseFourierTransform(fftFiltered, inputTomo);

	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindex = fnOut.find_last_of(".");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameFilteredVolume;
    outputFileNameFilteredVolume = rawname + "_filter.mrc";

	Image<double> saveImage;
	saveImage() = inputTomo;
	saveImage.write(outputFileNameFilteredVolume);
	#endif
		std::cout << "check 3 " << std::endl;

}



void ProgImagePeakHighContrast::getHighContrastCoordinates(MultidimArray<double> volFiltered)
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Picking coordinates..." << std::endl;
	#endif

	size_t centralSlice = zSize/2;
	size_t minSamplingSlice = centralSlice - (numberSampSlices / 2);
	size_t maxSamplingSlice = centralSlice + (numberSampSlices / 2);

	#ifdef DEBUG_HCC
	std::cout << "Number of sampling slices: " << numberSampSlices << std::endl;
	std::cout << "Sampling region from slice " << minSamplingSlice << " to " << maxSamplingSlice << std::endl;
	#endif

	// Calculate threshols value for the central slices of the volume
	std::vector<int> sliceVector;

	for(size_t k = minSamplingSlice; k < maxSamplingSlice; ++k)
	{
		for(size_t j = 0; j < ySize; ++j)
		{
			for(size_t i = 0; i < xSize; ++i)
			{
				sliceVector.push_back(DIRECT_ZYX_ELEM(volFiltered, k, i ,j));
			}
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

	#ifdef VERBOSE_OUTPUT
	std::cout << "Threshold value = " << threshold << std::endl;
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

				if (value < threshold)
				{
					DIRECT_A2D_ELEM(binaryCoordinatesMapSlice, i, j) = 1.0;
				}
			}
		}
		#ifdef DEBUG_HCC
		std::cout << "Labeling slice " << k << std::endl;
		#endif

		// The value 8 is the neighbourhood
		int colour = labelImage2D(binaryCoordinatesMapSlice, labelCoordiantesMapSlice, 8);

		#ifdef DEBUG_HCC
		std::cout << "Colour: " << colour << std::endl;
		#endif

		std::vector<std::vector<int>> coordinatesPerLabelX (colour);
		std::vector<std::vector<int>> coordinatesPerLabelY (colour);

		for(size_t i = 0; i < ySize; i++)
		{
            for(size_t j = 0; j < xSize; ++j)
			{
				int value = DIRECT_A2D_ELEM(labelCoordiantesMapSlice, i, j);

				if(value!=0)
				{
					coordinatesPerLabelX[value-1].push_back(j);
					coordinatesPerLabelY[value-1].push_back(i);
				}
			}
		}

		size_t numberOfCoordinatesPerValue;

		// Trim coordinates based on the characteristics of the labeled region
		for(size_t value = 0; value < colour; value++)
		{
			numberOfCoordinatesPerValue =  coordinatesPerLabelX[value].size();

			int xCoor = 0;
			int yCoor = 0;

			for(size_t coordinate=0; coordinate < coordinatesPerLabelX[value].size(); coordinate++)
			{
				xCoor += coordinatesPerLabelX[value][coordinate];
				yCoor += coordinatesPerLabelY[value][coordinate];
			}

			double xCoorCM = xCoor/numberOfCoordinatesPerValue;
			double yCoorCM = yCoor/numberOfCoordinatesPerValue;


			bool keep = filterLabeledRegions(coordinatesPerLabelX[value], coordinatesPerLabelY[value], xCoorCM, yCoorCM);

			if(keep)
			{
				Point3D<double> point3D(xCoorCM, yCoorCM, k);
				coordinates3D.push_back(point3D);
				// coordinates3Dx.push_back(xCoorCM);
				// coordinates3Dy.push_back(yCoorCM);
				// coordinates3Dz.push_back(k);
			}
		}

		#ifdef DEBUG_HCC
		std::cout << "Colour: " << colour << std::endl;
		#endif
    }

	#ifdef VERBOSE_OUTPUT
	std::cout << "Number of peaked coordinates: " << coordinates3D.size() << std::endl;
	#endif

	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindex = fnOut.find_last_of(".");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameLabeledVolume;
    outputFileNameLabeledVolume = rawname + "_label.mrc";

	Image<double> saveImage;
	saveImage() = labelCoordiantesMap; 
	saveImage.write(outputFileNameLabeledVolume);
	#endif
}



void ProgImagePeakHighContrast::clusterHighContrastCoordinates()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Clustering coordinates..." << std::endl;
	#endif

	// These vectors accumulate each coordinate attracted by every center of mass of calculate its mean at the end
	std::vector<std::vector<int>> centerOfMassXAcc;
	std::vector<std::vector<int>> centerOfMassYAcc;
	std::vector<std::vector<int>> centerOfMassZAcc;

	for(int i=0;i<numberCenterOfMass;i++)
	{
		int randomIndex = rand() % coordinates3D.size();

		int cx = coordinates3D[randomIndex].x;
		int cy = coordinates3D[randomIndex].y;
		int cz = coordinates3D[randomIndex].z;
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

	for(size_t i = 0; i < coordinates3D.size(); i++)
	{
		// Check if the coordinate is attracted to any centre of mass
		attractedToMassCenter = false; 

		int xCoor = coordinates3D[i].x;
		int yCoor = coordinates3D[i].y;
		int zCoor = coordinates3D[i].y;

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
}

void ProgImagePeakHighContrast::centerCoordinates(MultidimArray<double> volFiltered)
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Centering coordinates..." << std::endl;
	#endif

	size_t halfBoxSize = boxSize / 2;
	size_t correlationWedge = boxSize / 3;
	size_t numberOfFeatures = centerOfMassX.size();
	MultidimArray<double> feature, symmetricFeature, correlationVolumeR;

	// Construct feature and its symmetric

	for(size_t n = 0; n < numberOfFeatures; n++)
	{
		feature.initZeros(boxSize, boxSize, boxSize);
		symmetricFeature.initZeros(boxSize, boxSize, boxSize);
		
		for(int k = 0; k < boxSize; k++) // zDim
		{	
			for(int j = 0; j < boxSize; j++) // xDim
			{
				for(int i = 0; i < boxSize; i++) // yDim
				{
					int cmXhalf = centerOfMassX[n] - halfBoxSize;
					int cmYhalf = centerOfMassY[n] - halfBoxSize;
					int cmZhalf = centerOfMassZ[n] - halfBoxSize;

					DIRECT_A3D_ELEM(feature, k, i, j) = 
					DIRECT_A3D_ELEM(volFiltered, 
									cmZhalf + k, 
									cmYhalf + i, 
									cmXhalf + j);

					DIRECT_A3D_ELEM(symmetricFeature, boxSize -1 - k, boxSize -1 - i, boxSize -1 - j) = 
					DIRECT_A3D_ELEM(volFiltered, 
									cmZhalf + k, 
									cmYhalf + i,
									cmXhalf + j);
				}
			}
		}

		// Shift the particle respect to its symmetric to look for the maximum correlation displacement
		CorrelationAux aux;
		correlation_matrix(feature, symmetricFeature, correlationVolumeR, aux, true);

		double maximumCorrelation = MINDOUBLE;
		double xDisplacement = 0, yDisplacement = 0, zDisplacement = 0;

		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(correlationVolumeR)
		{
			double value = DIRECT_A3D_ELEM(correlationVolumeR, k, j, i);
			
			if (value > maximumCorrelation)
			{
				maximumCorrelation = value;
				xDisplacement = j;
				yDisplacement = i;
				zDisplacement = k;
			}
		}

		// Update coordinate
		centerOfMassX[n] += ((int) xDisplacement - boxSize / 2) / 2;
		centerOfMassY[n] += ((int) yDisplacement - boxSize / 2) / 2;
		centerOfMassZ[n] += ((int) zDisplacement - boxSize / 2) / 2;
	}
}


	void ProgImagePeakHighContrast::writeOutputCoordinates()
{
	MetaDataVec md;
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
	std::cout << "Output coordinates metadata saved at: " << fnOut << std::endl;
	#endif

}


// --------------------------- MAIN ----------------------------------

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

	xSize = XSIZE(inputTomo);
	ySize = YSIZE(inputTomo);
	zSize = ZSIZE(inputTomo);

	#ifdef DEBUG_DIM
	std::cout << "------------------ Input tomogram dimensions:" << std::endl;
	std::cout << "x " << XSIZE(inputTomo) << std::endl;
	std::cout << "y " << YSIZE(inputTomo) << std::endl;
	std::cout << "z " << ZSIZE(inputTomo) << std::endl;
	std::cout << "n " << NSIZE(inputTomo) << std::endl;
	#endif

 	preprocessVolume(inputTomo);

	#ifdef DEBUG_DIM
	std::cout << "------------------ Filtered tomogram dimensions:" << std::endl;
	std::cout << "x " << XSIZE(volFiltered) << std::endl;
	std::cout << "y " << YSIZE(volFiltered) << std::endl;
	std::cout << "z " << ZSIZE(volFiltered) << std::endl;
	std::cout << "n " << NSIZE(volFiltered) << std::endl;
	#endif
	
	getHighContrastCoordinates(inputTomo);

	clusterHighContrastCoordinates();

	if(centerFeatures==true)
	{
		centerCoordinates(inputTomo);
	}

	writeOutputCoordinates();
	
	auto t2 = high_resolution_clock::now();
	/* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
}



// --------------------------- UTILS functions ----------------------------

bool ProgImagePeakHighContrast::filterLabeledRegions(std::vector<int> coordinatesPerLabelX, std::vector<int> coordinatesPerLabelY, double centroX, double centroY)
{
	// Check number of elements of the label
	if(coordinatesPerLabelX.size() < numberOfCoordinatesThr)
	{
		return false;
	}

	// Calculate the furthest point of the region from the centroid
	double maxSquareDistance = 0;
	double distance;

	#ifdef DEBUG_FILTERLABEL
	size_t debugN;
	#endif

	for(size_t n = 0; n < coordinatesPerLabelX.size(); n++)
	{
		distance = (coordinatesPerLabelX[n]-centroX)*(coordinatesPerLabelX[n]-centroX)+(coordinatesPerLabelY[n]-centroY)*(coordinatesPerLabelY[n]-centroY);

		if(distance >= maxSquareDistance)
		{
			#ifdef DEBUG_FILTERLABEL
			debugN = n;
			#endif

			maxSquareDistance = distance;
		}
	}

	double maxDistace;
	maxDistace = sqrt(maxSquareDistance);

	// Check sphericity of the labeled region
	double circumscribedArea = PI * (maxDistace * maxDistace);;
	double area = 0.0 + (double)coordinatesPerLabelX.size();
	double ocupation;

	ocupation = area / circumscribedArea;

	#ifdef DEBUG_FILTERLABEL
	std::cout << "debugN " << debugN << std::endl;
	std::cout << "x max distance " << coordinatesPerLabelX[debugN] << std::endl;
	std::cout << "y max distance " << coordinatesPerLabelY[debugN] << std::endl;
	std::cout << "centroX " << centroX << std::endl;
	std::cout << "centroY " << centroY << std::endl;
	std::cout << "area " << area << std::endl;
	std::cout << "circumscribedArea " << circumscribedArea << std::endl;
	std::cout << "maxDistace " << maxDistace << std::endl;
	std::cout << "ocupation " << ocupation << std::endl;
	#endif

	if(ocupation < 0.5)
	{
		#ifdef DEBUG_FILTERLABEL
		std::cout << "COORDINATE REMOVED AT " << centroX << " , " << centroY << " BECAUSE OF OCCUPATION"<< std::endl;
		#endif
		return false;
	}
}