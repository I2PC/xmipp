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

#include "tomo_detect_misalignment_trajectory.h"
#include <chrono>

// --------------------------- INFO functions ----------------------------

void ProgTomoDetectMisalignmentTrajectory::readParams()
{
	fnVol = getParam("-i");
	fnTiltAngles = getParam("--tlt");
	fnOut = getParam("-o");

	samplingRate = getDoubleParam("--samplingRate");
	fiducialSize = getDoubleParam("--fiducialSize");

	thrSDHCC = getIntParam("--thrSDHCC");
	thrNumberCoords = getIntParam("--thrNumberCoords");
	thrChainDistanceAng = getDoubleParam("--thrChainDistanceAng");
	
	checkInputCoord = checkParam("--inputCoord");
	if(checkInputCoord)
	{
		fnInputCoord = getParam("--inputCoord");
	}
}


void ProgTomoDetectMisalignmentTrajectory::defineParams()
{
	addUsageLine("This function determines the location of high contrast features in a volume.");
	addParamsLine("  -i <mrcs_file=\"\">                   					: Input tilt-series.");
	addParamsLine("  --tlt <xmd_file=\"\">      							: Input file containning the tilt angles of the tilt-series in .xmd format.");
	addParamsLine("  [-o <output=\"./alignemntReport.xmd\">]       			: Output file containing the alignemnt report.");

	addParamsLine("  [--samplingRate <samplingRate=1>]						: Sampling rate of the input tomogram (A/px).");
	addParamsLine("  [--fiducialSize <fiducialSize=100>]					: Fiducial size in Angstroms (A).");


	addParamsLine("  [--thrSDHCC <thrSDHCC=5>]      						: Threshold number of SD a coordinate value must be over the mean to consider that it belongs to a high contrast feature.");
  	addParamsLine("  [--thrNumberCoords <thrNumberCoords=10>]				: Threshold minimum number of coordinates attracted to a center of mass to consider it as a high contrast feature.");
	addParamsLine("  [--thrChainDistanceAng <thrChainDistanceAng=20>]		: Threshold maximum distance in angstroms of a detected landmark to consider it belongs to a chain.");

	addParamsLine("  [--inputCoord <output=\"\">]							: Input coordinates of the 3D landmarks to calculate the residual vectors.");
}


// --------------------------- HEAD functions ----------------------------

void ProgTomoDetectMisalignmentTrajectory::generateSideInfo()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Generating side info..." << std::endl;
	#endif

	// Read tilt angles file
	MetaDataVec inputTiltAnglesMd;
	double tiltAngle;
	size_t objId;

	inputTiltAnglesMd.read(fnTiltAngles);

	tiltAngleStep=0;

	for(size_t objId : inputTiltAnglesMd.ids())
	{
		inputTiltAnglesMd.getValue(MDL_ANGLE_TILT, tiltAngle, objId);
		tiltAngles.push_back(tiltAngle);

		tiltAngleStep += tiltAngle;
	}

	tiltAngleStep /= tiltAngles.size();

	#ifdef VERBOSE_OUTPUT
	std::cout << "Input tilt angles read from: " << fnTiltAngles << std::endl;
	#endif


	// Initialize local alignment vector (depends on the number of acquisition angles)
	localAlignment.resize(nSize, true);

	// Update thresholds depending on input tilt-series sampling rate
	minDistancePx = minDistanceAng * samplingRate;
	thrChainDistancePx = thrChainDistanceAng * samplingRate;

	#ifdef VERBOSE_OUTPUT
	std::cout << "Thresholds:" << std::endl;
	std::cout << "thrChainDistancePx: "<< thrChainDistancePx << std::endl;
	std::cout << "minDistancePx: "<< minDistancePx << std::endl;
	std::cout << "thrTop10Chain: "<< thrTop10Chain << std::endl;
	std::cout << "thrLMChain: "<< thrLMChain << std::endl;
	std::cout << "numberOfElementsInChainThreshold: "<< numberOfElementsInChainThreshold << std::endl;
	#endif

	#ifdef VERBOSE_OUTPUT
	std::cout << "Side info generated succesfully!" << std::endl;
	#endif
}

void ProgTomoDetectMisalignmentTrajectory::bandPassFilter(MultidimArray<double> &tiltImage)
{
	FourierTransformer transformer1(FFTW_BACKWARD);
	MultidimArray<std::complex<double>> fftImg;
	transformer1.FourierTransform(tiltImage, fftImg, true);

	normDim = (xSize>ySize) ? xSize : ySize;

	// 43.2 = 1440 * 0.03. This 43.2 value makes w = 0.03 (standard value) for an image whose bigger dimension is 1440 px.
	double w = 43.2 / normDim;

    double lowFreqFilt = samplingRate/(1.1*fiducialSize);
	double highFreqFilt = samplingRate/(0.9*fiducialSize);

	double tail_high = highFreqFilt + w;
    double tail_low = lowFreqFilt - w;

	double delta = PI / w;

    double uy;
	double ux;
	double u;
	double uy2;

	long n=0;

	for(size_t i=0; i<YSIZE(fftImg); ++i)
	{
		FFT_IDX2DIGFREQ(i, ySize, uy);
		uy2=uy*uy;

		for(size_t j=0; j<XSIZE(fftImg); ++j)
		{
			FFT_IDX2DIGFREQ(j, xSize, ux);
			u=sqrt(uy2+ux*ux);

			if (u > tail_high || u < tail_low)
			{
				DIRECT_MULTIDIM_ELEM(fftImg, n) = 0;
			}
			else
			{
				if (u >= highFreqFilt && u <=tail_high)
				{
					DIRECT_MULTIDIM_ELEM(fftImg, n) *= 0.5*(1+cos((u-highFreqFilt)*delta));
				}

				if (u <= lowFreqFilt && u >= tail_low)
				{
					DIRECT_MULTIDIM_ELEM(fftImg, n) *= 0.5*(1+cos((u-lowFreqFilt)*delta));
				}
			}

			++n;
		}
	}

	transformer1.inverseFourierTransform(fftImg, tiltImage);
}


void ProgTomoDetectMisalignmentTrajectory::getHighContrastCoordinates(MultidimArray<double> tiltSeriesFiltered)
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Picking high contrast coordinates..." << std::endl;
	#endif

	// *** reutilizar binaryCoordinatesMapSlice slice a slice y descartar labelCoordiantesMap	
    MultidimArray<double> binaryCoordinatesMapSlice;
    MultidimArray<double> labelCoordiantesMapSlice;
    MultidimArray<double> labelCoordiantesMap;

	labelCoordiantesMap.initZeros(nSize, zSize, ySize, xSize);

	for(size_t k = 0; k < nSize; ++k)
	{
		std::vector<int> sliceVector;

		// search in the cosine streched region common for all the images
		int xSizeCS = (int)xSize * abs(cos(tiltAngles[k] * PI/180.0));
		int xCSmin = (int)(xSize-xSizeCS)/2;
		int xCSmax = (int)(xSize+xSizeCS)/2;

		#ifdef DEBUG_HCC
		std::cout << "Tilt angle: "<< tiltAngles[k] << "º" << std::endl;
		std::cout << "Cosine streched searching region: (" << xCSmin << ", " <<  xCSmax << ")" << std::endl;
		std::cout << "Cosine streched searching region size: " << xSizeCS << std::endl;
		#endif

		// Calculate threshold value for each image of the series
        for(size_t i = 0; i < ySize; ++i)
        {
			// search in the cosine streched region common for all the images
            for(size_t j = xCSmin; j < xCSmax; ++j)
            {
				/// *** enhance performance: do not use slice vector, sum directly from image
                sliceVector.push_back(DIRECT_NZYX_ELEM(tiltSeriesFiltered, k, 0, i ,j));
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

        double threshold = average - thrSDHCC * standardDeviation;


        #ifdef DEBUG_HCC
		std::cout << "Slice: " << k+1 << " Average: " << average << " SD: " << standardDeviation << " Threshold: " << threshold << std::endl;
        #endif

		binaryCoordinatesMapSlice.initZeros(ySize, xSize);

		#ifdef DEBUG_HCC
		int numberOfPointsAddedBinaryMap = 0;
		#endif

		for(size_t i = 0; i < ySize; i++)
		{
			// Search in the cosine streched region common for all the images
            for(size_t j = xCSmin; j < xCSmax; ++j)
			{
				double value = DIRECT_A3D_ELEM(tiltSeriesFiltered, k, i, j);

				if (value < threshold)
				{
					DIRECT_A2D_ELEM(binaryCoordinatesMapSlice, i, j) = 1.0;
					
					#ifdef DEBUG_HCC
					numberOfPointsAddedBinaryMap += 1;
					#endif
				}
			}
		}

		#ifdef DEBUG_HCC
		std::cout << "Number of points in the binary map: " << numberOfPointsAddedBinaryMap << std::endl;
		#endif

		// The value 8 is the neighbourhood
		int colour = labelImage2D(binaryCoordinatesMapSlice, labelCoordiantesMapSlice, 8);

		for(size_t i = 0; i < ySize; i++)
		{
			// search in the cosine streched region common for all the images
            for(size_t j = xCSmin; j < xCSmax; ++j)
			{
				double value = DIRECT_A2D_ELEM(labelCoordiantesMapSlice, i, j);
				
				if (value > 0)
				{			
					DIRECT_NZYX_ELEM(labelCoordiantesMap, k, 0, i, j) = value;
				}
			}
		}

		#ifdef DEBUG_HCC
		std::cout << "Colour: " << colour << std::endl;
		#endif

		std::vector<std::vector<int>> coordinatesPerLabelX (colour);
		std::vector<std::vector<int>> coordinatesPerLabelY (colour);

		for(size_t i = 0; i < ySize; i++)
		{
			// search in the cosine streched region common for all the images
            for(size_t j = xCSmin; j < xCSmax; ++j)
			{
				int value = DIRECT_A2D_ELEM(labelCoordiantesMapSlice, i, j);

				if(value!=0)
				{
					coordinatesPerLabelX[value-1].push_back(j);
					coordinatesPerLabelY[value-1].push_back(i);
				}
			}
		}

		#ifdef DEBUG_HCC
		int numberOfNewPeakedCoordinates = 0;
		#endif

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

				#ifdef DEBUG_HCC
				numberOfNewPeakedCoordinates += 1;
				#endif
			
			}
		}

		#ifdef DEBUG_HCC
		std::cout << "Number of coordinates added: " << numberOfNewPeakedCoordinates <<std::endl;
		std::cout << "Accumulated number of coordinates: " << coordinates3D.size() <<std::endl;
		#endif

    }

	#ifdef VERBOSE_OUTPUT
	std::cout << "Number of peaked coordinates: " << coordinates3D.size() << std::endl;
	#endif

	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindex = fnOut.find_last_of("\\/");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameLabeledVolume;
    outputFileNameLabeledVolume = rawname + "/ts_labeled.mrcs";

	Image<double> saveImage;
	saveImage() = labelCoordiantesMap; 
	saveImage.write(outputFileNameLabeledVolume);
	#endif

	#ifdef VERBOSE_OUTPUT
	std::cout << "High contrast coordinates picked succesfully!" << std::endl;
	#endif
}


void ProgTomoDetectMisalignmentTrajectory::detectLandmarkChains()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Detecting landmark chains..." << std::endl;
	#endif

	std::vector<int> counterLinesOfLandmarkAppearance(ySize);

	// Calculate the number of landmarks per row (y index)
	for(size_t i = 0; i < ySize; i++)
	{
		for(int j = 0; j < coordinates3D.size(); j++)
		{
			if(coordinates3D[j].y == i)
			{
				counterLinesOfLandmarkAppearance[i] += 1;
			}

			else if (coordinates3D[j-1].y == i && j-1 > 0)
			{
				counterLinesOfLandmarkAppearance[i] += 1;
			}

			else if (coordinates3D[j+1].y == i && j+1 < coordinates3D.size())
			{
				counterLinesOfLandmarkAppearance[i] += 1;
			}
		}
	}

	// Calculate poisson lambda
	int numberEmptyRows = std::count(counterLinesOfLandmarkAppearance.begin(), counterLinesOfLandmarkAppearance.end(), 0);
	std::vector<int> histogramOfLandmarkAppearanceSorted (counterLinesOfLandmarkAppearance.size()-numberEmptyRows); 

	size_t sideIndex = 0;
	for (size_t i = 0; i < counterLinesOfLandmarkAppearance.size(); i++)
	{
		if(counterLinesOfLandmarkAppearance[i]!=0)
		{
			histogramOfLandmarkAppearanceSorted[sideIndex] = counterLinesOfLandmarkAppearance[i];
			sideIndex += 1;
		}
	}
	

	// *** TODO: optimize, get n maxima elements without sorting
	sort(histogramOfLandmarkAppearanceSorted.begin(), histogramOfLandmarkAppearanceSorted.end(), std::greater<int>());

	// Poisson lambda
	std::cout <<  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" <<std::endl;
	for (size_t p = 0; p < counterLinesOfLandmarkAppearance.size(); p++)
	{
		std::cout <<  counterLinesOfLandmarkAppearance[p] <<std::endl;
	}


	std::cout <<  "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB" <<std::endl;


	for (size_t p = 0; p < histogramOfLandmarkAppearanceSorted.size(); p++)
	{
		std::cout <<  histogramOfLandmarkAppearanceSorted[p] <<std::endl;
	}


	float absolutePossionPercetile = histogramOfLandmarkAppearanceSorted.size()*poissonLandmarkPercentile;
	std::cout << absolutePossionPercetile <<std::endl;
	std::cout << (int)absolutePossionPercetile <<std::endl;

	float poissonAverage = histogramOfLandmarkAppearanceSorted[(int)absolutePossionPercetile];
	
	std::vector<size_t> chainIndexesY;

	// Test possion probability
	for (size_t i = 0; i < counterLinesOfLandmarkAppearance.size(); i++)
	{
		// Normalize the input values (make lambda=100 and make k=100(k'/lambda)) to fix a threshold value for the distribution.
		if (testPoissonDistribution(100*(poissonAverage/poissonAverage), 100*(counterLinesOfLandmarkAppearance[i]/poissonAverage)) > 0.001)
		{
			#ifdef DEBUG_POISSON
			std::cout << "Index " << i << " added with testPoissonDistribution=" << testPoissonDistribution(100*(poissonAverage/poissonAverage), 
						 100*(counterLinesOfLandmarkAppearance[i]/poissonAverage)) << std::endl;
			#endif

			chainIndexesY.push_back(i);
		}
	}

	globalAlignment = detectGlobalAlignmentPoisson(counterLinesOfLandmarkAppearance, chainIndexesY);

	#ifdef DEBUG_CHAINS
	std::cout << "chainIndexesY.size()=" << chainIndexesY.size() << std::endl;
	#endif

	// Compose and cluster chains
	chain2dMap.initZeros(ySize, xSize);

	#ifdef DEBUG_OUTPUT_FILES
	MultidimArray<int> clustered2dMap;
	clustered2dMap.initZeros(ySize, xSize);
	#endif

	for (size_t i = 0; i < chainIndexesY.size(); i++)
	{
		// Compose chains
		size_t chainIndexY = chainIndexesY[i];

		#ifdef DEBUG_CHAINS
		std::cout << "-----------------COMPOSING LINE " << i << "-----------------" << std::endl;
		#endif

		// Binary vector with one's in the x coordinates belonging to each y coordinate
		std::vector<size_t> chainLineY(xSize, 0);

		// Vector containing the angles of the selected coordinates
		std::vector<float> chainLineYAngles(xSize, 0);
		
		for (size_t j = 0; j < chainLineY.size() ; j++)
		{
			for(int x = 0; x < coordinates3D.size(); x++)
			{
				Point3D<double> coordinate3D = coordinates3D[x];

				if(coordinate3D.y == chainIndexY)
				{
					chainLineY[coordinate3D.x] = 1;
					
					if(abs(tiltAngles[coordinate3D.z])>abs(chainLineYAngles[coordinate3D.x]))
					{
						chainLineYAngles[coordinate3D.x] = tiltAngles[coordinate3D.z];
					}
				}

				else if(coordinate3D.y == chainIndexY-1 && x-1 > 0)
				{
					chainLineY[coordinate3D.x] = 1;
					
					if(abs(tiltAngles[coordinate3D.z])>abs(chainLineYAngles[coordinate3D.x]))
					{
						chainLineYAngles[coordinate3D.x] = tiltAngles[coordinate3D.z];
					}
				}

				else if(coordinate3D.y == chainIndexY+1 && x+1 < coordinates3D.size())
				{
					chainLineY[coordinate3D.x] = 1;

					if(abs(tiltAngles[coordinate3D.z])>abs(chainLineYAngles[coordinate3D.x]))
					{
						chainLineYAngles[coordinate3D.x] = tiltAngles[coordinate3D.z];
					}
				}	
			}
		}

		// Cluser chains
		std::vector<size_t> clusteredChainLineY(xSize, 0);
		size_t clusterId = 2;
		size_t clusterIdSize = 0;
		std::vector<size_t> clusterSizeVector;
		int landmarkDisplacementThreshold;

		#ifdef DEBUG_CHAINS
		std::cout << "-----------------CLUSTERING LINE " << i << "-----------------" << std::endl;
		#endif

		for (size_t j = 0; j < chainLineY.size(); j++)
		{
			if(chainLineY[j] != 0){

				// Check angle range to calculate landmarkDisplacementThreshold does not go further that the size of the image
				if (chainLineYAngles[j]+thrNumberDistanceAngleChain*tiltAngleStep > tiltAngles[tiltAngles.size()-1])
				{
					#ifdef DEBUG_CHAINS
					std::cout << chainLineYAngles[j] << "calculateLandmarkProjectionDiplacement DEFAULT"<< std::endl;
					#endif
					landmarkDisplacementThreshold = calculateLandmarkProjectionDiplacement(chainLineYAngles[j], tiltAngles[tiltAngles.size()-1], j); 
				}
				else
				{
					#ifdef DEBUG_CHAINS
					std::cout << chainLineYAngles[j] << "calculateLandmarkProjectionDiplacement CALCULATED"<< std::endl;
					#endif
					landmarkDisplacementThreshold = calculateLandmarkProjectionDiplacement(chainLineYAngles[j], chainLineYAngles[j]+thrNumberDistanceAngleChain*tiltAngleStep, j);
				}

				#ifdef DEBUG_CHAINS
				std::cout << "landmarkDisplacementThreshold=" << landmarkDisplacementThreshold << std::endl;
				#endif

				if(chainLineY[j]==1)
				{
					if(clusterIdSize > 0)
					{
						#ifdef DEBUG_CHAINS
						std::cout << "CASE: chainLineY[j]==1 --> clusterSizeVector.push_back " <<  clusterIdSize << std::endl;
						#endif

						clusterSizeVector.push_back(clusterIdSize);
						clusterIdSize = 0;
						clusterId += 1;
					}

					chainLineY[j] = clusterId;
					clusterIdSize += 1;

					for (size_t k = 1; k <= landmarkDisplacementThreshold; k++)
					{
						if(chainLineY[j+k]==1)
						{
							chainLineY[j+k] = clusterId;
							clusterIdSize += 1;
						}
					}
				}

				else
				{
					bool found = false;

					for (size_t k = 1; k <= landmarkDisplacementThreshold; k++)
					{
						// Check for new points added to the clusted
						if(chainLineY[j+k]==1)
						{
							chainLineY[j+k] = clusterId;
							clusterIdSize += 1;
							found = true;
						} 
						
						// Check for forward points already belonging to the cluster
						else if (chainLineY[j+k]!=0)
						{
							found = true;
						}
					}

					if (!found)
					{
						#ifdef DEBUG_CHAINS
						std::cout << "CASE: chainLineY[j]!=1 --> claverageusterSizeVector.push_back " <<  clusterIdSize << std::endl;
						#endif

						clusterSizeVector.push_back(clusterIdSize);
						clusterIdSize = 0;
						clusterId += 1;
					}
				}
			}
		}

		if(clusterIdSize>0)
		{
			#ifdef DEBUG_CHAINS
			std::cout << "CASE: clusterIdSize>0 --> clusterSizeVector.push_back " <<  clusterIdSize << std::endl;
			#endif

			clusterSizeVector.push_back(clusterIdSize);
			clusterIdSize = 0;
		}

		// Complete the overall 2D chain map
		#ifdef DEBUG_CHAINS
		std::cout << "clusterSizeVector.size()=" << clusterSizeVector.size() << std::endl; 

		for (size_t x = 0; x < clusterSizeVector.size(); x++)
		{
			std::cout << "clusterSizeVector[" << x << "]=" << clusterSizeVector[x] << std::endl;
		}
		#endif
		
		size_t lastClusterId=0, lastIndex=0, firstIndex=0;

		for (size_t j = 0; j < chainLineY.size(); j++)
		{	
			size_t clusterValue=chainLineY[j];

			if (clusterSizeVector[clusterValue-2] > numberOfElementsInChainThreshold && clusterValue != 0)
			{
				#ifdef DEBUG_CHAINS
				std::cout << "clusterValue=" << clusterValue << std::endl;
				std::cout << "lastClusterId=" << lastClusterId << std::endl;
				std::cout << "firstIndex=" << firstIndex << std::endl;
				std::cout << "lastIndex=" << lastIndex << std::endl;
				std::cout << "-----------------" << std::endl;
				#endif

				#ifdef DEBUG_OUTPUT_FILES
				DIRECT_A2D_ELEM(clustered2dMap, chainIndexY, j) = chainLineY[j];
				#endif
				
				// New label
				if (clusterValue != lastClusterId && lastClusterId != 0)
				{
					#ifdef DEBUG_CHAINS
					std::cout << "Chain with label " << chainLineY[j] << " and label size " << clusterSizeVector[chainLineY[j]-2] << " ADDED" << std::endl;
					std::cout << "First index: " << firstIndex << std::endl;
					std::cout << "Last index: " << lastIndex << std::endl;
					#endif

					for (size_t k = firstIndex; k < lastIndex+1; k++)
					{
						DIRECT_A2D_ELEM(chain2dMap, chainIndexY, k) = 1;
					}
					

					lastClusterId=clusterValue;
					firstIndex = j;
					lastIndex = j;
				}

				// Find first cluster
				else if (clusterValue != lastClusterId && lastClusterId == 0)
				{
					lastClusterId=clusterValue;
					firstIndex = j;
					lastIndex = j;
				}
				

				// Add elements to cluster
				else if (clusterValue==lastClusterId)
				{
					lastIndex = j;
				}
			}

			// Add last cluster
			if (firstIndex != lastIndex)
			{
				for (size_t k = firstIndex; k < lastIndex+1; k++)
				{
					DIRECT_A2D_ELEM(chain2dMap, chainIndexY, k) = 1;
				}
			}
			
		}
		
		clusterSizeVector.clear();
	}

	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindex = fnOut.find_last_of("\\/");
	std::string rawname = fnOut.substr(0, lastindex);

	std::string outputFileNameChain2dMap;
	std::string outputFileNameClustered2dMap;

    outputFileNameChain2dMap = rawname + "/ts_filteredChains.mrc";
    outputFileNameClustered2dMap = rawname + "/ts_clusteredChains.mrc";

	Image<int> saveImageBis;
	saveImageBis() = clustered2dMap;
	saveImageBis.write(outputFileNameClustered2dMap);
	
	Image<int> saveImage;
	saveImage() = chain2dMap;
	saveImage.write(outputFileNameChain2dMap);
	#endif

	#ifdef VERBOSE_OUTPUT
	std::cout << "Landmark chains detected succesfully!" << std::endl;
	#endif
}


void ProgTomoDetectMisalignmentTrajectory::detectMisalignedTiltImages()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Detecting misaligned tilt-images..." << std::endl;
	#endif

	std::vector<Point2D<double>> coordinatesInSlice;
	std::vector<size_t> lmOutRange(nSize, 0);

	for (size_t n = 0; n < nSize; n++)
	{
		// Calculate distances
		coordinatesInSlice = getCoordinatesInSlice(n);

		if(coordinatesInSlice.size() > 0)
		{

			#ifdef DEBUG_LOCAL_MISALI
			// Vector holding the distance of each landmark to its closest chain
			std::vector<double> vectorDistance;
			#endif

			for (size_t coord = 0; coord < coordinatesInSlice.size(); coord++)
			{
				Point2D<double> coord2D = coordinatesInSlice[coord];
				size_t matchCoordX = (size_t)-1; // Maximum possible size_t datatype
				size_t matchCoordY = (size_t)-1; // Maximum possible size_t datatype

				bool found = false;

				// Find distance to closest neighbour
				// *** optimizar: cada vez que se aumenta la distancia se revisitan los pixeles ya comprobados en distancias menores
				for (int distance = 1; distance < thrChainDistancePx; distance++)
				{
					for (int i = -distance; i < distance; i++)
					{
						for (int j = -(distance - abs(i)); j <= (distance - abs(i)); j++)
						{
							if (j + coord2D.y > 0 && i + coord2D.x  > 0 && j + coord2D.y < ySize && i + coord2D.x < xSize )
							{
								if ((abs(j)+abs(i) == distance) && (DIRECT_A2D_ELEM(chain2dMap, (int)(j + coord2D.y), (int)(i + coord2D.x)) != 0))
								{
									if(std::min(matchCoordX, matchCoordY) > std::min(i, j))
									{
										#ifdef DEBUG_LOCAL_MISALI
										//std::cout << "Found!! (" <<j<<"+"<<coord2D.y<<", "<<i<<"+"<<coord2D.x<<", "<< n << ")" << std::endl;
										#endif

										found = true;
										matchCoordX = i;
										matchCoordX = j;

										// Here we could break the loop but we do not to get the minimum distance to a chain (as a measurement of quality)*** no estoy seguro de esto
										// break;
									}
								}
							}
						}
					}
					
					if(found)
					{
						break;
					}
				}

				if(!found)
				{
					#ifdef DEBUG_LOCAL_MISALI
					//std::cout << "Not found!! (" <<coord2D.y<<", "<<coord2D.x<<", "<< n << ")" << std::endl;
					#endif

					vectorDistance.push_back(0);
					lmOutRange[n] += 1;
				}
				#ifdef DEBUG_LOCAL_MISALI
				else
				{
					vectorDistance.push_back(sqrt(matchCoordX*matchCoordX + matchCoordY*matchCoordY));
				}
				#endif
			}

			#ifdef DEBUG_LOCAL_MISALI
			for (size_t i = 0; i < vectorDistance.size(); i++)
			{
				std::cout << vectorDistance[i] << "  ";
			}
			
			std::cout << "\nlmOutRange[" << n << "]=" << lmOutRange[n] << "/" << coordinatesInSlice.size() << "=" << 
			float(lmOutRange[n])/float(coordinatesInSlice.size()) << "\n"<< std::endl;
			#endif
		}
		else
		{
			#ifdef VERBOSE_OUTPUT
			std::cout << "No landmarks detected in slice " << n << ". IMPOSSIBLE TO DETECT POTENTIAL MISALIGNMENT IN THIS IMAGE." << std::endl;
			#endif

			lmOutRange[n] = 0;
			#ifdef DEBUG_LOCAL_MISALI
			std::cout << "lmOutRange[" << n << "]=" << lmOutRange[n] << "\n"<< std::endl;
			#endif
		}
	}

	// Detect misalignment
	double sum = 0, sum2 = 0;
	int Nelems = 0;
	double m = 0;
	double sd = 0;

	for(size_t n = 0; n < nSize; n++)
	{
		int value = lmOutRange[n];
		sum += value;
		sum2 += value*value;
		++Nelems;
	}

	m = sum / nSize;
	sd = sqrt(sum2/Nelems - m*m);

	for(size_t n = 0; n < nSize; n++)
	{
		if(lmOutRange[n] > (m + 3*sd))
		{
			localAlignment[n] = false;

			#ifdef VERBOSE_OUTPUT
			std::cout << "MISALIGNMENT DETECTED IN IMAGE " << n << std::endl;
			#endif
		}
	}

	#ifdef VERBOSE_OUTPUT
	std::cout << "Misalignment in tilt-images succesfully detected!" << std::endl;
	#endif
}


/**void ProgTomoDetectMisalignmentTrajectory::calculateResidualVectors(MetaDataVec inputCoordMd)
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Calculating residual vectors" << std::endl;
	#endif

	size_t objId;
	size_t maxIndex;
	double tiltAngle;
	double distance;
	double maxDistance;

	int goldBeadX, goldBeadY, goldBeadZ;

	Matrix2D<double> projectionMatrix;
	Matrix1D<double> goldBead3d;
	Matrix1D<double> projectedGoldBead;

	std::vector<Matrix1D<double>> coordinatesInSlice;
	std::vector<Matrix1D<double>> projectedGoldBeads;

	Matrix2D<double> A_alignment;
	Matrix1D<double> T_alignment;
	Matrix2D<double> invW_alignment;
	Matrix2D<double> alignment_matrix;


	goldBead3d.initZeros(3);

	// Iterate through every tilt-image
	for(size_t n = 0; n<tiltAngles.size(); n++)
	{	
		tiltAngle = tiltAngles[n];

		# ifdef DEBUG
		std::cout << "Calculating residual vectors at slice " << n << " with tilt angle " << tiltAngle << "º" << std::endl;
		#endif

		coordinatesInSlice = getCoordinatesInSlice(n);

		Matrix2D<double> projectionMatrix = getProjectionMatrix(tiltAngle);

		#ifdef DEBUG_RESID
		std::cout << "Projection matrix------------------------------------"<<std::endl;
		std::cout << MAT_ELEM(projectionMatrix, 0, 0) << " " << MAT_ELEM(projectionMatrix, 0, 1) << " " << MAT_ELEM(projectionMatrix, 0, 2) << std::endl;
		std::cout << MAT_ELEM(projectionMatrix, 1, 0) << " " << MAT_ELEM(projectionMatrix, 1, 1) << " " << MAT_ELEM(projectionMatrix, 1, 2) << std::endl;
		std::cout << MAT_ELEM(projectionMatrix, 2, 0) << " " << MAT_ELEM(projectionMatrix, 2, 1) << " " << MAT_ELEM(projectionMatrix, 2, 2) << std::endl;
		std::cout << "------------------------------------"<<std::endl;
		#endif 

		// Iterate through every input 3d gold bead coordinate and project it onto the tilt image
		for(size_t objId : inputCoordMd.ids())
		{
			maxDistance = MAXDOUBLE;

			inputCoordMd.getValue(MDL_XCOOR, goldBeadX, objId);
			inputCoordMd.getValue(MDL_YCOOR, goldBeadY, objId);
			inputCoordMd.getValue(MDL_ZCOOR, goldBeadZ, objId);

			XX(goldBead3d) = (double) goldBeadX;
			YY(goldBead3d) = (double) goldBeadY;
			ZZ(goldBead3d) = (double) goldBeadZ;

			projectedGoldBead = projectionMatrix * goldBead3d;

			projectedGoldBeads.push_back(projectedGoldBead);
		}

		std::vector<size_t> randomIndexes = getRandomIndexes(projectedGoldBead.size());

		for(size_t i = 0; i < coordinatesInSlice.size(); i ++)
		{
			for(size_t j = 0; j < coordinatesInSlice.size(); j ++)
			{
				for(size_t k = 0; k < coordinatesInSlice.size(); k ++)
				{
					// def_affinity(XX(projectedGoldBeads[randomIndexes[0]]),
					// 			 YY(projectedGoldBeads[randomIndexes[0]]),
					// 			 XX(projectedGoldBeads[randomIndexes[1]]),
					// 			 YY(projectedGoldBeads[randomIndexes[1]]),
					// 			 XX(projectedGoldBeads[randomIndexes[2]]),
					// 			 YY(projectedGoldBeads[randomIndexes[2]]),
					// 			 XX(coordinatesInSlice[i]),
					// 			 YY(coordinatesInSlice[i]),
					// 			 XX(coordinatesInSlice[j]),
					// 			 YY(coordinatesInSlice[j]),
					// 			 XX(coordinatesInSlice[k]),
					// 			 YY(coordinatesInSlice[k]),
					// 			 A_alignment,
					// 			 T_alignment,
					// 			 invW_alignment)

					MAT_ELEM(alignment_matrix, 0, 0) = MAT_ELEM(A_alignment, 0, 0);
					MAT_ELEM(alignment_matrix, 0, 1) = MAT_ELEM(A_alignment, 0, 1);
					MAT_ELEM(alignment_matrix, 1, 0) = MAT_ELEM(A_alignment, 1, 0);
					MAT_ELEM(alignment_matrix, 1, 1) = MAT_ELEM(A_alignment, 1, 1);
					MAT_ELEM(alignment_matrix, 0, 2) = XX(T_alignment);
					MAT_ELEM(alignment_matrix, 1, 2) = YY(T_alignment);
					MAT_ELEM(alignment_matrix, 2, 0) = 0;
					MAT_ELEM(alignment_matrix, 2, 1) = 0;
					MAT_ELEM(alignment_matrix, 2, 2) = 1;
				}
			}
		}

			// #ifdef DEBUG_RESID
			// std::cout << XX(goldBead3d) << " " << YY(goldBead3d) << " " << ZZ(goldBead3d) << std::endl;
			// std::cout << XX(projectedGoldBead) << " " << YY(projectedGoldBead) << " " << ZZ(projectedGoldBead) << std::endl;
			// std::cout << "------------------------------------"<<std::endl;
			// #endif

			// // Iterate though every coordinate in the tilt-image and calculate the maximum distance
			// for(size_t i = 0; i < coordinatesInSlice.size(); i++)
			// {
			// 	distance = abs(XX(projectedGoldBead) - XX(coordinatesInSlice[i])) + abs(YY(projectedGoldBead) - YY(coordinatesInSlice[i]));

			// 	if(maxDistance > distance)
			// 	{
			// 		maxDistance = distance;
			// 		maxIndex = i;
			// 	}
			// }
			
			residualX.push_back(XX(coordinatesInSlice[maxIndex]) - XX(projectedGoldBead));
			residualY.push_back(YY(coordinatesInSlice[maxIndex]) - YY(projectedGoldBead));
			residualCoordinateX.push_back(XX(projectedGoldBead));
			residualCoordinateY.push_back(YY(projectedGoldBead));
			residualCoordinateZ.push_back(n);

	}

	#ifdef VERBOSE_OUTPUT
	std::cout << "Residual vectors calculated: " << residualX.size() << std::endl;
	#endif
}*/


// --------------------------- I/O functions ----------------------------

void ProgTomoDetectMisalignmentTrajectory::writeOutputAlignmentReport()
{
	size_t lastindexInputTS = fnVol.find_last_of(":");
	std::string rawnameTS = fnVol.substr(0, lastindexInputTS);
	
	MetaDataVec md;
	FileName fn;
	size_t id;


	if(!globalAlignment)
	{
		for(size_t i = 0; i < nSize; i++)
		{
			fn.compose(i + FIRST_IMAGE, rawnameTS);
			id = md.addObject();
			
			md.setValue(MDL_ENABLED, -1, id);
			md.setValue(MDL_IMAGE, fn, id);
		}
	}
	else
	{
		for(size_t i = 0; i < localAlignment.size(); i++)
		{
			fn.compose(i + FIRST_IMAGE, rawnameTS);
			id = md.addObject();

			if(localAlignment[i])
			{
				md.setValue(MDL_ENABLED, 1, id);
			}
			else
			{
				md.setValue(MDL_ENABLED, -1, id);
			}

			md.setValue(MDL_IMAGE, fn, id);
		}
	}

	md.write(fnOut);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Alignment report saved at: " << fnOut << std::endl;
	#endif
}

void ProgTomoDetectMisalignmentTrajectory::writeOutputCoordinates()
{
	size_t lastindex = fnOut.find_last_of("\\/");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameLandmarkCoordinates;
    outputFileNameLandmarkCoordinates = rawname + "/ts_landmarkCoordinates.xmd";

	MetaDataVec md;
	size_t id;

	for(size_t i = 0; i < coordinates3D.size(); i++)
	{
		id = md.addObject();
		md.setValue(MDL_XCOOR, (int)coordinates3D[i].x, id);
		md.setValue(MDL_YCOOR, (int)coordinates3D[i].y, id);
		md.setValue(MDL_ZCOOR, (int)coordinates3D[i].z, id);
	}


	md.write(outputFileNameLandmarkCoordinates);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Output coordinates metadata saved at: " << outputFileNameLandmarkCoordinates << std::endl;
	#endif

}


void ProgTomoDetectMisalignmentTrajectory::writeOutputResidualVectors()
{
	MetaDataVec md;
	size_t id;

	for(size_t i = 0; i < residualX.size(); i++)
	{
		id = md.addObject();
		md.setValue(MDL_X, residualX[i], id);
		md.setValue(MDL_Y, residualY[i], id);
		md.setValue(MDL_XCOOR, residualCoordinateX[i], id);
		md.setValue(MDL_YCOOR, residualCoordinateY[i], id);
		md.setValue(MDL_ZCOOR, residualCoordinateZ[i], id);
	}

	size_t lastindex = fnOut.find_last_of("\\/");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string fnOutResiduals;
    fnOutResiduals = rawname + "/residuals2d.xmd";

	md.write(fnOutResiduals);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Residuals metadata saved at: " << fnOutResiduals << std::endl;
	#endif

}


// --------------------------- MAIN ----------------------------------

void ProgTomoDetectMisalignmentTrajectory::run()
{
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;	

	auto t1 = high_resolution_clock::now();

	std::cout << "Starting..." << std::endl;

	size_t Xdim, Ydim;

	MetaDataVec tiltseriesmd;
    ImageGeneric tiltSeriesImages;

    if (fnVol.isMetaData())
    {
        tiltseriesmd.read(fnVol);
    }
    else
    {
        tiltSeriesImages.read(fnVol, HEADER);

        size_t Zdim, Ndim;
        tiltSeriesImages.getDimensions(Xdim, Ydim, Zdim, Ndim);

        if (fnVol.getExtension() == "mrc" and Ndim == 1)
            Ndim = Zdim;

        size_t id;
        FileName fn;
        for (size_t i = 0; i < Ndim; i++) 
        {
            id = tiltseriesmd.addObject();
            fn.compose(i + FIRST_IMAGE, fnVol);
            tiltseriesmd.setValue(MDL_IMAGE, fn, id);
        }
    }

	tiltSeriesImages.getDimensions(xSize, ySize, zSize, nSize);

	//#ifdef DEBUG_DIM
	std::cout << "Input tilt-series dimensions:" << std::endl;
	std::cout << "x " << xSize << std::endl;
	std::cout << "y " << ySize << std::endl;
	std::cout << "z " << zSize << std::endl;
	std::cout << "n " << nSize << std::endl;
	//#endif

	generateSideInfo();

	FileName fnTSimg;
	size_t objId, objId_ts;
	Image<double> imgTS;

	MultidimArray<double> &ptrImg = imgTS();
    MultidimArray<double> projImgTS;
    MultidimArray<double> filteredImg;
    MultidimArray<double> freqMap;

	projImgTS.initZeros(Ydim, Xdim);

	size_t Ndim, counter = 0;
	Ndim = tiltseriesmd.size();

	MultidimArray<double> filteredTiltSeries;
	filteredTiltSeries.initZeros(Ndim, 1, Ydim, Xdim);

	for(size_t objId : tiltseriesmd.ids())
	{
		tiltseriesmd.getValue(MDL_IMAGE, fnTSimg, objId);

		#ifdef DEBUG_PREPROCESS
        std::cout << "Preprocessing slice: " << fnTSimg << std::endl;
		#endif

        imgTS.read(fnTSimg);

        bandPassFilter(ptrImg);

        for (size_t i = 0; i < Ydim; ++i)
        {
            for (size_t j = 0; j < Xdim; ++j)
            {
				DIRECT_NZYX_ELEM(filteredTiltSeries, counter, 0, i, j) = DIRECT_A2D_ELEM(ptrImg, i, j);
			}
		}

		counter++;
	}
	
	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindex = fnOut.find_last_of("\\/");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameFilteredVolume;
    outputFileNameFilteredVolume = rawname + "/ts_filtered.mrcs";

	Image<double> saveImage;
	saveImage() = filteredTiltSeries;
	saveImage.write(outputFileNameFilteredVolume);
	#endif

	#ifdef DEBUG_DIM
	std::cout << "Filtered tilt-series dimensions:" << std::endl;
	std::cout << "x " << xSize << std::endl;
	std::cout << "y " << ySize << std::endl;
	std::cout << "z " << zSize << std::endl;
	std::cout << "n " << nSize << std::endl;
	#endif

	getHighContrastCoordinates(filteredTiltSeries);

	MultidimArray<int> proyectedCoordinates;
	proyectedCoordinates.initZeros(ySize, xSize);

	for(size_t n; n < coordinates3D.size(); n++)
	{
		DIRECT_A2D_ELEM(proyectedCoordinates, (int)coordinates3D[n].y, (int)coordinates3D[n].x) = 1;
	}

	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindexBis = fnOut.find_last_of("\\/");
	std::string rawnameBis = fnOut.substr(0, lastindexBis);
	std::string outputFileNameFilteredVolumeBis;
    outputFileNameFilteredVolumeBis = rawnameBis + "/ts_proyected.mrc";

	Image<int> saveImageBis;
	saveImageBis() = proyectedCoordinates;
	saveImageBis.write(outputFileNameFilteredVolumeBis);
	#endif

	#ifdef DEBUG_OUTPUT_FILES
	writeOutputCoordinates();
	#endif

	detectLandmarkChains();
	if(globalAlignment){
		detectMisalignedTiltImages();
	}

	writeOutputAlignmentReport();

	// if(checkInputCoord)
	// {
	// 	MetaDataVec inputCoordMd;
	// 	inputCoordMd.read(fnInputCoord);

	// 	calculateResidualVectors(inputCoordMd);
	// 	writeOutputResidualVectors();
	// }
	
	auto t2 = high_resolution_clock::now();
	/* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
}


// --------------------------- UTILS functions ----------------------------


bool ProgTomoDetectMisalignmentTrajectory::filterLabeledRegions(std::vector<int> coordinatesPerLabelX, std::vector<int> coordinatesPerLabelY, double centroX, double centroY)
{
	// Check number of elements of the label
	if(coordinatesPerLabelX.size() < thrNumberCoords)
	{
		return false;
	}

	// Check spehricity of the label
	double maxSquareDistance = 0;
	double distance;

	#ifdef DEBUG_FILTERLABEL
	size_t debugN;
	#endif

	for(size_t n = 0; n < coordinatesPerLabelX.size(); n++)
	{
		distance = (coordinatesPerLabelX[n]-centroX)*(coordinatesPerLabelX[n]-centroX)+(coordinatesPerLabelY[n]-centroY)*(coordinatesPerLabelY[n]-centroY);

		if(distance > maxSquareDistance)
		{
			#ifdef DEBUG_FILTERLABEL
			debugN = n;
			#endif

			maxSquareDistance = distance;
		}
	}

	double maxDistace;
	maxDistace = sqrt(maxSquareDistance);
	
	double area;
	double ocupation;

	area = PI * (maxDistace * maxDistace);

	ocupation = 0.0 + (double)coordinatesPerLabelX.size();
	ocupation = ocupation  / area;

	#ifdef DEBUG_FILTERLABEL
	std::cout << "x max distance " << coordinatesPerLabelX[debugN] << std::endl;
	std::cout << "y max distance " << coordinatesPerLabelY[debugN] << std::endl;
	std::cout << "centroX " << centroX << std::endl;
	std::cout << "centroY " << centroY << std::endl;
	std::cout << "area " << area << std::endl;
	std::cout << "maxDistace " << maxDistace << std::endl;
	std::cout << "ocupation " << ocupation << std::endl;
	#endif

	if(ocupation > 0.65)
	{
		return true;
	}
	if(ocupation <= 0.65)
	{
		return false;
	}
}

// bool ProgTomoDetectMisalignmentTrajectory::filterLabeledRegions(std::vector<std::vector<int>> coordinatesPerLabelX, 
// 																std::vector<std::vector<int>> coordinatesPerLabelY)
// {
// 	std::vector<int> coordsX;
// 	std::vector<int> coordsY;

// 	std::vector<float> sphericityOccupationVector (coordinatesPerLabelX.size());



// 	for (size_t i = 0; i < coordinatesPerLabelX.size(); i++)
// 	{
// 		coordsX = coordinatesPerLabelX[i];
// 		coordsY = coordinatesPerLabelY[i];

// 		// Only consider coordinates with enough number of elements
// 		if(coordsX.size() < thrNumberCoords)
// 		{
// 			// Calculate the center of mass of the label			
// 			numberOfCoordinatesPerValue =  coordsX.size();

// 			int xCoor = 0;
// 			int yCoor = 0;

// 			for(size_t coordinate=0; coordinate < coordsX.size(); coordinate++)
// 			{
// 				xCoor += coordsX[value][coordinate];
// 				yCoor += coordsX[value][coordinate];
// 			}

// 			double xCoorCM = xCoor/numberOfCoordinatesPerValue;
// 			double yCoorCM = yCoor/numberOfCoordinatesPerValue;

// 			// Calculate sphericity of the label
// 			double maxSquareDistance = 0;
// 			double distance;

// 			#ifdef DEBUG_FILTERLABEL
// 			size_t debugN;
// 			#endif

// 			for(size_t n = 0; n < coordsX.size(); n++)
// 			{
// 				distance = (coordsX[n]-xCoorCM)*(coordsX[n]-xCoorCM)+(coordsY[n]-yCoorCM)*(coordsY[n]-yCoorCM);

// 				if(distance > maxSquareDistance)
// 				{
// 					#ifdef DEBUG_FILTERLABEL
// 					debugN = n;
// 					#endif

// 					maxSquareDistance = distance;
// 				}
// 			}

// 			double maxDistace;
// 			maxDistace = sqrt(maxSquareDistance);
			
// 			double area;
// 			double ocupation;

// 			area = PI * (maxDistace * maxDistace);

// 			ocupation = 0.0 + (double)coordinatesPerLabelX.size();
// 			ocupation = ocupation  / area;

// 			#ifdef DEBUG_FILTERLABEL
// 			std::cout << "x max distance " << coordinatesPerLabelX[debugN] << std::endl;
// 			std::cout << "y max distance " << coordinatesPerLabelY[debugN] << std::endl;
// 			std::cout << "xCoorCM " << xCoorCM << std::endl;
// 			std::cout << "yCoorCM " << yCoorCM << std::endl;
// 			std::cout << "area " << area << std::endl;
// 			std::cout << "maxDistace " << maxDistace << std::endl;
// 			std::cout << "ocupation " << ocupation << std::endl;
// 			#endif

// 			sphericityOccupationVector.push_back(ocupation);
// 		}
// 		else
// 		{
// 			sphericityOccupationVector.push_back(0);
// 		}
// 	}

// 	std::vector<int> sphericityOccupationVectorSorted;
// 	sphericityOccupationVectorSorted = sphericityOccupationVector;

// 	// *** TODO: optimize, get n maxima elements without sorting
// 	sort(sphericityOccupationVectorSorted.begin(), sphericityOccupationVectorSorted.end(), std::greater<int>());

// 	// *** TODO: convert in input variable
// 	int numberOfLandmarksThr = 20;
// 	float sphericityOccupationThr = sphericityOccupationVectorSorted[numberOfLandmarksThr];

// 	for (size_t i = 0; i < sphericityOccupationVector.size(); i++)
// 	{
// 		if(sphericityOccupationVector[i] > sphericityOccupationThr)
// 		{
// 			// Add coordinate
// 			return true;
// 		}
// 		if(sphericityOccupationVector[i] <= sphericityOccupationThr)
// 		{
// 			// Remove coordinate
// 			return false;
// 		}
// 	}
// }


bool ProgTomoDetectMisalignmentTrajectory::detectGlobalAlignmentPoisson(std::vector<int> counterLinesOfLandmarkAppearance, std::vector<size_t> chainIndexesY)
{
	// float totalLM = coordinates3D.size();
	float totalLM = 0;
	float totalChainLM = 0;
	float totalIndexes = chainIndexesY.size();
	float top10LM = 0;

	for (size_t i = 0; i < counterLinesOfLandmarkAppearance.size(); i++)
	{
		totalLM += counterLinesOfLandmarkAppearance[i];
	}

	for (size_t i = 0; i < totalIndexes; i++)
	{
		totalChainLM += counterLinesOfLandmarkAppearance[(int)chainIndexesY[i]];
	}

	sort(counterLinesOfLandmarkAppearance.begin(), counterLinesOfLandmarkAppearance.end(), std::greater<int>());

	for (size_t i = 0; i < 10; i++)
	{
		top10LM += counterLinesOfLandmarkAppearance[i];
	}

	// Thresholds calculation
	float top10Chain = 100 * (top10LM / totalLM); 								// Compare to thrTop10Chain
	float lmChain = 100 * (totalChainLM / (totalLM)); 	// Compare to thrLMChain

	// Thresholds comparison
	bool top10ChainBool = top10Chain < thrTop10Chain;
	bool lmChainBool = lmChain < thrLMChain;

	#ifdef DEBUG_GLOBAL_MISALI
	std::cout << "Global misalignment detection parameters:" << std::endl;
	std::cout << "Total number of landmarks: " << totalLM << std::endl;
	std::cout << "Total number of landmarks belonging to the selected chains: " << totalChainLM << std::endl;
	std::cout << "Total number of landmarks belonging to the top 10 most populated indexes: " << top10LM << std::endl;

	std::cout << "Precentage of LM belonging to the top 10 populated chains respecto to the total number of LM: " << top10Chain << std::endl;
	std::cout << "Percentage of number LM belonging to the selected chains respect to the number of populated lines and the total number of LM: " << lmChain << std::endl;

	std::cout << "Compare top10Chain < thrTop10Chain (" << top10Chain << "<" << thrTop10Chain << "): " << top10ChainBool << std::endl;
	std::cout << "Compare lmChain < thrLMChain (" << lmChain << "<" << thrLMChain << "): " << lmChainBool << std::endl;
	#endif

	if(top10ChainBool || lmChainBool)
	{
		#ifdef VERBOSE_OUTPUT
		std::cout << "GLOBAL MISALIGNMENT DETECTED IN TILT-SERIES" << std::endl;
		std::cout << "Compare top10Chain < thrTop10Chain (" << top10Chain << "<" << thrTop10Chain << "): " << top10ChainBool << std::endl;
		std::cout << "Compare lmChain < thrLMChain (" << lmChain << "<" << thrLMChain << "): " << lmChainBool << std::endl;
		#endif

		return false;
	}
	else
	{
		return true;
	}
}


Matrix2D<double> ProgTomoDetectMisalignmentTrajectory::getProjectionMatrix(double tiltAngle)
{
	double cosTiltAngle = cos(tiltAngle * PI/180.0);
	double sinTiltAngle = sin(tiltAngle * PI/180.0);

	Matrix2D<double> projectionMatrix(3,3);

	MAT_ELEM(projectionMatrix, 0, 0) = cosTiltAngle;
	// MAT_ELEM(projectionMatrix, 0, 1) = 0;
	MAT_ELEM(projectionMatrix, 0, 2) = sinTiltAngle;
	// MAT_ELEM(projectionMatrix, 1, 0) = 0;
	MAT_ELEM(projectionMatrix, 1, 1) = 1;
	// MAT_ELEM(projectionMatrix, 1, 2) = 0;
	MAT_ELEM(projectionMatrix, 2, 0) = -sinTiltAngle;
	// MAT_ELEM(projectionMatrix, 2, 1) = 0;
	MAT_ELEM(projectionMatrix, 2, 2) = cosTiltAngle;

	return projectionMatrix;
}


std::vector<Point2D<double>> ProgTomoDetectMisalignmentTrajectory::getCoordinatesInSlice(size_t slice)
{
	std::vector<Point2D<double>> coordinatesInSlice;
	Point2D<double> coordinate(0,0);

	for(size_t n = 0; n < coordinates3D.size(); n++)
	{
		if(slice == coordinates3D[n].z)
		{
			coordinate.x = coordinates3D[n].x;
			coordinate.y = coordinates3D[n].y;
			coordinatesInSlice.push_back(coordinate);
		}
	}

	return coordinatesInSlice;
}


std::vector<size_t> ProgTomoDetectMisalignmentTrajectory::getRandomIndexes(size_t size)
{
	std::vector<size_t> indexes;
	size_t randomIndex;

	randomIndex = rand() % size;

	indexes.push_back(randomIndex);

	while (indexes.size() != 3)
	{
		randomIndex = rand() % size;

		for(size_t n = 0; n < indexes.size(); n++)
		{
			if(indexes[n] != randomIndex)
			{
				indexes.push_back(randomIndex);
				break;
			}
		}
	}
	
	return indexes;
}


float ProgTomoDetectMisalignmentTrajectory::testPoissonDistribution(float lambda, size_t k)
{
	double quotient=1;

	#ifdef DEBUG_POISSON
	std::cout << "k="<< k <<std::endl;
	std::cout << "lambda="<< lambda <<std::endl;
	#endif

	// Since k! can not be holded we calculate the quotient lambda^k/k!= (lambda/k) * (lambda/(k-1)) * ... * (lambda/1)
	for (size_t i = 1; i < k+1; i++)
	{
		quotient *= lambda / i;
	}

	#ifdef DEBUG_POISSON
	std::cout << "quotient="<< quotient <<std::endl;
	std::cout << "quotient*exp(-lambda)="<< quotient*exp(-lambda) <<std::endl;
	#endif
	
	return quotient*exp(-lambda);

}


float ProgTomoDetectMisalignmentTrajectory::calculateLandmarkProjectionDiplacement(float theta1, float theta2, float coordinateProjX)
{
	float xCoor = coordinateProjX - xSize/2;

	#ifdef DEBUG_CHAINS
	std::cout << "coordinateProjX=" << coordinateProjX << std::endl;
	std::cout << "xCoor=" << xCoor << std::endl;
	std::cout << "theta1=" << theta1 << std::endl;
	std::cout << "theta2=" << theta2 << std::endl;	
	std::cout << "theta1 * PI/180.0=" << theta1 * PI/180.0 << std::endl;
	std::cout << "theta2 * PI/180.0=" << theta2 * PI/180.0 << std::endl;
	#endif

	float distance = abs(((cos(theta2 * PI/180.0)/cos(theta1 * PI/180.0))-1)*xCoor);

	if (distance<minDistancePx)
	{
		return (int)minDistancePx;
	}
	
	return (int)distance;
}

