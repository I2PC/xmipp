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
	fnOut = getParam("-o");
	fnTiltAngles = getParam("-inputTiltAngleFile");
	samplingRate = getDoubleParam("--samplingRate");
	fiducialSize = getDoubleParam("--fiducialSize");
	sdThreshold = getIntParam("--sdThreshold");
	numberOfCoordinatesThr = getIntParam("--numberOfCoordinatesThr");
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
	addParamsLine("  -inputTiltAngleFile <xmd_file=\"\">      				: Input file containning the tilt angles of the tilt-series in .xmd format.");
	addParamsLine("  [-o <output=\"coordinates3D.xmd\">]       				: Output file containing the 3D coordinates.");
	addParamsLine("  [--sdThreshold <sdThreshold=5>]      					: Number of SD a coordinate value must be over the mean to conisder that it belongs to a high contrast feature.");
  	addParamsLine("  [--numberOfCoordinatesThr <numberOfCoordinatesThr=10>]	: Minimum number of coordinates attracted to a center of mass to consider it.");
  	addParamsLine("  [--samplingRate <samplingRate=1>]						: Sampling rate of the input tomogram (A/px).");
	addParamsLine("  [--fiducialSize <fiducialSize=100>]					: Fiducial size in Angstroms (A).");
	addParamsLine("  [--inputCoord <output=\"\">]							: Input coordinates of the 3D landmarks to calculate the residual vectors.");
}


// --------------------------- HEAD functions ----------------------------

void ProgTomoDetectMisalignmentTrajectory::generateSideInfo()
{
	MetaData inputTiltAnglesMd;
	double tiltAngle;
	size_t objId;

	inputTiltAnglesMd.read(fnTiltAngles);

	tiltAngleStep=0;

	FOR_ALL_OBJECTS_IN_METADATA(inputTiltAnglesMd)
	{
		objId = __iter.objId;
		inputTiltAnglesMd.getValue(MDL_ANGLE_TILT, tiltAngle, objId);
		tiltAngles.push_back(tiltAngle);

		tiltAngleStep += tiltAngle;
	}

	tiltAngleStep /= tiltAngles.size();

}

void ProgTomoDetectMisalignmentTrajectory::bandPassFilter(MultidimArray<double> &inputTiltSeries) //*** tiltImage*
{
	FourierTransformer transformer1(FFTW_BACKWARD);
	MultidimArray<std::complex<double>> fftImg;
	transformer1.FourierTransform(inputTiltSeries, fftImg, true);

	double w = 0.03;

    double lowFreqFilt = samplingRate/(1.1*fiducialSize);
	double highFreqFilt = samplingRate/(0.9*fiducialSize);

	double tail_high = highFreqFilt + w;
    double tail_low = lowFreqFilt - w;

	double delta = PI / w;

    double uy, ux, u, uy2;

    size_t ydimImg = YSIZE(inputTiltSeries);
    size_t xdimImg = XSIZE(inputTiltSeries);

	long n=0;

	for(size_t i=0; i<YSIZE(fftImg); ++i)
	{
		FFT_IDX2DIGFREQ(i, ydimImg, uy);
		uy2=uy*uy;

		for(size_t j=0; j<XSIZE(fftImg); ++j)
		{
			FFT_IDX2DIGFREQ(j, xdimImg, ux);
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

	transformer1.inverseFourierTransform(fftImg, inputTiltSeries);
}


void ProgTomoDetectMisalignmentTrajectory::getHighContrastCoordinates(MultidimArray<double> tiltSeriesFiltered)
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Picking coordinates..." << std::endl;
	#endif

    MultidimArray<double> binaryCoordinatesMapSlice;
    MultidimArray<double> labelCoordiantesMapSlice;
    MultidimArray<double> labelCoordiantesMap;

	labelCoordiantesMap.initZeros(nSize, zSize, ySize, xSize);
	
	for(size_t k = 0; k < nSize; ++k)
	{
		std::vector<int> sliceVector;
		
		// Calculate threshold value for each image of the series
        for(size_t i = 0; i < ySize; ++i)
        {
            for(size_t j = 0; j < xSize; ++j)
            {
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

        double threshold = average - sdThreshold * standardDeviation;


        #ifdef VERBOSE_OUTPUT
		std::cout << "Slice: " << k+1 << " Average: " << average << " SD: " << standardDeviation << " Threshold: " << threshold << std::endl;
        #endif

		binaryCoordinatesMapSlice.initZeros(ySize, xSize);

		#ifdef DEBUG
		int numberOfPointsAddedBinaryMap = 0;
		#endif

		for(size_t i = 0; i < ySize; i++)
		{
			for(size_t j = 0; j < xSize; j++)
			{
				double value = DIRECT_A3D_ELEM(tiltSeriesFiltered, k, i, j);

				if (value < threshold)
				{
					DIRECT_A2D_ELEM(binaryCoordinatesMapSlice, i, j) = 1.0;
					
					#ifdef DEBUG
					numberOfPointsAddedBinaryMap += 1;
					#endif
				}
			}
		}

		#ifdef DEBUG
		std::cout << "Number of points in the binary map: " << numberOfPointsAddedBinaryMap << std::endl;
		#endif

		// The value 8 is the neighbourhood
		int colour = labelImage2D(binaryCoordinatesMapSlice, labelCoordiantesMapSlice, 8);

        for (size_t i = 0; i < ySize; ++i)
        {
            for (size_t j = 0; j < xSize; ++j)
            {
				double value = DIRECT_A2D_ELEM(labelCoordiantesMapSlice, i, j);
				
				if (value > 0)
				{			
					DIRECT_NZYX_ELEM(labelCoordiantesMap, k, 0, i, j) = value;
				}
			}
		}

		#ifdef DEBUG
		std::cout << "Colour: " << colour << std::endl;
		#endif

		std::vector<std::vector<int>> coordinatesPerLabelX (colour);
		std::vector<std::vector<int>> coordinatesPerLabelY (colour);


		for(size_t j = 0; j < xSize; j++)
		{
			for(size_t i = 0; i < ySize; i++)
			{
				int value = DIRECT_A2D_ELEM(labelCoordiantesMapSlice, i, j);

				if(value!=0)
				{
					coordinatesPerLabelX[value-1].push_back(j);
					coordinatesPerLabelY[value-1].push_back(i);
				}
			}
		}

		#ifdef DEBUG
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
				coordinates3Dx.push_back(xCoorCM);
				coordinates3Dy.push_back(yCoorCM);
				coordinates3Dn.push_back(k);

				#ifdef DEBUG
				numberOfNewPeakedCoordinates += 1;
				#endif
			
			}
		}

		#ifdef DEBUG
		std::cout << "Number of coordinates added: " << numberOfNewPeakedCoordinates <<std::endl;
		std::cout << "Accumulated number of coordinates: " << coordinates3Dx.size() <<std::endl;
		#endif

    }

	#ifdef VERBOSE_OUTPUT
	std::cout << "Number of peaked coordinates: " << coordinates3Dx.size() << std::endl;
	#endif

	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindex = fnOut.find_last_of(".");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameLabeledVolume;
    outputFileNameLabeledVolume = rawname + "_label.mrcs";

	Image<double> saveImage;
	saveImage() = labelCoordiantesMap; 
	saveImage.write(outputFileNameLabeledVolume);
	#endif
}


// void ProgTomoDetectMisalignmentTrajectory::detectLandmarkChains()
// {
// 	std::vector<size_t> counterLinesOfLandmarkAppearance(ySize);

// 	for(size_t i = 0; i < ySize; i++)
// 	{
// 		for(int j = 0; j < coordinates3Dy.size(); j++)
// 		{
// 			if(coordinates3Dy[j] == i)
// 			{
// 				counterLinesOfLandmarkAppearance[i] += 1;
// 			}

// 			if (coordinates3Dy[j-1] == i && j-1 > 0)
// 			{
// 				counterLinesOfLandmarkAppearance[i] += 1;
// 			}

// 			else if (coordinates3Dy[j+1] == i && j+1 < coordinates3Dy.size())
// 			{
// 				counterLinesOfLandmarkAppearance[i] += 1;
// 			}
// 		}
// 	}

// 	std::cout << "Test 1" << std::endl;

// 	// Take the average of the three most populated indexes as the expected value for a Possion distribution testing
// 	size_t numberOfIndexesForPossionAverage = 3;
// 	float poissonAverage = 0;

// 	std::vector<size_t> histogramOfLandmarkAppearanceSorted = counterLinesOfLandmarkAppearance;

// 	// *** TODO: optimize, get n maxima elements without sorting
// 	sort(histogramOfLandmarkAppearanceSorted.begin(), histogramOfLandmarkAppearanceSorted.end());
	
// 	for (size_t i = 0; i < numberOfIndexesForPossionAverage; i++)
// 	{
// 		poissonAverage += histogramOfLandmarkAppearanceSorted[histogramOfLandmarkAppearanceSorted.size()-i];
// 	}

// 	poissonAverage /= 3;

// 	std::vector<size_t> chainIndexesY;

// 	std::cout << "Test 2" << std::endl;


// 	// Test possion probability
// 	for (size_t i = 0; i < counterLinesOfLandmarkAppearance.size(); i++)
// 	{
// 		if (testPoissonDistribution(poissonAverage, counterLinesOfLandmarkAppearance[i])<0.01)
// 		{
// 			chainIndexesY.push_back(i);
// 		}
// 	}

// 	// SPLIT CHAINS ---> make function of this and interate in the loop
// 	MultidimArray<int> chain2dMap;
// 	chain2dMap.initZeros(ySize, xSize);

// 	std::cout << "Test 3" << std::endl;
// 	std::cout << "------------------------------------------------------------chainIndexesY.size()" << chainIndexesY.size() << std::endl;


// 	for (size_t i = 0; i < chainIndexesY.size(); i++)
// 	{
// 		size_t chainIndexY = chainIndexesY[i];

// 		// Binary vector with one's in the x coordinates belonging to each y coordinate
// 		std::vector<size_t> chainLineY(xSize, 0);

// 		// Vector containing the angles of the selected coordinates
// 		std::vector<size_t> chainLineYAngles(xSize, 0);
		
// 		for (size_t j = 0; j < chainLineY.size() ; j++)
// 		{
// 			for(int x = 0; x < coordinates3Dx.size(); x++)
// 			{
// 				if(coordinates3Dy[x] == chainIndexY)
// 				{
// 					chainLineY[coordinates3Dx[x]] = 1;
// 					chainLineYAngles[coordinates3Dx[x]] = tiltAngles[coordinates3Dn[x]];
// 				}		
// 				else if(x-1 > 0 && coordinates3Dy[x-1] == chainIndexY)
// 				{
// 					chainLineY[coordinates3Dx[x-1]] = 1;
// 					chainLineYAngles[coordinates3Dx[x-1]] = tiltAngles[coordinates3Dn[x-1]];

// 				}	
// 				else if(x+1 < coordinates3Dy.size() && coordinates3Dy[x+1] == chainIndexY)
// 				{
// 					chainLineY[coordinates3Dx[x+1]] = 1;
// 					chainLineYAngles[coordinates3Dx[x+1]] = tiltAngles[coordinates3Dn[x+1]];

// 				}	
// 			}
// 		}

// 		std::cout << "Test 4" << std::endl;


// 		// Cluser the chainLineY vector
// 		std::vector<size_t> clusteredChainLineY(xSize, 0);
// 		size_t clusterId = 1;
// 		size_t clusterIdSize = 0;
// 		std::vector<size_t> clusterSizeVector;

// 		// std::cout << "[" << std::endl;

// 		// for (size_t j = 0; j < chainLineY.size(); j++)
// 		// {
// 		// 	std::cout << chainLineY[j]  << std::endl ;
// 		// }
// 		// std::cout << "]" << std::endl;

// 		for (size_t j = 0; j < chainLineY.size(); j++)
// 		{
// 			if(chainLineY[j]==1 && clusteredChainLineY[j]==0)
// 			{
// 				clusteredChainLineY[j] = clusterId;
// 				clusterIdSize += 1;

// 				float landmarkDisplacementThreshold = calculateLandmarkProjectionDiplacement(chainLineYAngles[j], chainLineYAngles[j]+2*tiltAngleStep, j);

// 				for (size_t k = 1; k <= landmarkDisplacementThreshold; k++)
// 				{
// 					if(chainLineY[j+k]==1)
// 					{
// 						clusteredChainLineY[j+k] = clusterId;
// 						clusterIdSize += 1;
// 					}
// 				}
// 			}
// 			else if (chainLineY[j]==1 && clusteredChainLineY[j]!=0)
// 			{
// 				float landmarkDisplacementThreshold = calculateLandmarkProjectionDiplacement(chainLineYAngles[j], chainLineYAngles[j]+2*tiltAngleStep, j);

// 				int found = 0;

// 				for (size_t k = 1; k <= landmarkDisplacementThreshold; k++)
// 				{
// 					if(chainLineY[j+k]==1)
// 					{
// 						found = 1;
// 						clusteredChainLineY[j+k] = clusterId;
// 						clusterIdSize += 1;
// 					}
// 				}

// 				if (found==0)
// 				{
// 					clusterId += 1;
// 					clusterSizeVector.push_back(clusterIdSize);
// 					clusterIdSize = 0;
// 				}
// 			}
// 		}

// 		if(clusterIdSize>0)
// 		{
// 			clusterSizeVector.push_back(clusterIdSize);
// 			clusterIdSize = 0;
// 		}

// 		std::cout << "Test 5" << std::endl;

// 		// Complete the overall 2D chain map
// 		// *** TODO Make threshold more clever
// 		size_t numberOfElementsInChainThreshold = 10;

// 		// std::cout << "clusteredChainLineY.size()=" << clusteredChainLineY.size() << std::endl; 
// 		// std::cout << "clusterSizeVector.size()=" << clusterSizeVector.size() << std::endl; 

// 		for (size_t j = 0; j < clusteredChainLineY.size(); j++)
// 		{	
// 			// std::cout << "j=" << j << std::endl; 
// 			// std::cout << "clusteredChainLineY[j]=" << clusteredChainLineY[j] << std::endl; 
// 			// std::cout << "clusterSizeVector[clusteredChainLineY[j]]=" << clusterSizeVector[clusteredChainLineY[j]] << std::endl; 

// 			if(clusteredChainLineY[j] != 0)
// 			{
// 				if (clusterSizeVector[clusteredChainLineY[j]] > numberOfElementsInChainThreshold)
// 				{
// 					DIRECT_A2D_ELEM(chain2dMap, chainIndexY, j) = 1;
// 				}			
// 			}
// 		}
// 	}

// 	std::cout << "Test 6" << std::endl;


// 	#ifdef DEBUG_OUTPUT_FILES
// 	size_t lastindex = fnOut.find_last_of(".");
// 	std::string rawname = fnOut.substr(0, lastindex);
// 	std::string outputFileNameFilteredVolume;
//     outputFileNameFilteredVolume = rawname + "_filteredChains.mrc";

// 	Image<int> saveImage;
// 	saveImage() = chain2dMap;
// 	saveImage.write(outputFileNameFilteredVolume);
// 	#endif
// }

void ProgTomoDetectMisalignmentTrajectory::detectLandmarkChains(MultidimArray<int> projectedCoordinates)
{
	std::vector<size_t> counterLinesOfLandmarkAppearance(ySize);

	for(size_t i = 0; i < ySize; i++)
	{
		for(int j = 0; j < coordinates3Dy.size(); j++)
		{
			if(coordinates3Dy[j] == i)
			{
				counterLinesOfLandmarkAppearance[i] += 1;
			}

			if (coordinates3Dy[j-1] == i && j-1 > 0)
			{
				counterLinesOfLandmarkAppearance[i] += 1;
			}

			else if (coordinates3Dy[j+1] == i && j+1 < coordinates3Dy.size())
			{
				counterLinesOfLandmarkAppearance[i] += 1;
			}
		}
	}

	std::cout << "Test 1" << std::endl;

	// Take the average of the three most populated indexes as the expected value for a Possion distribution testing
	size_t numberOfIndexesForPossionAverage = 3;
	float poissonAverage = 0;

	std::vector<size_t> histogramOfLandmarkAppearanceSorted = counterLinesOfLandmarkAppearance;

	// *** TODO: optimize, get n maxima elements without sorting
	sort(histogramOfLandmarkAppearanceSorted.begin(), histogramOfLandmarkAppearanceSorted.end());
	
	for (size_t i = 0; i < numberOfIndexesForPossionAverage; i++)
	{
		poissonAverage += histogramOfLandmarkAppearanceSorted[histogramOfLandmarkAppearanceSorted.size()-i];
	}

	poissonAverage /= 3;

	std::vector<size_t> chainIndexesY;

    std::cout << "Test 2" << std::endl;

	// Test possion probability
	for (size_t i = 0; i < counterLinesOfLandmarkAppearance.size(); i++)
	{
		if (testPoissonDistribution(poissonAverage, counterLinesOfLandmarkAppearance[i])<0.05)
		{
			chainIndexesY.push_back(i);
		}
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CHAIN INDEXESS

	MultidimArray<int> chainCoordinates;
	chainCoordinates.initZeros(ySize, xSize);

	for (size_t j = 0; j < chainIndexesY.size(); j++)
	{
		size_t yCoor = chainIndexesY[j];

		std::vector<size_t> clusteredChainLineY(xSize, 0);

		for(size_t i = 0; i < xSize; i++)
		{
			if(DIRECT_A2D_ELEM(projectedCoordinates, j, i) == 1 ||
			   DIRECT_A2D_ELEM(projectedCoordinates, i, j+1) == 1 ||
			   DIRECT_A2D_ELEM(projectedCoordinates, i, j-1) == 1 ||)
			{
				
			}

		}
	}
	

	// SPLIT CHAINS 
	MultidimArray<int> chain2dMap;
	chain2dMap.initZeros(ySize, xSize);

	std::cout << "Test 3" << std::endl;
	std::cout << "------------------------------------------------------------chainIndexesY.size()" << chainIndexesY.size() << std::endl;


	for (size_t i = 0; i < chainIndexesY.size(); i++)
	{
		size_t chainIndexY = chainIndexesY[i];

		// Binary vector with one's in the x coordinates belonging to each y coordinate
		std::vector<size_t> chainLineY(xSize, 0);

		// Vector containing the angles of the selected coordinates
		std::vector<size_t> chainLineYAngles(xSize, 0);
		
		for (size_t j = 0; j < chainLineY.size() ; j++)
		{
			for(int x = 0; x < coordinates3Dx.size(); x++)
			{
				if(coordinates3Dy[x] == chainIndexY)
				{
					chainLineY[coordinates3Dx[x]] = 1;
					chainLineYAngles[coordinates3Dx[x]] = tiltAngles[coordinates3Dn[x]];
				}		
				else if(x-1 > 0 && coordinates3Dy[x-1] == chainIndexY)
				{
					chainLineY[coordinates3Dx[x-1]] = 1;
					chainLineYAngles[coordinates3Dx[x-1]] = tiltAngles[coordinates3Dn[x-1]];

				}	
				else if(x+1 < coordinates3Dy.size() && coordinates3Dy[x+1] == chainIndexY)
				{
					chainLineY[coordinates3Dx[x+1]] = 1;
					chainLineYAngles[coordinates3Dx[x+1]] = tiltAngles[coordinates3Dn[x+1]];

				}	
			}
		}

		std::cout << "Test 4" << std::endl;


		// Cluser the chainLineY vector
		std::vector<size_t> clusteredChainLineY(xSize, 0);
		size_t clusterId = 1;
		size_t clusterIdSize = 0;
		std::vector<size_t> clusterSizeVector;

		// std::cout << "[" << std::endl;

		// for (size_t j = 0; j < chainLineY.size(); j++)
		// {
		// 	std::cout << chainLineY[j]  << std::endl ;
		// }
		// std::cout << "]" << std::endl;

		for (size_t j = 0; j < chainLineY.size(); j++)
		{
			if(chainLineY[j]==1 && clusteredChainLineY[j]==0)
			{
				clusteredChainLineY[j] = clusterId;
				clusterIdSize += 1;

				float landmarkDisplacementThreshold = calculateLandmarkProjectionDiplacement(chainLineYAngles[j], chainLineYAngles[j]+2*tiltAngleStep, j);

				for (size_t k = 1; k <= landmarkDisplacementThreshold; k++)
				{
					if(chainLineY[j+k]==1)
					{
						clusteredChainLineY[j+k] = clusterId;
						clusterIdSize += 1;
					}
				}
			}
			else if (chainLineY[j]==1 && clusteredChainLineY[j]!=0)
			{
				float landmarkDisplacementThreshold = calculateLandmarkProjectionDiplacement(chainLineYAngles[j], chainLineYAngles[j]+2*tiltAngleStep, j);

				int found = 0;

				for (size_t k = 1; k <= landmarkDisplacementThreshold; k++)
				{
					if(chainLineY[j+k]==1)
					{
						found = 1;
						clusteredChainLineY[j+k] = clusterId;
						clusterIdSize += 1;
					}
				}

				if (found==0)
				{
					clusterId += 1;
					clusterSizeVector.push_back(clusterIdSize);
					clusterIdSize = 0;
				}
			}
		}

		if(clusterIdSize>0)
		{
			clusterSizeVector.push_back(clusterIdSize);
			clusterIdSize = 0;
		}

		std::cout << "Test 5" << std::endl;

		// Complete the overall 2D chain map
		// *** TODO Make threshold more clever
		size_t numberOfElementsInChainThreshold = 10;

		// std::cout << "clusteredChainLineY.size()=" << clusteredChainLineY.size() << std::endl; 
		// std::cout << "clusterSizeVector.size()=" << clusterSizeVector.size() << std::endl; 

		for (size_t j = 0; j < clusteredChainLineY.size(); j++)
		{	
			// std::cout << "j=" << j << std::endl; 
			// std::cout << "clusteredChainLineY[j]=" << clusteredChainLineY[j] << std::endl; 
			// std::cout << "clusterSizeVector[clusteredChainLineY[j]]=" << clusterSizeVector[clusteredChainLineY[j]] << std::endl; 

			if(clusteredChainLineY[j] != 0)
			{
				if (clusterSizeVector[clusteredChainLineY[j]] > numberOfElementsInChainThreshold)
				{
					DIRECT_A2D_ELEM(chain2dMap, chainIndexY, j) = 1;
				}			
			}
		}
	}

	std::cout << "Test 6" << std::endl;


	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindex = fnOut.find_last_of(".");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameFilteredVolume;
    outputFileNameFilteredVolume = rawname + "_filteredChains.mrc";

	Image<int> saveImage;
	saveImage() = chain2dMap;
	saveImage.write(outputFileNameFilteredVolume);
	#endif
}

void ProgTomoDetectMisalignmentTrajectory::calculateResidualVectors(MetaData inputCoordMd)
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
		FOR_ALL_OBJECTS_IN_METADATA(inputCoordMd)
		{
			maxDistance = MAXDOUBLE;

			objId = __iter.objId;
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
}


// --------------------------- I/O functions ----------------------------

void ProgTomoDetectMisalignmentTrajectory::writeOutputCoordinates()
{
	MetaData md;
	size_t id;

	for(size_t i = 0; i < coordinates3Dx.size(); i++)
	{
		id = md.addObject();
		md.setValue(MDL_XCOOR, (int)coordinates3Dx[i], id);
		md.setValue(MDL_YCOOR, (int)coordinates3Dy[i], id);
		md.setValue(MDL_ZCOOR, coordinates3Dn[i], id);
	}

	md.write(fnOut);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Coordinates metadata saved at: " << fnOut << std::endl;
	#endif

}


void ProgTomoDetectMisalignmentTrajectory::writeOutputResidualVectors()
{
	MetaData md;
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

	generateSideInfo();
	
	MetaData tiltseriesmd;
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

	#ifdef DEBUG_DIM
	size_t checkXdim, checkYdim, checkZdim, checkNdim;
	tiltSeriesImages.getDimensions(checkXdim, checkYdim, checkZdim, checkNdim);

	std::cout << "Input tilt-series dimensions:" << std::endl;
	std::cout << "x " << checkXdim << std::endl;
	std::cout << "y " << checkYdim << std::endl;
	std::cout << "z " << checkZdim << std::endl;
	std::cout << "n " << checkNdim << std::endl;
	#endif

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

	FOR_ALL_OBJECTS_IN_METADATA(tiltseriesmd)
	{
		objId = __iter.objId;
		tiltseriesmd.getValue(MDL_IMAGE, fnTSimg, objId);

		#ifdef DEBUG
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
	size_t lastindex = fnOut.find_last_of(".");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameFilteredVolume;
    outputFileNameFilteredVolume = rawname + "_filter.mrc";

	Image<double> saveImage;
	saveImage() = filteredTiltSeries;
	saveImage.write(outputFileNameFilteredVolume);
	#endif

	xSize = XSIZE(filteredTiltSeries);
	ySize = YSIZE(filteredTiltSeries);
	zSize = ZSIZE(filteredTiltSeries);
	nSize = NSIZE(filteredTiltSeries);

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

	for(size_t n; n < coordinates3Dn.size(); n++)
	{
		DIRECT_A2D_ELEM(proyectedCoordinates, (int)coordinates3Dy[n], (int)coordinates3Dx[n]) = 1;
	}

	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindexBis = fnOut.find_last_of(".");
	std::string rawnameBis = fnOut.substr(0, lastindexBis);
	std::string outputFileNameFilteredVolumeBis;
    outputFileNameFilteredVolumeBis = rawnameBis + "_proyected.mrc";

	Image<int> saveImageBis;
	saveImageBis() = proyectedCoordinates;
	saveImageBis.write(outputFileNameFilteredVolumeBis);
	#endif

	writeOutputCoordinates();
	detectLandmarkChains(proyectedCoordinates);

	if(checkInputCoord)
	{
		MetaData inputCoordMd;
		inputCoordMd.read(fnInputCoord);

		calculateResidualVectors(inputCoordMd);
		writeOutputResidualVectors();
	}
	
	auto t2 = high_resolution_clock::now();
	/* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
}


// --------------------------- UTILS functions ----------------------------


bool ProgTomoDetectMisalignmentTrajectory::filterLabeledRegions(std::vector<int> coordinatesPerLabelX, std::vector<int> coordinatesPerLabelY, double centroX, double centroY)
{
	// Check number of elements of the label
	if(coordinatesPerLabelX.size() < numberOfCoordinatesThr)
	{
		return false;
	}

	// Check spehricity of the label
	double maxSquareDistance = 0;
	double distance;

	size_t debugN;

	for(size_t n = 0; n < coordinatesPerLabelX.size(); n++)
	{
		distance = (coordinatesPerLabelX[n]-centroX)*(coordinatesPerLabelX[n]-centroX)+(coordinatesPerLabelY[n]-centroY)*(coordinatesPerLabelY[n]-centroY);

		if(distance > maxSquareDistance)
		{
			debugN = n;
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


std::vector<Matrix1D<double>> ProgTomoDetectMisalignmentTrajectory::getCoordinatesInSlice(int slice)
{
	std::vector<Matrix1D<double>> coordinatesInSlice;
	Matrix1D<double> coordinate(2);

	for(size_t n = 0; n < coordinates3Dx.size(); n++)
	{
		if(slice == coordinates3Dn[n])
		{
			XX(coordinate) = coordinates3Dx[n];
			YY(coordinate) = coordinates3Dy[n];
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

	// Since k! can not be holded we calculate the quotient lambda^k/k!= (lambda/n) * (lambda/(n-1)) * ... * (lambda/1)
	for (size_t i = 1; i < k+1; i++)
	{
		quotient *= lambda / i;
	}
	
	return quotient*exp(-lambda);

}


float ProgTomoDetectMisalignmentTrajectory::calculateLandmarkProjectionDiplacement(float theta1, float theta2, float coordinateProjX)
{

	return ((cos(theta2)/cos(theta1))-1)*coordinateProjX;

}



// float ProgTomoDetectMisalignmentTrajectory::getSortedProyectedCoordinatesInLine(size_t y, size_t tolerance)
// {
// 	size_t numberOfCoordinates = coordinates3Dn.size()

// 	for (size_t i = 0; i < numberOfCoordinates; i++)
// 	{
// 		size_t coordY = coordinates3Dy[i];

// 		if (size_t t = -tolerance; i <= tolerance; t++)
// 		{
			
// 		}
		
// 	}
	

// }



