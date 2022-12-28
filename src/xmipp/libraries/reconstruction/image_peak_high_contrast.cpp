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



void ProgImagePeakHighContrast::generateSideInfo()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Generating side info..." << std::endl;
	#endif

	fiducialSizePx = fiducialSize / samplingRate;

	#ifdef VERBOSE_OUTPUT
	std::cout << "Side info generated successfully!" << std::endl;
	#endif
}



void ProgImagePeakHighContrast::preprocessVolume(MultidimArray<double> &inputTomo)
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Preprocessing volume..." << std::endl;
	#endif


	// -- Average
	#ifdef VERBOSE_OUTPUT
	std::cout << "Averaging volume..." << std::endl;
	#endif

	// Image<double> sliceTomo;
	MultidimArray<double> slice(ySize, xSize);
	MultidimArray<double> sliceU(ySize, xSize);
	MultidimArray<double> sliceD(ySize, xSize);
	MultidimArray<double> sliceU2(ySize, xSize);
	MultidimArray<double> sliceD2(ySize, xSize);

	for (int k = 2; k < zSize-2; k++)
	{
		if(k==2)
		{
			inputTomo.getSlice(k-2, sliceU2);
			inputTomo.getSlice(k-1, sliceU);
			inputTomo.getSlice(k,   slice);
			inputTomo.getSlice(k+1, sliceD);
			inputTomo.getSlice(k+2, sliceD2);
		}
		else
		{
			sliceU2 = sliceU;
			sliceU = slice;
			slice  = sliceD;
			sliceD  = sliceD2;
			inputTomo.getSlice(k+2, sliceD2);
		}

		for (int i = 0; i < ySize; i++)
		{
			for (int j = 0; j < xSize; j++)
			{			
				DIRECT_A3D_ELEM(inputTomo, k, i, j) = DIRECT_A2D_ELEM(sliceD2, i , j) +
													  DIRECT_A2D_ELEM(sliceD,  i , j) +
													  DIRECT_A2D_ELEM(slice,   i , j) +
													  DIRECT_A2D_ELEM(sliceU,  i , j) +
													  DIRECT_A2D_ELEM(sliceU2, i , j);
			}
		}
	}

	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindex = fnOut.find_last_of(".");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameFilteredVolume;
    outputFileNameFilteredVolume = rawname + "_average.mrc";

	V.write(outputFileNameFilteredVolume);
	#endif

	#ifdef VERBOSE_OUTPUT
	std::cout << "Volume averaging finished succesfully!" << std::endl;
	#endif


	// -- Band-pass filtering
	MultidimArray< std::complex<double> > fftV;
	transformer.FourierTransform(inputTomo, fftV, false);

	#ifdef VERBOSE_OUTPUT
	std::cout << "Applying bandpass filter to volume..." << std::endl;
	#endif

	int ux, uy, uz, uz2, uz2y2;

	int n=0;

	double freqLow = samplingRate / (fiducialSize*1.1);
	double freqHigh = samplingRate/(fiducialSize*0.9);
	
	double w; // = 0.02 
	double cutoffFreqHigh = freqHigh + w;
	double cutoffFreqLow = freqLow - w;
	double delta = PI / w;

	normDim = (xSize>ySize) ? xSize : ySize;

	// 43.2 = 1440 * 0.03. This 43.2 value makes w = 0.03 (standard value) for an image whose bigger dimension is 1440 px.
	w = 43.2 / normDim;

	#ifdef DEBUG_PREPROCESS
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

		FFT_IDX2DIGFREQ(k,zSize,uz);
		uz2=uz*uz;

		for(size_t i=0; i<YSIZE(fftV); ++i)
		{
			FFT_IDX2DIGFREQ(i,ySize,uy);
			uz2y2=uz2+uy*uy;

			for(size_t j=0; j<XSIZE(fftV); ++j)
			{
				FFT_IDX2DIGFREQ(j,xSize,ux);
				double u=sqrt(uz2y2+ux*ux);

				if(u > cutoffFreqHigh || u < cutoffFreqLow)
				{
					DIRECT_MULTIDIM_ELEM(fftV, n) = 0;
				} 
				else
				{
					if(u >= freqHigh && u < cutoffFreqHigh)
					{
						DIRECT_MULTIDIM_ELEM(fftV, n) *= 0.5*(1+cos((u-freqHigh)*delta));
					}
				
					if (u <= freqLow && u > cutoffFreqLow)
					{
						DIRECT_MULTIDIM_ELEM(fftV, n) *= 0.5*(1+cos((u-freqLow)*delta));
					}
				}
				
				++n;
			}
		}
	}

	transformer.inverseFourierTransform();

	#ifdef DEBUG_OUTPUT_FILES
	lastindex = fnOut.find_last_of(".");
	rawname = fnOut.substr(0, lastindex);
	outputFileNameFilteredVolume;
    outputFileNameFilteredVolume = rawname + "_bandpass.mrc";

	V.write(outputFileNameFilteredVolume);
	#endif

	#ifdef VERBOSE_OUTPUT
	std::cout << "Bandpass filter applied to volume succesfully!" << std::endl;
	#endif
	

	// -- Apply Laplacian to tomo with kernel:
	//     0  0 0    0 -1  0    0 0 0
	// k = 0 -1 0    -1 4 -1    0 -1 0
	//     0  0 0    0 -1  0    0 0 0
		
	#ifdef VERBOSE_OUTPUT
	std::cout << "Applying laplacian filter to volume..." << std::endl;
	#endif

	for (int k = 1; k < zSize-1; k++)
	{
		MultidimArray<double> slice(ySize, xSize);
		inputTomo.getSlice(k, slice);

		for (int i = 1; i < ySize-1; i++)
		{
			for (int j = 1; j < xSize-1; j++)
			{				
				DIRECT_A3D_ELEM(inputTomo, k, i, j) = -2 * DIRECT_A2D_ELEM(slice, i,   j-1) +
													  -2 * DIRECT_A2D_ELEM(slice, i,   j+1) +
													  -2 * DIRECT_A2D_ELEM(slice, i-1, j) +
													  -2 * DIRECT_A2D_ELEM(slice, i+1, j) +
													   8 * DIRECT_A2D_ELEM(slice, i,   j);
			}
		}
	} 

	#ifdef VERBOSE_OUTPUT
	std::cout << "Laplacian filter applied to volume succesfully!" << std::endl;
	#endif


	// -- Set extreme slices to 0 (unafected by average filter)
	for (int k = 0; k < zSize; k++)
	{
		if(k<2 || k > zSize-2)
		{
			for (int i = 0; i < ySize; i++)
			{
				for (int j = 0; j < xSize; j++)
				{	
					DIRECT_A3D_ELEM(inputTomo, k, i, j) = 0.0;
				}
			}
		}
	}

	#ifdef DEBUG_OUTPUT_FILES
	lastindex = fnOut.find_last_of(".");
	rawname = fnOut.substr(0, lastindex);
	outputFileNameFilteredVolume;
    outputFileNameFilteredVolume = rawname + "_preprocess.mrc";

	V.write(outputFileNameFilteredVolume);
	#endif

	#ifdef VERBOSE_OUTPUT
	std::cout << "Volume preprocessed succesfully!" << std::endl;
	#endif
}



void ProgImagePeakHighContrast::getHighContrastCoordinates(MultidimArray<double> &inputTomo)
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
	double sum = 0;
	double sum2 = 0;
	int Nelems = xSize * ySize * numberSampSlices;
	
	for(size_t k = minSamplingSlice; k < maxSamplingSlice; ++k)
	{
		for(size_t j = 0; j < ySize; ++j)
		{
			for(size_t i = 0; i < xSize; ++i)
			{
				double value = DIRECT_ZYX_ELEM(inputTomo, k, i ,j);
				sum += value;
				sum2 += value*value;
			}
		}
	}

	double average = sum / Nelems;
	double standardDeviation = sqrt(sum2/Nelems - average*average);

	double thresholdL = average-sdThreshold*standardDeviation;
	double thresholdU = average+sdThreshold*standardDeviation;

	#ifdef DEBUG_HCC
	std::cout << "ThresholdU value = " << thresholdU << std::endl;
	std::cout << "ThresholdL value = " << thresholdL << std::endl;
	#endif

	MultidimArray<double> binaryCoordinatesMapSlice;
	MultidimArray<double> labelCoordiantesMapSlice;
	
	for(size_t k = 1; k < zSize-1; k++)
	{	
		binaryCoordinatesMapSlice.initZeros(ySize, xSize);

		#ifdef DEBUG_HCC
		int numberOfPointsAddedBinaryMap = 0;
		#endif

		for(size_t j = 0; j < xSize; j++)
		{
			for(size_t i = 0; i < ySize; i++)
			{
				double value = DIRECT_A3D_ELEM(inputTomo, k, i, j);

				// if (value < thresholdL || value > thresholdU) 
				// {
				if (value < thresholdL) 
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

		#ifdef DEBUG_HCC
		std::cout << "Labeling slice " << k << std::endl;
		#endif


		// MultidimArray<double> labelCoordiantesMapSliceClosing(ySize, xSize);
		// std::cout << "check1" << std::endl;
		// closing2D(binaryCoordinatesMapSlice, labelCoordiantesMapSliceClosing, 8, 0, 8);  // neigh, count, size
		// std::cout << "check2" << std::endl;

		// The value 8 is the neighbourhood
		int colour = labelImage2D(binaryCoordinatesMapSlice, labelCoordiantesMapSlice, 8);


		#ifdef DEBUG_OUTPUT_FILES
		for (size_t j = 0; j < xSize; j++)
		{
			for (size_t i = 0; i < ySize; i++)
			{
				double value = DIRECT_A2D_ELEM(labelCoordiantesMapSlice, i, j);

				if (value > 0)
				{
					DIRECT_A3D_ELEM(inputTomo, k, i, j) = value;
				}
				else
				{
					DIRECT_A3D_ELEM(inputTomo, k, i, j) = 0;
				}
			}
		}
		#endif

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

				if(value != 0)
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
			}
		}
    }

	#ifdef VERBOSE_OUTPUT
	std::cout << "Number of high contrast features found: " << coordinates3D.size() << std::endl;
	#endif

	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindex = fnOut.find_last_of(".");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameLabeledVolume;
    outputFileNameLabeledVolume = rawname + "_label.mrc";

	V.write(outputFileNameLabeledVolume);
	#endif
}



void ProgImagePeakHighContrast::clusterHCC()
{
	std::vector<size_t> coordinatesInSlice;
	std::vector<size_t> coordinatesInSlice_up;
	std::vector<size_t> coordinatesInSlice_down;

	std::vector<size_t> coord3DVotes_V(coordinates3D.size(), 0);

	float thrVottingDistance2 = (fiducialSizePx/2)*(fiducialSizePx/2);

	#ifdef DEBUG_CLUSTER
	std::cout << "thrVottingDistance2 " << thrVottingDistance2 << std::endl;
	#endif
	
	size_t deletedIndexes;

	#ifdef DEBUG_CLUSTER
	size_t iteration = 0;
	#endif

	// -- Erase non-consistent coordinates with the voting systen
	do
	{
		#ifdef DEBUG_CLUSTER
		std::cout << "--- ITERATION " << iteration << std::endl;
		#endif

		deletedIndexes = 0;

		// Votting step	
		for (int k = 0; k < zSize; k++)
		{
			if (k == 0)	// Skip up-image for first slice
			{
				coordinatesInSlice = getCoordinatesInSliceIndex(k);
				coordinatesInSlice_down = getCoordinatesInSliceIndex(k+1);
			}
			else if (k == (nSize-1)) // Skip down-image for last slice
			{
				coordinatesInSlice_up = coordinatesInSlice;
				coordinatesInSlice = coordinatesInSlice_down;
			}
			else // Non-extrema slices
			{
				coordinatesInSlice_up = coordinatesInSlice;
				coordinatesInSlice = coordinatesInSlice_down;
				coordinatesInSlice_down = getCoordinatesInSliceIndex(k+1);
			}

			for(size_t i = 0; i < coordinatesInSlice.size(); i++)
			{
				Point3D<double> c = coordinates3D[coordinatesInSlice[i]];

				// Skip for first image in the series
				if (k != 0)
				{
					for (size_t j = 0; j < coordinatesInSlice_up.size(); j++)
					{
						Point3D<double> cu = coordinates3D[coordinatesInSlice_up[j]];
						float distance2 = (c.x-cu.x)*(c.x-cu.x)+(c.y-cu.y)*(c.y-cu.y);

						if(distance2 < thrVottingDistance2)
						{
							coord3DVotes_V[coordinatesInSlice[i]] += 1;
						}
					}
				}

				// Skip for last image in the series
				if (k != (nSize-1))
				{		
					for (size_t j = 0; j < coordinatesInSlice_down.size(); j++)
					{
						Point3D<double> cd = coordinates3D[coordinatesInSlice_down[j]];
						float distance2 = (c.x-cd.x)*(c.x-cd.x)+(c.y-cd.y)*(c.y-cd.y);

						if(distance2 < thrVottingDistance2)
						{
							coord3DVotes_V[coordinatesInSlice[i]] += 1;
						}
					}
				}
			}
		}

		// Trimming step
		for (size_t i = 0; i < coord3DVotes_V.size(); i++)
		{
			if (coord3DVotes_V[i] == 0)
			{
				#ifdef DEBUG_CLUSTER
				std::cout << "Deleted coordinate " << i << std::endl;
				#endif

				coordinates3D.erase(coordinates3D.begin()+i);
				coord3DVotes_V.erase(coord3DVotes_V.begin()+i);
				deletedIndexes++;
				i--;
			}
		}

		#ifdef DEBUG_CLUSTER
		std::cout << "DeletedIndexes: " << deletedIndexes << std::endl; 
		iteration++;
		#endif

	}
	while(deletedIndexes > 0);

	#ifdef DEBUG_CLUSTER
	std::cout << "coord3DVotes_V.size() " << coord3DVotes_V.size() << std::endl;
	std::cout << "coordinates3D.size() " << coordinates3D.size() << std::endl;

	for (size_t i = 0; i < coord3DVotes_V.size(); i++)
	{
		std::cout << coord3DVotes_V[i] << " ";
	}
	std::cout << std::endl;
	#endif


	// -- Cluster non-unvoted coordinates
	std::vector<size_t> coord3DId_V(coordinates3D.size(), 0);
	size_t currentId = 1;

	// Initialize ID's in the first slice
	coordinatesInSlice = getCoordinatesInSliceIndex(0);	

	for(size_t i = 0; i < coordinatesInSlice.size(); i++)
	{
		coord3DId_V[coordinatesInSlice[i]] = currentId;
		currentId++;
	}

	// Extend ID's for coordinates in the whole volume
	for (int k = 1; k < zSize; k++)
	{
		coordinatesInSlice_up = coordinatesInSlice;
		coordinatesInSlice = getCoordinatesInSliceIndex(k);	

		for(size_t i = 0; i < coordinatesInSlice.size(); i++)
		{
			Point3D<double> c = coordinates3D[coordinatesInSlice[i]];

			double match = false;
			for (size_t j = 0; j < coordinatesInSlice_up.size(); j++)
			{
				Point3D<double> cu = coordinates3D[coordinatesInSlice_up[j]];
				float distance2 = (c.x-cu.x)*(c.x-cu.x)+(c.y-cu.y)*(c.y-cu.y);

				if(distance2 < thrVottingDistance2)
				{
					coord3DId_V[coordinatesInSlice[i]] = coord3DId_V[coordinatesInSlice_up[j]];
					match = true;
					break;
				}
			}

			if (!match)
			{
				coord3DId_V[coordinatesInSlice[i]] = currentId;
				currentId++;
			}
		}
	}


	#ifdef DEBUG_CLUSTER
	std::cout << "coord3DId_V.size() " << coord3DId_V.size() << std::endl;

	for (size_t i = 0; i < coord3DId_V.size(); i++)
	{
		std::cout << coord3DId_V[i] << " ";
	}
	std::cout << std::endl;
	#endif


	#ifdef VERBOSE_OUTPUT
	std::cout << "Number of clusters identified: " << (currentId-1) << std::endl;
	#endif

	// -- Average coordinates with the same ID
    std::vector<Point3D<double>> coordinates3D_avg;
	
	for (size_t id = 1; id < currentId; id++)
	{
		// Sum coordinate components with the same ID
		Point3D<double> coord3D_avg(0,0,0);
		int nCoords = 0;

		for (int n = 0; n < coord3DId_V.size(); n++)
		{
			if (coord3DId_V[n] == id)
			{
				coord3D_avg.x += coordinates3D[n].x;
				coord3D_avg.y += coordinates3D[n].y;
				coord3D_avg.z += coordinates3D[n].z;
				nCoords++;

				coordinates3D.erase(coordinates3D.begin()+n);
				coord3DId_V.erase(coord3DId_V.begin()+n);
				n--;
			}
		}

		coord3D_avg.x /= nCoords;
		coord3D_avg.y /= nCoords;
		coord3D_avg.z /= nCoords;

		coordinates3D_avg.push_back(coord3D_avg);
	}

	coordinates3D = coordinates3D_avg;


	#ifdef VERBOSE_OUTPUT
	std::cout << "Number of coordinates obtained after clustering: " << coordinates3D.size() << std::endl;
	std::cout << "Clustering of coordinates finished successfully!!" << std::endl;
	#endif

	// // Generate output labeled and filtered series
	// // *** generate labeled and filtered outputs
	// #ifdef DEBUG_OUTPUT_FILES
	// MultidimArray<int> filteredLabeledTS;
	// filteredLabeledTS.initZeros(nSize, 1, ySize, xSize);

	// std::vector<Point2D<double>> cis;

	// for (size_t n = 0; n < nSize; n++)
	// {
	// 	cis = getCoordinatesInSlice(n);

	// 	MultidimArray<int> filteredLabeledTS_Image;
	// 	filteredLabeledTS_Image.initZeros(ySize, xSize);

	// 	for(size_t i = 0; i < cis.size(); i++)
	// 	{
	// 		fillImageLandmark(filteredLabeledTS_Image, (int)cis[i].x, (int)cis[i].y, 1);
	// 	}

	// 	for (size_t i = 0; i < ySize; ++i)
	// 	{
	// 		for (size_t j = 0; j < xSize; ++j)
	// 		{
	// 			DIRECT_NZYX_ELEM(filteredLabeledTS, n, 0, i, j) = DIRECT_A2D_ELEM(filteredLabeledTS_Image, i, j);
	// 		}
	// 	}
	// }
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



// void ProgImagePeakHighContrast::centerCoordinates(MultidimArray<double> volFiltered)
// {
// 	#ifdef VERBOSE_OUTPUT
// 	std::cout << "Centering coordinates..." << std::endl;
// 	#endif

// 	size_t halfBoxSize = boxSize / 2;
// 	size_t numberOfFeatures = coordinates3D.size();

// 	MultidimArray<double> feature;
// 	MultidimArray<double> mirrorFeature;
// 	MultidimArray<double> correlationVolumeR;

// 	int coordHalfX;
// 	int coordHalfY;
// 	int coordHalfZ;

// 	for(size_t n = 0; n < numberOfFeatures; n++)
// 	{
// 		// Construct feature and its mirror symmetric
// 		feature.initZeros(boxSize, boxSize, boxSize);
// 		mirrorFeature.initZeros(boxSize, boxSize, boxSize);
		
// 		for(int k = 0; k < boxSize; k++) // zDim
// 		{	
// 			for(int j = 0; j < boxSize; j++) // xDim
// 			{
// 				for(int i = 0; i < boxSize; i++) // yDim
// 				{
// 					coordHalfX = coordinates3D[n].x - halfBoxSize;
// 					coordHalfY = coordinates3D[n].y - halfBoxSize;
// 					coordHalfZ = coordinates3D[n].z - halfBoxSize;

// 					// Check coordinate is not out of volume
// 					if ((coordHalfZ + k) < 0 || (coordHalfZ + k) > zSize ||
// 					    (coordHalfY + i) < 0 || (coordHalfY + i) > ySize ||
// 						(coordHalfX + j) < 0 || (coordHalfX + j) > xSize)
// 					{
// 						DIRECT_A3D_ELEM(feature, k, i, j) = 0;

// 						DIRECT_A3D_ELEM(mirrorFeature, boxSize -1 - k, boxSize -1 - i, boxSize -1 - j) = 0;
// 					}
// 					else
// 					{
// 						DIRECT_A3D_ELEM(feature, k, i, j) = DIRECT_A3D_ELEM(volFiltered, 
// 																			coordHalfZ + k, 
// 																			coordHalfY + i, 
// 																			coordHalfX + j);

// 						DIRECT_A3D_ELEM(mirrorFeature, boxSize -1 - k, boxSize -1 - i, boxSize -1 - j) = 
// 						DIRECT_A3D_ELEM(volFiltered, 
// 										coordHalfZ + k, 
// 										coordHalfY + i,
// 										coordHalfX + j);
// 					}
// 				}
// 			}
// 		}

// 		// Shift the particle respect to its symmetric to look for the maximum correlation displacement
// 		CorrelationAux aux;
// 		correlation_matrix(feature, mirrorFeature, correlationVolumeR, aux, true);

// 		double maximumCorrelation = MINDOUBLE;
// 		double xDisplacement = 0;
// 		double yDisplacement = 0;
// 		double zDisplacement = 0;

// 		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(correlationVolumeR)
// 		{
// 			double value = DIRECT_A3D_ELEM(correlationVolumeR, k, j, i);
			
// 			if (value > maximumCorrelation)
// 			{
// 				maximumCorrelation = value;
// 				xDisplacement = j;
// 				yDisplacement = i;
// 				zDisplacement = k;
// 			}
// 		}

// 		#ifdef DEBUG_CENTER_COORDINATES
// 		std::cout << "--------------------" << std::endl;
// 		std::cout << "maximumCorrelation " << maximumCorrelation << std::endl;
// 		std::cout << "xDisplacement " << ((int) xDisplacement - boxSize / 2) / 2 << std::endl;
// 		std::cout << "yDisplacement " << ((int) yDisplacement - boxSize / 2) / 2 << std::endl;
// 		std::cout << "zDisplacement " << ((int) zDisplacement - boxSize / 2) / 2 << std::endl;
// 		#endif


// 		// Update coordinate and remove if it is moved out of the volume
// 		double updatedCoordinateX = coordinates3D[n].x + ((int) xDisplacement - boxSize / 2) / 2;
// 		double updatedCoordinateY = coordinates3D[n].y + ((int) yDisplacement - boxSize / 2) / 2;
// 		double updatedCoordinateZ = coordinates3D[n].z + ((int) zDisplacement - boxSize / 2) / 2;

// 		int deletedCoordinates = 0;
	
// 		if (updatedCoordinateZ < 0 || updatedCoordinateZ > zSize ||
// 			updatedCoordinateY < 0 || updatedCoordinateY > ySize ||
// 			updatedCoordinateX < 0 || updatedCoordinateX > xSize)
// 		{
// 			coordinates3D.erase(coordinates3D.begin()+n-deletedCoordinates);
// 			deletedCoordinates++;
// 		}
// 		else
// 		{
// 			coordinates3D[n].x = updatedCoordinateX;
// 			coordinates3D[n].y = updatedCoordinateY;
// 			coordinates3D[n].z = updatedCoordinateZ;
// 		}
// 	}

// 	#ifdef VERBOSE_OUTPUT
// 	std::cout << "Centering of coordinates finished successfully!!" << std::endl;
// 	#endif
// }



void ProgImagePeakHighContrast::centerCoordinates(MultidimArray<double> volFiltered)
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Centering coordinates..." << std::endl;
	#endif

	size_t numberOfFeatures = coordinates3D.size();

	MultidimArray<double> feature;
	MultidimArray<double> mirrorFeature;
	MultidimArray<double> correlationVolumeR;

	int coordHalfX;
	int coordHalfY;
	int coordHalfZ;

	int doubleBoxSize = boxSize * 2;

	for(size_t n = 0; n < numberOfFeatures; n++)
	{
		// Construct feature and its mirror symmetric
		feature.initZeros(doubleBoxSize, doubleBoxSize, doubleBoxSize);
		mirrorFeature.initZeros(doubleBoxSize, doubleBoxSize, doubleBoxSize);
		
		coordHalfX = coordinates3D[n].x - boxSize;
		coordHalfY = coordinates3D[n].y - boxSize;
		coordHalfZ = coordinates3D[n].z - boxSize;

		for(int k = 0; k < doubleBoxSize; k++) // zDim
		{	
			for(int j = 0; j < doubleBoxSize; j++) // xDim
			{
				for(int i = 0; i < doubleBoxSize; i++) // yDim
				{
					// Check coordinate is not out of volume
					if ((coordHalfZ + k) < 0 || (coordHalfZ + k) > zSize ||
					    (coordHalfY + i) < 0 || (coordHalfY + i) > ySize ||
						(coordHalfX + j) < 0 || (coordHalfX + j) > xSize)
					{
						DIRECT_A3D_ELEM(feature, k, i, j) = 0;

						DIRECT_A3D_ELEM(mirrorFeature, doubleBoxSize -1 - k, doubleBoxSize -1 - i, doubleBoxSize -1 - j) = 0;
					}
					else
					{
						DIRECT_A3D_ELEM(feature, k, i, j) = DIRECT_A3D_ELEM(volFiltered, 
																			coordHalfZ + k, 
																			coordHalfY + i, 
																			coordHalfX + j);

						DIRECT_A3D_ELEM(mirrorFeature, doubleBoxSize -1 - k, doubleBoxSize -1 - i, doubleBoxSize -1 - j) = 
						DIRECT_A3D_ELEM(volFiltered, 
										coordHalfZ + k, 
										coordHalfY + i,
										coordHalfX + j);
					}
				}
			}
		}

		#ifdef DEBUG_CENTER_COORDINATES
		Image<double> subtomo;

		subtomo() = feature;
		size_t lastindex = fnOut.find_last_of(".");
		std::string rawname = fnOut.substr(0, lastindex);
		std::string outputFileNameSubtomo;
		outputFileNameSubtomo = rawname + "_" + std::to_string(n) + "_feature.mrc";
		subtomo.write(outputFileNameSubtomo);

		subtomo() = mirrorFeature;
		outputFileNameSubtomo = rawname + "_" + std::to_string(n) + "_mirrorFeature.mrc";
		subtomo.write(outputFileNameSubtomo);
		#endif

		// Shift the particle respect to its symmetric to look for the maximum correlation displacement
		CorrelationAux aux;
		correlation_matrix(feature, mirrorFeature, correlationVolumeR, aux, true);

		double maximumCorrelation = MINDOUBLE;
		double xDisplacement = 0;
		double yDisplacement = 0;
		double zDisplacement = 0;

		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(correlationVolumeR)
		{
			double value = DIRECT_A3D_ELEM(correlationVolumeR, k, i, j);
			
			if (value > maximumCorrelation)
			{
				maximumCorrelation = value;
				xDisplacement = j;
				yDisplacement = i;
				zDisplacement = k;
			}
		}

		#ifdef DEBUG_CENTER_COORDINATES
		std::cout << "-------------------- coordinate " << n << std::endl;
		std::cout << "maximumCorrelation " << maximumCorrelation << std::endl;
		std::cout << "xDisplacement " << ((int) xDisplacement - boxSize) / 2 << std::endl;
		std::cout << "yDisplacement " << ((int) yDisplacement - boxSize) / 2 << std::endl;
		std::cout << "zDisplacement " << ((int) zDisplacement - boxSize) / 2 << std::endl;
		#endif


		// Update coordinate and remove if it is moved out of the volume
		double updatedCoordinateX = coordinates3D[n].x - ((int) xDisplacement - boxSize) / 2;
		double updatedCoordinateY = coordinates3D[n].y - ((int) yDisplacement - boxSize) / 2;
		double updatedCoordinateZ = coordinates3D[n].z - ((int) zDisplacement - boxSize) / 2;

		int deletedCoordinates = 0;
	
		if (updatedCoordinateZ < 0 || updatedCoordinateZ > zSize ||
			updatedCoordinateY < 0 || updatedCoordinateY > ySize ||
			updatedCoordinateX < 0 || updatedCoordinateX > xSize)
		{
			coordinates3D.erase(coordinates3D.begin()+n-deletedCoordinates);
			deletedCoordinates++;
		}
		else
		{
			coordinates3D[n].x = updatedCoordinateX;
			coordinates3D[n].y = updatedCoordinateY;
			coordinates3D[n].z = updatedCoordinateZ;
		}

		#ifdef DEBUG_CENTER_COORDINATES
		// Construct and save the centered feature
		MultidimArray<double> centerFeature;
		
		// centerFeature.initZeros(boxSize, boxSize, boxSize);
		// int halfBoxSize = boxSize / 2;

		// coordHalfX = coordinates3D[n].x - halfBoxSize;
		// coordHalfY = coordinates3D[n].y - halfBoxSize;
		// coordHalfZ = coordinates3D[n].z - halfBoxSize;

		// for(int k = 0; k < boxSize; k++) // zDim
		// {	
		// 	for(int j = 0; j < boxSize; j++) // xDim
		// 	{
		// 		for(int i = 0; i < boxSize; i++) // yDim
		// 		{
		// 			// Check coordinate is not out of volume
		// 			if ((coordHalfZ + k) < 0 || (coordHalfZ + k) > zSize ||
		// 			    (coordHalfY + i) < 0 || (coordHalfY + i) > ySize ||
		// 				(coordHalfX + j) < 0 || (coordHalfX + j) > xSize)
		// 			{
		// 				DIRECT_A3D_ELEM(centerFeature, k, i, j) = 0;
		// 			}
		// 			else
		// 			{
		// 				DIRECT_A3D_ELEM(centerFeature, k, i, j) = DIRECT_A3D_ELEM(volFiltered, 
		// 																		  coordHalfZ + k, 
		// 																		  coordHalfY + i, 
		// 																		  coordHalfX + j);
		// 			}
		// 		}
		// 	}
		// }

		centerFeature.initZeros(doubleBoxSize, doubleBoxSize, doubleBoxSize);

		coordHalfX = coordinates3D[n].x - boxSize;
		coordHalfY = coordinates3D[n].y - boxSize;
		coordHalfZ = coordinates3D[n].z - boxSize;

		for(int k = 0; k < doubleBoxSize; k++) // zDim
		{	
			for(int j = 0; j < doubleBoxSize; j++) // xDim
			{
				for(int i = 0; i < doubleBoxSize; i++) // yDim
				{
					// Check coordinate is not out of volume
					if ((coordHalfZ + k) < 0 || (coordHalfZ + k) > zSize ||
					    (coordHalfY + i) < 0 || (coordHalfY + i) > ySize ||
						(coordHalfX + j) < 0 || (coordHalfX + j) > xSize)
					{
						DIRECT_A3D_ELEM(centerFeature, k, i, j) = 0;
					}
					else
					{
						DIRECT_A3D_ELEM(centerFeature, k, i, j) = DIRECT_A3D_ELEM(volFiltered, 
																			      coordHalfZ + k, 
																			      coordHalfY + i, 
																			      coordHalfX + j);
					}
				}
			}
		}

		subtomo() = centerFeature;
		outputFileNameSubtomo = rawname + "_" + std::to_string(n) + "_centerFeature.mrc";
		subtomo.write(outputFileNameSubtomo);
		#endif
	}

	#ifdef VERBOSE_OUTPUT
	std::cout << "Centering of coordinates finished successfully!!" << std::endl;
	#endif
}



void ProgImagePeakHighContrast::removeDuplicatedCoordinates()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Removing duplicated coordinates..." << std::endl;
	#endif

	double maxDistance = fiducialSizePx * fiducialSizePx;
	size_t deletedCoordinates;

	#ifdef DEBUG_REMOVE_DUPLICATES
	size_t iteration = 0;
	std::cout << "maxDistance " << maxDistance  << std::endl;
	#endif

	do
	{
		#ifdef DEBUG_REMOVE_DUPLICATES
		iteration +=1;
		std::cout << "STARTING ITERATION " << iteration<< std::endl;
		std::cout << "coordinates3D.size() " << coordinates3D.size() << std::endl;
		#endif

		std::vector<Point3D<double>> newCoordinates3D;
		size_t numberOfFeatures = coordinates3D.size();
		std::vector<size_t> deleteCoordinatesVector(numberOfFeatures, 0);

		for(size_t i = 0; i < numberOfFeatures; i++)
		{
			for(size_t j = i+1; j < numberOfFeatures; j++)
			{
				double distance = (coordinates3D[i].x - coordinates3D[j].x) * (coordinates3D[i].x - coordinates3D[j].x) +
								  (coordinates3D[i].y - coordinates3D[j].y) * (coordinates3D[i].y - coordinates3D[j].y) +
								  (coordinates3D[i].z - coordinates3D[j].z) * (coordinates3D[i].z - coordinates3D[j].z);

				if (distance < maxDistance && deleteCoordinatesVector[i] == 0 && deleteCoordinatesVector[j] == 0)
				{
					#ifdef DEBUG_REMOVE_DUPLICATES
					std::cout << "distance match between coordinates " << i << " and " << j  << std::endl;
					#endif

					Point3D<double> p((coordinates3D[i].x + coordinates3D[j].x)/2, 
									  (coordinates3D[i].y + coordinates3D[j].y)/2, 
									  (coordinates3D[i].z + coordinates3D[j].z)/2);
					newCoordinates3D.push_back(p);
					
					deleteCoordinatesVector[i] = 1;
					deleteCoordinatesVector[j] = 1;
				}
			}
		}

		#ifdef DEBUG_REMOVE_DUPLICATES
		std::cout << "coordinates3D.size() " << coordinates3D.size() << std::endl;
		#endif

		deletedCoordinates = 0;
		for (size_t i = 0; i < deleteCoordinatesVector.size(); i++)
		{
			if (deleteCoordinatesVector[i] == 1)
			{
				coordinates3D.erase(coordinates3D.begin()+i-deletedCoordinates);
				deletedCoordinates++;
			}	
		}

		#ifdef DEBUG_REMOVE_DUPLICATES
		std::cout << "deletedCoordinates " << deletedCoordinates << std::endl;
		std::cout << "coordinates3D.size() " << coordinates3D.size() << std::endl;
		std::cout << "newCoordinates3D.size() " << newCoordinates3D.size() << std::endl;
		#endif

		for (size_t i = 0; i < newCoordinates3D.size(); i++)
		{
			coordinates3D.push_back(newCoordinates3D[i]);
		}

		#ifdef DEBUG_REMOVE_DUPLICATES
		std::cout << "coordinates3D.size() " << coordinates3D.size() << std::endl;
		#endif

		newCoordinates3D.clear();
	}
	while (deletedCoordinates>0);	

	#ifdef VERBOSE_OUTPUT
	std::cout << "Removing duplicated coordinates finished succesfully!!" << std::endl;
	#endif
}



void ProgImagePeakHighContrast::filterCoordinatesByCorrelation(MultidimArray<double> volFiltered)
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Filter coordinates by correlation..." << std::endl;
	#endif

	size_t halfBoxSize = boxSize / 2;
	size_t numberOfFeatures = coordinates3D.size();

	MultidimArray<double> feature;
	MultidimArray<double> mirrorFeature;
	double dotProduct = 0;
	double dotProductMirror = 0;

	int coordHalfX;
	int coordHalfY;
	int coordHalfZ;

	for(size_t n = 0; n < numberOfFeatures; n++)
	{
		// Construct feature and its mirror symmetric
		feature.initZeros(boxSize, boxSize, boxSize);
		mirrorFeature.initZeros(boxSize, boxSize, boxSize);
		
		for(int k = 0; k < boxSize; k++) // zDim
		{	
			for(int j = 0; j < boxSize; j++) // xDim
			{
				for(int i = 0; i < boxSize; i++) // yDim
				{
					coordHalfX = coordinates3D[n].x - halfBoxSize;
					coordHalfY = coordinates3D[n].y - halfBoxSize;
					coordHalfZ = coordinates3D[n].z - halfBoxSize;

					// Check coordinate is not out of volume
					if ((coordHalfZ + k) < 0 || (coordHalfZ + k) > zSize ||
					    (coordHalfY + i) < 0 || (coordHalfY + i) > ySize ||
						(coordHalfX + j) < 0 || (coordHalfX + j) > xSize)
					{
						DIRECT_A3D_ELEM(feature, k, i, j) = 0;

						DIRECT_A3D_ELEM(mirrorFeature, boxSize -1 - k, boxSize -1 - i, boxSize -1 - j) = 0;
					}
					else
					{
						DIRECT_A3D_ELEM(feature, k, i, j) = DIRECT_A3D_ELEM(volFiltered, 
																			coordHalfZ + k, 
																			coordHalfY + i, 
																			coordHalfX + j);

						DIRECT_A3D_ELEM(mirrorFeature, boxSize -1 - k, boxSize -1 - i, boxSize -1 - j) = 
						DIRECT_A3D_ELEM(volFiltered, 
										coordHalfZ + k, 
										coordHalfY + i,
										coordHalfX + j);
					}
				}
			}
		}

		// feature.rangeAdjust(-1, 1);
		// mirrorFeature.rangeAdjust(-1, 1);

		feature.statisticsAdjust(0.0, 1.0);
		mirrorFeature.statisticsAdjust(0.0, 1.0);

		// Calculate scalar product
		for(int k = 0; k < boxSize; k++) // zDim
		{	
			for(int j = 0; j < boxSize; j++) // xDim
			{
				for(int i = 0; i < boxSize; i++) // yDim
				{
					dotProduct += DIRECT_A3D_ELEM(feature, k, i, j) * DIRECT_A3D_ELEM(feature, k, i, j);
					dotProductMirror += DIRECT_A3D_ELEM(feature, k, i, j) * DIRECT_A3D_ELEM(mirrorFeature, k, i, j);
				}
			}
		}

		dotProduct /= boxSize *  boxSize * boxSize;
		dotProductMirror /= boxSize *  boxSize * boxSize;

		#ifdef DEBUG_FILTER_COORDINATES
		std::cout << "-------------------- coordinate " << n << std::endl;
		std::cout << "dot product: " << dotProduct << std::endl;
		std::cout << "dot product mirror: " << dotProductMirror << std::endl;
		std::cout << "quotient: " << dotProduct/dotProductMirror << std::endl;
		#endif
	}

	#ifdef VERBOSE_OUTPUT
	std::cout << "Filtering coordinates by correlation finished succesfully!!" << std::endl;
	#endif
}


void ProgImagePeakHighContrast::writeOutputCoordinates()
{
	MetaDataVec md;
	size_t id;

	for(size_t i = 0 ;i < coordinates3D.size(); i++)
	{
		id = md.addObject();
		md.setValue(MDL_XCOOR, (int)coordinates3D[i].x, id);
		md.setValue(MDL_YCOOR, (int)coordinates3D[i].y, id);
		md.setValue(MDL_ZCOOR, (int)coordinates3D[i].z, id);
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

	generateSideInfo();
	
	V.read(fnVol);

	MultidimArray<double> &inputTomo=V();

	xSize = XSIZE(inputTomo);
	ySize = YSIZE(inputTomo);
	zSize = ZSIZE(inputTomo);
	nSize = NSIZE(inputTomo);

	#ifdef DEBUG_DIM
	std::cout << "------------------ Input tomogram dimensions:" << std::endl;
	std::cout << "x " << xSize << std::endl;
	std::cout << "y " << ySize << std::endl;
	std::cout << "z " << zSize << std::endl;
	std::cout << "n " << nSize << std::endl;
	#endif

 	preprocessVolume(inputTomo);

	#ifdef DEBUG_DIM
	std::cout << "------------------ Preprocessed tomogram dimensions:" << std::endl;
	std::cout << "x " << xSize << std::endl;
	std::cout << "y " << ySize << std::endl;
	std::cout << "z " << zSize << std::endl;
	std::cout << "n " << nSize << std::endl;
	#endif
	
	getHighContrastCoordinates(inputTomo);

	// clusterHighContrastCoordinates();
	clusterHCC();

	if(centerFeatures==true)
	{
		V.read(fnVol);
		MultidimArray<double> &inputTomo=V();

		centerCoordinates(inputTomo);
	}

	removeDuplicatedCoordinates();

	filterCoordinatesByCorrelation(inputTomo);

	writeOutputCoordinates();
	
	auto t2 = high_resolution_clock::now();
	/* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
}



// --------------------------- UTILS functions ----------------------------

bool ProgImagePeakHighContrast::filterLabeledRegions(std::vector<int> coordinatesPerLabelX, std::vector<int> coordinatesPerLabelY, double centroX, double centroY)
{
	#ifdef DEBUG_FILTERLABEL
	// // Uncomment for phantom
	// std::cout << "No label filtering, phantom mode!!" << std::endl;
	// return true;
	#endif

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

	if(ocupation < 0.75)
	{
		#ifdef DEBUG_FILTERLABEL
		std::cout << "COORDINATE REMOVED AT " << centroX << " , " << centroY << " BECAUSE OF OCCUPATION"<< std::endl;
		#endif
		return false;
	}

	return true;
}



std::vector<size_t> ProgImagePeakHighContrast::getCoordinatesInSliceIndex(size_t slice)
{
	std::vector<size_t> coordinatesInSlice;

	#ifdef DEBUG_COORDS_IN_SLICE
	std::cout << "Geting coordinates from slice " << slice << std::endl;
	#endif

	for(size_t n = 0; n < coordinates3D.size(); n++)
	{
		if(slice == coordinates3D[n].z)
		{
			coordinatesInSlice.push_back(n);
		}
	}

	#ifdef DEBUG_COORDS_IN_SLICE
	std::cout << "Number of coordinates found:  " << coordinatesInSlice.size() << std::endl;
	#endif

	return coordinatesInSlice;
}


// std::vector<size_t> ProgImagePeakHighContrast::smoothingEdges(size_t slice)
// {
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
// }