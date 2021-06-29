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

void ProgTomoDetectMisalignmentTrajectory::readParams()
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


void ProgTomoDetectMisalignmentTrajectory::defineParams()
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


void ProgTomoDetectMisalignmentTrajectory::bandPassFilter(MultidimArray<double> &origImg)
{
	// imgTofilter.resizeNoCopy(origImg);
	FourierTransformer transformer1(FFTW_BACKWARD);
	MultidimArray<std::complex<double>> fftImg;
	transformer1.FourierTransform(origImg, fftImg, true);

	// Filter frequencies
	double highFreqFilt = samplingRate/fiducialSize;
	double tail = highFreqFilt + 0.02;

    double lowFreqFilt = samplingRate/fiducialSize;

	double idelta = PI/(highFreqFilt-tail);

    double uy, ux, u, uy2;

    size_t ydimImg = YSIZE(origImg);
    size_t xdimImg = XSIZE(origImg);

	long n=0;
	for(size_t i=0; i<YSIZE(fftImg); ++i)
	{
		FFT_IDX2DIGFREQ(i, ydimImg, uy);
		uy2=uy*uy;
		for(size_t j=0; j<XSIZE(fftImg); ++j)
		{
			FFT_IDX2DIGFREQ(j, xdimImg, ux);
			u=sqrt(uy2+ux*ux);
            if (u>=highFreqFilt && u<=lowFreqFilt)
            {
                DIRECT_MULTIDIM_ELEM(fftImg, n) *= 0.5*(1+cos((u-highFreqFilt)*idelta));//H;
            }
            else if (u>tail)
            {
                DIRECT_MULTIDIM_ELEM(fftImg, n) = 0;
            }
			++n;
		}
	}

	transformer1.inverseFourierTransform(fftImg, origImg);
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
		std::cout << "----------------------------------------------- Processing slide " << k << std::endl;
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

        #ifdef VERBOSE_OUTPUT
		std::cout << "Average: " << average << std::endl;
 		std::cout << "SD: " << standardDeviation << std::endl;
		#endif

        double threshold = average - sdThreshold * standardDeviation;

        #ifdef VERBOSE_OUTPUT
        std::cout<< "Threshold: " << threshold << std::endl;
        #endif

		binaryCoordinatesMapSlice.initZeros(ySize, xSize);

		// *** test
		int test = 0;

		for(size_t i = 0; i < ySize; i++)
		{
			for(size_t j = 0; j < xSize; j++)
			{
				double value = DIRECT_A3D_ELEM(tiltSeriesFiltered, k, i, j);

				if (value < threshold)
				{
					DIRECT_A2D_ELEM(binaryCoordinatesMapSlice, i, j) = 1.0;
					test += 1;
				}
			}
		}

		std::cout << "Number of points in the binary map: " << test << std::endl;

		#ifdef DEBUG
		std::cout << "Labelling slice " << k << std::endl;
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


		std::cout << "coordinatesPerLabelX" << coordinatesPerLabelX.size() << std::endl;
		std::cout << "coordinatesPerLabelY" << coordinatesPerLabelY.size() << std::endl;

		int fedetest = 0;
		size_t numberOfCoordinatesPerValue;

		// Trim coordinates thresholding the number of elements per label
		for(size_t value = 0; value < colour; value++)
		{
			numberOfCoordinatesPerValue =  coordinatesPerLabelX[value].size();

			if(numberOfCoordinatesPerValue > numberOfCoordinatesThr)
			{
				int xCoor = 0;
				int yCoor = 0;

				for(size_t coordinate=0; coordinate < coordinatesPerLabelX[value].size(); coordinate++)
				{
					xCoor += coordinatesPerLabelX[value][coordinate];
					yCoor += coordinatesPerLabelY[value][coordinate];
				}

				coordinates3Dx.push_back(xCoor/coordinatesPerLabelX[value].size());
				coordinates3Dy.push_back(yCoor/coordinatesPerLabelY[value].size());
				coordinates3Dn.push_back(k);
				fedetest += 1;
			}
		}

		#ifdef DEBUG
		std::cout << "Number of coordinates added " << fedetest <<std::endl;
		std::cout << "coordinates3Dx.size()=" << coordinates3Dx.size() <<std::endl;
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


void ProgTomoDetectMisalignmentTrajectory::clusterHighContrastCoordinates()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Clustering coordinates..." << std::endl;
	#endif

	// These vectors accumulate each coordinate attracted by every center of mass of calculate its mean at the end
	std::vector<std::vector<int>> centerOfMassXAcc;
	std::vector<std::vector<int>> centerOfMassYAcc;
	std::vector<std::vector<int>> centerOfMassZAcc;

	// for(int i=0;i<numberCenterOfMass;i++)
	// {
	// 	int randomIndex = rand() % coordinates3Dx.size();

	// 	int cx = coordinates3Dx[randomIndex];
	// 	int cy = coordinates3Dy[randomIndex];
	// 	int cz = coordinates3Dz[randomIndex];
	// 	centerOfMassX.push_back(cx);
	// 	centerOfMassY.push_back(cy);
	// 	centerOfMassZ.push_back(cz);

	// 	std::vector<int> newCenterOfMassX;
	// 	std::vector<int> newCenterOfMassY;
	// 	std::vector<int> newCenterOfMassZ;

	// 	newCenterOfMassX.push_back(cx);
	// 	newCenterOfMassY.push_back(cy);
	// 	newCenterOfMassZ.push_back(cz);

	// 	centerOfMassXAcc.push_back(newCenterOfMassX);
	// 	centerOfMassYAcc.push_back(newCenterOfMassY);
	// 	centerOfMassZAcc.push_back(newCenterOfMassZ);
	// }

	int squareDistanceThr = distanceThr*distanceThr;
	bool attractedToMassCenter = false;

	for(size_t m = 0; m < coordinates3Dx.size(); m++)
	{
		// Check if the coordinate is attracted to any centre of mass
		attractedToMassCenter = false; 

		int xCoor = coordinates3Dx[m];
		int yCoor = coordinates3Dy[m];
		int zCoor = coordinates3Dz[m];

		for(size_t n = 0; n < centerOfMassX.size(); n++)
		{
			int xCM = centerOfMassX[n];
			int yCM = centerOfMassY[n];
			int zCM = centerOfMassZ[n];

			int squareDistance = (xCoor-xCM)*(xCoor-xCM)+(yCoor-yCM)*(yCoor-yCM)+(zCoor-zCM)*(zCoor-zCM);
			
			#ifdef DEBUG_DIST
			std::cout << "-----------------------------------------------------------------------" << std::endl;
			std::cout << "distance: " << squareDistance<< std::endl;
			std::cout << "threshold: " << squareDistanceThr<< std::endl;
			#endif

			if(squareDistance < squareDistanceThr)
			{
				// Update center of mass with new coordinate
				centerOfMassX[n]=xCM+(xCoor-xCM)/2;
				centerOfMassY[n]=yCM+(yCoor-yCM)/2;
				centerOfMassZ[n]=zCM+(zCoor-zCM)/2;

				// Add all the coordinate vectors to each center of mass
				centerOfMassXAcc[n].push_back(xCoor);
				centerOfMassYAcc[n].push_back(yCoor);
				centerOfMassZAcc[n].push_back(zCoor);

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

void ProgTomoDetectMisalignmentTrajectory::centerCoordinates(MultidimArray<double> volFiltered)
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Centering coordinates..." << std::endl;
	#endif

	size_t halfBoxSize = boxSize / 2;
	size_t correlationWedge = boxSize / 3;
	size_t numberOfFeatures = centerOfMassX.size();
	MultidimArray<double> feature, symmetricFeature, auxSymmetricFeature;

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
					DIRECT_A3D_ELEM(feature, k, i, j) = 
					DIRECT_A3D_ELEM(volFiltered, 
									centerOfMassZ[n] - halfBoxSize + k, 
									centerOfMassY[n] - halfBoxSize + i, 
									centerOfMassX[n] - halfBoxSize + j);

					DIRECT_A3D_ELEM(symmetricFeature, boxSize -1 - k, boxSize -1 - i, boxSize -1 - j) = 
					DIRECT_A3D_ELEM(volFiltered, 
									centerOfMassZ[n] - halfBoxSize + k, 
									centerOfMassY[n] - halfBoxSize + i,
									centerOfMassX[n] - halfBoxSize + j);
				}
			}
		}

		// Shift the particle respect to its symmetric to look for the maximum correlation displacement
		int correlationWedge = boxSize / 3;
		int xDisplacement = 0, yDisplacement = 0, zDisplacement = 0;

		double maxCorrelation = correlationIndex(feature, symmetricFeature);

		for(int kaux = -1 * correlationWedge; kaux < correlationWedge; kaux++) // zDim
		{	
			for(int jaux = -1 * correlationWedge; jaux < correlationWedge; jaux++) // xDim
			{
				for(int iaux = -1 * correlationWedge; iaux < correlationWedge; iaux++) // yDim
				{
					
					auxSymmetricFeature.initZeros(boxSize, boxSize, boxSize);
					
					// Construct auxiliar symmetric feature shifting the symmetric feature to calculate 
					// the correlation with the extracted feature

					FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(auxSymmetricFeature)
					{
						if(k + kaux >= 0 && k + kaux < boxSize - 1 &&
							j + jaux >= 0 && j + jaux < boxSize - 1 &&
							i + iaux >= 0 && i + iaux < boxSize - 1)
						{
							DIRECT_A3D_ELEM(auxSymmetricFeature, k + kaux, i + iaux, j + jaux) = 
							DIRECT_A3D_ELEM(symmetricFeature, k, i, j);	
						}
					}

					double correlation = correlationIndex(feature, auxSymmetricFeature);
					std::cout << "correlationIndex(symmetricFeature, auxSymmetricFeature)" << correlation << std::endl;

					if(correlation > maxCorrelation)
					{
						maxCorrelation = correlation;
						xDisplacement = jaux;
						yDisplacement = iaux;
						zDisplacement = kaux;
					}
				}
			}
		}

		// Update coordinate
		centerOfMassX[n] = centerOfMassX[n] + xDisplacement;
		centerOfMassY[n] = centerOfMassY[n] + yDisplacement;
		centerOfMassZ[n] = centerOfMassZ[n] + zDisplacement;
	}
}


void ProgTomoDetectMisalignmentTrajectory::writeOutputCoordinates()
{
	MetaData md;
	size_t id;

	for(size_t i=0;i<coordinates3Dx.size();i++)
	{
		id = md.addObject();
		md.setValue(MDL_XCOOR, coordinates3Dx[i], id);
		md.setValue(MDL_YCOOR, coordinates3Dy[i], id);
		md.setValue(MDL_ZCOOR, coordinates3Dn[i], id);
		// md.setValue(MDL_XCOOR, centerOfMassX[i], id);
		// md.setValue(MDL_YCOOR, centerOfMassY[i], id);
		// md.setValue(MDL_ZCOOR, centerOfMassZ[i], id);
	}

	md.write(fnOut);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Coordinates metadata saved at: " << fnOut << std::endl;
	#endif

}

void ProgTomoDetectMisalignmentTrajectory::run()
{
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

	auto t1 = high_resolution_clock::now();

	std::cout << "Starting..." << std::endl;
	size_t Xdim, Ydim;
	
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
        std::cout << fnTSimg << std::endl;
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

	// clusterHighContrastCoordinates();

	// if(centerFeatures==true)
	// {
	// 	centerCoordinates(volFiltered);
	// }

	writeOutputCoordinates();
	
	auto t2 = high_resolution_clock::now();
	/* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
}
