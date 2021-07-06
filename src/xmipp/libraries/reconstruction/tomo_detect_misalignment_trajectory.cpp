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
	addParamsLine("  [-o <output=\"coordinates3D.xmd\">]       				: Output file containing the 3D coodinates.");
	addParamsLine("  [--sdThreshold <sdThreshold=5>]      					: Number of SD a coordinate value must be over the mean to conisder that it belongs to a high contrast feature.");
  	addParamsLine("  [--numberOfCoordinatesThr <numberOfCoordinatesThr=10>]	: Minimum number of coordinates attracted to a center of mass to consider it.");
  	addParamsLine("  [--samplingRate <samplingRate=1>]						: Sampling rate of the input tomogram (A/px).");
	addParamsLine("  [--fiducialSize <fiducialSize=100>]					: Fiducial size in Angstroms (A).");
	addParamsLine("  [--inputCoord <output=\"\">]							: Input coordinates of the 3D landmarks to calculate the residual vectors.");
}


// --------------------------- HEAD functions ----------------------------


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


void ProgTomoDetectMisalignmentTrajectory::calculateResidualVectors(MetaData inputCoordMd)
{
	size_t objId;
	double tiltAngle;
	double distance;
	double maxDistance;

	Matrix2D<double> projectionMatrix;
	Matrix1D<int> goldBead3d;
	Matrix1D<double> projectedGoldBead;

	std::vector<Matrix1D<int>> coordinatesInSlice;

	goldBead3d.initZeros(3);

	// Iterate through every tilt-image
	for(size_t n = 0; n<tiltAngles.size(); n++)
	{	
		tiltAngle = tiltAngles[n];

		coordinatesInSlice = getCoordinatesInSlice(n);

		Matrix2D<double> projectionMatrix = getProjectionMatrix(tiltAngle);

		// Iterate through every input 3d gold bead coordinate
		FOR_ALL_OBJECTS_IN_METADATA(inputCoordMd)
		{
			maxDistance = 0;

			objId = __iter.objId;
			inputCoordMd.getValue(MDL_XCOOR, XX(goldBead3d), objId);
			inputCoordMd.getValue(MDL_YCOOR, YY(goldBead3d), objId);
			inputCoordMd.getValue(MDL_ZCOOR, ZZ(goldBead3d), objId);

			projectedGoldBead = projectionMatrix.operator*(goldBead3d);

			// Iterate though every coordinate in the tilt-image and calculate the maximum distance
			for(size_t i = 0; i < coordinatesInSlice.size(); i++)
			{
				distance = abs(XX(projectedGoldBead) - XX(coordinatesInSlice[i])) + abs(YY(projectedGoldBead) - YY(coordinatesInSlice[i]));

				if(maxDistance > distance)
				{
					maxDistance = distance;

					residualX.push_back(XX(coordinatesInSlice[i]) - XX(projectedGoldBead));
					residualY.push_back(YY(coordinatesInSlice[i]) - YY(projectedGoldBead));
					residualZ.push_back(n);
				}
			}
		}
	}
}


// --------------------------- I/O functions ----------------------------

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
	}

	md.write(fnOut);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Coordinates metadata saved at: " << fnOut << std::endl;
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
		DIRECT_A2D_ELEM(proyectedCoordinates, coordinates3Dy[n], coordinates3Dx[n]) = 1;
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

	if(checkInputCoord)
	{
		MetaData inputTiltAnglesMd;
		double tiltAngle;
		inputTiltAnglesMd.read(fnTiltAngles);

		FOR_ALL_OBJECTS_IN_METADATA(inputTiltAnglesMd)
		{
			objId = __iter.objId;
			inputTiltAnglesMd.getValue(MDL_ANGLE_TILT, tiltAngle, objId);
			tiltAngles.push_back(tiltAngle);
		}

		MetaData inputCoordMd;
		inputCoordMd.read(fnInputCoord);

		calculateResidualVectors(inputCoordMd);
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
	double cosTiltAngle = cos(tiltAngle);
	double sinTiltAngle = sin(tiltAngle);

	Matrix2D<double> projectionMatrix(3,3);

	MAT_ELEM(projectionMatrix, 0, 0) = cosTiltAngle;
	// MAT_ELEM(projectionMatrix, 0, 0) = 0;
	MAT_ELEM(projectionMatrix, 0, 0) = sinTiltAngle;
	// MAT_ELEM(projectionMatrix, 0, 0) = 0;
	MAT_ELEM(projectionMatrix, 0, 0) = 1;
	// MAT_ELEM(projectionMatrix, 0, 0) = 0;
	MAT_ELEM(projectionMatrix, 0, 0) = -sinTiltAngle;
	// MAT_ELEM(projectionMatrix, 0, 0) = 0;
	MAT_ELEM(projectionMatrix, 0, 0) = cosTiltAngle;

	return projectionMatrix;
}

std::vector<Matrix1D<int>> ProgTomoDetectMisalignmentTrajectory::getCoordinatesInSlice(int slice)
{
	std::vector<Matrix1D<int>> coordinatesInSlice;
	Matrix1D<int> coordinate(2);

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
