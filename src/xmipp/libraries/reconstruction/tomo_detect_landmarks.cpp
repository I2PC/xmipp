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

#include "tomo_detect_landmarks.h"
#include <chrono>



// --------------------------- INFO functions ----------------------------
void ProgTomoDetectLandmarks::readParams()
{
	fnVol = getParam("-i");
	fnOut = getParam("-o");

	samplingRate = getDoubleParam("--samplingRate");
	fiducialSize = getDoubleParam("--fiducialSize");

    targetFS = getDoubleParam("--targetLMsize");
    thrSD = getIntParam("--thrSD");
}


void ProgTomoDetectLandmarks::defineParams()
{
	addUsageLine("This function determines the location of high contrast features in a volume.");
	addParamsLine("  -i <mrcs_file=\"\">                   	    : Input tilt-series.");
	addParamsLine("  [-o <output=\"./landmarkCoordinates.xmd\">]    : Output file containing the alignemnt report.");

	addParamsLine("  [--samplingRate <samplingRate=1>]			: Sampling rate of the input tomogram (A/px).");
	addParamsLine("  [--fiducialSize <fiducialSize=100>]		: Fiducial size in Angstroms (A).");

	addParamsLine("  [--targetLMsize <targetLMsize=8>]		    : Targer size of landmark when downsampling (px).");
	addParamsLine("  [--thrSD <thrSD=5>]		    		: Number of times over the mean has to be a pixel valur to consider it an outlier.");
}


void ProgTomoDetectLandmarks::generateSideInfo()
{
	fiducialSizePx = fiducialSize / samplingRate; 

    ds_factor = targetFS / fiducialSizePx; 
    xSize_d = xSize * ds_factor;
    ySize_d = ySize * ds_factor;

    std::cout << "Generating side info: " << std::endl;
    std::cout << "fiducialSizePx: " << fiducialSizePx << std::endl;
    std::cout << "ds_factor: " << ds_factor << std::endl;
    std::cout << "xSize_d: " << xSize_d << std::endl;
    std::cout << "ySize_d: " << ySize_d << std::endl;
}



// --------------------------- HEAD functions ----------------------------
void ProgTomoDetectLandmarks::downsample(MultidimArray<double> &tiltImage, MultidimArray<double> &tiltImage_ds)
{
    MultidimArray<std::complex<double>> fftImg;
	MultidimArray<std::complex<double>> fftImg_ds;

	FourierTransformer transformer1;
	FourierTransformer transformer2;

    fftImg_ds.initZeros(ySize_d, xSize_d/2+1);
    transformer1.FourierTransform(tiltImage, fftImg, false);


    for (size_t i = 0; i < ySize_d/2; ++i)
    {
        for (size_t j = 0; j < xSize_d/2; ++j)
        {
            DIRECT_A2D_ELEM(fftImg_ds, i, j) = DIRECT_A2D_ELEM(fftImg, i, j);
            DIRECT_A2D_ELEM(fftImg_ds, (ySize_d/2)+i, j) = DIRECT_A2D_ELEM(fftImg, ySize-ySize_d/2+i, j);
        }
    }

    transformer2.inverseFourierTransform(fftImg_ds, tiltImage_ds);
}


void ProgTomoDetectLandmarks::sobelFiler(MultidimArray<double> &tiltImage)
{  
    // Create the gradient images for x and y directions
    MultidimArray<double>  gradX;
    MultidimArray<double>  gradY;

    gradX.initZeros(ySize_d, xSize_d);
    gradY.initZeros(ySize_d, xSize_d);

    // Apply the Sobel filter in the x-direction
    for (int i = 1; i < ySize_d - 1; ++i)
    {
        for (int j = 1; j < xSize_d - 1; ++j)
        {
            double pixelValue = 0;

            for (int k = -1; k <= 1; ++k)
            {
                for (int l = -1; l <= 1; ++l)
                {
                    pixelValue += A2D_ELEM(tiltImage, i + k, j + l) * sobelX[k + 1][l + 1];
                }
            }

            A2D_ELEM(gradX, i, j) = pixelValue;
        }
    }

    // Apply the Sobel filter in the y-direction
    for (int i = 1; i < ySize_d - 1; ++i)
    {
        for (int j = 1; j < xSize_d - 1; ++j)
        {
            double pixelValue = 0;

            for (int k = -1; k <= 1; ++k)
            {
                for (int l = -1; l <= 1; ++l)
                {
                    pixelValue += A2D_ELEM(tiltImage, i + k, j + l) * sobelY[k + 1][l + 1];
                }
            }

            A2D_ELEM(gradY, i, j) = pixelValue;
        }
    }

    // Compute the gradient magnitude   
    tiltImage.initZeros(ySize_d, xSize_d);

    for (int i = 0; i < ySize_d; ++i)
    {
        for (int j = 0; j < xSize_d; ++j)
        {

            A2D_ELEM(tiltImage, i, j) = sqrt(A2D_ELEM(gradX, i, j) * A2D_ELEM(gradX, i, j) + 
                                             A2D_ELEM(gradY, i, j) * A2D_ELEM(gradY, i, j));
        }
    }
}


void ProgTomoDetectLandmarks::enhanceLandmarks(MultidimArray<double> &tiltImage)
{
	MultidimArray<double> tiltImage_enhanced;
	CorrelationAux aux;

	correlation_matrix(tiltImage, landmarkReference, tiltImage_enhanced, aux, true);
	tiltImage = tiltImage_enhanced;
}


void ProgTomoDetectLandmarks::getHighContrastCoordinates(MultidimArray<double> tiltSeriesFiltered)
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Picking high contrast coordinates..." << std::endl;
	#endif

    MultidimArray<double> binaryCoordinatesMapSlice;
    MultidimArray<double> labelCoordiantesMapSlice;
    MultidimArray<double> labelCoordiantesMap;

	labelCoordiantesMap.initZeros(nSize, zSize, ySize_d, xSize_d);

	for(size_t k = 0; k < nSize; ++k)
	{
		#ifdef DEBUG_HCC
		std::cout <<  "Searching for high contrast coordinates in tilt-image " << k << std::endl;
		#endif

		std::vector<int> sliceVector;

		for (size_t j = 0; j < ySize_d; j++)
		{
			for (size_t i = 0; i < xSize_d; i++)
			{
				sliceVector.push_back(DIRECT_NZYX_ELEM(tiltSeriesFiltered, k, 0, j ,i));
			}
		}

        double sum = 0;
		double sum2 = 0;
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

        double thresholdU = average + thrSD * standardDeviation;

        #ifdef DEBUG_HCC
		std::cout << "------------------------------------------------------" << std::endl;
		std::cout << "Slice: " << k+1 << " Average: " << average << " SD: " << standardDeviation << std::endl;
        #endif

        binaryCoordinatesMapSlice.initZeros(ySize_d, xSize_d);
        labelCoordiantesMapSlice.initZeros(ySize_d, xSize_d);

        for(size_t i = 0; i < ySize_d; i++)
        {
            for(size_t j = 0; j < xSize_d; ++j)
            {
                double value = DIRECT_A3D_ELEM(tiltSeriesFiltered, k, i, j);

                if (value > thresholdU)
                {
                    DIRECT_A2D_ELEM(binaryCoordinatesMapSlice, i, j) = 1.0;
                }
            }
        }

        int colour;
        colour = labelImage2D(binaryCoordinatesMapSlice, labelCoordiantesMapSlice, 8);

        #ifdef DEBUG_HCC
        std::cout << "Colour: " << colour << std::endl;
        #endif

        std::vector<std::vector<int>> coordinatesPerLabelX (colour);
        std::vector<std::vector<int>> coordinatesPerLabelY (colour);

        for(size_t i = 0; i < ySize_d; i++)
        {
            for(size_t j = 0; j < xSize_d; ++j)
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
		#ifdef DEBUG_FILTERLABEL
		std::cout <<  "Filtering labels in tilt-image " << k << std::endl;
		#endif

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
        		Point3D<double> point3D(xCoorCM/ds_factor, yCoorCM/ds_factor, k);
        		coordinates3D.push_back(point3D);
        	}
        }

        for(size_t i = 0; i < ySize_d; i++)
        {
            for(size_t j = 0; j < xSize_d; ++j)
            {
                double value = DIRECT_A2D_ELEM(labelCoordiantesMapSlice, i, j);

                if (value > 0)
                {
                    DIRECT_NZYX_ELEM(labelCoordiantesMap, k, 0, i, j) = value;
                }
            }
        }
    }

	#ifdef VERBOSE_OUTPUT
	std::cout << "Number of peaked coordinates: " << coordinates3D.size() << std::endl;
	#endif

    #ifdef DEBUG_HCC
    std::cout << "--- List do peaked coordinates: " << std::endl;
    for (size_t i = 0; i < coordinates3D.size(); i++)
    {
        std::cout << "(" << coordinates3D[i].x << ", " << 
                            coordinates3D[i].y << ", " << 
                            coordinates3D[i].z << ")"  << std::endl;
    }
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

    // Generate output labeled_filtered series
	#ifdef DEBUG_OUTPUT_FILES
	MultidimArray<int> filteredLabeledTS;
	filteredLabeledTS.initZeros(nSize, zSize, ySize_d, xSize_d);

    std::vector<Point2D<double>> cis;

	for (size_t n = 0; n < nSize; n++)
	{
        cis.clear();

        // Get coordinate in slice n
		Point2D<double> coordinate(0, 0);

        for(size_t k = 0; k < coordinates3D.size(); k++)
        {
            if(n == coordinates3D[k].z)
            {
                coordinate.x = coordinates3D[k].x;
                coordinate.y = coordinates3D[k].y;
                cis.push_back(coordinate);
            }
        }

        std::cout << "--- cis at image " << n << std::endl;
        for (size_t i = 0; i < cis.size(); i++)
        {
            std::cout << "(" << cis[i].x << ", " << 
                                cis[i].y << ")"  << std::endl;
        }

        // Paint dots in tilt-image
		MultidimArray<int> filteredLabeledTS_Image;
		filteredLabeledTS_Image.initZeros(ySize_d, xSize_d);

		for(size_t i = 0; i < cis.size(); i++)
		{
            int x = (int)cis[i].x;
            int y = (int)cis[i].y;
            int value = 1;

            int beadSize = (int)(targetFS*0.75);

            for (int i = -beadSize; i <= beadSize; i++)
            {
                for (int j = -beadSize; j <= beadSize; j++)
                {
                    if (j + y > 0 && i + x  > 0 && j + y < ySize_d && i + x < xSize_d && i*i+j*j <= beadSize*beadSize)
                    {
                        DIRECT_A2D_ELEM(filteredLabeledTS_Image, (j + y), (i + x)) = value;
                    }
                }
            }
		}

		for (size_t i = 0; i < ySize_d; ++i)
		{
			for (size_t j = 0; j < xSize_d; ++j)
			{
				DIRECT_NZYX_ELEM(filteredLabeledTS, n, 0, i, j) = DIRECT_A2D_ELEM(filteredLabeledTS_Image, i, j);
			}
		}
	}

	size_t lastindexBis = fnOut.find_last_of("\\/");
	std::string rawnameBis = fnOut.substr(0, lastindexBis);
	std::string outputFileNameFilteredVolumeBis;
    outputFileNameFilteredVolumeBis = rawnameBis + "/ts_labeled_filtered.mrcs";

	Image<int> saveImageBis;
	saveImageBis() = filteredLabeledTS;
	saveImageBis.write(outputFileNameFilteredVolumeBis);
	#endif
}


void ProgTomoDetectLandmarks::centerCoordinates(MultidimArray<double> tiltSeries)
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
	int ti;

	int boxSize = int(fiducialSizePx);
	int doubleBoxSize = fiducialSizePx * 2;

	#ifdef DEBUG_CENTER_COORDINATES
	std::cout << "Tilt-series dimensions:" << std::endl;
	std::cout << "x " << XSIZE(tiltSeries) << std::endl;
	std::cout << "y " << YSIZE(tiltSeries) << std::endl;
	std::cout << "z " << ZSIZE(tiltSeries) << std::endl;
	std::cout << "n " << NSIZE(tiltSeries) << std::endl;
	#endif

	for(size_t n = 0; n < numberOfFeatures; n++)
	{
		#ifdef DEBUG_CENTER_COORDINATES
		std::cout << "-------------------- coordinate " << n << " (" << coordinates3D[n].x << ", " << coordinates3D[n].y << ", " << coordinates3D[n].z << ")" << std::endl;
		#endif

		// Construct feature and its mirror symmetric. We quadruple the size to include a feature two times
		// the box size plus padding to avoid incoherences in the shift sign
		feature.initZeros(2 * doubleBoxSize, 2 * doubleBoxSize);
		mirrorFeature.initZeros(2 * doubleBoxSize, 2 * doubleBoxSize);

		coordHalfX = coordinates3D[n].x - boxSize;
		coordHalfY = coordinates3D[n].y - boxSize;
		ti = coordinates3D[n].z;

		for(int i = 0; i < doubleBoxSize; i++) // yDim
		{
			for(int j = 0; j < doubleBoxSize; j++) // xDim
			{
				// Check coordinate is not out of volume
				if ((coordHalfY + i) < 0 || (coordHalfY + i) > ySize ||
					(coordHalfX + j) < 0 || (coordHalfX + j) > xSize)
				{
					DIRECT_A2D_ELEM(feature, i + boxSize, j + boxSize) = 0;

					DIRECT_A2D_ELEM(mirrorFeature, doubleBoxSize + boxSize -1 - i, doubleBoxSize + boxSize -1 - j) = 0;
				}
				else
				{
					DIRECT_A2D_ELEM(feature, i + boxSize, j + boxSize) = DIRECT_A3D_ELEM(tiltSeries, 
																		ti, 
																		coordHalfY + i, 
																		coordHalfX + j);

					DIRECT_A2D_ELEM(mirrorFeature, doubleBoxSize + boxSize -1 - i, doubleBoxSize + boxSize -1 - j) = 
					DIRECT_A3D_ELEM(tiltSeries, 
									ti, 
									coordHalfY + i,
									coordHalfX + j);
				}
			}
		}

		#ifdef DEBUG_CENTER_COORDINATES
		Image<double> image;

		std::cout << "Feature dimensions (" << XSIZE(feature) << ", " << YSIZE(feature) << ", " << ZSIZE(feature) << ")" << std::endl;
		image() = feature;
		size_t lastindex = fnOut.find_last_of(".");
		std::string rawname = fnOut.substr(0, lastindex);
		std::string outputFileName;
		outputFileName = rawname + "_" + std::to_string(n) + "_feature.mrc";
		image.write(outputFileName);

		std::cout << "Mirror feature dimensions (" << XSIZE(mirrorFeature) << ", " << YSIZE(mirrorFeature) << ", " << ZSIZE(mirrorFeature) << ")" << std::endl;
		image() = mirrorFeature;
		outputFileName = rawname + "_" + std::to_string(n) + "_mirrorFeature.mrc";
		image.write(outputFileName);
		#endif

		// Shift the particle respect to its symmetric to look for the maximum correlation displacement
		CorrelationAux aux;
		correlation_matrix(feature, mirrorFeature, correlationVolumeR, aux, true);

		auto maximumCorrelation = MINDOUBLE;
		int xDisplacement = 0;
		int yDisplacement = 0;

		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(correlationVolumeR)
		{
			if (n==0)
			{
				std::cout << "Pixel (" << i << ", " << j << ")" << std::endl;
			}

			double value = DIRECT_A2D_ELEM(correlationVolumeR, i, j);

			if (value > maximumCorrelation)
			{
				std::cout << "new maximumCorrelation " << value << " at (" << i << ", " << j << ")" << std::endl;

				maximumCorrelation = value;
				xDisplacement = j;
				yDisplacement = i;
			}
		}

		#ifdef DEBUG_CENTER_COORDINATES
		std::cout << "maximumCorrelation " << maximumCorrelation << std::endl;
		std::cout << "xDisplacement " << (xDisplacement - doubleBoxSize) / 2 << std::endl;
		std::cout << "yDisplacement " << (yDisplacement - doubleBoxSize) / 2 << std::endl;

		std::cout << "Correlation volume dimensions (" << XSIZE(correlationVolumeR) << ", " << YSIZE(correlationVolumeR) << ")" << std::endl;
		#endif


		// Update coordinate and remove if it is moved out of the volume
		double updatedCoordinateX = coordinates3D[n].x + (xDisplacement - doubleBoxSize) / 2;
		double updatedCoordinateY = coordinates3D[n].y + (yDisplacement - doubleBoxSize) / 2;

		int deletedCoordinates = 0;

		if (updatedCoordinateY < 0 || updatedCoordinateY > ySize ||
			updatedCoordinateX < 0 || updatedCoordinateX > xSize)
		{
			coordinates3D.erase(coordinates3D.begin()+n-deletedCoordinates);
			deletedCoordinates++;
		}
		else
		{
			coordinates3D[n].x = updatedCoordinateX;
			coordinates3D[n].y = updatedCoordinateY;
		}

		#ifdef DEBUG_CENTER_COORDINATES
		// Construct and save the centered feature
		MultidimArray<double> centerFeature;

		centerFeature.initZeros(doubleBoxSize, doubleBoxSize);

		coordHalfX = coordinates3D[n].x - boxSize;
		coordHalfY = coordinates3D[n].y - boxSize;

		for(int j = 0; j < doubleBoxSize; j++) // xDim
		{
			for(int i = 0; i < doubleBoxSize; i++) // yDim
			{
				// Check coordinate is not out of volume
				if ((coordHalfY + i) < 0 || (coordHalfY + i) > ySize ||
					(coordHalfX + j) < 0 || (coordHalfX + j) > xSize)
				{
					DIRECT_A2D_ELEM(centerFeature, i, j) = 0;
				}
				else
				{
					DIRECT_A2D_ELEM(centerFeature, i, j) = DIRECT_A3D_ELEM(tiltSeries,
																				ti,
																				coordHalfY + i,
																				coordHalfX + j);
				}
			}
		}

		std::cout << "Centered feature dimensions (" << XSIZE(centerFeature) << ", " << YSIZE(centerFeature) << ")" << std::endl;

		image() = centerFeature;
		outputFileName = rawname + "_" + std::to_string(n) + "_centerFeature.mrc";
		image.write(outputFileName);
		#endif
	}

	#ifdef DEBUG_CENTER_COORDINATES
	std::cout << "3D coordinates after centering: " << std::endl;

	for(size_t n = 0; n < numberOfFeatures; n++)
	{
		std::cout << "Coordinate " << n << " (" << coordinates3D[n].x << ", " << coordinates3D[n].y << ", " << coordinates3D[n].z << ")" << std::endl;

	}
	#endif

	#ifdef VERBOSE_OUTPUT
	std::cout << "Centering of coordinates finished successfully!" << std::endl;
	#endif
}



// --------------------------- I/O functions ----------------------------
void ProgTomoDetectLandmarks::writeOutputCoordinates()
{
	MetaDataVec md;
	size_t id;

	for(size_t i = 0; i < coordinates3D.size(); i++)
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
void ProgTomoDetectLandmarks::run()
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

	#ifdef DEBUG_DIM
	std::cout << "Input tilt-series dimensions:" << std::endl;
	std::cout << "x " << xSize << std::endl;
	std::cout << "y " << ySize << std::endl;
	std::cout << "z " << zSize << std::endl;
	std::cout << "n " << nSize << std::endl;
	#endif

	generateSideInfo();

	FileName fnTSimg;
	size_t objId, objId_ts;
	Image<double> imgTS;

	MultidimArray<double> &ptrImg = imgTS();

	size_t Ndim, counter = 0;
	Ndim = tiltseriesmd.size();

	MultidimArray<double> filteredTiltSeries;
	filteredTiltSeries.initZeros(Ndim, 1, ySize_d, xSize_d);

	MultidimArray<double> tiltSeries;
	tiltSeries.initZeros(Ndim, 1, ySize, xSize);

    #ifdef DEBUG_DIM
	std::cout << "Filtered tilt-series dimensions:" << std::endl;
	std::cout << "x " << XSIZE(filteredTiltSeries) << std::endl;
	std::cout << "y " << YSIZE(filteredTiltSeries) << std::endl;
	std::cout << "z " << ZSIZE(filteredTiltSeries) << std::endl;
	std::cout << "n " << NSIZE(filteredTiltSeries) << std::endl;
	#endif

	// Create phantom for landmark reference
    createLandmarkTemplate();

	for(size_t objId : tiltseriesmd.ids())
	{
		tiltseriesmd.getValue(MDL_IMAGE, fnTSimg, objId);

        imgTS.read(fnTSimg);

        MultidimArray<double> tiltImage_ds;
        tiltImage_ds.initZeros(ySize_d, xSize_d);

        #ifdef DEBUG_DOWNSAMPLE
        std::cout << "Aplying sobel filter to image " << counter << std::endl;
        #endif
        downsample(ptrImg, tiltImage_ds);

        #ifdef DEBUG_DIM
        std::cout << "Tilt-image dimensions before dowsampling:" << std::endl;
        std::cout << "x " << XSIZE(ptrImg) << std::endl;
        std::cout << "y " << YSIZE(ptrImg) << std::endl;

		std::cout << "Tilt-image dimensions after dowsampling:" << std::endl;
		std::cout << "x " << XSIZE(tiltImage_ds) << std::endl;
        std::cout << "y " << YSIZE(tiltImage_ds) << std::endl;
        #endif

		// Contruct normalized tilt-series for posterior coordinates centering
		ptrImg.statisticsAdjust(0.0, 1.0);

		for (size_t i = 0; i < ySize; ++i)
        {
            for (size_t j = 0; j < xSize; ++j)
            {
				DIRECT_NZYX_ELEM(tiltSeries, counter, 0, i, j) = DIRECT_A2D_ELEM(ptrImg, i, j);
			}
		}


        #ifdef DEBUG_SOBEL
        std::cout << "Aplying sobel filter to image " << counter << std::endl;
        #endif
        sobelFiler(tiltImage_ds);
        enhanceLandmarks(tiltImage_ds);

        for (size_t i = 0; i < ySize_d; ++i)
        {
            for (size_t j = 0; j < xSize_d; ++j)
            {
				DIRECT_NZYX_ELEM(filteredTiltSeries, counter, 0, i, j) = DIRECT_A2D_ELEM(tiltImage_ds, i, j);
			}
		}

		counter++;
	}

	#ifdef DEBUG_DIM
	std::cout << "Filtered tilt-series dimensions:" << std::endl;
	std::cout << "x " << xSize_d << std::endl;
	std::cout << "y " << ySize_d << std::endl;
	std::cout << "z " << zSize << std::endl;
	std::cout << "n " << nSize << std::endl;
	#endif

	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindex = fnOut.find_last_of("\\/");
	std::string rawname = fnOut.substr(0, lastindex);
	std::string outputFileNameFilteredVolume;
    outputFileNameFilteredVolume = rawname + "/ts_filtered.mrcs";

	Image<double> saveImage;
	saveImage() = filteredTiltSeries;
	saveImage.write(outputFileNameFilteredVolume);

	std::string tsFN;
    tsFN = rawname + "/ts.mrcs";

	saveImage() = tiltSeries;
	saveImage.write(tsFN);
	#endif

    getHighContrastCoordinates(filteredTiltSeries);

	centerCoordinates(tiltSeries);

    // Write output coordinates
    writeOutputCoordinates();

	auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1); 	// Getting number of milliseconds as an integer
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
}



// --------------------------- UTILS functions ----------------------------
bool ProgTomoDetectLandmarks::filterLabeledRegions(std::vector<int> coordinatesPerLabelX, std::vector<int> coordinatesPerLabelY, double centroX, double centroY)
{
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

	if(ocupation < 0.25)
	{
		#ifdef DEBUG_FILTERLABEL
		std::cout << "COORDINATE REMOVED AT (" << centroX << " , " << centroY << ") BECAUSE OF OCCUPATION"<< std::endl;
		std::cout << "-------------------------------------------"  << std::endl;
		#endif
		return false;
	}

	// Check the relative area compared with the expected goldbead
	double expectedArea = PI * ((targetFS) * targetFS);
	double relativeArea = (4*area)/expectedArea;  // Due to filtering and labelling processes labeled gold beads tend to reduce its radius in half

	#ifdef DEBUG_FILTERLABEL
	std::cout << "expectedArea " << expectedArea << std::endl;
	std::cout << "relativeArea " << relativeArea << std::endl;
	#endif


	if (relativeArea > 4 || relativeArea < 0.25)
	{
		#ifdef DEBUG_FILTERLABEL
		std::cout << "COORDINATE REMOVED AT " << centroX << " , " << centroY << " BECAUSE OF RELATIVE AREA"<< std::endl;
		std::cout << "-------------------------------------------"  << std::endl;
		#endif
		return false;
	}
	#ifdef DEBUG_FILTERLABEL
	std::cout << "COORDINATE NO REMOVED AT " << centroX << " , " << centroY << std::endl;
	std::cout << "-------------------------------------------"  << std::endl;
	#endif
	return true;
}


void ProgTomoDetectLandmarks::createLandmarkTemplate()
{
	// Generate first reference
    int targetFS_half = targetFS/2;
    int targetFS_half_sq = targetFS_half*targetFS_half;

    landmarkReference.initZeros(ySize_d, xSize_d);

    // Create tilt-image with a single landamrk
    for (int k = -targetFS_half; k <= targetFS_half; ++k)
    {
        for (int l = -targetFS_half; l <= targetFS_half; ++l)
        {
            if ((k*k+l*l) < targetFS_half_sq)
            {
                A2D_ELEM(landmarkReference, ySize_d/2 + k, xSize_d/2 + l) = 1;
            }
        }
    }

    // Apply Sobel filer to reference
    sobelFiler(landmarkReference);

    // Save reference
    #ifdef DEBUG_REFERENCE
    size_t li = fnOut.find_last_of("\\/");
	std::string rn = fnOut.substr(0, li);
	std::string outFN;
    outFN = rn + "/landmarkReference.mrcs";

	Image<double> si;
	si() = landmarkReference;
	si.write(outFN);
    #endif
}
