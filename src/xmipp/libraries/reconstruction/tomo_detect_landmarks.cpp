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

	#ifdef VERBOSE_OUTPUT
    std::cout << "Generating side info: " << std::endl;
    std::cout << "fiducialSizePx: " << fiducialSizePx << std::endl;
    std::cout << "ds_factor: " << ds_factor << std::endl;
    std::cout << "xSize_d: " << xSize_d << std::endl;
    std::cout << "ySize_d: " << ySize_d << std::endl;
    std::cout << "fnVol: " << fnVol << std::endl;
    std::cout << "fnOut: " << fnOut << std::endl;
    std::cout << "fiducialSize: " << fiducialSize << std::endl;
    std::cout << "targetFS: " << targetFS << std::endl;
    std::cout << "thrSD: " << thrSD << std::endl;
	#endif
}

void ProgTomoDetectLandmarks::bandpassFilter(MultidimArray<double> &tiltImage)
{
	MultidimArray< std::complex<double> > fftTI;
	transformer.FourierTransform(tiltImage, fftTI, false);

	#ifdef VERBOSE_OUTPUT
	std::cout << "Applying bandpass filter to volume..." << std::endl;
	#endif

	int n=0;

	double freqLow = samplingRate / (fiducialSize*1.2);
	double freqHigh = samplingRate/(fiducialSize*0.8);
	
	double w= 0.05;
	double cutoffFreqHigh = freqHigh + w;
	double cutoffFreqLow = freqLow - w;
	double delta = PI / w;

	normDim = (xSize>ySize) ? xSize : ySize;

	// 43.2 = 1440 * 0.03. This 43.2 value makes w = 0.03 (standard value) for an image whose bigger dimension is 1440 px.
	//w = 43.2 / normDim;

	#ifdef DEBUG_PREPROCESS
	std::cout << "samplingRate " << samplingRate << std::endl;
	std::cout << "fiducialSize " << fiducialSize << std::endl;
	std::cout << "freqLow " << freqLow << std::endl;
	std::cout << "freqHigh " << freqHigh << std::endl;
	std::cout << "cutoffFreqLow " << cutoffFreqLow << std::endl;
	std::cout << "cutoffFreqHigh " << cutoffFreqHigh << std::endl;
	#endif

	for(size_t i=0; i<YSIZE(fftTI); ++i)
	{
		double uy;
		double ux;
		double uy2;

		FFT_IDX2DIGFREQ(i,ySize,uy);
		uy2=uy*uy;

		for(size_t j=0; j<XSIZE(fftTI); ++j)
		{
			FFT_IDX2DIGFREQ(j,xSize,ux);
			double u=sqrt(uy2+ux*ux);

			if(u > cutoffFreqHigh || u < cutoffFreqLow)
			{
				DIRECT_MULTIDIM_ELEM(fftTI, n) = 0;
			} 
			else
			{
				if(u >= freqHigh && u < cutoffFreqHigh)
				{
					DIRECT_MULTIDIM_ELEM(fftTI, n) *= 0.5*(1+cos((u-freqHigh)*delta));
				}
			
				if (u <= freqLow && u > cutoffFreqLow)
				{
					DIRECT_MULTIDIM_ELEM(fftTI, n) *= 0.5*(1+cos((u-freqLow)*delta));
				}
			}
			
			++n;
		}
	}

	transformer.inverseFourierTransform();
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


void ProgTomoDetectLandmarks::detectInterpolationEdges(MultidimArray<double> &tiltImage)
{
	// Detect interpolation region
	MultidimArray<double> tmpImage = tiltImage;

	for (size_t i = 1; i < xSize-1; i++)
	{
		for (size_t j = 1; j < ySize-1; j++)
		{
			DIRECT_A2D_ELEM(tmpImage, j ,i) = (-1 * DIRECT_A2D_ELEM(tiltImage, j-1 ,i) +
											   -1 * DIRECT_A2D_ELEM(tiltImage, j+1 ,i) +
											   -1 * DIRECT_A2D_ELEM(tiltImage, j ,i-1) +
											   -1 * DIRECT_A2D_ELEM(tiltImage, j ,i+1) +
									 		    4 * DIRECT_A2D_ELEM(tiltImage, j ,i));
		}
	}

	// Background value as the median of the corners
	std::vector<double> corners{DIRECT_A2D_ELEM(tiltImage, 0, 0),
								DIRECT_A2D_ELEM(tiltImage, 0, xSize-1),
								DIRECT_A2D_ELEM(tiltImage, ySize-1, 0),
								DIRECT_A2D_ELEM(tiltImage, ySize-1, xSize-1)};

	sort(corners.begin(), corners.end(), std::greater<double>());

	backgroundValue = (corners[1]+corners[2])/2;

	// Margin thickness
	marginThickness = (int)(fiducialSizePx);

	// Fill borders (1 px) with backgound value (no affected by Laplacian)
	for (size_t j = 0; j < xSize; j++)
	{
		// First row
		DIRECT_A2D_ELEM(tiltImage, 0, j) = backgroundValue;
		DIRECT_A2D_ELEM(tmpImage, 0, j) = 0;

		// Last row
		DIRECT_A2D_ELEM(tiltImage, ySize-1, j) = backgroundValue;
		DIRECT_A2D_ELEM(tmpImage, ySize-1, j) = 0;
	}

	for (size_t i = 0; i < ySize; i++)
	{
		// First column
		DIRECT_A2D_ELEM(tiltImage, i, 0) = backgroundValue;
		DIRECT_A2D_ELEM(tmpImage, i, 0) = 0;

		// Last column
		DIRECT_A2D_ELEM(tiltImage, i, xSize-1) = backgroundValue;
		DIRECT_A2D_ELEM(tmpImage, i, xSize-1) = 0;
	}

	// Detect edges
	auto epsilon = MINDOUBLE;

	std::vector<Point2D<int>> interpolationLimits;
	Point2D<int> limitIgnoreRow (0, 0);
	interpolationLimits.push_back(limitIgnoreRow); // Limit to ignore first row

	int xMin;
	int xMax;

	#ifdef DEBUG_INTERPOLATION_EDGES
	std::cout << "Background value: " << backgroundValue << std::endl;
	std::cout << "Margin thickness: " << marginThickness << std::endl;
	#endif

	for (size_t j = 1; j < ySize-1; j++)
	{
		for (size_t i = 0; i < xSize; i++)
		{		
			if(abs(DIRECT_A2D_ELEM(tmpImage, j, i)) > epsilon)
			{
				xMin = ((i + marginThickness)>xSize) ? xSize : (i + marginThickness);

				// Fill margin thickness with background value
				for (size_t a = 0; a < xMin; a++)
				{
					DIRECT_A2D_ELEM(tiltImage, j, a) = backgroundValue;
				}
				
				break;
			}
		}

		for (size_t i = xSize-1; i > 0; i--)
		{
			if(abs(DIRECT_A2D_ELEM(tmpImage, j, i)) > epsilon)
			{
				xMax = ((i - marginThickness)<0) ? 0 : (i - marginThickness);

				// Fill margin thickness with background value
				for (size_t a = xMax; a < xSize; a++)
				{
					DIRECT_A2D_ELEM(tiltImage, j, a) = backgroundValue;
				}

				break;
			}
		}

		if (xMin >= xMax)
		{
			int value = (int) (((xMax+marginThickness)+(xMin-marginThickness))/2);
			xMax = value;
			xMin = value;
		}
		
		Point2D<int> limit (xMin, xMax);
		interpolationLimits.push_back(limit);
	}

	interpolationLimits.push_back(limitIgnoreRow); // Limit to ignore last row
	interpolationLimitsVector.push_back(interpolationLimits);

	// Apply DS factor to interpolation limits
	std::vector<Point2D<int>> interpolationLimits_ds;

	double meanXmin;
	double meanXmax;

	int windowSize = (int)(1/ds_factor);
	int windowSizeCounter = windowSize;

	for (size_t i = 0; i < interpolationLimits.size(); i++)
	{
		meanXmax += interpolationLimits[i].x * ds_factor;
		meanXmin += interpolationLimits[i].y * ds_factor;

		windowSizeCounter -= 1;
		
		if (windowSizeCounter == 0)
		{
			Point2D<int> limit ((int) (meanXmax / windowSize), 
								(int) (meanXmin / windowSize));

			interpolationLimits_ds.push_back(limit);

			windowSizeCounter = windowSize;
			meanXmax = 0;
			meanXmin = 0;
		}
	}

	interpolationLimitsVector_ds.push_back(interpolationLimits_ds);
}


void ProgTomoDetectLandmarks::sobelFiler(MultidimArray<double> &tiltImage, int ti)
{  
    // Create the gradient images for x and y directions
    MultidimArray<double>  gradX;
    MultidimArray<double>  gradY;

    gradX.initZeros(ySize_d, xSize_d);
    gradY.initZeros(ySize_d, xSize_d);

	if (ti == -1) // Do not consider interpolation edges (only for reference)
	{
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
	else // Consider interpolation edges
	{
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
}


void ProgTomoDetectLandmarks::enhanceLandmarks(MultidimArray<double> &tiltImage)
{
	MultidimArray<double> tiltImage_enhanced;
	CorrelationAux aux;

	correlation_matrix(tiltImage, landmarkReference, tiltImage_enhanced, aux, true);
	correlation_matrix(tiltImage_enhanced, landmarkReference_Gaussian, tiltImage, aux, true);

	substractBackgroundRollingBall(tiltImage, 2*targetFS);
}


void ProgTomoDetectLandmarks::getHighContrastCoordinates(MultidimArray<double> tiltSeriesFiltered)
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Picking high contrast coordinates..." << std::endl;
	#endif

    MultidimArray<double> binaryCoordinatesMapSlice;
    MultidimArray<double> labelCoordiantesMapSlice;
    MultidimArray<double> equalizedMap;
    MultidimArray<double> labelCoordiantesMap;
	MultidimArray<double> tiltImage;

	labelCoordiantesMap.initZeros(nSize, zSize, ySize_d, xSize_d);
	equalizedMap.initZeros(nSize, zSize, ySize_d, xSize_d);

	for(size_t k = 0; k < nSize; ++k)
	{	
		#ifdef DEBUG_HCC
		std::cout <<  "Searching for high contrast coordinates in tilt-image " << k << std::endl;
		#endif

		std::vector<Point2D<int>> interLim = interpolationLimitsVector_ds[k];

		// Z-SCORE THRESHOLDING ----------------------------------------------
		std::vector<int> sliceVector;
		
		for (size_t i = 0; i < ySize_d; i++)
		{
			Point2D<int> il = interLim[i];

			for (size_t j = il.x; j < il.y; ++j)
			{
				sliceVector.push_back(DIRECT_NZYX_ELEM(tiltSeriesFiltered, k, 0, i, j));
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
		std::cout << "thresholdU: " << thresholdU << std::endl;
        #endif

		tiltImage.initZeros(ySize_d, xSize_d);

		for(size_t i = 0; i < ySize_d; i++)
        {
            for(size_t j = 0; j < xSize_d; ++j)
            {
                double value = DIRECT_NZYX_ELEM(tiltSeriesFiltered, k, 0, i, j);

                if (value > thresholdU)
                {
					DIRECT_A2D_ELEM(tiltImage, i, j) = value;
               }
            }
        }

				// MAX POOLING ------------------------------------------------
		maxPooling(tiltImage, targetFS);
		filterFourierDirections(tiltImage);

		CorrelationAux aux;
		MultidimArray<double> imageTmp;
		imageTmp = tiltImage;
		correlation_matrix(imageTmp, landmarkReference_FTdir, tiltImage, aux, true);

		// Save equalized tilt-series  
		for(size_t i = 0; i < ySize_d; i++)
        {
 			Point2D<int> il = interLim[i];

            for(size_t j = il.x; j < il.y; ++j)
            {
				DIRECT_NZYX_ELEM(equalizedMap, k, 0, i, j) = DIRECT_A2D_ELEM(tiltImage, i, j);
            }
        }

		// OTSU THRESHOLDING ----------------------------------------------
		// std::vector<double> otsuVector;

        // for(size_t i = 0; i < ySize_d; i++)
        // {
        //     for(size_t j = 0; j < xSize_d; ++j)
        //     {
        //         double value = DIRECT_NZYX_ELEM(tiltSeriesFiltered, k, 0, i, j);

        //         if (value > thresholdU)
        //         {
		// 			otsuVector.push_back(DIRECT_A2D_ELEM(tiltImage, i, j));
        //         }
        //     }
        // }

		// MultidimArray<double> otsuMultidim;
		// otsuMultidim.initZeros(otsuVector.size());

		// for (size_t i =0; i<otsuVector.size(); i++)
		// {
		// 	DIRECT_MULTIDIM_ELEM(otsuMultidim, i) = otsuVector[i];
		// }

		// double otsuThr = OtsuSegmentation(otsuMultidim);

		// std::cout << "otsu thr ---------------------------------------------------------------> " << otsuThr << std::endl;

		// LABELLING --------------------------------------------------------------------
		binaryCoordinatesMapSlice.initZeros(ySize_d, xSize_d);

		double avg;
		double std;

		sum = 0;
		sum2 = 0;
        Nelems = 0;
        average = 0;
        standardDeviation = 0;

		for(size_t i = 0; i < ySize_d; i++)
        {
 			Point2D<int> il = interLim[i];

            for(size_t j = il.x; j < il.y; ++j)
            {
				int value = DIRECT_A2D_ELEM(tiltImage, i, j);
				sum += value;
				sum2 += value*value;
				++Nelems;
            }
        }

        average = sum / sliceVectorSize;
        standardDeviation = sqrt(sum2/Nelems - average*average);
		
		std::vector<int> dataVector;

		for(size_t i = 0; i < ySize_d; i++)
        {
 			Point2D<int> il = interLim[i];

            for(size_t j = il.x; j < il.y; ++j)
            {
				if (DIRECT_A2D_ELEM(tiltImage, i, j) > average + thrSD * standardDeviation)
				{
					DIRECT_A2D_ELEM(binaryCoordinatesMapSlice, i, j) = 1.0;
				}
            }
        }

        labelCoordiantesMapSlice.initZeros(ySize_d, xSize_d);
        int colour;
        colour = labelImage2D(binaryCoordinatesMapSlice, labelCoordiantesMapSlice, 8);
		
		#ifdef DEBUG_HCC
        std::cout << "Colour: " << colour << std::endl;
        #endif

		// MultidimArray<double> labelCopy = labelCoordiantesMapSlice;
		// #include<data/morphology.h>
		// erode2D(labelCopy, labelCoordiantesMapSlice, 4, 2, 512); // in, out, neigh, count, size

 		// FILTER LABELLED REGIONS --------------------- -----------------------------------
        std::vector<std::vector<int>> coordinatesPerLabelX (colour);
        std::vector<std::vector<int>> coordinatesPerLabelY (colour);

        for(size_t i = 0; i < ySize_d; i++)
        {
 			Point2D<int> il = interLim[i];

            for(size_t j = il.x; j < il.y; ++j)
            {
                int value = DIRECT_A2D_ELEM(labelCoordiantesMapSlice, i, j);

                if(value!=0)
                {
                    coordinatesPerLabelX[value-1].push_back(j);
                    coordinatesPerLabelY[value-1].push_back(i);
                }
            }
        }

		computeAvgStddev

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
    std::cout << "--- List of peaked coordinates: " << std::endl;
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
		
	// save tilt series only with thr of sd ------------------------------------ 
	outputFileNameLabeledVolume = rawname + "/ts_equalized.mrcs";
	saveImage() = equalizedMap;
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
                coordinate.x = coordinates3D[k].x * ds_factor;
                coordinate.y = coordinates3D[k].y * ds_factor;
                cis.push_back(coordinate);
            }
        }

		#ifdef DEBUG_HCC
        std::cout << "--- cis at image " << n << std::endl;
        for (size_t i = 0; i < cis.size(); i++)
        {
            std::cout << "(" << cis[i].x << ", " << 
                                cis[i].y << ")"  << std::endl;
        }
		#endif

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
			#ifdef DEBUG_CENTER_COORDINATES
			if (n==0)
			{
				std::cout << "Pixel (" << i << ", " << j << ")" << std::endl;
			}
			#endif

			double value = DIRECT_A2D_ELEM(correlationVolumeR, i, j);

			if (value > maximumCorrelation)
			{
				#ifdef DEBUG_CENTER_COORDINATES
				std::cout << "new maximumCorrelation " << value << " at (" << i << ", " << j << ")" << std::endl;
				#endif

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
    Image<double> tiltSeriesImages;

    if (fnVol.isMetaData())
    {
        tiltseriesmd.read(fnVol);
    }
    else
    {
        tiltSeriesImages.read(fnVol);

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
	createLandmarkTemplate_Gaussian();
	createLandmarkTemplate_FTdir();

	for(size_t objId : tiltseriesmd.ids())
	{
		tiltseriesmd.getValue(MDL_IMAGE, fnTSimg, objId);

        imgTS.read(fnTSimg);

        MultidimArray<double> tiltImage_ds;
        tiltImage_ds.initZeros(ySize_d, xSize_d);

		// Detect interpolation edges
		#ifdef DEBUG_INTERPOLATION_EDGES
		std::cout << "Detecting interpolation edges for image " << counter << std::endl;
		#endif

		detectInterpolationEdges(ptrImg);
		// bandpassFilter(ptrImg);

		#ifdef DEBUG_INTERPOLATION_EDGES
		std::cout << "Interpolation edges for image " << counter << std::endl;

		for (size_t i = 0; i < interpolationLimitsVector[counter].size(); i++)
		{
			std::cout << "y: " << i << " xMin: " << interpolationLimitsVector[counter][i].x << " xMax: " << interpolationLimitsVector[counter][i].y << std::endl;
		}
		#endif

		// Downsample
        #ifdef DEBUG_DOWNSAMPLE
        std::cout << "Downsampling image " << counter << std::endl;
        #endif
        downsample(ptrImg, tiltImage_ds);

		// Normalized dowsmapled image
		tiltImage_ds.statisticsAdjust(0.0, 1.0);	// Normalize image mean: 0 and std: 1

        #ifdef DEBUG_DIM
        std::cout << "Tilt-image dimensions before downsampling:" << std::endl;
        std::cout << "x " << XSIZE(ptrImg) << std::endl;
        std::cout << "y " << YSIZE(ptrImg) << std::endl;

		std::cout << "Tilt-image dimensions after downsampling:" << std::endl;
		std::cout << "x " << XSIZE(tiltImage_ds) << std::endl;
        std::cout << "y " << YSIZE(tiltImage_ds) << std::endl;
        #endif

		// Contruct normalized tilt-series for posterior coordinates centering
		ptrImg.statisticsAdjust(0.0, 1.0);	// Normalize image mean: 0 and std: 1

		for (size_t i = 0; i < ySize; ++i)
        {
            for (size_t j = 0; j < xSize; ++j)
            {
				DIRECT_NZYX_ELEM(tiltSeries, counter, 0, i, j) = DIRECT_A2D_ELEM(ptrImg, i, j);
			}
		}

		// Sobel filter and landmark enhancement
        #ifdef DEBUG_SOBEL
        std::cout << "Aplying sobel filter to image " << counter << std::endl;
        #endif
        // sobelFiler(tiltImage_ds, counter);
        enhanceLandmarks(tiltImage_ds);

		std::vector<Point2D<int>> interLim = interpolationLimitsVector_ds[counter];
        for (size_t i = 0; i < ySize_d; ++i)
        {
			Point2D<int> il = interLim[i];
            for (size_t j = il.x; j < il.y; ++j)
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

	// Check sphericity of the labeled region
	double circumscribedArea = PI * (maxSquareDistance);;
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
	std::cout << "ocupation " << ocupation << std::endl;
	#endif

	// if(ocupation < 0.25)
	// {
	// 	#ifdef DEBUG_FILTERLABEL
	// 	std::cout << "COORDINATE REMOVED AT (" << centroX << " , " << centroY << ") BECAUSE OF OCCUPATION"<< std::endl;
	// 	std::cout << "-------------------------------------------"  << std::endl;
	// 	#endif
	// 	return false;
	// }

	// Check the relative area compared with the expected goldbead
	double expectedArea = PI * ((targetFS) * targetFS);
	double relativeArea = (4*area)/expectedArea;  // Due to filtering and labelling processes labeled gold beads tend to reduce its radius in half

	#ifdef DEBUG_FILTERLABEL
	std::cout << "expectedArea " << expectedArea << std::endl;
	std::cout << "relativeArea " << relativeArea << std::endl;
	#endif


	if (relativeArea > 8 || relativeArea < 0.1)
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
    int targetFS_half = 1.1*(targetFS/2);
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
    // sobelFiler(landmarkReference, -1);

    // Save reference
    #ifdef DEBUG_REFERENCE
    size_t li = fnOut.find_last_of("\\/");
	std::string rn = fnOut.substr(0, li);
	std::string outFN;
    outFN = rn + "/landmarkReference.mrc";

	Image<double> si;
	si() = landmarkReference;
	si.write(outFN);
    #endif
}

void ProgTomoDetectLandmarks::createLandmarkTemplate_Gaussian()
{
	// Generate first reference
    int targetFS_half = 1.1*(targetFS/2);
    int targetFS_half_sq = targetFS_half*targetFS_half;
	int targetFS_sq = targetFS * targetFS;

    landmarkReference_Gaussian.initZeros(ySize_d, xSize_d);
    landmarkReference_Gaussian.initConstant(1);

	// double sigma = targetFS_half/3;
	double sigma = targetFS/3;

    // Create tilt-image with a single landamrk
    for (int k = 0; k < ySize_d; ++k)
    {
        for (int l = 0; l < xSize_d; ++l)
        {
			int k_p = k-ySize_d/2;
			int l_p = l-xSize_d/2;

			double mod2 = (k_p*k_p + l_p*l_p);
			A2D_ELEM(landmarkReference_Gaussian, k, l) = 1 - exp(-mod2 /(2*sigma*sigma))/(sigma*sqrt(2*PI));
        }
    }

    // Apply Sobel filer to reference
    // sobelFiler(landmarkReference_Gaussian, -1);

    // Save reference
    #ifdef DEBUG_REFERENCE
    size_t li = fnOut.find_last_of("\\/");
	std::string rn = fnOut.substr(0, li);
	std::string outFN;
    outFN = rn + "/landmarkReference_Gaussian.mrc";

	Image<double> si;
	si() = landmarkReference_Gaussian;
	si.write(outFN);
    #endif
}


void ProgTomoDetectLandmarks::createLandmarkTemplate_FTdir()
{
	// Generate first reference
    landmarkReference_FTdir.initZeros(ySize_d, xSize_d);

    // Create tilt-image with a single landamrk
    for (int k = 0; k < ySize_d; ++k)
    {
        for (int l = 0; l < xSize_d; ++l)
        {
			int k_p = k-ySize_d/2;
			int l_p = l-xSize_d/2;

			double mod = sqrt(k_p*k_p+l_p*l_p);

            if (mod < targetFS)
            {
                A2D_ELEM(landmarkReference_FTdir, k, l) = 1;
            }
			else if (mod < targetFS + 10)
			{
				A2D_ELEM(landmarkReference_FTdir, k, l) = -1 + (mod - targetFS) * 0.1;
			}
        }
    }

	
    // Save reference
    #ifdef DEBUG_REFERENCE
    size_t li = fnOut.find_last_of("\\/");
	std::string rn = fnOut.substr(0, li);
	std::string outFN;
    outFN = rn + "/landmarkReference_FTdir.mrc";

	Image<double> si;
	si() = landmarkReference_FTdir;
	si.write(outFN);
    #endif
}


// // ------------------------------------------------------------------------------------------------------------------------------
// // MAXPOOLING
void ProgTomoDetectLandmarks::maxPooling(MultidimArray<double> &image, size_t windowSize) 
{
	MultidimArray<double> window;
	std::vector<double> aa;

	// Ensure windowSize is an odd number
    windowSize = (windowSize % 2 == 0) ? windowSize + 1 : windowSize;
	int halfWindowSize = (int)windowSize / 2;

	MultidimArray<double> imageTmp = image;
	image.initConstant(0);

    for (int i = 0; i < ySize_d; i++) 
	{
        for (int j = 0; j < xSize_d; j++) 
		{
            int y0 = std::max(0, int(i - windowSize / 2));
            int x0 = std::max(0, int(j - windowSize / 2));
            int yF = std::min(ySize_d - 1, i + windowSize / 2);
            int xF = std::min(xSize_d - 1, j + windowSize / 2);

			for (int k = -halfWindowSize; k <= halfWindowSize ; k++)
            {
                for (int l = -halfWindowSize ; l <= halfWindowSize ; l++)
                {
                    if (i+k>=0 && j+l>=0 && j+l<xSize_d && i+k<ySize_d && k*k+l*l<=halfWindowSize*halfWindowSize)
                    {
                        aa.push_back(DIRECT_A2D_ELEM(imageTmp, (i + k), (j + l)));
                    }
                }
            }

			// window2D(imageTmp, window, y0, x0, yF, xF); 
			// double elem_max = *std::max_element(MULTIDIM_ARRAY(window), MULTIDIM_ARRAY(window)+NZYXSIZE(window));

			double elem_max = *std::max_element(aa.begin(), aa.end());
			DIRECT_A2D_ELEM(image, i, j) = elem_max;
			aa.clear();
        }
    }
}


// // ------------------------------------------------------------------------------------------------------------------------------
// // ADAPTATIVE HISTOGRAM EQUALIZATION
void ProgTomoDetectLandmarks::adaptiveHistogramEqualization(MultidimArray<double> &image, size_t windowSize) 
{
	MultidimArray<double> window;

	// Ensure windowSize is an odd number
    windowSize = (windowSize % 2 == 0) ? windowSize + 1 : windowSize;
	size_t halfWindowSize = windowSize / 2;

    // Apply adaptive histogram equalization
    for (int i = (halfWindowSize-1); i < ySize_d; i += windowSize) 
	{
        for (int j = (halfWindowSize-1); j < xSize_d; j += windowSize) 
		{
            int y0 = std::max(0, int(i - windowSize / 2));
            int x0 = std::max(0, int(j - windowSize / 2));
            int yF = std::min(ySize_d - 1, i + windowSize / 2);
            int xF = std::min(xSize_d - 1, j + windowSize / 2);

			std::cout << "y0 " << y0 << std::endl;
			std::cout << "x0 " << x0 << std::endl;
			std::cout << "yF " << yF << std::endl;
			std::cout << "xF " << xF << std::endl;

			window2D(image, window, y0, x0, yF, xF); 
			// for (size_t k = y0; k < yF; k++)
			// {
			// 	for (size_t l = x0; l < xF; l++)
			// 	{
			// 		DIRECT_A2D_ELEM(window, k-y0, l-x0) = DIRECT_A2D_ELEM(image, k, l);
			// 	}
			// }

			histogram_equalization(window, 255);

            for (size_t k = y0; k < yF; k++)
			{
				for (size_t l = x0; l < xF; l++)
				{
					DIRECT_A2D_ELEM(image, k, l) = DIRECT_A2D_ELEM(window, k-y0, l-x0);
				}
			}
        }
    }
}

// // // ------------------------------------------------------------------------------------------------------------------------------
// // // ADAPTATIVE HISTPGRAM EQUALIZATION ----------------------------------------------------------------------------------> FAIL
// void ProgTomoDetectLandmarks::adaptiveHistogramEqualization(MultidimArray<double> &image, size_t windowSize) 
// {
// 	MultidimArray<double> window;

// 	MultidimArray<double> imageRangeAdjusted = image;
// 	imageRangeAdjusted.rangeAdjust(0, 255);

//     // Iterate through each pixel in the image
//     for (size_t i = 0; i < ySize_d; ++i) 
// 	{
//         for (size_t j = 0; j < xSize_d; ++j) 
// 		{
//             // Define the local region based on the windowSize
//             int y0 = std::max(0, int(i - windowSize / 2));
//             int x0 = std::max(0, int(j - windowSize / 2));
//             int yF = std::min(ySize_d - 1, i + windowSize / 2);
//             int xF = std::min(xSize_d - 1, j + windowSize / 2);

//             // Construct window
// 			window2D(imageRangeAdjusted, window, y0, x0, yF, xF); 

// 			// Xmipp histogram equalization
// 			int bins = 200;
// 			histogram_equalization(window, 255);
			
// 			double newValue = DIRECT_A2D_ELEM(image, i, j) * (DIRECT_A2D_ELEM(window, windowSize / 2, windowSize / 2)/DIRECT_A2D_ELEM(imageRangeAdjusted, i, j));
// 			DIRECT_A2D_ELEM(image, i, j) = newValue;

//             // // Compute the local histogram
//             // std::vector<int> histogram(256, 0);

// 			// for (int k = 0; k < yF-y0; ++k)
// 			// {
// 			// 	for (int l = 0; l < xF-x0; ++l)
// 			// 	{
// 			// 		std::cout << "(int)DIRECT_A2D_ELEM(window, k, l) " << (int)DIRECT_A2D_ELEM(window, k, l) << std::endl;
// 			// 		++histogram[(int)DIRECT_A2D_ELEM(window, k, l)];
// 			// 	}
// 			// }
// 			// std::cout << "b " << std::endl;

//             // // Compute the cumulative distribution function (CDF)
//             // std::vector<int> cdf(256, 0);
//             // cdf[0] = histogram[0];

//             // for (int k = 1; k < 256; ++k) 
// 			// {
//             //     cdf[k] = cdf[k - 1] + histogram[k];
//             // }

// 			// std::cout << "c " << std::endl;


//             // // Compute the new intensity value for the current pixel
//             // int currentValue = DIRECT_A2D_ELEM(image, i, j);
//             // int newValue = currentValue * (cdf[DIRECT_A2D_ELEM(imageRangeAdjusted, i, j)] - cdf[0]) / (ySize_d * xSize_d - cdf[0]);

// 			// std::cout << "currentValue " << currentValue << std::endl;
// 			// std::cout << "newValue " << newValue << std::endl;

//             // // Update the pixel value in the image
//             // DIRECT_A2D_ELEM(image, i, j) = newValue;
			
// 			// std::cout << "d " << std::endl;

//         }
//     }
// }


// // ------------------------------------------------------------------------------------------------------------------------------
// // HOUGH TRANSFORM
// void ProgTomoDetectLandmarks::houghTransform(MultidimArray<double> &image, int minRadius, int maxRadius) 
// {
// 	// Define accumulator array
// 	std::vector<std::vector<std::vector<int>>> accumulator(
// 		xSize_d, 
// 		std::vector<std::vector<int>>(
// 			ySize_d, 
// 			std::vector<int>(maxRadius - minRadius + 1, 0)));

// 	for (int i = 0; i < ySize; ++i) {
// 		for (int j = 0; j < xSize; ++j) {
			
// 			// Check if the pixel is part of an edge (assuming 1 for foreground, 0 for background)
// 			if (DIRECT_A2D_ELEM(image, i, j) > 0) {
				
// 				// Iterate over possible circle radii
// 				for (int r = minRadius; r <= maxRadius; ++r) {
					
// 					// Iterate over possible circle centers
// 					for (int theta = 0; theta < 360; ++theta) {
// 						int cx = i + r * cos(theta * M_PI / 180.0);
// 						int cy = j + r * sin(theta * M_PI / 180.0);

// 						// Check if the circle center is within the image boundaries
// 						if (cx >= 0 && cx < xSize && cy >= 0 && cy < ySize) {
// 							// Accumulate a vote for this circle
// 							accumulator[cx][cy][r - minRadius]++;
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}

// 	// Process accumulator to find circle parameters
// 	for (int i = 0; i < ySize; ++i) {
// 		for (int j = 0; j < xSize; ++j) {
// 			for (int r = minRadius; r <= maxRadius; ++r) {
// 				if (accumulator[j][i][r - minRadius] > 0) {
// 					std::cout << "Detected circle at center (" << j << ", " << i << ") with radius " << r << std::endl;
// 				}
// 			}
// 		}
// 	}
// }
// // ------------------------------------------------------------------------------------------------------------------------------

void ProgTomoDetectLandmarks::filterFourierDirections(MultidimArray<double> &image) 
{
	MultidimArray<double> imageTmp;
	MultidimArray<double> imageOut;
	imageOut.resizeNoCopy(image);
	imageOut.initConstant(1);
	
	size_t numberOfDirections = 8;
	double angleStep = PI / numberOfDirections;

	for (size_t n = 0; n < numberOfDirections; n++)
	{
	 	imageTmp = image;
		// std::cout << "xdir= " << cos(n*angleStep) << ", ydir=" << sin(n*angleStep) << std::endl;
		
		directionalFilterFourier(imageTmp, cos(n*angleStep), sin(n*angleStep));
		imageTmp.statisticsAdjust(0.0, 1.0);
		// imageTmp.binarize(0.0);

		// FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(imageTmp)
		// {
		// 	if(DIRECT_A1D_ELEM(imageTmp, i) < 0)
		// 	{
		// 		DIRECT_A1D_ELEM(imageTmp, i)=0;
		// 	}
		// }

		imageOut = imageOut + imageTmp;
	}

	image = image * imageOut;
	image.statisticsAdjust(0.0, 1.0);
	
	// std::cout << "Discrete sumation mode" << std::endl;
	// MultidimArray<double> imageDirX;
	// imageDirX = image;
	// MultidimArray<double> imageDirY;
	// imageDirY = image;
	// // MultidimArray<double> imageDirYX;
	// // imageDirYX = image;

	// directionalFilterFourier(imageDirX, 1, 0);
	// directionalFilterFourier(imageDirY, 0, 1);
	// // directionalFilterFourier(imageDirYX, 1.0/sqrt(2), 1.0/sqrt(2));

	// image = imageDirX * imageDirY;
	// // image = imageDirX * imageDirY * imageDirYX;
}


void ProgTomoDetectLandmarks::directionalFilterFourier(MultidimArray<double> &image, double xdir, double ydir) 
{
    MultidimArray<std::complex<double>> fftImg;
	auto xvoldim = XSIZE(image);
	auto yvoldim = YSIZE(image);
	// auto xvoldim = xSize_d;
	// auto yvoldim = ySize_d;

	FourierTransformer transformer1;

    transformer1.FourierTransform(image, fftImg, false);

	auto YdimFT=(int)YSIZE(fftImg);
	auto XdimFT=(int)XSIZE(fftImg);

	Matrix1D<double> freq_fourier_x;
	Matrix1D<double> freq_fourier_y;
	MultidimArray< double > freqMap;

	// Initializing the frequency vectors
	freq_fourier_x.initZeros(XSIZE(fftImg));
	freq_fourier_y.initZeros(YSIZE(fftImg));

	// u is the frequency
	double u;

	// Defining frequency components. First element should be 0, it is set as the smallest number to avoid singularities
	VEC_ELEM(freq_fourier_y,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<YdimFT; ++k){
		FFT_IDX2DIGFREQ(k,yvoldim, u);
		VEC_ELEM(freq_fourier_y, k) = u;
	}

	VEC_ELEM(freq_fourier_x,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<XdimFT; ++k){
		FFT_IDX2DIGFREQ(k,xvoldim, u);
		VEC_ELEM(freq_fourier_x, k) = u;
	}

	//Initializing map with frequencies
	freqMap.resizeNoCopy(fftImg);
	freqMap.initConstant(1.9);  //Nyquist is 2, we take 1.9 greater than Nyquist



	// Directional frequencies along each direction
	double uy, ux, uy2;
	long n=0;
	for(size_t i=0; i<YdimFT; ++i)
	{
		uy = VEC_ELEM(freq_fourier_y, i);
		uy2 = uy*uy;

		for(size_t j=0; j<XdimFT; ++j)
		{
			ux = VEC_ELEM(freq_fourier_x, j);
			ux = sqrt(uy2 + ux*ux);

			if	(ux<=0.5)
			{
				DIRECT_MULTIDIM_ELEM(freqMap,n) = 1/ux;
				if ((j == 0) && (uy<0))
				{
					DIRECT_MULTIDIM_ELEM(freqMap,n) = 1.9;
				}
				if ((i == 0) && (j == 0))
				{
					DIRECT_MULTIDIM_ELEM(freqMap,n) = 1.9;
				}
					
			}				
			++n;
		}
	}
	// Image<double> img;
	// img() = freqMap;
	// img.write("freqMap.mrc");
	double cosAngle;
	// cosAngle = 1.0/sqrt(2);
	cosAngle = 0.9397;  // 20º cosine
	auto aux = (8.0/((cosAngle -1)*(cosAngle -1)));
	//cosine = expf( -((cosine -1)*(cosine -1))*aux); 

	double lowerBound, upperBound;

	lowerBound = 1/targetFS-0.1;
	upperBound = 1/targetFS+0.1;

	n = 0;
	for (int i=0; i<YdimFT; i++)
	{
		double uy = VEC_ELEM(freq_fourier_y, i);
		for (int j=0; j<XdimFT; j++)
		{
			double ux = VEC_ELEM(freq_fourier_x, j);

			double iun = DIRECT_MULTIDIM_ELEM(freqMap,n);


			if (1/iun<upperBound && 1/iun>lowerBound)
			{
				// Si nos queremos y nos amamos porque no nos besamos por donde meamos
				auto ux_norm = ux*iun;
				auto uy_norm = uy*iun;
				double cosine = fabs(xdir*ux_norm + ydir*uy_norm);

				if (cosine >= cosAngle)	
				{
					//------

					cosine = exp( -((cosine -1)*(cosine -1))*aux); 
					DIRECT_MULTIDIM_ELEM(fftImg, n) *= cosine;
					//------
					
					// DIRECT_MULTIDIM_ELEM(fftImg, n) = 0.0;
				}
				else
				{
					DIRECT_MULTIDIM_ELEM(fftImg, n) = 0.0;
				}
			}

			// ------
			else{
				DIRECT_MULTIDIM_ELEM(fftImg, n) = 0.0;
			}
			// ------
			n++;
		}
	}

	transformer1.inverseFourierTransform(fftImg, image);
	// Image<double> img;
	// img() = image;
	// img.write("dirMap.mrc");
}

// FILTER FOURIER DIRECTIONS: inteto con una sola FT
// -----------------------------------------------------------------------------------------------------
// void ProgTomoDetectLandmarks::filterFourierDirections(MultidimArray<double> &image) 
// {
// 	MultidimArray<std::complex<double>> fftImg;
// 	MultidimArray<std::complex<double>> fftImgTmp;
// 	// MultidimArray<std::complex<double>> fftImgOut;

// 	MultidimArray<double> imageTmp;
// 	imageTmp.resizeNoCopy(image);

// 	MultidimArray<double> imageOut;
// 	imageOut.resizeNoCopy(image);
// 	imageOut.initConstant(1);

// 	FourierTransformer transformer1;
//     transformer1.FourierTransform(image, fftImg, false);
// 	// fftImgOut.resizeNoCopy(fftImg);
// 	// fftImgOut.initConstant(0);
	
// 	// size_t numberOfDirections = 2;
// 	// double angleStep = PI / numberOfDirections;

// 	// for (size_t i = 0; i < numberOfDirections; i++)
// 	// {
// 	// 	 fftImgTmp= fftImg;
// 	// 	directionalFilterFourier(fftImgTmp, cos(i*angleStep), sin(i*angleStep));
		
// 	// 	transformer1.inverseFourierTransform(fftImgTmp, imageTmp);

// 	// 	imageOut = imageOut * imageTmp;
// 	// }

// 	// image = imageOut;
	
// 	std::cout << "Discrete sumation mode" << std::endl;
// 	MultidimArray<double> imageDirX;
// 	imageDirX.resizeNoCopy(image);
// 	MultidimArray<double> imageDirY;
// 	imageDirY.resizeNoCopy(image);
// 	// MultidimArray<double> imageDirYX;
// 	// imageDirYX.resizeNoCopy(image);

// 	fftImgTmp=fftImg;
// 	directionalFilterFourier(fftImgTmp, 1, 0);
// 	transformer1.inverseFourierTransform(fftImgTmp, imageDirX);

// 	fftImgTmp=fftImg;
// 	directionalFilterFourier(fftImgTmp, 0, 1);
// 	transformer1.inverseFourierTransform(fftImgTmp, imageDirY);

// 	// fftImgTmp=fftImg;
// 	// directionalFilterFourier(fftImgTmp, 1.0/sqrt(2), 1.0/sqrt(2));
// 	// transformer1.inverseFourierTransform(fftImgTmp, imageDirYX);

// 	image = imageDirX * imageDirY;
// 