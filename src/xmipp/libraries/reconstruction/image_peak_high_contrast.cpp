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
	numberOfInitialCoordinates = getIntParam("--numberInitialCoor");
    numberSampSlices = getIntParam("--numberSampSlices");
	numberCenterOfMass = getIntParam("--numberCenterOfMass");
	distanceThr = getIntParam("--distanceThr");
	numberOfCoordinatesThr = getIntParam("--numberOfCoordinatesThr");
	samplingRate = getDoubleParam("--samplingRate");

}


void ProgImagePeakHighContrast::defineParams()
{
	addUsageLine("This function determines the location of the outliers points in a volume");
	addParamsLine("  --vol <vol_file=\"\">                   				: Input volume. It is recommended that it is gaussian filtered previously.");
	addParamsLine("  -o <output=\"coordinates3D.xmd\">        				: Output file containing the 3D coodinates.");
  	addParamsLine("  [--boxSize <boxSize=32>]								: Box size of the peaked coordinates.");
	addParamsLine("  [--numberInitialCoor <numberInitialCoor=15000>]      	: Threshold to detect outlier pixes values.");
  	addParamsLine("  [--numberSampSlices <numberSampSlices=10>]     		: Number of slices to use to determin the threshold value.");
  	addParamsLine("  [--numberCenterOfMass <numberCenterOfMass=10>]			: Number of initial center of mass to trim coordinates.");
  	addParamsLine("  [--distanceThr <distanceThr=10>]						: Minimum distance to consider two coordinates belong to the same center of mass.");
  	addParamsLine("  [--numberOfCoordinatesThr <numberOfCoordinatesThr=10>]	: Minimum number of coordinates attracted to a center of mass to consider it.");
  	addParamsLine("  [--fiducialSize <fiducialSize=100>]						: Fiducial size in Angstroms (A)");
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

				if(u > cutoffFreqHigh)
				{
					DIRECT_MULTIDIM_ELEM(fftFiltered, n) = 0;
				} 

				if(u < cutoffFreqLow)
				{
					DIRECT_MULTIDIM_ELEM(fftFiltered, n) = 0;
				} 
				
				if(u >= freqHigh && u < cutoffFreqHigh)
				{
					DIRECT_MULTIDIM_ELEM(fftFiltered, n) *= 0.5*(1+cos((u-freqHigh)*delta));
				}
				
				if (u <= freqLow && u > cutoffFreqLow)
				{
					DIRECT_MULTIDIM_ELEM(fftFiltered, n) *= 0.5*(1+cos((u-freqLow)*delta));

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

	Image<double> saveImage;
	saveImage() = volFiltered; 
	saveImage.write(outputFileNameFilteredVolume);

	return volFiltered;
}


	void ProgImagePeakHighContrast::getHighContrastCoordinates(MultidimArray<double> volFiltered)
{
	std::cout << "Picking coordinates..." << std::endl;

	size_t centralSlice = ZSIZE(volFiltered)/2;
	std::vector<double> tomoVector(0);

	#ifdef DEBUG
	std::cout << "Number of sampling slices: " << numberSampSlices << std::endl;
	std::cout << "Number of initial coordinates: " << numberOfInitialCoordinates << std::endl;

	std::cout << "Sampling region from slice " << centralSlice - (numberSampSlices/2) << " to " 
	<< centralSlice + (numberSampSlices / 2) << std::endl;
	#endif

	for(size_t k = centralSlice - (numberSampSlices/2); k <= centralSlice + (numberSampSlices / 2); ++k)
	{
		for(size_t j = 0; j < YSIZE(volFiltered); ++j)
		{
			for(size_t i = 0; i < XSIZE(volFiltered); ++i)
			{
				#ifdef DEBUG_DIM
				std::cout << "i: " << i << " j: " << j << " k:" << k << std::endl;
				#endif

				tomoVector.push_back(DIRECT_ZYX_ELEM(volFiltered, k, i ,j));
			}
		}
	}
	
	std::sort(tomoVector.begin(),tomoVector.end());


	double highThresholdValue = tomoVector[size_t(tomoVector.size()-(numberOfInitialCoordinates/2))];
    double lowThresholdValue = tomoVector[size_t(numberOfInitialCoordinates/2)];

	#ifdef DEBUG
	std::cout << "High threshold value = " << highThresholdValue << std::endl;
    std::cout << "Low threshold value = " << lowThresholdValue << std::endl;
	#endif

    std::vector<int> coordinates3Dx(0);
    std::vector<int> coordinates3Dy(0);
    std::vector<int> coordinates3Dz(0);

    FOR_ALL_ELEMENTS_IN_ARRAY3D(volFiltered)
    {
        double value = A3D_ELEM(volFiltered, k, i, j);

        if (value<=lowThresholdValue or value>=highThresholdValue)
        {
            coordinates3Dx.push_back(j);
            coordinates3Dy.push_back(i);
            coordinates3Dz.push_back(k);
        }
    }

	#ifdef DEBUG
	std::cout << "Number of peaked coordinates: " << coordinates3Dx.size() << std::endl;
	#endif

	clusterHighContrastCoordinates(volFiltered,
								   coordinates3Dx,
								   coordinates3Dy,
								   coordinates3Dz);
}


	void ProgImagePeakHighContrast::clusterHighContrastCoordinates(MultidimArray<double> &volFiltered,
																   std::vector<int> coordinates3Dx,
																   std::vector<int> coordinates3Dy,
																   std::vector<int> coordinates3Dz)
{
	std::cout << "Clustering coordinates..." << std::endl;


	// These vectors keep track of the position of the center of mass
	std::vector<int> centerOfMassX(0);
    std::vector<int> centerOfMassY(0);
    std::vector<int> centerOfMassZ(0);

	// These vectors accumulate each coordinate attracted by every center of mass of calculate its mean at the end
	std::vector<int> centerOfMassXAcc(0);
    std::vector<int> centerOfMassYAcc(0);
    std::vector<int> centerOfMassZAcc(0);
	
	std::vector<int> numberOfCoordsPerCM(0);

	for(int i=0;i<numberCenterOfMass;i++)
	{
		int randomIndex = rand() % coordinates3Dx.size();

		centerOfMassX.push_back(coordinates3Dx[randomIndex]);
		centerOfMassY.push_back(coordinates3Dy[randomIndex]);
		centerOfMassZ.push_back(coordinates3Dz[randomIndex]);

		centerOfMassXAcc.push_back(coordinates3Dx[randomIndex]);
		centerOfMassYAcc.push_back(coordinates3Dy[randomIndex]);
		centerOfMassZAcc.push_back(coordinates3Dz[randomIndex]);
		
		numberOfCoordsPerCM.push_back(1);
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
				centerOfMassXAcc[j] = centerOfMassXAcc[j] + coordinates3Dx[i];
				centerOfMassYAcc[j] = centerOfMassYAcc[j] + coordinates3Dy[i];
				centerOfMassZAcc[j] = centerOfMassZAcc[j] + coordinates3Dz[i];

				numberOfCoordsPerCM[j]++;

				attractedToMassCenter = true;
				break;
			}
		}

		if (attractedToMassCenter == false)
		{
			centerOfMassX.push_back(coordinates3Dx[i]);
			centerOfMassY.push_back(coordinates3Dy[i]);
			centerOfMassZ.push_back(coordinates3Dz[i]);

			centerOfMassXAcc.push_back(coordinates3Dx[i]);
			centerOfMassYAcc.push_back(coordinates3Dy[i]);
			centerOfMassZAcc.push_back(coordinates3Dz[i]);

			numberOfCoordsPerCM.push_back(1);
		}
	}

	// Check that coordinates at the border of the volume are not outside when considering the box size
	for(size_t i=0;i<numberOfCoordsPerCM.size();i++)
	{
		if(centerOfMassX[i]<boxSize/2 or XSIZE(volFiltered)-centerOfMassX[i]<boxSize/2 or
		   centerOfMassY[i]<boxSize/2 or YSIZE(volFiltered)-centerOfMassY[i]<boxSize/2 or
		   centerOfMassZ[i]<boxSize/2 or ZSIZE(volFiltered)-centerOfMassZ[i]<boxSize/2)
		{
			numberOfCoordsPerCM.erase(numberOfCoordsPerCM.begin()+i);
			centerOfMassX.erase(centerOfMassX.begin()+i);
			centerOfMassY.erase(centerOfMassY.begin()+i);
			centerOfMassZ.erase(centerOfMassZ.begin()+i);
			i--;
		}
	}

	// Check number of coordinates per center of mass
	for(size_t i=0;i<numberOfCoordsPerCM.size();i++)
	{
		if(numberOfCoordsPerCM[i]<numberOfCoordinatesThr)
		{
			numberOfCoordsPerCM.erase(numberOfCoordsPerCM.begin()+i);
			centerOfMassX.erase(centerOfMassX.begin()+i);
			centerOfMassY.erase(centerOfMassY.begin()+i);
			centerOfMassZ.erase(centerOfMassZ.begin()+i);
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
	
	getHighContrastCoordinates(volFiltered);
}
