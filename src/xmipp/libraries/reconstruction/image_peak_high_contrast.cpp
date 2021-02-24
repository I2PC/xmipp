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
	thr = getDoubleParam("--thr");
    samp = getIntParam("--samp");
	numberCenterOfMass = getIntParam("--numberCenterOfMass");
	distanceThr = getIntParam("--distanceThr");
	numberOfCoordinatesThr = getIntParam("--numberOfCoordinatesThr");

}

void ProgImagePeakHighContrast::defineParams()
{
	addUsageLine("This function determines the location of the outliers points in a volume");
	addParamsLine("  --vol <vol_file=\"\">                   		: Input volume");
	addParamsLine("  -o <output=\"coordinates3D.xmd\">        		: Output file containing the 3D coodinates");
	addParamsLine("  [--thr <thr=0.1>]                		 		: Threshold to detect outlier pixes values");
  	addParamsLine("  [--samp <samp=10>]                		 		: Number of slices to use to determin the threshold value");
  	addParamsLine("  [--numberCenterOfMass <numberCenterOfMass=10>]	: Number of initial center of mass to trim coordinates");
  	addParamsLine("  [--distanceThr <distanceThr=10>]				: Minimum distance to consider two coordinates belong to the same center of mass");
  	addParamsLine("  [--numberOfCoordinatesThr <distanceThr=10>]	: Minimum number of coordinates attracted to a center of mass to consider it");

}

void ProgImagePeakHighContrast::getHighContrastCoordinates()
{
	std::cout << "Starting..." << std::endl;

	#define DEBUG
	// #define DEBUG_DIM
	// #define DEBUG_COOR
	// #define DEBUG_DIM
	// #define DEBUG_DIST

	#ifdef DEBUG
	std::cout << "Number of sampling slices: " << samp << std::endl;
	std::cout << "Threshold: " << thr << std::endl;
	#endif

	Image<double> inputVolume;
	inputVolume.read(fnVol);

	MultidimArray<double> &inputTomo=inputVolume();
	std::vector<double> tomoVector(0);

	size_t centralSlice = ZSIZE(inputTomo)/2;

	#ifdef DEBUG_DIM
	std::cout << "x " << XSIZE(inputTomo) << std::endl;
	std::cout << "y " << YSIZE(inputTomo) << std::endl;
	std::cout << "z " << ZSIZE(inputTomo) << std::endl;
	std::cout << "n " << NSIZE(inputTomo) << std::endl;
	#endif

	//////////////////////////////////////////////////////////////////////////// CONTRAST ENHANCEMENT
	// Image <double> enhancedVolume;

	// enhancedVolume().resizeNoCopy(inputVolume)
	// enhancedVolume().initConstant(255) // TODO: This shold be the image average
	

	///////////////////////////////////////////////////////////////////////////////// PICK OUTLIERS

	#ifdef DEBUG
	std::cout << "Sampling region from slice " << centralSlice - (samp/2) << " to " 
	<< centralSlice + (samp / 2) << std::endl;
	#endif

	for(size_t k = centralSlice - (samp/2); k <= centralSlice + (samp / 2); ++k)
	{
		for(size_t j = 0; j < YSIZE(inputTomo); ++j)
		{
			for(size_t i = 0; i < XSIZE(inputTomo); ++i)
			{
				#ifdef DEBUG_DIM
				std::cout << "i: " << i << " j: " << j << " k:" << k << std::endl;
				#endif

				tomoVector.push_back(DIRECT_ZYX_ELEM(inputTomo, k, i ,j));
			}
		}
	}
	
	std::sort(tomoVector.begin(),tomoVector.end());


	double highThresholdValue = tomoVector[size_t(tomoVector.size()*(1-(thr/2)))];
    double lowThresholdValue = tomoVector[size_t(tomoVector.size()*(thr/2))];

	#ifdef DEBUG
	std::cout << "High threshold value = " << highThresholdValue << std::endl;
    std::cout << "Low threshold value = " << lowThresholdValue << std::endl;
	#endif

	#ifdef DEBUG_COOR
	std::cout << "Peaked coordinates" << std::endl;
	std::cout << "-----------------------------" << std::endl;
	#endif

	std::vector<int> coordinates3Dx(0);
    std::vector<int> coordinates3Dy(0);
    std::vector<int> coordinates3Dz(0);

    FOR_ALL_ELEMENTS_IN_ARRAY3D(inputTomo)
    {
        double value = A3D_ELEM(inputTomo, k, i, j);

        if (value<=lowThresholdValue or value>=highThresholdValue)
        {
			#ifdef DEBUG_COOR
            std::cout << "(" << i << "," << j << "," << k << ")" << std::endl;
			#endif

            coordinates3Dx.push_back(j);
            coordinates3Dy.push_back(i);
            coordinates3Dz.push_back(k);
        }
    }

	#ifdef DEBUG
	std::cout << "Number of peaked coordinates: " << coordinates3Dx.size() << std::endl;
	#endif

	////////////////////////////////////////////////////////////////////////////////// GENERATE CENTERS OF MASS

	std::vector<int> centerOfMassX(0);
    std::vector<int> centerOfMassY(0);
    std::vector<int> centerOfMassZ(0);
	
	std::vector<int> numberOfCoordsPerCM(0);

	for(int i=0;i<numberCenterOfMass;i++)
	{
		int randomIndex = rand() % coordinates3Dx.size();

		centerOfMassX.push_back(coordinates3Dx[randomIndex]);
		centerOfMassY.push_back(coordinates3Dy[randomIndex]);
		centerOfMassZ.push_back(coordinates3Dz[randomIndex]);
		
		numberOfCoordsPerCM.push_back(1);
	}

	int squareDistanceThr = distanceThr*distanceThr;
	bool attractedToMassCenter = false;

	for(size_t i=0;i<coordinates3Dx.size();i++)
	{
		// Check if the coordinate is attracted to any centre of mass
		attractedToMassCenter = false; 

		int xCoor = coordinates3Dx[i];
		int yCoor = coordinates3Dy[i];
		int zCoor = coordinates3Dz[i];

		for(size_t j=0;j<centerOfMassX.size();j++)
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
				centerOfMassX[j]=centerOfMassX[j]+(coordinates3Dx[i]-centerOfMassX[j])/2;
				centerOfMassY[j]=centerOfMassY[j]+(coordinates3Dy[i]-centerOfMassY[j])/2;
				centerOfMassZ[j]=centerOfMassZ[j]+(coordinates3Dz[i]-centerOfMassZ[j])/2;

				numberOfCoordsPerCM[j]++;

				attractedToMassCenter = true;
				break;
			}
		}

		if (attractedToMassCenter==false)
		{
			centerOfMassX.push_back(coordinates3Dx[i]);
			centerOfMassY.push_back(coordinates3Dy[i]);
			centerOfMassZ.push_back(coordinates3Dz[i]);

			numberOfCoordsPerCM.push_back(1);
		}
	}

	#ifdef DEBUG
	std::cout << "Number of centers of mass: " << centerOfMassX.size() << std::endl;
	#endif

	///////////////////////////////////////////////////////////////////////////////////////////////TRIM CENTER OF MASS
	std::cout  << numberOfCoordsPerCM.size() << std::endl;
	std::cout  << centerOfMassX.size() << std::endl;
	std::cout  << centerOfMassY.size() << std::endl;
	std::cout  << centerOfMassZ.size() << std::endl;

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

	///// probar sin numeros aleatorios 
	///// probar por separado los mas claros y los mas oscuros

	////////////////////////////////////////////////////////////////////////////////////////////// SAVE COORDINATES
	MetaData md;
	size_t id;


	for(size_t i=0;i<centerOfMassX.size();i++)
	{
		id = md.addObject();
		// md.setValue(MDL_XCOOR, centerOfMassX[i], id);
		// md.setValue(MDL_YCOOR, centerOfMassY[i], id);
		// md.setValue(MDL_ZCOOR, centerOfMassZ[i], id);
	}

	md.write(fnOut);

}

// void ProgImagePealHighContrast::writeOutputCoordinates()
// {
// 	std::ofstream outputFile;
// 	outputFile.open(fnOut);

// 	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(coordinates3Dx){
// 		outputFile << coordinates3Dx(i) << "\t" << coordinates3Dy(i) << "\t"<< coordinates3Dz(i) << "\n" 
// 	}

// 	outputFile.close()
// }

void ProgImagePeakHighContrast::run()
{
	getHighContrastCoordinates();
}
