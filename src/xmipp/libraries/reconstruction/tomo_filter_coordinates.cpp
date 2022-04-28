/***************************************************************************
 *
 * Authors:    Federico P. de Isidro-Gómez    fp.deisidro@cnb.csic.es (2021)
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

#include "tomo_filter_coordinates.h"
#include <core/metadata_extension.h>
#include <core/metadata_vec.h>
#include <iostream>
#include <core/utils/time_utils.h>


// --------------------------- INFO functions ----------------------------

void ProgTomoFilterCoordinates::readParams()
{
	fnInTomo = getParam("--inTomo");
	fnMask = getParam("--mask");
	fnInCoord = getParam("--coordinates");
	radius = getIntParam("--radius");
   	fnOut = getParam("-o");
}


void ProgTomoFilterCoordinates::defineParams()
{
	addUsageLine("This program generates a new metatada coordinates file with local information given by the input tomogram (e.g. local ");
	addUsageLine("resolution map). \n"); 
	addUsageLine("Given a set of coordinates and a tomogram, local statistics will be calculated for each coordinate (given by the radius). "); 
	addUsageLine("Optionally a mask can be provided, and then those coordinates outside of the mask (value = 0) will be removed.");
	
	addParamsLine("  --inTomo <mrcs_file=\"\">                     : Input tomogram (density or resolution tomogram).");
	addParamsLine("  [--mask <xmd_file=\"\">]                      : Input xmd file containing the mask.");
	addParamsLine("  --coordinates <xmd_file=\"\">                 : Input xmd file containing the 3D coordinates.");
	addParamsLine("  --radius <radius=50>                          : Radius in pixels of the neighbourhood of the coordinates to get resolution score.");
	addParamsLine("  -o <outCoord=\"filteredCoordinates3D.xmd\">   : Output file containing the filtered 3D coordinates.");
}


// --------------------------- HEAD functions ----------------------------

void ProgTomoFilterCoordinates::filterCoordinatesWithMask(MultidimArray<int> &inputVolume)
{
	Point3D<int> coord3D;
	for (int i = 0; i < inputCoords.size(); i++)
	{
		coord3D = inputCoords[i];

		if(coord3D.z < 0 || coord3D.z > (zDim-1) || coord3D.y < 0 || coord3D.y > (yDim-1) || coord3D.x < 0 || coord3D.x > (xDim-1))
		{
			std::cout << "WARNNING: Coordinate at (x=" << coord3D.x<< ", y=" << coord3D.y << ", z=" << coord3D.z << ") erased due to its out of the mask." << std::endl;
			inputCoords.erase(inputCoords.begin()+i);
			i--;
		}

		else if(DIRECT_A3D_ELEM(inputVolume, coord3D.z, coord3D.y, coord3D.x) == 0)
		{
			#ifdef VERBOSE_OUTPUT
			std::cout << "Coordinate erased with value " << DIRECT_A3D_ELEM(inputVolume, coord3D.z, coord3D.y, coord3D.x) << " at (x=" << coord3D.x << ", y=" << coord3D.y << ", z=" << coord3D.z << ")" << std::endl;
			#endif 
			inputCoords.erase(inputCoords.begin()+i);
			i--;
		}
		
		#ifdef VERBOSE_OUTPUT
		else
		{
			std::cout << "Coordinate saved with value " << DIRECT_A3D_ELEM(inputVolume, coord3D.z, coord3D.y, coord3D.x) << " at (x=" << coord3D.x << ", y=" << coord3D.y << ", z=" << coord3D.z << ")" << std::endl;
		}
		#endif 
	}
}


void ProgTomoFilterCoordinates::calculateCoordinateStatistics(MultidimArray<double> &tom)
{
	MetaDataVec scoredMd;
	MDRowVec row;

	for (size_t i = 0; i < inputCoords.size(); i++)
	{
		Point3D coor = inputCoords[i];

		if (((coor.z - radius) < 0) || ((coor.z + radius) > (zDim-1)) || ((coor.y - radius) < 0) || ((coor.y + radius) > (yDim-1)) || ((coor.x - radius) < 0) || ((coor.x + radius) > (xDim-1)))
		{
			std::cout << "WARNNING: Coordinate at (x=" << coor.x<< ", y=" << coor.y << ", z=" << coor.z << ") masked out." << std::endl;
			continue;
		}
				
		double meanCoor = 0;
		double meanCoor2 = 0;
		double stdCoor = 0;
		// double medianCoor = 0;
		// double madCoor = 0;
		double value = 0;
		size_t Nelems = 0;
		

		for (int i = -radius; i < radius; i++)
		{
			size_t i2 = i*i;
			for (int j = -radius; j < radius; j++)
			{
				size_t j2i2 = i2 + j*j;
				for (int k = -radius; k < radius; k++)
				{
					size_t r2 = j2i2 + k*k;
					
					if (r2 <= radius)
					{
						auto value = DIRECT_A3D_ELEM(tom, coor.z + k, coor.y + i, coor.x + j);
						meanCoor += value;
						Nelems++;
						meanCoor2 += value*value;
					}
				}
			}
		}

		meanCoor /= Nelems;
		stdCoor = sqrt(meanCoor2/Nelems - meanCoor*meanCoor);
		
		row.setValue(MDL_XCOOR, coor.x);
		row.setValue(MDL_YCOOR, coor.y);
		row.setValue(MDL_ZCOOR, coor.z);
		row.setValue(MDL_AVG, meanCoor);
		row.setValue(MDL_STDDEV, stdCoor);
		// row.setValue(MDL_VOLUME_SCORE1, medianCoor);
		// row.setValue(MDL_VOLUME_SCORE2, madCoor);
		scoredMd.addRow(row);
	}
	scoredMd.write(fnOut);
}

// --------------------------- I/O functions ----------------------------

void ProgTomoFilterCoordinates::readInputCoordinates()
{
	MetaDataVec inCoordMd;
	inCoordMd.read(fnInCoord);

	size_t objId;
	Point3D<int> coordinate3D;

	for(size_t objId : inCoordMd.ids())
	{
		inCoordMd.getValue(MDL_XCOOR, coordinate3D.x, objId);
		inCoordMd.getValue(MDL_YCOOR, coordinate3D.y, objId);
		inCoordMd.getValue(MDL_ZCOOR, coordinate3D.z, objId);

		inputCoords.push_back(coordinate3D);
	}


	#ifdef VERBOSE_OUTPUT
	std::cout << "Input coordinates metadata read from: " << fnInCoord << std::endl;
	#endif
}

void ProgTomoFilterCoordinates::writeOutputCoordinates()
{
	MetaDataVec outCoordMd;

	for(size_t i = 0; i < inputCoords.size(); i++)
	{
		MDRowVec row;
		row.setValue(MDL_XCOOR, inputCoords[i].x);
		row.setValue(MDL_YCOOR, inputCoords[i].y);
		row.setValue(MDL_ZCOOR, inputCoords[i].z);
		outCoordMd.addRow(row);
	}
	outCoordMd.write(fnOut);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Output coordinates metadata saved at: " << fnOut << std::endl;
	#endif
}


// --------------------------- MAIN ----------------------------------

void ProgTomoFilterCoordinates::run()
{
	timeUtils::reportTimeMs("Execution time", [&]{

	std::cout << "Starting..." << std::endl;

	// Reading coordinates
	readInputCoordinates();

	// Reading mask if exists
	if (fnMask !="")
	{	
		Image<int> maskImg;
		maskImg.read(fnMask);
		auto &mask = maskImg();

		filterCoordinatesWithMask(mask);
	}
	
	// Reading input tomogram
	Image<double> tomoMap;
	tomoMap.read(fnInTomo);
	auto &tom = tomoMap();

	xDim = XSIZE(tom);
	yDim = YSIZE(tom);
	zDim = ZSIZE(tom);

	calculateCoordinateStatistics(tom);
	});
}