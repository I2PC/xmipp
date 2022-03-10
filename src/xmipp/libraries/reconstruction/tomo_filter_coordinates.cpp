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
#include <chrono>


// --------------------------- INFO functions ----------------------------

void ProgTomoFilterCoordinates::readParams()
{
	fnInTomo = getParam("--inTomo");
	fnMask = getParam("--mask");
	fnInCoord = getParam("--coordinates");
	radius = getIntParam("--radius");
    checkResThr = checkParam("--threshold");
   	fnOut = getParam("-o");
}


void ProgTomoFilterCoordinates::defineParams()
{
	addUsageLine("This program filters a set of coordinates with one of two criteria:.");
	addUsageLine("\nUsing a mask. In this case a set of coordinates and a mask with the regions to preserve ");
    addUsageLine("coordinates different than 0 need to be input. If only these two options are input this ");
    addUsageLine("criteria will be applied, if not the second one will be applied");
	addUsageLine("\nUsing a resolution. In this case a set of coordinates, a resolution map, and a resolution ");
    addUsageLine("percentile to select the number of coordinates to be saved after the scoring need o be ");
    addUsageLine("input. If these three options are input then this criteria will be applied.");
	addParamsLine("  --inTomo <mrcs_file=\"\">                                : Input volume (mask or resolution map).");
	addParamsLine("  [--mask <xmd_file=\"\">]                               : Input xmd file containing the 3D coordinates.");
	addParamsLine("  --coordinates <xmd_file=\"\">                               : Percentile resolution threshold.");
	addParamsLine("  --radius <radius=50>                               : Radius of the neighbourhood of the coordinates to get resolution score.");
    addParamsLine("  [--threshold <outCoord=\"filteredCoordinates3D.xmd\">]   : Output file containing the filtered 3D coordinates.");
	addParamsLine("  -o <outCoord=\"filteredCoordinates3D.xmd\">   : Output file containing the filtered 3D coordinates.");
}


// --------------------------- HEAD functions ----------------------------

void ProgTomoFilterCoordinates::filterCoordinatesWithMask(MultidimArray<int> &inputVolume)
{
	Point3D<int> coord3D;
	for (size_t i = 0; i < inputCoords.size(); i++)
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


void ProgTomoFilterCoordinates::takeCoordinateFromTomo(MultidimArray<double> &tom)
{
	MetaDataVec scoredMd;
	MDRowVec row;

	for (size_t i = 0; i < inputCoords.size(); i++)
	{
		Point3D coor = inputCoords[i];

		if (((coor.z - radius) < 0) || ((coor.z + radius) > (zDim-1)) || ((coor.y - radius) < 0) || ((coor.y + radius) > (yDim-1)) || ((coor.x - radius) < 0) || ((coor.x + radius) > (xDim-1)))
		{
			std::cout << "WARNNING: Coordinate at (x=" << coor.x<< ", y=" << coor.y << ", z=" << coor.z << ") erased due to its out of the mask." << std::endl;
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
						double value;
						value = DIRECT_A3D_ELEM(tom, coor.z + k, coor.y + i, coor.x + j);
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
	size_t id;

	for(size_t i = 0; i < inputCoords.size(); i++)
	{
		id = outCoordMd.addObject();
		outCoordMd.setValue(MDL_XCOOR, inputCoords[i].x, id);
		outCoordMd.setValue(MDL_YCOOR, inputCoords[i].y, id);
		outCoordMd.setValue(MDL_ZCOOR, inputCoords[i].z, id);
	}

	outCoordMd.write(fnOut);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Output coordinates metadata saved at: " << fnOut << std::endl;
	#endif
}


// --------------------------- MAIN ----------------------------------

void ProgTomoFilterCoordinates::run()
{
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;	

	auto t1 = high_resolution_clock::now();

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

	// CHECK COORDINATES WITH MASK IF APPEARS AS INPUT
	takeCoordinateFromTomo(tom);
	
	auto t2 = high_resolution_clock::now();
	
	/* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
}