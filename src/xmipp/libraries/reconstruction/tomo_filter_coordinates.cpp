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
#include <chrono>


// --------------------------- INFO functions ----------------------------

void ProgTomoFilterCoordinates::readParams()
{
	fnInVol = getParam("-inVol");
	fnInCoord = getParam("-inCoord");
    
    checkResThr = checkParam("-resThr");

	if(checkResThr)
	{
		resThr = getDoubleParam("-resThr");
		execMode = 1;
	}else
	{
		execMode = 0;
	}

   	fnOutCoord = getParam("-outCoord");
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
	addParamsLine("  -inVol <mrcs_file=\"\">                                : Input volume (mask or resolution map).");
	addParamsLine("  -inCoord <xmd_file=\"\">                               : Input xmd file containing the 3D coordinates.");
	addParamsLine("  [-resThr <resThr=0.25>]                               : Percentile resolution threshold.");
    addParamsLine("  [-outCoord <outCoord=\"filteredCoordinates3D.xmd\">]   : Output file containing the filtered 3D coordinates.");
}


// --------------------------- HEAD functions ----------------------------

void ProgTomoFilterCoordinates::filterCoordinatesWithMask(MultidimArray<int> &inputVolume)
{
	Point3D<double> coord3D;
	for (size_t i = 0; i < inputCoords.size(); i++)
	{
		coord3D = inputCoords[i];

		if(DIRECT_A3D_ELEM(inputVolume, (int)coord3D.z, (int)coord3D.y, (int)coord3D.x) == 0)
		{
			inputCoords.erase(inputCoords.begin()+i);
			i--;
		}
	}
}


// --------------------------- I/O functions ----------------------------

void ProgTomoFilterCoordinates::readInputCoordinates()
{
	MetaDataVec inCoordMd;
	inCoordMd.read(fnInCoord);

	size_t objId;
	Point3D<double> coordinate3D;

	// FOR_ALL_OBJECTS_IN_METADATA(inCoordMd)
	// {
	// 	// objId = __iter.objId;
	// 	objId = inCoordMd.ids;

	// 	inCoordMd.getValue(MDL_XCOOR, coordinate3D.x, objId);
	// 	inCoordMd.getValue(MDL_YCOOR, coordinate3D.y, objId);
	// 	inCoordMd.getValue(MDL_ZCOOR, coordinate3D.z, objId);

	// 	inputCoords.push_back(coordinate3D);
	// }

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

	outCoordMd.write(fnOutCoord);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Output coordinates metadata saved at: " << fnOutCoord << std::endl;
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

	if(execMode)
	{
		Image<double> inputVolume;
		inputVolume.read(fnInVol);

	}else
	{
		Image<int> inputVolume;
		inputVolume.read(fnInVol);

		auto &inVol=inputVolume();

		xDim = XSIZE(inVol);
		yDim = YSIZE(inVol);
		zDim = ZSIZE(inVol);

		#ifdef DEBUG_DIM
		std::cout << "Input volume dimensions:" << std::endl;
		std::cout << "x " << XSIZE(inputTomo) << std::endl;
		std::cout << "y " << YSIZE(inputTomo) << std::endl;
		std::cout << "z " << ZSIZE(inputTomo) << std::endl;
		std::cout << "n " << NSIZE(inputTomo) << std::endl;
		#endif

		readInputCoordinates();

		#ifdef VERBOSE_OUTPUT
		std::cout << "Number of coordinates before filtering: " << inputCoords.size() << std::endl;
		#endif


		if(execMode)
		{
			std::cout << "Filtering with resolution map..." << std::endl;
		}else
		{
			std::cout << "Filtering with mask..." << std::endl;
			filterCoordinatesWithMask(inVol);
		}

		#ifdef VERBOSE_OUTPUT
		std::cout << "Number of coordinates after filtering: " << inputCoords.size() << std::endl;
		#endif

		writeOutputCoordinates();
	}
	
	auto t2 = high_resolution_clock::now();
	/* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
}