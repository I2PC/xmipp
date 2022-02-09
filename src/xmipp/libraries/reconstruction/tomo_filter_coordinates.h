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

#ifndef __TOMO_FILTER_COORDINATES
#define __TOMO_FILTER_COORDINATES


#include <iostream>
#include <core/xmipp_program.h>
#include <core/xmipp_filename.h>
#include <core/xmipp_image.h>
#include <data/point3D.h>
#include <core/metadata_vec.h>

#define VERBOSE_OUTPUT
//#define DEBUG_DIM


class ProgTomoFilterCoordinates : public XmippProgram
{

public:
    /** Filenames */
    FileName fnInTomo, fnMask, fnInCoord, fnOutCoord;

    /** Threshold */
    double resThr;

    /** Check params **/
    bool checkResThr;

    /** Execution mode: 0 -> mask, 1 -> resolution **/
    bool execMode;

    /** Radius map amalysis*/
    size_t radius;


private:
    /** Tomogram size */
    size_t xDim, yDim, zDim;

    /** Input coordinates */
    std::vector<Point3D<int>> inputCoords;


public:
    // --------------------------- INFO functions ----------------------------

    void readParams();

    void defineParams();


    // --------------------------- HEAD functions ----------------------------

    void filterCoordinatesWithMask(MultidimArray<double> &inputVolume);


    // --------------------------- I/O functions ----------------------------

    void readInputCoordinates();

    void writeOutputCoordinates();

    void takeCoordinateFromTomo(MultidimArray<double> &tom);


    // --------------------------- MAIN ----------------------------------

    void run();


};

#endif
