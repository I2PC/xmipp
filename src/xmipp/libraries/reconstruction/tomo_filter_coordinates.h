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

#include <core/xmipp_filename.h>
#include <data/point3D.h>
#include <core/xmipp_image.h>
#include <core/xmipp_program.h>


#define VERBOSE_OUTPUT
//#define DEBUG_DIM


class ProgTomoFilterCoordinates : public XmippProgram
{

private:
    /** Filenames */
    FileName fnInTomo, fnMask, fnInCoord, fnOut;

    /** Threshold */
    double resThr;

    /** Execution mode: 0 -> mask, 1 -> resolution **/
    bool execMode;

    /** Radius map amalysis*/
    int radius;

    /** Tomogram size */
    size_t xDim, yDim, zDim;

    /** Input coordinates */
    std::vector<Point3D<int>> inputCoords;


private:
    // --------------------------- INFO functions ----------------------------

    void readParams();

    void defineParams();


    // --------------------------- HEAD functions ----------------------------

    void filterCoordinatesWithMask(MultidimArray<int> &inputVolume);

    void defineSphere(MultidimArray<int> &sphere);

    void extractStatistics(MultidimArray<double> &tomo, MultidimArray<int> &sphere);


    // --------------------------- I/O functions ----------------------------

    void readInputCoordinates();

    void writeOutputCoordinates();

    void calculateCoordinateStatistics(MultidimArray<double> &tom);


    // --------------------------- MAIN ----------------------------------

    void run();


};

#endif
