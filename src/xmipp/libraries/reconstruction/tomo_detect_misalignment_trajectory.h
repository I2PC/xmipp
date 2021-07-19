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

#ifndef __IMAGE_PEAK_HIGH_CONTRAST
#define __IMAGE_PEAK_HIGH_CONTRAST

#include <iostream>
#include <core/xmipp_program.h>
#include <core/xmipp_image.h>
#include <core/metadata.h>
#include <core/xmipp_fft.h>
#include <core/xmipp_fftw.h>
#include <core/geometry.h>
#include <math.h>
#include <limits>
#include <complex>
#include <data/fourier_filter.h>
#include <data/filters.h>
#include <string>
#include "symmetrize.h"
#include <data/morphology.h>
#include "core/xmipp_image_generic.h"

#define VERBOSE_OUTPUT
#define DEBUG
// #define DEBUG_DIM
// #define DEBUG_FILTERLABEL
#define DEBUG_RESID
#define DEBUG_OUTPUT_FILES

class ProgTomoDetectMisalignmentTrajectory : public XmippProgram
{

public:
    /** Filenames */
    FileName fnVol, fnOut, fnInputCoord, fnTiltAngles;

    /** Threshold */
    double fiducialSize, samplingRate, sdThreshold;

    /** Number of slices and original centers of mass */
    int boxSize, numberSampSlices, numberCenterOfMass, distanceThr, numberOfCoordinatesThr;

    /** Center features **/
    bool checkInputCoord;
    
private:
    /** Input tilt-series dimensions */
    size_t xSize;
	size_t ySize;
	size_t zSize;
    size_t nSize;

    /** Vector containig the tilt angles from the series */
    std::vector<double> tiltAngles;

    /** Vectors for peaked coordinates components */
    std::vector<double> coordinates3Dx;
    std::vector<double> coordinates3Dy;
    std::vector<int> coordinates3Dn;

    /** Vectors for calculated residuals components */
    std::vector<double> residualX;
    std::vector<double> residualY;
    std::vector<int> residualCoordinateX;
    std::vector<int> residualCoordinateY;
    std::vector<int> residualCoordinateZ;

public:

    // --------------------------- INFO functions ----------------------------

    void readParams();

    void defineParams();


    // --------------------------- HEAD functions ----------------------------

    /**
     * Bandpass filtering the input tilt-series.
     *
     * @param
     * @return
     *
    */
    void bandPassFilter(MultidimArray<double> &inputTiltSeries);

    /**
     * Peaks high contrast regions in a volume.
     *
     * @param
     * @return
     *
    */
    void getHighContrastCoordinates(MultidimArray<double> tiltSeriesFiltered);

    /**
     * Calculate residual vectors from the 3D landmark and the obtained coordinates.
     *
     * @param
     * @return
     *
    */
    void calculateResidualVectors(MetaData inputCoordMd);


    // --------------------------- I/O functions ----------------------------

    /**
     * Write obtained coordinates in output file.
     *
     * @param
     * @return
     *
    */
    void writeOutputCoordinates();


    /**
     * Write obtained coordinates in output file.
     *
     * @param
     * @return
     *
    */
    void writeOutputResidualVectors();


    // --------------------------- UTILS functions ----------------------------

    /**
     * Filter labeled regions.
     *
     * @param
     * @return
     *
    */
    bool filterLabeledRegions(std::vector<int> coordinatesPerLabelX, std::vector<int> coordinatesPerLabelY, double centroX, double centroY);

    /**
     * Calculation of the projection matrix given the projection angle.
     *
     * @param
     * @return
     *
    */
    Matrix2D<double> getProjectionMatrix(double tiltAngle);

    /**
     * Retrieve all coordinates peaked from the same slice.
     *
     * @param
     * @return
     *
    */
    std::vector<Matrix1D<double>> getCoordinatesInSlice(int slice);


    // --------------------------- MAIN ----------------------------------

    void run();
};

#endif