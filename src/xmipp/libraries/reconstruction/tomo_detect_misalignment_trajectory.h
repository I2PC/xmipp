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
#include <core/metadata_vec.h>
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
#include <data/point3D.h>
#include <data/point2D.h>

#define VERBOSE_OUTPUT
#define DEBUG
// #define DEBUG_DIM
// #define DEBUG_FILTERLABEL
// #define DEBUG_POISSON
// #define DEBUG_CHAINS
#define DEBUG_MISALI
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
    size_t biggestSize;

    /** Vector containig the tilt angles from the series */
    std::vector<double> tiltAngles;

    /** Angle step */
    float tiltAngleStep;

    /** Vector for peaked coordinates components */
    std::vector<Point3D<double>> coordinates3D;

    /** Map of clustered and filtered chains */
    MultidimArray<int> chain2dMap;

    /** Vectors for calculated residuals components */
    std::vector<double> residualX;
    std::vector<double> residualY;
    std::vector<int> residualCoordinateX;
    std::vector<int> residualCoordinateY;
    std::vector<int> residualCoordinateZ;

    /** Thresholds */
    size_t poissonLandmarkPercentile = 50;          //*** update with more clever meassurement
    size_t numberOfElementsInChainThreshold = 6;    // Minimum number of landmarks to keep a chain

    // Thresholds are saved in angstroms in order to be independent of the sampling rate and image size
    float minDistanceAng = 70;                      // Minimum distance to cosider that 2 landmarks belong to the same chain
    float thrChainDistanceAng = 100;                // Maximum distance of a detected landmark to a chain
    
    // Thresholds measured in pixels updated in generateSideInfo function
    float minDistancePx;                          
    double thrChainDistancePx;

    /** Alignment report. True = aligned - False = misaligned */
    bool globalAlignment;
    std::vector<bool> localAlignment;

public:

    bool detectGlobalAlignmentPoisson(std::vector<int> counterLinesOfLandmarkAppearance, std::vector<size_t> chainIndexesY);
    void writeOutputAlignmentReport();



    // --------------------------- INFO functions ----------------------------

    void readParams();

    void defineParams();


    // --------------------------- HEAD functions ----------------------------

    /**
     * Generate side info usefull for the rest of protocols
     *
     * @param
     * @return
     *
    */
    void generateSideInfo();

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
     * Detect landmark chains from landmark coordinates.
     *
     * @param
     * @return
     *
    */
    void detectLandmarkChains();

    /**
     * Detect images from the tilt-series misaligned from the detected landmark chains.
     *
     * @param
     * @return
     *
    */
    void detectMisalignedTiltImages();

    /**
     * Calculate residual vectors from the 3D landmark and the obtained coordinates.
     *
     * @param
     * @return
     *
    */
    // void calculateResidualVectors(MetaDataVec inputCoordMd);


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
    std::vector<Point2D<double>> getCoordinatesInSlice(size_t slice);


    /**
     * Retrieve a vector contaiing 3 different indexes as i > 0 && i > size.
     *
     * @param
     * @return
     *
    */
    std::vector<size_t> getRandomIndexes(size_t size);


    /**
     * Description ***
     *
     * @param 
     * @return
     *
    */
    float testPoissonDistribution(float poissonAverage, size_t numberOfOcccurrences);


    /**
     * Description ***
     *
     * @param 
     * @return
     *
    */
    float calculateLandmarkProjectionDiplacement(float theta1, float theta2, float coordinateProjX);



    // --------------------------- MAIN ----------------------------------

    void run();
};

#endif