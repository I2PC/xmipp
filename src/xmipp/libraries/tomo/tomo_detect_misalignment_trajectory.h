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
#include "reconstruction/symmetrize.h"
#include <data/morphology.h>
#include "core/xmipp_image_generic.h"
#include <data/point3D.h>
#include <data/point2D.h>
#include <stdio.h>
#include <fstream>
#include "tomo_detect_landmarks.h"

#define VERBOSE_OUTPUT
#define GENERATE_RESIDUAL_STATISTICS

// #define DEBUG_DIM
// #define DEBUG_PREPROCESS
// #define DEBUG_HCC
// #define DEBUG_CENTER_COORDINATES
// #define DEBUG_VOTTING
// #define DEBUG_FILTERLABEL
// #define DEBUG_POISSON
// #define DEBUG_CHAINS
// #define DEBUG_GLOBAL_MISALI
// #define DEBUG_LOCAL_MISALI
// #define DEBUG_RESID
// #define DEBUG_COORDS_CS
// #define DEBUG_RESIDUAL_ANALYSIS
#define DEBUG_PRUNE_RESIDUALS
// #define DEBUG_RESIDUAL_STATISTICS_FILE
#define DEBUG_OUTPUT_FILES

class ProgTomoDetectMisalignmentTrajectory : public XmippProgram
{

public:
    /** Landmark detector */
    ProgTomoDetectLandmarks lmDetector;

    /** Params specific for landmark detectior */
    double targetFS;

    /** Filenames */
    FileName fnVol;
    FileName fnOut;
    FileName fnTiltAngles;
    FileName fnInputCoord;

    /** Input info */
    double fiducialSize;    // Fiducial size in Angstroms
    float fiducialSizePx;   // Fiducial size in pixels
    double samplingRate;

    /** Thresholds */
    float thrSDHCC;              // Threshold number of SD a coordinate value must be over the mean to consider that it belongs to a high contrast feature.
    float thrFiducialDistance;   // Maximum distance of a detected landmark to consider it belongs to a chain


    // Coordinate model structure
    struct CM {
        Point3D<double> detectedCoordinate;     // Coordinate detected in each tilt-image
        Point3D<double> coordinate3d;           // 3D coordinate whose porjection is the closest
        Point2D<double> residuals;              // Residual vector from detected to projected
        size_t id;                              // ID common for all the CM belonging to the same coordinate 3D
        double mahalanobisDistance;             // Mahalanobis distance to the expected residual distribution
    };
    
    /** Input tilt-series dimensions */
    size_t xSize;
	size_t ySize;
	size_t zSize;
    size_t nSize;
    size_t normDim;

    /** Array of coordinate model structures */
    std::vector<CM> vCM;

    /** Vector containig the tilt angles from the series */
    std::vector<double> tiltAngles;

    /** Vector containig the input 3D coorinates used for alignment */
    std::vector<Point3D<int>> inputCoords;
    size_t numberOfInputCoords = inputCoords.size();

    /** Angle step */
    float tiltAngleStep;

    /** Vector for peaked coordinates components */
    std::vector<Point3D<double>> coordinates3D;

    /** Thresholds */
    // float avgResidPercentile_LocalAlignment;

    /** Alignment report. True = aligned / False = misaligned */
    bool globalAlignment = true;
    std::vector<bool> localAlignment;

    /** Avg and STD of Mahalanobis distance for each tilt image */
    std::vector<double> avgMahalanobisDistanceV;
    std::vector<double> stdMahalanobisDistanceV;

public:

    // --------------------------- INFO functions ----------------------------

    void readParams();

    void defineParams();


    // --------------------------- HEAD functions ----------------------------

    /**
     * Generate side info usefull for the rest of protocols
     *
     * @param
     * @return
    */
    void generateSideInfo();

    /**
     * Calculate residual vectors from the 3D landmark and the obtained coordinates.
     *
     * @param
     * @return
     *
    */
    void calculateResidualVectors();

    void detectMisalignmentFromResiduals();
    void detectMisalignmentFromResidualsMahalanobis();

    void generateResidualStatiscticsFile();

    void pruneResidualVectors();


    // --------------------------- I/O functions ----------------------------

    /**
     * Write obtained coordinates in output file.
     *
     * @param
     * @return
     *
    */
   void writeOutputVCM();

   void writeOutputAlignmentReport(); 


    // --------------------------- UTILS functions ----------------------------

    /**
     * Filter labeled regions.
     *
     * @param
     * @return
     *
    */
    void fillImageLandmark(MultidimArray<int> &proyectedImage, int x, int y, int value);

    /**
     * Bandpass filtering the input tilt-series.
     *
     * @param
     * @return
     *
    */
    void adjustCoordinatesCosineStreching();

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

    void getCMFromCoordinate(int x, int y, int z, std::vector<CM> &vCM);

    bool checkProjectedCoordinateInInterpolationEdges(Matrix1D<double> projectedCoordinate, size_t slice);

    void getCMbyFiducial(size_t fiducialNumber, std::vector<CM> &vCM_fiducial);

    void getCMbyImage(size_t tiltImageNumber, std::vector<CM> &vCM_image);


    // --------------------------- MAIN ----------------------------------

    void run();


    // --------------------------- UNUSED FUNCTIONS ----------------------------------

    // void bandPassFilter(MultidimArray<double> &inputTiltSeries, int imageNumber);

    // void bandPassFilterBis(MultidimArray<double> &tiltImage, MultidimArray<double> &tiltImageBis);

    // bool votingHCC();

    // void localAmplitude(MultidimArray<double> &tiltImage, MultidimArray<double> &amplitude);

    // void factorial(size_t base, size_t fact);

    // double binomialTest(int x, int n, float p);

    // std::vector<CM> getCMFromCoordinate(int x, int y, int z);

    // float calculateLandmarkProjectionDiplacement(float theta1, float theta2, float coordinateProjX);

    // float testPoissonDistribution(float poissonAverage, size_t numberOfOcccurrences);

    // std::vector<size_t> getRandomIndexes(size_t size);

    // std::vector<size_t> getCoordinatesInSliceIndex(size_t slice);

    // int calculateTiltAxisIntersection(Point2D<double> p1, Point2D<double> p2);

    // bool detectGlobalMisalignment();

    // std::vector<std::vector<Point2D<double>>> splitCoordinatesInHalfImage(std::vector<Point2D<double>> inCoords);

    // bool filterLabeledRegions(std::vector<int> coordinatesPerLabelX, std::vector<int> coordinatesPerLabelY, double centroX, double centroY);
    
    // void closing2D(MultidimArray<double> binaryImage, int size, int count, int neig);

    // void detectInterpolationEdges(MultidimArray<double> &tiltImage);

    // void writeOutputResidualVectors();

    // void writeOutputCoordinates();

    // void detectMisalignedTiltImages();

    // void getHighContrastCoordinates(MultidimArray<double> tiltSeriesFiltered);

    // void centerCoordinates(MultidimArray<double> tiltSeriesFiltered);

    // void detectLandmarkChains();

    // bool detectGlobalAlignmentPoisson(std::vector<int> counterLinesOfLandmarkAppearance, std::vector<size_t> chainIndexesY);

    // void detectMisalignmentFromResidualsBis();

};

#endif