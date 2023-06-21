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
#include <stdio.h>

#include <fstream>

#define VERBOSE_OUTPUT
// #define GENERATE_RESIDUAL_STATISTICS

// #define DEBUG_DIM
// #define DEBUG_PREPROCESS
#define DEBUG_HCC
// #define DEBUG_CENTER_COORDINATES
// #define DEBUG_VOTTING
// #define DEBUG_FILTERLABEL
// #define DEBUG_POISSON
// #define DEBUG_CHAINS
// #define DEBUG_GLOBAL_MISALI
// #define DEBUG_LOCAL_MISALI
// #define DEBUG_RESID
// #define DEBUG_COORDS_CS
#define DEBUG_RESIDUAL_ANALYSIS
#define DEBUG_OUTPUT_FILES

class ProgTomoDetectMisalignmentTrajectory : public XmippProgram
{

public:
    /** Filenames */
    FileName fnVol;
    FileName fnOut;
    FileName fnTiltAngles;
    FileName fnInputCoord;

    /** Input info */
    double fiducialSize;
    double samplingRate;

    bool checkInputCoord;

    /** Thresholds */
    int thrNumberCoords;         // Threshold minimum number of coordinates attracted to a center of mass to consider it as a high contrast feature.
    float thrSDHCC;              // Threshold number of SD a coordinate value must be over the mean to consider that it belongs to a high contrast feature.
    float thrChainDistanceAng;   // Maximum distance of a detected landmark to consider it belongs to a chain
    float thrFiducialDistance;   // Maximum distance of a detected landmark to consider it belongs to a chain


    // Coordinate model structure
    struct CM {
        Point3D<double> detectedCoordinate;     // Coordinate detected in each tilt-image
        Point3D<double> coordinate3d;           // 3D coordinate whose porjection is the closest
        Point2D<double> residuals;              // Residual vector from detected to projected
        size_t id;                              // ID common for all the CM belonging to the same coordinate 3D
    };


    // Vector of points saving the interpolation limits for each tilt-image
    std::vector<std::vector<Point2D<int>>> interpolationLimitsVector;


    // Interpolation corners structure for each tilt-image
    // struct IC {
    //     int x1;  // Top-left corner (x1, 0)
    //     int x2;  // Top-right corner (x2, 0)
    //     int x3;  // Bottom-left corner (x3, ySize)
    //     int x4;  // Bottom-right corner (x4, ySize)
    //     int y1;  // Top-left corner (0, y1)
    //     int y2;  // Top-right corner (xSize, y2)
    //     int y3;  // Bottom-left corner (0, y3)
    //     int y4;  // Bottom-right corner (xSize, y4)
    //     double m1;   // Slope of top-left edge
	//     double m2;   // Slope of top-right edge
	//     double m3;   // Slope of bottom-left edge
	//     double m4;   // Slope of bottom-right edge
    // };

    /** Array of interpolation corner structures */
    // std::vector<IC> vIC;


    
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

    /** Map of clustered and filtered chains */
    MultidimArray<int> chain2dMap;

    /** Vectors for calculated residuals components */
    std::vector<double> residualX;
    std::vector<double> residualY;
    std::vector<int> residualCoordinateX;
    std::vector<int> residualCoordinateY;
    std::vector<int> residualCoordinateZ;

    /** Thresholds */
    float poissonLandmarkPercentile = 0.2;          // Percentencile of the number of landmarks per row among the populated rows (exclude empty rows), taken as the lambda for poisson probability calculation
    size_t numberOfElementsInChainThreshold = 6;    // Minimum number of landmarks to keep a chain
    size_t thrNumberDistanceAngleChain = 3;         // Angular distance (number of angular steps) for two coordinates to belong to the same chain, multiplied by the distance to the tilt axis
    float avgResidPercentile_LocalAlignment;

    // Distance thresholds are saved in angstroms in order to be independent of the sampling rate and image size
    float minDistanceAng = 20;                      // Minimum distance to consider that 2 landmarks belong to the same chain
    
    // Thresholds measured in pixels updated in generateSideInfo function
    float minDistancePx;                          
    double thrChainDistancePx;
    float fiducialSizePx;

    // Global alignment thresholds
    float thrTop10Chain = 20;                       // Percentage of LM belonging to the top 10 populated chains (top10ChainLM/coordinates3D.size())
    float thrLMChain = 30;                          // Percentage of number of average LM belonging to the selected chains (avgChainLM/(chainIndexes.seiz()*coordinates3D.size()))

    /** Alignment report. True = aligned / False = misaligned */
    bool globalAlignment = true;
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
    void bandPassFilter(MultidimArray<double> &inputTiltSeries, int imageNumber);

    void bandPassFilterBis(MultidimArray<double> &tiltImage, MultidimArray<double> &tiltImageBis);

    bool votingHCC();


    /**
     * Peaks high contrast regions in a volume.
     *
     * @param
     * @return
     *
    */
    void getHighContrastCoordinates(MultidimArray<double> tiltSeriesFiltered);


    void centerCoordinates(MultidimArray<double> tiltSeriesFiltered);


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
    void calculateResidualVectors();


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

        /**
     * Write obtained coordinates in output file.
     *
     * @param
     * @return
     *
    */
   void writeOutputVCM();


    // --------------------------- UTILS functions ----------------------------

    /**
     * Filter labeled regions.
     *
     * @param
     * @return
     *
    */
    bool filterLabeledRegions(std::vector<int> coordinatesPerLabelX, std::vector<int> coordinatesPerLabelY, double centroX, double centroY);
    void closing2D(MultidimArray<double> binaryImage, int size, int count, int neig);

    /**
     * Filter labeled regions.
     *
     * @param
     * @return
     *
    */
    void fillImageLandmark(MultidimArray<int> &proyectedImage, int x, int y, int value);

    /**
     * Filter labeled regions.
     *
     * @param
     * @return
     *
    */
    bool detectGlobalMisalignment();

    std::vector<std::vector<Point2D<double>>> splitCoordinatesInHalfImage(std::vector<Point2D<double>> inCoords);



    /**
     * Bandpass filtering the input tilt-series.
     *
     * @param
     * @return
     *
    */
    void adjustCoordinatesCosineStreching();


    /**
     * Filter labeled regions.
     *
     * @param
     * @return
     *
    */
    int calculateTiltAxisIntersection(Point2D<double> p1, Point2D<double> p2);


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


    std::vector<size_t> getCoordinatesInSliceIndex(size_t slice);

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

    /**
     * Description ***
     *
     * @param 
     * @return
     *
    */

    // std::vector<CM> getCMFromCoordinate(int x, int y, int z);

    void getCMFromCoordinate(int x, int y, int z, std::vector<CM> &vCM);

    void detectMisalignmentFromResiduals();

    bool checkProjectedCoordinateInInterpolationEdges(Matrix1D<double> projectedCoordinate, size_t slice);


    // void factorial(size_t base, size_t fact);

    double binomialTest(int x, int n, float p);

    void getCMbyFiducial(size_t fiducialNumber, std::vector<CM> &vCM_fiducial);

    void getCMbyImage(size_t tiltImageNumber, std::vector<CM> &vCM_image);

    // void localAmplitude(MultidimArray<double> &tiltImage, MultidimArray<double> &amplitude);



    // --------------------------- MAIN ----------------------------------

    void run();
};

#endif