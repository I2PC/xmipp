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
#include <math.h>
#include <limits>
#include <complex>
#include <data/fourier_filter.h>
#include <data/filters.h>
#include <string>
#include "symmetrize.h"
#include <data/morphology.h>
#include <data/point3D.h>
#include <data/point2D.h>
#include <data/basic_pca.h>

#define VERBOSE_OUTPUT
// #define DEBUG_OUTPUT_FILES
// #define DEBUG_DIM
// #define DEBUG_PREPROCESS
// #define DEBUG_HCC
// #define DEBUG_FILTERLABEL
// #define DEBUG_CLUSTER
// #define DEBUG_COORDS_IN_SLICE
// #define DEBUG_DIST
// #define DEBUG_CENTER_COORDINATES
// #define DEBUG_REMOVE_DUPLICATES
#define DEBUG_FILTER_COORDINATES
// #define DEBUG_RADIAL_AVERAGE
// #define DEBUG_MAHALANOBIS_DISTANCE

class ProgImagePeakHighContrast : public XmippProgram
{

public:
    /** Filenames */
    FileName fnVol;
    FileName fnOut;

    /** Fiducial size in angstroms */
    double fiducialSize;

    /** Sampling rate */
    double samplingRate;

    /** Box size */
    int boxSize;

    /** Number of samplig slices to calculate mean and SD of the tomogram*/
    int numberSampSlices;

    /** Thresholds */
    double sdThr;                       // Number of SD over the mean to consider a coordinate value as an outlier
    int numberOfCoordinatesThr;         // Minimum number of coordinates to keep a label
    double mirrorCorrelationThr;        // Minimum correlation between a fiducial and its mirror
    double mahalanobisDistanceThr;      // Maximum mahalanobis distance (empirical value)

    /** Toggle to use relaxed mode*/
    bool relaxedMode;               // Relaxed mode keeps coordinates when none of them pass the mirror correlation filter

    /** Fiducial size in pixels */
     double fiducialSizePx;

    /** Centralized Fourier transformer */
   	FourierTransformer transformer;

    /** Centralized Image for handlign tomogram */
	Image<double> V;

    
private:
    /** Input tomogram dimensions */
    size_t xSize;
	size_t ySize;
	size_t zSize;
	size_t nSize;

    /** Biggest tomogram size used for normalization */
	size_t normDim;

    /** Vectors saving centers of mass components after coordinates clusterings */
    std::vector<int> centerOfMassX;
    std::vector<int> centerOfMassY;
    std::vector<int> centerOfMassZ;

    /** Vector saving 3D coordinates */
    std::vector<Point3D<double>> coordinates3D;

public:

    // --------------------- IN/OUT FUNCTIONS -----------------------------

    /**
     * Read input program parameters.
    */
    void readParams();

    /**
     * Define input program parameters.
    */
    void defineParams();

    /**
     * Write output coordinates metadata.
    */
    void writeOutputCoordinates();


    // ---------------------- MAIN FUNCTIONS -----------------------------

    /**
     * Proprocessing of the input volume. Slice averaging (5 slices), bandpass filtering at 
     * the gold bead size, and apply laplacian. 
    */
    void preprocessVolume(MultidimArray<double> &inputTomo);

    /**
     * Peak high contrast coordinates in a volume. Detect coordinates with an outlier value, 
     * generate a binary map to posterior label it, and filter the labeled regions depending on
     * size and shape. Keep
    */
    void getHighContrastCoordinates(MultidimArray<double> &volFiltered);

    /**
     * Smoothing and filtering the input volume. Use a votting system remove those coodinates
     * that are not cosistent through different slices, cluster those coordinates that survived
     * the votting, and averga those coordinates belonging to the same cluster.
    */
    void clusterHCC();

    /**
     * Center the obtained coordinates from clustering. Shift coordinates to keep the fiducial
     * centered by calculating the maximum correlation between the peaked feature and its mirror.
    */
    void centerCoordinates(MultidimArray<double> volFiltered);

    /**
     * Remove duplicated coordinates. Iteratively, merge those coordinates referred to the same 
     * high contrast feature based on a minimum distance threshold (the fiducial size).
    */
    void removeDuplicatedCoordinates(MultidimArray<double> volFiltered);

    /**
     * Filter coordinates by the correlation. Calculate the dot product between each feature and 
     * its mirror, and compare to a correlation threshold.
    */
    void filterCoordinatesByCorrelation(MultidimArray<double> volFiltered);


    // --------------------------- UTILS functions ----------------------------

    /**
     * Calculate global parameters used through the program.
    */
    void generateSideInfo();

    /**
     * Filter labeled regions. Util method to remove those labeles regions based on their size 
     * (minimum number of points that should contain) and shape (present a globular structure,
     * as expected from a gold bead).
    */
    bool filterLabeledRegions(std::vector<int> coordinatesPerLabelX, std::vector<int> coordinatesPerLabelY, double centroX, double centroY);

    /**
     * Get index coordinates from slice. Return the index in the 3D coordinates vector of those 
     * coordinates belonging to the specified slice.
    */
    std::vector<size_t> getCoordinatesInSliceIndex(size_t slice);

    /**
     * Calculate the radial average of the numSlices slices centered in the feature volume 
     * (Z direction).
    */
    void radialAverage(MultidimArray<float> &feature, MultidimArray<float> &radialAverage, size_t numSlices);
    
    /**
     * 
    */
    void mahalanobisDistance(std::vector<MultidimArray<float>> &setOfFeatures_RA, MultidimArray<double> &mahalanobisDistance_List);

    // --------------------------- MAIN ----------------------------------

    /**
     * Run main program.
    */
    void run();
};

#endif