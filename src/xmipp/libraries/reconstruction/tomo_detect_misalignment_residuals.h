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
// #define DEBUG_RESID
// #define DEBUG_RESIDUAL_ANALYSIS
// #define DEBUG_RESIDUAL_STATISTICS_FILE

class ProgTomoDetectMisalignmentResiduals : public XmippProgram
{

public:
    /** Filenames */
    FileName fnInputTS;
    FileName fnResidualInfo;
    FileName fnOut;

    /** Input info */
    double fiducialSize;    // Fiducial size in Angstroms
    float fiducialSizePx;   // Fiducial size in pixels
    double samplingRate;

    /** Thresholds */
    float thrFiducialDistance;   // Maximum distance of a detected landmark to consider it belongs to a chain


    // Coordinate model structure
    struct resMod {
        Point3D<double> landmarkCoord;     // Coordinate detected in each tilt-image
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
    std::vector<resMod> vResMod;

    /** Vector containig the input 3D coorinates used for alignment */
    size_t numberOfInputCoords;

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
     *
    */
    void generateSideInfo();

    void detectMisalignmentFromResiduals();
    void detectMisalignmentFromResidualsMahalanobis();
    void detectMisalignmentFromResidualsMahalanobisRobust();
    void generateResidualStatiscticsFile();
    void contructResidualMatrix();


    // --------------------------- I/O functions ----------------------------

    /**
     * Write obtained coordinates in output file.
     *
     * @param
     * @return
     *
    */
   void readInputResiduals();

   void writeOutputAlignmentReport(); 
   void writeWeightedResiduals();



    // --------------------------- UTILS functions ----------------------------

    void getResModByFiducial(size_t fiducialNumber, std::vector<resMod> &vResMod_fiducial);

    void getResModByImage(size_t tiltImageNumber, std::vector<resMod> &vResMod_image);


    // --------------------------- MAIN ----------------------------------

    void run();

};

#endif