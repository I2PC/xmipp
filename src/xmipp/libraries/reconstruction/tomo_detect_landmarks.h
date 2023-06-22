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

#ifndef __TOMO_DETECT_LANDMARKS
#define __TOMO_DETECT_LANDMARKS

#include <iostream>
#include <math.h>
#include <limits>
#include <complex>
#include <string>
#include <stdio.h>
#include <core/xmipp_filename.h>
#include <core/multidim_array.h>
#include <core/xmipp_program.h>
#include <core/metadata_label.h>
#include <core/metadata_vec.h>
#include <core/xmipp_image.h>
#include <core/xmipp_fft.h>
#include <core/xmipp_fftw.h>
#include <core/xmipp_image_generic.h>
#include <data/point3D.h>
#include <data/filters.h>

#include <fstream>

#define VERBOSE_OUTPUT

#define DEBUG_DIM
// #define DEBUG_DOWNSAMPLE
// #define DEBUG_SOBEL
#define DEBUG_HCC
#define DEBUG_OUTPUT_FILES

class ProgTomoDetectLandmarks : public XmippProgram
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
   
    /** Input tilt-series dimensions */
    size_t xSize;
	size_t ySize;
	size_t zSize;
    size_t nSize;
    size_t normDim;

    /** Input tilt-series dimensions after downsampling */
    size_t xSize_d;
	size_t ySize_d;
	size_t zSize_d;
    size_t nSize_d;
    size_t normDim_d;

    /** Target fiducial size and downsampling factor */
    size_t targetFS = 16;
    double ds_factor;



    /** Vector for peaked coordinates components */
    std::vector<Point3D<double>> coordinates3D;

    /** Fiducial size in pixels */
    float fiducialSizePx;

    // Define the Sobel kernels
    std::vector<std::vector<double>> sobelX = {{-1, 0, 1},
                                            {-2, 0, 2},
                                            {-1, 0, 1}};
                                  
    std::vector<std::vector<double>> sobelY = {{-1, -2, -1},
                                            { 0,  0,  0},
                                            { 1,  2,  1}};

public:

    // --------------------------- INFO functions ----------------------------
    
    void readParams();

    void defineParams();

    void generateSideInfo();


    // --------------------------- HEAD functions ----------------------------
    
    void downsample(MultidimArray<double> &tiltImage, MultidimArray<double> &tiltImage_ds);

    void sobelFiler(MultidimArray<double> &tiltImage);

    void getHighContrastCoordinates(MultidimArray<double> tiltSeriesFiltered);

    // --------------------------- I/O functions ----------------------------
    
    void writeOutputCoordinates();


    // --------------------------- MAIN ----------------------------------

    void run();

};

#endif