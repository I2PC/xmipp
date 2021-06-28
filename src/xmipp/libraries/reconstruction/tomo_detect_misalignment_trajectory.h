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
#define DEBUG_FILTERPARAMS
// #define DEBUG_DIM
// #define DEBUG_DIST
#define DEBUG_OUTPUT_FILES

class ProgTomoDetectMisalignmentTrajectory : public XmippProgram
{

public:
    /** Filenames */
    FileName fnVol, fnOut;

    /** Threshold */
    double fiducialSize, samplingRate, sdThreshold;

    /** Number of slices and original centers of mass */
    int boxSize, numberSampSlices, numberCenterOfMass, distanceThr, numberOfCoordinatesThr;

    /** Center features **/
    bool centerFeatures;
    
private:
    /** Input tomogram dimensions */
    size_t xSize;
	size_t ySize;
	size_t zSize;

    /** Vectors for peaked coordinates components */
    std::vector<int> coordinates3Dx;
    std::vector<int> coordinates3Dy;
    std::vector<int> coordinates3Dz;

    /** Vectors for centers of mass components after coordinates clusterings */
    std::vector<int> centerOfMassX;
    std::vector<int> centerOfMassY;
    std::vector<int> centerOfMassZ;

    /** Fourier Transform */
    

public:

    void readParams();
    void defineParams();

    /**
     * Bandpass filtering the input volume.
     *
     * @param
     * @return
     *
    */
    void bandPassFilter(MultidimArray<double> &origImg);


    /**
     * Smoothing and filtering the input volume.
     *
     * @param
     * @return
     *
    */
    MultidimArray<double> preprocessVolume(MultidimArray<double> &inputTomo);

    /**
     * Peaks the high contrast regions in a volume.
     *
     * @param
     * @return
     *
    */
    void getHighContrastCoordinates(MultidimArray<double> volFiltered);

    /**
     * Cluster 3d coordinates into its center of mass.
     *
     * @param
     * @return
     *
    */
    void clusterHighContrastCoordinates();

    /**
     * Write obtained coordinates in output file.
     *
     * @param
     * @return
     *
    */
    void centerCoordinates(MultidimArray<double> volFiltered);

    /**
     * Center the picked features into the box.
     *
     * @param
     * @return
     *
    */
    void writeOutputCoordinates();

    void run();
};

#endif