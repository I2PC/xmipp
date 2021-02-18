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


class ProgImagePeakHighContrast : public XmippProgram
{
public:
    /** Filenames */
    FileName fnVol, fnOut;

    /** Threshold */
    double thr;

    /** Number of slices and original centers of mass */
    int samp, numberCenterOfMass, distanceThr;

    /** Output coordinates */
    // extern std::vector<int> coordinates3Dx(0);
    // extern std::vector<int> coordinates3Dy(0);
    // extern std::vector<int> coordinates3Dz(0);

public:

    void readParams();
    void defineParams();

    /**
     * Peaks the high contrast regions in a volume.
     *
     * @param
     * @return
     *
    */
    void getHighContrastCoordinates();

    /**
     * Write obtained coordinates in output file.
     *
     * @param
     * @return
     *
    */
    // void writeOutputCoordinates();

    void run();
};

#endif