/***************************************************************************
 *
 * Authors:    Federico P. de Isidro Gomez	federico.pdeisidro@astx.com (2024)
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

#ifndef _LOCAL_PARTICLE_ALINGMENT
#define _LOCAL_PARTICLE_ALINGMENT

#include "core/xmipp_program.h"
#include <core/metadata_vec.h>
#include "core/xmipp_image_generic.h"


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


class ProgLocalParticleAlignment : public XmippProgram
{

public:
    /** Filenames */
    FileName fnIn;
	FileName fnOut;

	/** Input info */
	Matrix1D<double> alignmentCenter;

	/** Alingment and particle information */
	MultidimArray<double> particles;

	/** Dimensions */
	size_t xDim;
	size_t yDim;
	size_t nDim;

private:
    /** Input tomogram dimensions */
    size_t xSize;
	size_t ySize;
	size_t zSize;


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


    // ---------------------- MAIN FUNCTIONS -----------------------------

    /**
     *  
    */
    void recenterParticle();


    // --------------------------- UTILS functions ----------------------------

    /**
     *  
    */
    void calculateShiftDisplacement(Matrix2D<double> particleAlignment, Matrix2D<double> shifts);

	void getParticleSize();



    // --------------------------- MAIN ----------------------------------

    /**
     * Run main program.
    */
    void run();
};

#endif