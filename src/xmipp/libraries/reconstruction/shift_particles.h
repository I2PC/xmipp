// /***************************************************************************
// *
// * Authors:    Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
// *
// * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
// *
// * This program is free software; you can redistribute it and/or modify
// * it under the terms of the GNU General Public License as published by
// * the Free Software Foundation; either version 2 of the License, or
// * (at your option) any later version.
// *
// * This program is distributed in the hope that it will be useful,
// * but WITHOUT ANY WARRANTY; without even the implied warranty of
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// * GNU General Public License for more details.
// *
// * You should have received a copy of the GNU General Public License
// * along with this program; if not, write to the Free Software
// * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
// * 02111-1307  USA
// *
// *  All comments concerning this program package may be sent to the
// *  e-mail address 'xmipp@cnb.csic.es'
// ***************************************************************************/

#ifndef LIBRARIES_RECONSTRUCTION_SHIFT_PARTICLES_H_
#define LIBRARIES_RECONSTRUCTION_SHIFT_PARTICLES_H_

 #include "core/metadata.h"
 #include "core/xmipp_program.h"
 #include "core/xmipp_image.h"
 #include "data/fourier_filter.h"
 #include "data/fourier_projection.h"


 class ProgShiftParticles: public XmippProgram
 {
 public:
    /** Filename of the reference volume */
    FileName fnVolR;
    /// Filename of input xmd particles and particle images
    FileName fnParticles, fnImage;
    /// Filename of output particles
    FileName fnOut;

 public:
    // Input particles metadata
    MetaData mdParticles;
 	MDRow row;
    // Input center coordinates, rot, tilt, psi and shifts from metadata, output box size
    double x0, y0, z0, rot, tilt, psi, shiftx, shifty;
    int boxSize;
    // Particle rotation matrix, new shift matrix, aux matrix
 	Matrix2D<double> R, A, RA;
    // Input particle, output particle
 	Image<double> I, Iout;

 public:
    /// Empty constructor
// 	ProgSubtractProjection();

    /// Read argument from command line
    void readParams();

    /// Show
    void show();

    /// Define parameters
    void defineParams();

    /// Project volume and subtraction
    void run();

 };
 //@}
 #endif
