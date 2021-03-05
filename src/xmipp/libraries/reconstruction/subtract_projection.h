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

 #ifndef _PROG_SUBTRACT_PROJECTION
 #define _PROG_SUBTRACT_PROJECTION

 #include "core/metadata.h"
 #include "core/xmipp_program.h"
 #include "core/xmipp_image.h"
 #include "data/fourier_filter.h"
 #include "data/fourier_projection.h"


 class ProgSubtractProjection: public XmippProgram
 {
 public:
    /** Filename of the reference volume */
    FileName fnVolR;
    /// Filename of input particles
    FileName fnParticles;
    /// Filename of output particles
    FileName fnOut;
    /// Filename of input mask
    FileName fnMask;

 public:
    // Input particles metadata
    MetaData mdParticles;
    // Input subtraction parameters
    double cutFreq, lambda;
 	int sigma, iter;
    // Input image
 	Image<double> V, I;
 	// Theoretical projection
 	Projection P;
 	// Filter
    FourierFilter filter;
    // CTF Check
    bool hasCTF;
    // CTF params
 	double defocusU, defocusV, ctfAngle;
 	// CTF
 	CTFDescription ctf;
    // CTF filter
    FourierFilter FilterCTF;
 	// CTF image

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
