// /***************************************************************************
//  *
//  * Authors:    Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
//  *
//  * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
//  *
//  * This program is free software; you can redistribute it and/or modify
//  * it under the terms of the GNU General Public License as published by
//  * the Free Software Foundation; either version 2 of the License, or
//  * (at your option) any later version.
//  *
//  * This program is distributed in the hope that it will be useful,
//  * but WITHOUT ANY WARRANTY; without even the implied warranty of
//  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  * GNU General Public License for more details.
//  *
//  * You should have received a copy of the GNU General Public License
//  * along with this program; if not, write to the Free Software
//  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
//  * 02111-1307  USA
//  *
//  *  All comments concerning this program package may be sent to the
//  *  e-mail address 'xmipp@cnb.csic.es'
//  ***************************************************************************/

 #include "shift_particles.h"
 #include "core/transformations.h"
 #include "core/xmipp_image_extension.h"
 #include "core/xmipp_image_generic.h"
 #include "core/xmipp_fft.h"
 #include "core/xmipp_fftw.h"
 #include "data/projection.h"
 #include "data/mask.h"
 #include <iostream>
 #include <string>
 #include <sstream>
 #include "data/image_operate.h"


 // Read arguments ==========================================================
 void ProgShiftParticles::readParams()
 {
	fnParticles = getParam("-i");
 	fnVolR = getParam("--ref");
	x0=getDoubleParam("--x0");
	y0=getDoubleParam("--y0");
	z0=getDoubleParam("--z0");
	fnOut=getParam("-o");
	if (fnOut=="")
		fnOut="output_particle_";
 }

 // Show ====================================================================
 void ProgShiftParticles::show()
 {
    if (!verbose)
        return;
	std::cout
	<< "Input particles:   	" << fnParticles << std::endl
	<< "Reference volume:   " << fnVolR      << std::endl
	<< "New center:  		" << x0 << ", " << y0 << ", "  << z0 << std::endl
	<< "Output particles: 	" << fnOut 	     << std::endl
	;
 }

 // usage ===================================================================
 void ProgShiftParticles::defineParams()
 {
     //Usage
     addUsageLine("");
     //Parameters
     addParamsLine("-i <particles>         : Particles metadata (.xmd file)");
     addParamsLine("--ref <volume>         : Reference volume to subtract");
     addParamsLine("[-o <structure=\"\">]  : Output filename suffix for shifted particles");
     addParamsLine("                       : If no name is given, then output_particles.xmd");
     addParamsLine("[--x0 <x=0>]           : New center x coordinate");
     addParamsLine("[--y0 <y=0>]           : New center y coordinate");
     addParamsLine("[--z0 <z=0>]           : New center z coordinate");
     addExampleLine("A typical use is:",false);
     addExampleLine("xmipp_shift_particles -i input_particles.xmd --ref input_map.mrc -o output_particles --x0  --y0 0 --z0 0");
 }

 void ProgShiftParticles::run()
 {
	show();
	// Read input volume and particles.xmd
	V.read(fnVolR);
	V().setXmippOrigin();
 	MultidimArray<double> &mV=V();
 	mdParticles.read(fnParticles);

 	int ix_particle = 0;

    FOR_ALL_OBJECTS_IN_METADATA(mdParticles)
    {
    	// Read particle image
    	mdParticles.getRow(row,__iter.objId);
    	row.getValue(MDL_IMAGE, fnImage);
		std::cout<< "Particle: " << fnImage << std::endl;
    	I.read(fnImage);
    	I().setXmippOrigin();
		I.write("I.mrc");
		// Read particle metadata
     	row.getValue(MDL_ANGLE_ROT, rot);
     	row.getValue(MDL_ANGLE_TILT, tilt);
     	row.getValue(MDL_ANGLE_PSI, psi);

    	// Compute projection of the volume
    	projectVolume(mV, P, (int)XSIZE(I()), (int)XSIZE(I()), rot, tilt, psi);
		P.write("P.mrc");

		// Save subtracted particles in metadata
		ix_particle++;
		FileName out = formatString("%d@%s.mrcs", ix_particle, fnOut.c_str());
		I.write(out);
		mdParticles.setValue(MDL_IMAGE, out, ix_particle);
    }
    mdParticles.write(fnParticles);
 }
