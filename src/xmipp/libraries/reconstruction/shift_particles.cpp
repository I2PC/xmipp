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
	x0=getDoubleParam("--center",0);
	y0=getDoubleParam("--center",1);
	z0=getDoubleParam("--center",2);
	boxSize=getIntParam("--boxSize");
	fnOut=getParam("-o");
	if (fnOut=="")
		fnOut="output_particles.mrcs";
 }

 // Show ====================================================================
 void ProgShiftParticles::show()
 {
    if (!verbose)
        return;
	std::cout
	<< "Input particles:   	" << fnParticles << std::endl
	<< "New center:  		" << x0 << ", "  << y0 << ", "  << z0 << std::endl
	<< "Output box size: 	" << boxSize     << std::endl
	<< "Output particles: 	" << fnOut 	     << std::endl
	;
 }

 // usage ===================================================================
 void ProgShiftParticles::defineParams()
 {
     //Usage
     addUsageLine("Center particles into a selected point of a volume.");
     //Parameters
     addParamsLine("-i <particles>         			: Particles metadata (.xmd file)");
     addParamsLine("[-o <structure=\"\">]  			: Output filename for shifted particles");
     addParamsLine("                       			: If no name is given, then output_particles.mrcs is used");
     addParamsLine("[--center <x0=0> <y0=0> <z0=0>] : New center coordinates x,y,z");
     addParamsLine("[--boxSize <b=300>]             : Output box size");
     addExampleLine("A typical use is:",false);
     addExampleLine("xmipp_shift_particles -i input_particles.xmd -o output_particles.xmd --center 0 0 0 --boxSize 300");
 }

 void ProgShiftParticles::run()
 {
	show();

	Matrix1D<double> pos, posp;
    FileName fnImage;
 	MDRow row;
 	Matrix2D<double> R, A;
 	Image<double> I, Iout;
    double rot, tilt, psi, shiftx, shifty, shiftz;

	pos.initZeros(3);
	pos(0) = x0;
	pos(1) = y0;
	pos(2) = z0;
	mdParticles.read(fnParticles);
	int ix_particle = 0;

    FOR_ALL_OBJECTS_IN_METADATA(mdParticles)
	{
		mdParticles.getRow(row,__iter.objId);
		row.getValue(MDL_IMAGE, fnImage);
		std::cout<< "Particle: " << fnImage << std::endl;
		I.read(fnImage);
		row.getValue(MDL_ANGLE_ROT, rot);
		row.getValue(MDL_ANGLE_TILT, tilt);
		row.getValue(MDL_ANGLE_PSI, psi);
		row.getValue(MDL_SHIFT_X, shiftx);
		row.getValue(MDL_SHIFT_Y, shifty);
		row.getValue(MDL_SHIFT_Z, shiftz);

		R.initIdentity(3);
		Euler_angles2matrix(rot, tilt, psi, R, false);
		R = R.inv();
		posp = R * pos;

		MDRow rowGeo;
		rowGeo.setValue(MDL_SHIFT_X, -posp(0));
		rowGeo.setValue(MDL_SHIFT_Y, -posp(1));
		A.initIdentity(3);
		geo2TransformationMatrix(rowGeo, A, true);

		I().setXmippOrigin();
		Iout().resize(1, 1, boxSize, boxSize, false);
		Iout().setXmippOrigin();
		applyGeometry(LINEAR, Iout(), I(), A, IS_NOT_INV, false, 0.);

		ix_particle++;
		FileName out = formatString("%d@%s", ix_particle, fnOut.c_str());
		Iout.write(out);
		mdParticles.setValue(MDL_IMAGE, out, ix_particle);
		mdParticles.setValue(MDL_SHIFT_X, shiftx+posp(0), ix_particle);
		mdParticles.setValue(MDL_SHIFT_Y, shifty+posp(1), ix_particle);
    }
    mdParticles.write(fnParticles);
 }
