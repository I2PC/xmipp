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

void geo2TransformationMatrixShift(const MDRow &imageGeo, Matrix2D<double> &A)
{
    double shiftX = 0., shiftY = 0., shiftZ = 0.;

    imageGeo.getValue(MDL_SHIFT_X, shiftX);
    imageGeo.getValue(MDL_SHIFT_Y, shiftY);
    imageGeo.getValue(MDL_SHIFT_Z, shiftZ);
    int dim = A.Xdim() - 1;
    //This check the case when matrix A is not initialized with correct size
    if (dim < 2 || dim > 3)
    {
        dim = 3;
        A.resizeNoCopy(dim + 1, dim + 1);
    }
	A.initIdentity();
    dMij(A, 0, dim) = shiftX;
    dMij(A, 1, dim) = shiftY;
    dMij(A, 2, dim) = shiftZ;
}

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
		fnOut="output_particle_";
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
     addParamsLine("[-o <structure=\"\">]  			: Output filename suffix for shifted particles");
     addParamsLine("                       			: If no name is given, then output_particles.xmd");
     addParamsLine("[--center <x0=0> <y0=0> <z0=0>] : New center coordinates x,y,z");
     addParamsLine("[--boxSize <b=300>]             : Output box size");
     addExampleLine("A typical use is:",false);
     addExampleLine("xmipp_shift_particles -i input_particles.xmd --ref input_map.mrc -o output_particles --center 0 0 0 --boxSize 300");
 }

 void ProgShiftParticles::run()
 {
	show();
	// Read input center
	Matrix1D<double> pos;
	pos.initZeros(3);
	pos(0) = x0;
	pos(1) = y0;
	pos(2) = z0;
	// Read input particles.xmd
 	mdParticles.read(fnParticles);
 	int ix_particle = 0;

    FOR_ALL_OBJECTS_IN_METADATA(mdParticles)
    {
    	// Read particle image and metadata
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
		pos = R * pos;
		std::cout<< "pos: " << pos << std::endl;

		MDRow rowGeo;
		rowGeo.setValue(MDL_SHIFT_X, -pos(0));
		rowGeo.setValue(MDL_SHIFT_Y, -pos(1));
		rowGeo.setValue(MDL_SHIFT_Z, -pos(2));

	    A.initIdentity(3);
	    geo2TransformationMatrix(rowGeo, A, true);
//	    geo2TransformationMatrixShift(rowGeo, A);
		std::cout<< "A: " << A << std::endl;

    	I().setXmippOrigin();
    	Iout().resize(1, 1, boxSize, boxSize, false);
    	Iout().setXmippOrigin();
		applyGeometry(LINEAR, Iout(), I(), A, IS_NOT_INV, true, 0.);

		// Save shifted particles in metadata
		ix_particle++;
		FileName out = formatString("%d@%s.mrcs", ix_particle, fnOut.c_str());
		Iout.write(out);
		mdParticles.setValue(MDL_IMAGE, out, ix_particle);
		// CHECK!!!
		mdParticles.setValue(MDL_SHIFT_X, shiftx+pos(0), ix_particle);
		mdParticles.setValue(MDL_SHIFT_Y, shifty+pos(1), ix_particle);
		mdParticles.setValue(MDL_SHIFT_Z, shiftz+pos(2), ix_particle);
    }
    mdParticles.write(fnParticles);
 }
