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

 #include "subtract_projection.h"
 #include "core/transformations.h"
 #include "core/xmipp_image_extension.h"
 #include "core/xmipp_image_generic.h"
 #include "data/projection.h"
 #include "data/mask.h"

 // Empty constructor =======================================================
// ProgSubtractProjection::ProgSubtractProjection()
// {
//    produces_a_metadata = true;
//    each_image_produces_an_output = false;
// }

 // Read arguments ==========================================================
 void ProgSubtractProjection::readParams()
 {
	fnParticles = getParam("-i");
 	fnVolR = getParam("--ref");
	if (fnOut=="")
		fnOut="output_particles.xmd";
	fnMask=getParam("--mask");
	iter=getIntParam("--iter");
	sigma=getIntParam("--sigma");
	cutFreq=getDoubleParam("--cutFreq");
	lambda=getDoubleParam("--lambda");
//	fnVol1F=getParam("--saveV1");
//	if (fnVol1F=="")
//		fnVol1F="volume1_filtered.mrc";
//	fnVol2A=getParam("--saveV2");
//	if (fnVol2A=="")
//		fnVol2A="volume2_adjusted.mrc";
 }

 // Show ====================================================================
 void ProgSubtractProjection::show()
 {
    if (!verbose)
        return;
 	XmippMetadataProgram::show();
	std::cout
	<< "Input particles:   	" << fnParticles << std::endl
	<< "Reference volume:   " << fnVolR      << std::endl
	<< "Mask:    	   		" << fnMask      << std::endl
	<< "Sigma:				" << sigma       << std::endl
	<< "Iterations:    	   	" << iter        << std::endl
	<< "Cutoff frequency:  	" << cutFreq     << std::endl
	<< "Relaxation factor:  " << lambda      << std::endl
	<< "Output particles: 	" << fnOut 	     << std::endl
	;
 }

 // usage ===================================================================
 void ProgSubtractProjection::defineParams()
 {
     //Usage
     addUsageLine("");
     //Parameters
     addParamsLine("-i <particles>          : Particles metadata (.xmd file)");
     addParamsLine("--ref <volume>      	: Reference volume to subtract");
     addParamsLine("[-o <structure=\"\">] 	: Output metadata (.xmd) of subtracted particles");
     addParamsLine("                      	: If no name is given, then output_particles.xmd");
     addParamsLine("[--mask <mask=\"\">]  	: Mask for the region of subtraction");
     addParamsLine("[--sigma <s=3>]    		: Decay of the filter (sigma) to smooth the mask transition");
     addParamsLine("[--iter <n=1>]        	: Number of iterations");
     addParamsLine("[--cutFreq <f=0>]       	: Cutoff frequency (<0.5)");
     addParamsLine("[--lambda <l=0>]       	: Relaxation factor for Fourier Amplitude POCS (between 0 and 1)");
//        addParamsLine("[--saveV1 <structure=\"\"> ]  : Save subtraction intermediate files (vol1 filtered)");
//        addParamsLine("[--saveV2 <structure=\"\"> ]  : Save subtraction intermediate files (vol2 adjusted)");
    addExampleLine("A typical use is:",false);
    addExampleLine("");
 }

 void ProgSubtractProjection::run()
 {
	show();
	V.read(fnVolR);
 	MultidimArray<double> &mV=V();
 	mdParticles.read(fnParticles);
 	MDRow row;
 	double rot, tilt, psi;
 	FileName fnImage;

    FOR_ALL_OBJECTS_IN_METADATA(mdParticles)
    {
    	mdParticles.getRow(row,__iter.objId);
    	row.getValue(MDL_IMAGE, fnImage);
    	I.read(fnImage);
    	I().setXmippOrigin();
     	row.getValue(MDL_ANGLE_ROT, rot);
     	row.getValue(MDL_ANGLE_TILT, tilt);
     	row.getValue(MDL_ANGLE_PSI, psi);
    	projectVolume(mV, P, (int)XSIZE(I()), (int)XSIZE(I()), rot, tilt, psi);
    }

// 	FileName fn(fnOut);

 }



