/***************************************************************************
 * Authors:     Erney Ramirez-Aportela  (eramirez@cnb.csic.es)
 *              Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *
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


#include "transform_window_from_volume.h"

ProgTransFromVol::ProgTransFromVol()
{
	produces_an_output=true;
	produces_a_metadata=true;
	each_image_produces_an_output=true;
}

void ProgTransFromVol::readParams()
{
	XmippMetadataProgram::readParams();
//    fnVol = getParam("--vol");
//    outVol = getParam("--outvol");
    fnMask = getParam("--mask");
    boxSize = getIntParam("--boxSize");
//    initvol = checkParam("--vol");
}

void ProgTransFromVol::defineParams()
{
	addUsageLine("Project a center of mass of the map into image stack");
	XmippMetadataProgram::defineParams();
//    addParamsLine("  [--vol <vol_file=\"\">]   : Input volume");
    addParamsLine("  --mask <vol_file=\"\">   : Input mask");
//    addParamsLine("  [--outvol <vol_file=\"\">]   : Output resize volume");
    addParamsLine("  --boxSize <box>         : Box size around the projection of the center");
    addExampleLine("Create images centered on the center of mass of the mask and with a size of 50x50: ", false);
    addExampleLine("xmipp_transform_window_from_volume -i particle.xmd --vol mask.vol --boxsize 50 -o particle_window.stk");
}

void ProgTransFromVol::show()
{
	XmippMetadataProgram::show();
	std::cout << "Input images: " << fnMdIn << std::endl;
}

void ProgTransFromVol::startProcessing()
{
    createEmptyFile(fn_out, boxSize, boxSize, zdimOut, mdInSize, true, WRITE_OVERWRITE);
	delete_output_stack = false;
	create_empty_stackfile = false;
	XmippMetadataProgram::startProcessing();
}

void ProgTransFromVol::preProcess()
{
    Image<double> Vorig, Vmask, Vout;
	Vmask.read(fnMask);
	Vmask().setXmippOrigin();
	Vmask().centerOfMass(center,0.1);
    center.resize(4);
    center(3)=1;

    if (verbose>0)
    	std::cout << "Center of mass: " << center << std::endl;

//    if (initvol)
//    {
//    	Vorig.read(fnVol);
//    	Vorig().setXmippOrigin();
//    	Vout().initZeros(Vorig());
//
//		int z0=round(center(0)-boxSize/2);
//		int y0=round(center(1)-boxSize/2);
//		int x0=round(center(2)-boxSize/2);
//		Vorig().window(Vout(),z0,y0,x0,z0+boxSize-1,y0+boxSize-1,x0+boxSize-1);
//		Vout.write(outVol);
//    }
}

void ProgTransFromVol::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{

	int dim;
	Iin.read(fnImg);
	Iin().setXmippOrigin();
	dim=Iin().getDim();
//	std::cout << "dimensions: " << dim << std::endl;

	if (dim==2)
	{
		geo2TransformationMatrix(rowIn, A);
		projectedCenter=A*center;

//	    if (verbose>0)
//	    	std::cout << "projectedCenter: " << projectedCenter << std::endl;

		int y0=round(YY(projectedCenter)-boxSize/2);
		int x0=round(XX(projectedCenter)-boxSize/2);
		Iin().window(Iout(),y0,x0,y0+boxSize-1,x0+boxSize-1);
		Iout.write(fnImgOut);

		rowOut=rowIn;
		rowOut.setValue(MDL_SHIFT_X,0.0);
		rowOut.setValue(MDL_SHIFT_Y,0.0);

		rowOut.setValue(MDL_IMAGE,fnImgOut);
	}

	if (dim==3)
	{
//    	Iout().initZeros(Iin());

		int z0=round(center(0)-boxSize/2);
		int y0=round(center(1)-boxSize/2);
		int x0=round(center(2)-boxSize/2);
		Iin().window(Iout(),z0,y0,x0,z0+boxSize-1,y0+boxSize-1,x0+boxSize-1);
		Iout.write(fnImgOut);
	}

}
