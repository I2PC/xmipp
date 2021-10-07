/***************************************************************************
 *
 * Authors:    Estrella Fernandez Gimenez   (me.fernandez@cnb.csic.es)
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

#include "shift_volume.h"
#include "core/transformations.h"

// Read arguments ==========================================================
void ProgShiftVolume::readParams()
{

    fn_vol = getParam("-i");
    fn_out = getParam("-o");
    if (fn_out=="")
    	fn_out=fn_vol;
}

// Show ====================================================================
void ProgShiftVolume::show() const
{
	if (verbose>0)
	{
		std::cout
		<< "Input volume:\t" << fn_vol << std::endl
		<< "Output (shifted) volume:\t" << fn_vol << std::endl;
	}
}

// usage ===================================================================
void ProgShiftVolume::defineParams()
{
	addUsageLine("This program takes a volume and shifts it to its original position.");
}

#define GET_VOL_COORD \
int xp=x+j;\
if (xp<0 || xp>=XSIZE(mOutVol))\
	continue;\
int yp=y+i;\
if (yp<0 || yp>=YSIZE(mOutVol))\
	continue;\
int zp=z+k;\
if (zp<0 || zp>=ZSIZE(mOutVol))\
	continue;\

void ProgShiftVolume::run()
{

    show();
    vol.read(fn_vol);
    const MultidimArray<double> &mVol=vol();
    outVol().resizeNoCopy(vol());
    outVol().initConstant(0.0);
    MultidimArray<double> &mOutVol=outVol();
    MultidimArray<double> shiftVol;
    Matrix2D<double> A;
    A.initIdentity(4);

	geo2TransformationMatrix(row,A);
	applyGeometry(LINEAR, shiftVol, mVol, A, IS_NOT_INV, DONT_WRAP);

	double avg=0;
	double avgN=0;
	FOR_ALL_ELEMENTS_IN_ARRAY3D(mOutVol)
	{
		GET_VOL_COORD
		double val=A3D_ELEM(mOutVol,k,i,j);
	}
    vol.write(fn_out);
}


