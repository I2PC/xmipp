/***************************************************************************
 *
 * Authors:    Estrella Fernandez Gimenez         me.fernandez@cnb.csic.es (2019)
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

#include "tomo_map_back.h"
#include "core/transformations.h"

// Read arguments ==========================================================
void ProgTomoMapBack::readParams()
{

    fn_tomo = getParam("-i");
    fn_geom = getParam("--geom");
    fn_ref = getParam("--ref");
    fn_out = getParam("-o");
    if (fn_out=="")
    	fn_out=fn_tomo;
    modeStr = getParam("--method");
    if (modeStr=="highlight")
    	K=getDoubleParam("--method",1);
    if (modeStr=="avg" || modeStr=="copy_binary")
    	threshold=getDoubleParam("--method",1);
}

// Show ====================================================================
void ProgTomoMapBack::show() const
{
	if (verbose>0)
	{
		std::cout
		<< "Input tomogram    : " << fn_tomo    << std::endl
		<< "Input geometry    : " << fn_geom    << std::endl
		<< "Input reference   : " << fn_ref     << std::endl
		<< "Painting mode     : " << modeStr    << std::endl
		;
	}
}

// usage ===================================================================
void ProgTomoMapBack::defineParams()
{
	addUsageLine("This program takes a tomogram, a reference subtomogram and a metadata with geometrical parameters");
	addUsageLine("(x,y,z) and places the reference subtomogram on the tomogram at the designated locations (map back).");
	addUsageLine("The program has several 'painting' options:");
	addUsageLine("   1. Copying the reference onto the tomogram");
	addUsageLine("   2. Setting the region occupied by the reference in the tomogram to the average value of that region");
	addUsageLine("   3. Add the reference multiplied by a constant to the location specified");
	addUsageLine("   4. Copy a binarized version of the reference onto the tomogram");
    addParamsLine("   -i <tomogram>           : Original tomogram");
    addParamsLine("  [-o <tomogram=\"\">]     : Output tomogram mapped back");
    addParamsLine("   --geom <geometry>       : Subtomograms coordinates and rotation angles (it must be a metadata)");
    addParamsLine("   --ref <reference>       : Subtomogram reference");
    addParamsLine("  [--method <mode=copy>]   : Painting mode");
    addParamsLine("     where <mode>");
    addParamsLine("        copy");
    addParamsLine("        avg <threshold=0.5>");
    addParamsLine("        highlight <K=1>");
    addParamsLine("        copy_binary <threshold=0.5>");
}

// Produce side information ================================================
void ProgTomoMapBack::produce_side_info()
{
    tomo.read(fn_tomo);
    mdGeom.read(fn_geom);
    reference.read(fn_ref);
    reference().setXmippOrigin();
    const MultidimArray<double> &mReference=reference();

    if (modeStr=="copy")
    	mode=1;
    else if (modeStr=="avg")
    	mode=2;
    else if (modeStr=="highlight")
    	mode=3;
    else if (modeStr=="copy_binary")
    	mode=4;

    if (mode==4 || mode==2)
    {
		FOR_ALL_ELEMENTS_IN_ARRAY3D(mReference)
    	    A3D_ELEM(mReference,k,i,j)=(A3D_ELEM(mReference,k,i,j)>threshold) ? 1:0;
    }
}

#define GET_TOMO_COORD \
int xp=x+j;\
if (xp<0 || xp>=XSIZE(mTomo))\
	continue;\
int yp=y+i;\
if (yp<0 || yp>=YSIZE(mTomo))\
	continue;\
int zp=z+k;\
if (zp<0 || zp>=ZSIZE(mTomo))\
	continue;\

void ProgTomoMapBack::run()
{
    show();
    produce_side_info();
    int x,y,z;
    const MultidimArray<double> &mReference=reference();
    MultidimArray<double> &mTomo=tomo();
    MultidimArray<double> referenceRotated;
    Matrix2D<double> A;
    A.initIdentity(4);
    MDRow row;

    FOR_ALL_OBJECTS_IN_METADATA(mdGeom)
    {
    	mdGeom.getRow(row,__iter.objId);
    	row.getValue(MDL_XCOOR,x);
    	row.getValue(MDL_YCOOR,y);
    	row.getValue(MDL_ZCOOR,z);
    	geo2TransformationMatrix(row,A);
    	applyGeometry(LINEAR, referenceRotated, mReference, A, IS_NOT_INV, DONT_WRAP);

    	double avg=0, avgN=0;
    	if (mode==2)
    	{
    		FOR_ALL_ELEMENTS_IN_ARRAY3D(referenceRotated)
			{
    			GET_TOMO_COORD
				avg+=DIRECT_A3D_ELEM(mTomo,zp,yp,xp);
    			avgN+=1;
			}
    		if (avgN>0)
    			avg/=avgN;
    	}

		FOR_ALL_ELEMENTS_IN_ARRAY3D(referenceRotated)
		{
    		GET_TOMO_COORD
			double val=A3D_ELEM(referenceRotated,k,i,j);
			switch (mode)
			{
			case 1:
			case 4:
				DIRECT_A3D_ELEM(mTomo,zp,yp,xp)=val;
				break;
			case 2:
				if (val>0)
					DIRECT_A3D_ELEM(mTomo,zp,yp,xp)=avg;
				break;
			case 3:
				DIRECT_A3D_ELEM(mTomo,zp,yp,xp)+=K*val;
				break;
			}
		}
    }

    tomo.write(fn_out);
}
