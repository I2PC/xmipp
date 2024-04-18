/***************************************************************************
 * Authors:     C.O.S. Sorzano (coss@cnb.csic.es)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your param) any later version.
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

#include <core/xmipp_program.h>
#include <core/metadata_label.h>
#include <core/metadata_vec.h>
#include <core/geometry.h>
#include "directions.h"

class ProgMetadataAngles: public XmippProgram
{
private:
    FileName fn_in, fn_out, fn_sym;
    MetaDataVec mdIn;
protected:
    void defineParams()
    {
        addUsageLine("Perform angular operations on a metadata file. ");
        addUsageLine("If the -o option is not used the original metadata will be modified.");

        addParamsLine(" -i <metadata>         : Input metadata file");
        addParamsLine("   [-o  <metadata>]    : Output metadata file, if not provided result will overwrite input file");

		addParamsLine("   [--bringToAsymmetricUnit <sym=\"c1\">]  : Bring all the angles (rot, tilt) to the asymmetric unit");
        
    }

    void readParams()
    {
        fn_in = getParam("-i");
        fn_out = checkParam("-o") ? getParam("-o") : fn_in;
        if (checkParam("--bringToAsymmetricUnit"))
        	fn_sym = getParam("--bringToAsymmetricUnit");
    }
    
    void doAsymmetric()
    {
        SymList SL;
		SL.readSymmetryFile(fn_sym);
        int imax=SL.symsNo();
		
		std::vector<double> rotList, tiltList;
		make_even_distribution(rotList, tiltList, 2.0, SL, false);

		Matrix2D<double> Laux, Raux;
        for (size_t objId : mdIn.ids())
        {
        	double rot, tilt, psi;
        	mdIn.getValue(MDL_ANGLE_ROT, rot, objId);
        	mdIn.getValue(MDL_ANGLE_TILT, tilt, objId);
        	mdIn.getValue(MDL_ANGLE_PSI, psi, objId);
        	
        	double bestrot=rot, besttilt=tilt, bestpsi=psi;
    		double bestDist=1e38;
			for (int i = 0; i < imax; i++)
			{
				SL.getMatrices(i, Laux, Raux, false);
	        	double newrot, newtilt, newpsi;
				Euler_apply_transf(Laux, Raux, rot, tilt, psi, newrot, newtilt, newpsi);

				int idx=find_nearest_direction(newrot, newtilt, rotList, tiltList, SL, Laux, Raux);
				double dist = distance_directions(newrot, newtilt, rotList[idx], tiltList[idx], false);
				if (dist<bestDist)
				{
					bestDist=dist;
					bestrot=newrot;
					besttilt=newtilt;
					bestpsi=newpsi;
				}
			}

            mdIn.setValue(MDL_ANGLE_ROT,bestrot,objId);
            mdIn.setValue(MDL_ANGLE_TILT,besttilt,objId);
            mdIn.setValue(MDL_ANGLE_PSI,bestpsi,objId);
        }
    }

public:
    void run()
    {
    	mdIn.read(fn_in);
        if (checkParam("--bringToAsymmetricUnit"))
            doAsymmetric();
		mdIn.write(fn_out);
    }
};
