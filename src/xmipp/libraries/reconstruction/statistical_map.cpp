/***************************************************************************
 *
 * Authors:     Federico P. de Isidro-Gomez (federico.pdeisidro@astx.com)
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

 #include "statistical_map.h"
 #include "core/transformations.h"
 #include "core/metadata_extension.h"
 #include "core/multidim_array.h"
 #include "core/xmipp_image_extension.h"
 #include "core/xmipp_image_generic.h"
 #include "core/xmipp_image_base.h"
 #include "core/xmipp_fft.h"
 #include "core/xmipp_fftw.h"
 #include "core/linear_system_helper.h"
 #include "data/projection.h"
 #include "data/mask.h"
 #include "data/filters.h"
 #include "data/morphology.h"
 #include <iostream>
 #include <string>
 #include <sstream>
 #include "data/image_operate.h"
 #include <iostream>
 #include <cstdlib>
 #include <vector>
 #include <utility>
 #include <chrono>



// I/O methods ===================================================================
void ProgStatisticalMap::readParams()
{
    fn_in = getParam("-i");
    fn_oroot = getParam("-oroot");
}

void ProgStatisticalMap::show() const
{
    if (!verbose)
        return;
	std::cout
	<< "Input volume pool metadata:\t" << fn_in << std::endl
	<< "Output location for statistical volumes:\t" << fn_oroot << std::endl;
}

void ProgStatisticalMap::defineParams()
{
	//Usage
    addUsageLine("This algorithm computes a statistical map that characterize the input map pool for posterior comparison \
                  to an input map in order to characterize the likelyness of its densities.");

    //Parameters
	XmippMetadataProgram::defineParams();
    addParamsLine("-i <i=\"\">          : Input metadata containing map pool for statistical map calculation.");
    addParamsLine("--oroot <oroot=\"\"> : Output location for saving statistical maps.");
}

 void ProgStatisticalMap::readVolume(const MDRow &r) 
 {
	r.getValue(MDL_IMAGE, fnImgI,);
	V.read(fnImgI);
	V().setXmippOrigin();
 }

 void ProgStatisticalMap::writeStatisticalMap(MDRow &rowOut, FileName fnImgOut, Image<double> &img, double avg, double std, double zScore) 
 {
    avgVolume.write(fn_out_avg_map);
 	stdVolume.write(fn_out_std_map); 
 }


// Main method ===================================================================
void ProgStatisticalMap::run()
{
	auto t1 = std::chrono::high_resolution_clock::now();

    generateSideInfo();

    mapPoolMD.read(fn_in);

    for (const auto& row : mapPoolMD)
	{
        row.getValue(MDL_IMAGE, fn_V);


    }
}


// Core methods ===================================================================
void ProgStatisticalMap::processVolume(FileName fn_vol)
 { 
	
}


// Utils methods ===================================================================
void ProgStatisticalMap::generateSideInfo()
{
    FileName fn_out_avg_map = fn_oroot + "statsMap_avg.mrc";
    FileName fn_out_std_map = fn_oroot + "statsMap_std.mrc";
}
