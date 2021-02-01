/***************************************************************************
 *
 * Authors:    Federico P. de Isidro Gomez			  fp.deisidro@cnb.csic.es
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

#include "image_peak_high_contrast.h"

void ProgPeakHighContrast::readParams()
{
	fnVol = getParam("--vol");
	fnOut = getParam("-o");
	thr = getDoubleParam("--thr");
    samp = getIntParam("--samp")
}

void ProgPeakHighContrast::defineParams()
{
	addUsageLine("This function determines the location of the outliers points in a volume");
	addParamsLine("  --vol <vol_file=\"\">                   : Input volume");
	addParamsLine("  -o <output=\"coordinaates3D.txt\">        : Output file containing the 3D coodinates");
	addParamsLine("  [--thr <thr=0.9>]                		 : Threshold");
  	addParamsLine("  [--samp <samp=10>]                		 : Number of slices to use to determin ");

}

void ProgPeakHighContrast ::getHighContrastCoordinates()
{
	std::cout << "Starting..." << std::endl;

	Image<double> inputVolume;
	inputVolume.read(fnVol);

	//inputVolume().setXmippOrigin();

	MultidimArray<double> &inputTomo=inputVolume();
	std::vector<double> tomoVector;
	
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(inputTomo)
		tomoVector.push_back(DIRECT_MULTIDIM_ELEM(inputTomo, n));
	
	std::sort(tomoVector.begin(),tomoVector.end());

	double thresholdValue = tomoVector[size_t(tomoVector.size()*thr)];

	std::cout << "threshold value = " << thresholdValue << std::endl;

    MultidimArray<int> coordinates3D;

    FOR_ALL_ELEMENTS_IN_ARRAY3D(inputTomo)
    {
        double res = A3D_ELEM(inputTomo, k, i, j);

        if (res<=thresholdValue)
        {
            //std::cout << "i " << i << " j " << j << " k" << k << std::endl;
            coordinates3D += [i, j, k];
        }
    }
