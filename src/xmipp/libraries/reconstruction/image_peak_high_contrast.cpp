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

void ProgImagePeakHighContrast::readParams()
{
	fnVol = getParam("--vol");
	fnOut = getParam("-o");
	thr = getDoubleParam("--thr");
    samp = getIntParam("--samp");
}

void ProgImagePeakHighContrast::defineParams()
{
	addUsageLine("This function determines the location of the outliers points in a volume");
	addParamsLine("  --vol <vol_file=\"\">                   : Input volume");
	addParamsLine("  -o <output=\"coordinaates3D.txt\">        : Output file containing the 3D coodinates");
	addParamsLine("  [--thr <thr=0.1>]                		 : Threshold");
  	addParamsLine("  [--samp <samp=10>]                		 : Number of slices to use to determin the threshold value");

}

void ProgImagePeakHighContrast::getHighContrastCoordinates()
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

	double highThresholdValue = tomoVector[size_t(tomoVector.size()*(thr/2))];
    double lowThresholdValue = tomoVector[size_t(tomoVector.size()*(1-(thr/2)))];

	std::cout << "high threshold value = " << highThresholdValue << std::endl;
    std::cout << "low threshold value = " << lowThresholdValue << std::endl;

    std::vector<int> coordinates3Dx(0);
    std::vector<int> coordinates3Dy(0);
    std::vector<int> coordinates3Dz(0);

    FOR_ALL_ELEMENTS_IN_ARRAY3D(inputTomo)
    {
        double value = A3D_ELEM(inputTomo, k, i, j);

        if (value<=lowThresholdValue or value>=highThresholdValue)
        {
            //std::cout << "i " << i << " j " << j << " k" << k << std::endl;
            coordinates3Dx.push_back(i);
            coordinates3Dy.push_back(j);
            coordinates3Dz.push_back(k);
        }
    }
}
