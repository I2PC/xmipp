/***************************************************************************
 *
 * Authors:    Jose Luis Vilas 					  jlvilas@cnb.csic.es
 * 			   Oier Lauzirika Zarrabeitia         oierlauzi@bizkaia.eu
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

#include "tomo_twofold_align.h"


// --------------------------- INFO functions ----------------------------

void ProgTomoTwofoldAlign::readParams()
{
    
	fnInMetadata = getParam("-i");
    fnOutMetadata = getParam("-o");
}


void ProgTomoTwofoldAlign::defineParams()
{
	addUsageLine("This program aligns all combinations of subtomograms"); // TODO
	
	addParamsLine("  -i <input_metadata>       			: Input metadata containing a set of volumes.");
	addParamsLine("  -o <output_metadata>   			: Output containing alignment among all possible pairs of volumes.");
}


// --------------------------- HEAD functions ----------------------------

void ProgTomoTwofoldAlign::twofoldAlign(std::size_t i, std::size_t j, 
										double &rot, double &tilt, double &psi)
{
	// TODO
}

void ProgTomoTwofoldAlign::readVolumes()
{
	FileName imageFilename;

	inputVolumes.reserve(inputVolumesMd.size());
	projectors.reserve(inputVolumesMd.size());
	centralProjections.reserve(inputVolumesMd.size());
	for (std::size_t objId : inputVolumesMd)
	{
		inputVolumesMd.getValue(MDL_IMAGE, imageFilename, objId);
		inputVolumes.emplace_back(imageFilename);
		projectors.emplace_back(inputVolumes.back(), 2.0, 1.0, xmipp_transformation::LINEAR ) // TODO
		projectors.back().project(0.0, 0.0, 0.0);
		centralProjections.emplace_back(projectors.back().projection());
	}
}

void ProgTomoTwofoldAlign::run() override
{
	std::string image1, image2;
	for(std::size_t i = 1; i < projectors.size(); ++i)
	{
		inputVolumesMd.getValue(MDL_IMAGE, image1, i+1);

		for (std::size_t j = 0; j < i; ++j)
		{
			inputVolumesMd.getValue(MDL_IMAGE, image2, j+1);

			// Perform the volume alignment
			double rot, tilt, psi;
			twofoldAlign(i, j, rot, tilt, psi);

			// Write to metadata
			std::size_t id = alignmentMd.addObject();
			inputVolumesMd.setValue(MDL_IMAGE1, image1, id);
			inputVolumesMd.setValue(MDL_IMAGE2, image2, id);
            alignmentMd.setValue(MDL_ANGLE_ROT, rot, id);
            alignmentMd.setValue(MDL_ANGLE_TILT, tilt, id);
            alignmentMd.setValue(MDL_ANGLE_PSI, psi, id);
		}
	}
}
