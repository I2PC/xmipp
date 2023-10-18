/***************************************************************************
 * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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

#include "aligned_3d_classification.h"

#include "core/metadata_label.h"

// Read arguments ==========================================================
void ProgAligned3dClassification::readParams()
{
    XmippMetadataProgram::readParams();
	useCtf = checkParam("--useCtf");
	padding = getDoubleParam("--padding");
	maxFreq = getDoubleParam("--max_frequency");
	referenceVolumesMdFilename = getParam("-r");
	if (checkParam("--mask"))
		mask.readParams(this);
	if (useCtf)
		ctfDescription.readParams(this);
}

// Define parameters ==========================================================
void ProgAligned3dClassification::defineParams()
{
	produces_a_metadata = true;

    addUsageLine("Perform a multireference 3D classification over a set of projection aligned images.");
    XmippMetadataProgram::defineParams();
    addParamsLine("   [--useCtf]		             : Consider CTF when projecting references");
    addParamsLine("   [--padding <padding=1.0>]	     : Padding factor used in Fourier");
    addParamsLine("   [--max_frequency <freq=1.0>]	 : Maximum digital frequency");
    addParamsLine("    -r <reference_metadata>		 : Metadata with all the reference volumes");
	mask.defineParams(this);
	ctfDescription.defineParams(this);
}

void ProgAligned3dClassification::preProcess()
{
	referenceVolumesMd.read(referenceVolumesMdFilename);
	readVolumes();
	createProjectors();
	createMaskProjector();
}

void ProgAligned3dClassification::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
	// Read the input image
	inputImage.read(fnImg);

	// Generate CTF image
	const MultidimArray<double>* ctf = nullptr;
	if (useCtf)
	{
		ctfDescription.readFromMdRow(rowIn);
		ctfDescription.generateCTF(inputImage(), ctfImage);
		ctf = &ctfImage;
	}

	// Read alignment
	double rot, tilt, psi, shiftX, shiftY;
	rowIn.getValue(MDL_ANGLE_ROT, rot);
	rowIn.getValue(MDL_ANGLE_TILT, tilt);
	rowIn.getValue(MDL_ANGLE_PSI, psi);
	rowIn.getValue(MDL_SHIFT_X, shiftX);
	rowIn.getValue(MDL_SHIFT_Y, shiftY);

	// Project the mask
	if (maskProjector)
	{
		maskProjector->project(rot, tilt, psi, shiftX, shiftY);
		maskProjector->projection().binarize();
	}
	
	// Generate projections and evaluate
	std::size_t best;
	double bestDistance = std::numeric_limits<double>::max();
	for (size_t i = 0; i < projectors.size(); ++i)
	{	
		auto& projector = projectors[i];
		projector.project(rot, tilt, psi, shiftX, shiftY, ctf);

		// Compute the squared euclidean distance
		auto& projection = projector.projection();
		projection -= inputImage();
		if(maskProjector) 
			projection *= maskProjector->projection();
		const auto dist2 = projection.sum2();

		// Update the best score
		if (dist2 < bestDistance)
		{
			best = i;
			bestDistance = dist2;
		}
	}

	rowOut.setValue(MDL_REF3D, best+1);
}

void ProgAligned3dClassification::readVolumes()
{
	FileName fn;
	referenceVolumes.resize(referenceVolumesMd.size());
	auto ite = referenceVolumes.begin();
	for (const auto& row : referenceVolumesMd)
	{
		row.getValue(MDL_IMAGE, fn);
		(ite++)->read(fn);
	}
}

void ProgAligned3dClassification::createProjectors()
{
	projectors.clear();
	projectors.reserve(referenceVolumes.size());
	for (auto& volume : referenceVolumes)
	{
		projectors.emplace_back(
			volume(), 
			padding, maxFreq, 
			xmipp_transformation::BSPLINE3
		);
	}
}

void ProgAligned3dClassification::createMaskProjector()
{
	if (mask.type)
	{
		mask.generate_mask(referenceVolumes.front()());
		mask.force_to_be_continuous();
		maskProjector = std::make_unique<FourierProjector>(
			mask.get_cont_mask(), 
			padding, maxFreq, 
			xmipp_transformation::BSPLINE3
		);
	}
}