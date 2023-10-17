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

#include "core/metadata_vec.h"
#include "core/metadata_label.h"

// Read arguments ==========================================================
void ProgAligned3dClassification::readParams()
{
    XmippMetadataProgram::readParams();
	ctfDescription.readParams(this);
}

// Define parameters ==========================================================
void ProgAligned3dClassification::defineParams()
{
    addUsageLine("Perform a multireference 3D classification over a set of projection aligned images.");
    XmippMetadataProgram::defineParams();
	ctfDescription.defineParams(this);
	referenceMdFilename = getParam("-r")
}

void ProgAligned3dClassification::preProcess() override
{
	MetaDataVec referenceVolumesMd(referenceMdFilename);

	// Read volumes
	FileName referenceVolumeFn;
	referenceVolumes.clear();
	referenceVolumes.reserve(referenceVolumesMd.size());
	for (auto objId : referenceVolumesMd)
	{
		referenceVolumesMd.getValue(MDL_IMAGE, referenceVolumeFn, objId);
		referenceVolumes.emplace_back(referenceFn);
	}

	// Generate projectors from metadata
	projectors.clear();
	projectors.reserve(referenceVolumes.size());
	for (auto& volume : referenceVolumes)
	{
		projectors.emplace_back(volume(), 1.0, 1.0, xmipp_transformation::BSPLINE3);
	}
}

void ProgAligned3dClassification::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
	// Read the input image
	inputImage.read(fnImg);

	// Generate CTF image
	ctfDescription.readFromMdRow(rowIn);
	ctfDescription.gereateCTF(inputImage(), ctfImage);

	// Generate projections and evaluate
	double rot, tilt, psi, shiftX, shiftY;
	rowIn.getValue(MDL_ROT, rot);
	rowIn.getValue(MDL_TILT, tilt);
	rowIn.getValue(MDL_PSI, psi);
	rowIn.getValue(MDL_SHIFTX, shiftX);
	rowIn.getValue(MDL_SHIFTY, shiftY);
	std::size_t best;
	double bestDistance = std::numeric_limits<double>::max();
	for (size_t i = 0; i < projectors.size(); ++i)
	{	
		auto& projector = projectors[i];
		projector.project(rot, tilt, psi, shiftX, shiftY, &ctfImage);

		// Compute the squared euclidean distance
		auto& projection = projector.projection;
		projection -= inputImage();
		const auto dist2 = projection.sum2();

		// Update
		if (dist2 < bestDistance)
		{
			best = i;
			bestDistance = dist2;
		}
	}

	rowOut.setValue(MDL_REF3D, best);
}
