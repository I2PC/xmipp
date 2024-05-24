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

#include "tomo_volume_align_twofold.h"


// --------------------------- INFO functions ----------------------------

void ProgTomoVolumeAlignTwofold::readParams()
{
    
	fnInMetadata = getParam("-i");
    fnOutMetadata = getParam("-o");
    angularSamplingRate = getDoubleParam("--angularSampling");
    maxTiltAngle = getDoubleParam("--maxTilt");
}


void ProgTomoVolumeAlignTwofold::defineParams()
{
	addUsageLine("This program aligns all combinations of subtomograms"); // TODO
	
	addParamsLine("  -i <input_metadata>       			: Input metadata containing a set of volumes.");
	addParamsLine("  -o <output_metadata>   			: Output containing alignment among all possible pairs of volumes.");
	addParamsLine("  --angularSampling <degrees>   		: Angular sampling rate in degrees.");
	addParamsLine("  --maxTilt <degrees>   				: Maximum tilt angle for the angular search in degrees.");
}


// --------------------------- HEAD functions ----------------------------

double ProgTomoVolumeAlignTwofold::twofoldAlign(std::size_t i, std::size_t j, 
											  double &rot, double &tilt, double &psi)
{
	const auto nPsi = static_cast<std::size_t>(360.0 / angularSamplingRate);
	const auto &directions = sphereSampling.no_redundant_sampling_points_angles;

	double bestCost = std::numeric_limits<double>::max();
	for(std::size_t i = 0; i < directions.size(); ++i)
	{
		for(std::size_t j = 0; j < nPsi; ++j)
		{
            const double rot1 = XX(directions[i]);
            const double tilt1 = YY(directions[i]);
            const double psi1 = ZZ(directions[i]) + (nPsi * angularSamplingRate);
			const double rot2 = -psi1;
			const double tilt2 = -tilt1;
			const double psi2 = -rot1;
			auto &projector1 = projectors[i];
			auto &projector2 = projectors[j];
			projector1.project(rot1, tilt1, psi1);
			projector2.project(rot2, tilt2, psi2);
			auto &projectionImage1 = projector1.projection();
			auto &projectionImage2 = projector2.projection();

			projectionImage1 -= centralProjections[j];
			projectionImage2 -= centralProjections[i];

			const auto cost = computeSquareSum(projectionImage1) + computeSquareSum(projectionImage2);
			if (cost < bestCost)
			{
				rot = rot1;
				tilt = tilt1;
				psi = psi1;
				bestCost = cost;
			}
		}
	}

	return bestCost;
}

double ProgTomoVolumeAlignTwofold::computeSquareSum(const MultidimArray<double> &x)
{
	double sum = 0.0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(x)
	{
		const auto &elem = DIRECT_MULTIDIM_ELEM(x, n);
		sum += elem*elem;
	}
	return sum;
}

void ProgTomoVolumeAlignTwofold::readVolumes()
{
	FileName imageFilename;

	inputVolumesMd.read(fnInMetadata);

	inputVolumes.reserve(inputVolumesMd.size());
	projectors.reserve(inputVolumesMd.size());
	centralProjections.reserve(inputVolumesMd.size());
	for (std::size_t objId : inputVolumesMd.ids())
	{
		// Read volume from MD
		inputVolumesMd.getValue(MDL_IMAGE, imageFilename, objId);
		inputVolumes.emplace_back();
		inputVolumes.back().read(imageFilename);

		// Create a projector for the volume
		projectors.emplace_back(
			inputVolumes.back()(), 
			2.0, 1.0, // TODO
			xmipp_transformation::BSPLINE3
		);

		// Obtain the central projection TODO do not use projector
		projectors.back().project(0.0, 0.0, 0.0);
		centralProjections.emplace_back(projectors.back().projection());
	}
}

void ProgTomoVolumeAlignTwofold::defineSampling()
{

    mysampling.setSampling(angularSamplingRate);
    //if (!mysampling.SL.isSymmetryGroup(fn_sym, symmetry, sym_order))
    //    REPORT_ERROR(ERR_VALUE_INCORRECT,
    //                 (std::string)"Invalid symmetry" +  fn_sym);
    mysampling.computeSamplingPoints(false, maxTiltAngle);
    //mysampling.SL.readSymmetryFile(fn_sym);
    //mysampling.fillLRRepository();
    //mysampling.removeRedundantPoints(symmetry, sym_order);
}

void ProgTomoVolumeAlignTwofold::run()
{
	readVolumes();

	std::string image1, image2;
	for(std::size_t i = 1; i < projectors.size(); ++i)
	{
		inputVolumesMd.getValue(MDL_IMAGE, image1, i+1);

		for (std::size_t j = 0; j < i; ++j)
		{
			inputVolumesMd.getValue(MDL_IMAGE, image2, j+1);

			// Perform the volume alignment
			double rot, tilt, psi;
			const auto cost = twofoldAlign(i, j, rot, tilt, psi);

			// Write to metadata
			std::size_t id = alignmentMd.addObject();
			alignmentMd.setValue(MDL_IMAGE1, image1, id);
			alignmentMd.setValue(MDL_IMAGE2, image2, id);
            alignmentMd.setValue(MDL_ANGLE_ROT, rot, id);
            alignmentMd.setValue(MDL_ANGLE_TILT, tilt, id);
            alignmentMd.setValue(MDL_ANGLE_PSI, psi, id);
            alignmentMd.setValue(MDL_COST, cost, id);
		}
	}

	alignmentMd.write(fnOutMetadata);
}
