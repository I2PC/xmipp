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
    maxFrequency = getDoubleParam("--maxFreq");
    padding = getDoubleParam("--padding");
	interp = getIntParam("--interp");
}


void ProgTomoVolumeAlignTwofold::defineParams()
{
	addUsageLine("This program aligns all combinations of subtomograms"); // TODO
	
	addParamsLine("  -i <input_metadata>       			: Input metadata containing a set of volumes.");
	addParamsLine("  -o <output_metadata>   			: Output containing alignment among all possible pairs of volumes.");
	addParamsLine("  --angularSampling <degrees>   		: Angular sampling rate in degrees.");
	addParamsLine("  --maxTilt <degrees>   				: Maximum tilt angle for the angular search in degrees.");
	addParamsLine("  --maxFreq <freq=0.5>  				: Maximum digital frequency");
	addParamsLine("  --padding <factor=2.0>  			: Padding factor");
	addParamsLine("  --interp <interp=1>  				: Interpolation method factor");
}


// --------------------------- HEAD functions ----------------------------

double ProgTomoVolumeAlignTwofold::twofoldAlign(std::size_t i, std::size_t j, 
											    double &rot, double &tilt, double &psi)
{
	const auto nPsi = static_cast<std::size_t>(360.0 / angularSamplingRate);
	const auto &directions = sphereSampling.no_redundant_sampling_points_angles;

	auto &projector1 = projectors[i];
	auto &projector2 = projectors[j];
	const auto &centralProjection1 = centralProjections[i];
	const auto &centralProjection2 = centralProjections[j];

	double bestCost = std::numeric_limits<double>::max();
	for(std::size_t k = 0; k < directions.size(); ++k)
	{
		for(std::size_t l = 0; l < nPsi; ++l)
		{
            const double rot1 = XX(directions[k]);
            const double tilt1 = YY(directions[k]);
            const double psi1 = l * angularSamplingRate;
			const double rot2 = -psi1;
			const double tilt2 = -tilt1;
			const double psi2 = -rot1;
			projector1.project(rot1, tilt1, psi1);
			projector2.project(rot2, tilt2, psi2);

			const auto cost = computeSquareDistance(projector1.projection(), centralProjection2) + 
							  computeSquareDistance(projector2.projection(), centralProjection1) ;

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

double ProgTomoVolumeAlignTwofold::computeSquareDistance(const MultidimArray<double> &x, 
														 const MultidimArray<double> &y )
{
	double sum = 0.0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(x)
	{
		const auto delta = DIRECT_MULTIDIM_ELEM(x, n) - DIRECT_MULTIDIM_ELEM(y, n);
		sum += delta*delta;
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
		inputVolumes.back()().setXmippOrigin();

		// Create a projector for the volume
		projectors.emplace_back(
			inputVolumes.back()(),
			padding, maxFrequency, interp
		);

		// Obtain the central projection TODO do not use projector
		projectors.back().project(0.0, 0.0, 0.0);
		centralProjections.emplace_back(projectors.back().projection());
	}
}

void ProgTomoVolumeAlignTwofold::defineSampling()
{
	FileName fnSymmetry = "c1"; //TODO
	int symmetry, sym_order;
    sphereSampling.setSampling(angularSamplingRate);
    if (!sphereSampling.SL.isSymmetryGroup(fnSymmetry, symmetry, sym_order))
        REPORT_ERROR(ERR_VALUE_INCORRECT,
                     (std::string)"Invalid symmetry" +  fnSymmetry);
    sphereSampling.computeSamplingPoints(false, maxTiltAngle);
    sphereSampling.SL.readSymmetryFile(fnSymmetry);
    sphereSampling.fillLRRepository();
    sphereSampling.removeRedundantPoints(symmetry, sym_order);
}

void ProgTomoVolumeAlignTwofold::run()
{
	readVolumes();
	defineSampling();

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

			std::cout << "(" << j << ", " << i << ")\n";
		}
	}

	alignmentMd.write(fnOutMetadata);
}
