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
	nThreads = getIntParam("--threads");
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
	addParamsLine("  --threads <threads=8>  			: Number of threads");
}


// --------------------------- HEAD functions ----------------------------

void ProgTomoVolumeAlignTwofold::alignPair(std::size_t i, std::size_t j)
{
	FileName image1;
	FileName image2;

	// Perform the volume alignment
	double rot, tilt, psi;
	const auto cost = twofoldAlign(i, j, rot, tilt, psi);

	// Write to metadata
	std::lock_guard<std::mutex> lock(outputMetadataMutex);
	std::size_t id = alignmentMd.addObject();
	inputVolumesMd.getValue(MDL_IMAGE, image1, i+1);
	inputVolumesMd.getValue(MDL_IMAGE, image2, j+1);
	alignmentMd.setValue(MDL_IMAGE1, image1, id);
	alignmentMd.setValue(MDL_IMAGE2, image2, id);
	alignmentMd.setValue(MDL_ANGLE_ROT, rot, id);
	alignmentMd.setValue(MDL_ANGLE_TILT, tilt, id);
	alignmentMd.setValue(MDL_ANGLE_PSI, psi, id);
	alignmentMd.setValue(MDL_COST, cost, id);
	  
	std::cout << "(" << j << ", " << i << ")" << std::endl;
}

double ProgTomoVolumeAlignTwofold::twofoldAlign(std::size_t i, std::size_t j, 
											    double &rot, double &tilt, double &psi)
{
	const auto nPsi = static_cast<std::size_t>(360.0 / angularSamplingRate);
	const auto &directions = sphereSampling.no_redundant_sampling_points_angles;

	auto &projector1 = projectors[i];
	auto &projector2 = projectors[j];
	const auto &centralSlices1 = centralSlices[i];
	const auto &centralSlices2 = centralSlices[j];
	auto &mutex1 = projectorMutex[i];
	auto &mutex2 = projectorMutex[j];

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

			double cost = 0.0;
			{
				std::lock_guard<std::mutex> lock(mutex1);
				projector1.projectToFourier(rot1, tilt1, psi1);

				cost += computeSquareDistance(projector1.projectionFourier, centralSlices2);
			}
			{
				std::lock_guard<std::mutex> lock(mutex2);
				projector2.projectToFourier(rot2, tilt2, psi2);

				cost += computeSquareDistance(projector2.projectionFourier, centralSlices1);
			}

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

double ProgTomoVolumeAlignTwofold::computeSquareDistance(const MultidimArray<std::complex<double>> &x, 
														 const MultidimArray<std::complex<double>> &y )
{
	double sum = 0.0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(x)
	{
		const auto delta = DIRECT_MULTIDIM_ELEM(x, n) - DIRECT_MULTIDIM_ELEM(y, n);
		sum += delta.real()*delta.real() + delta.imag()*delta.imag();
	}
	return sum;
}

void ProgTomoVolumeAlignTwofold::readVolumes()
{
	FileName imageFilename;
	inputVolumesMd.read(fnInMetadata);

	inputVolumes.reserve(inputVolumesMd.size());
	for (std::size_t objId : inputVolumesMd.ids())
	{
		inputVolumesMd.getValue(MDL_IMAGE, imageFilename, objId);
		inputVolumes.emplace_back();
		inputVolumes.back().read(imageFilename);
		inputVolumes.back()().setXmippOrigin();
	}
}

void ProgTomoVolumeAlignTwofold::createProjectors()
{
	projectors.reserve(inputVolumes.size());
	for(std::size_t i = 0; i < inputVolumes.size(); ++i)
	{
		projectors.emplace_back(
			inputVolumes[i](),
			padding, maxFrequency, interp
		);
	}
	projectorMutex = std::vector<std::mutex>(projectors.size());
}

void ProgTomoVolumeAlignTwofold::projectCentralSlices()
{
	centralSlices.reserve(inputVolumesMd.size());
	for(std::size_t i = 0; i < projectors.size(); ++i)
	{
		auto &projector = projectors[i];
		projector.projectToFourier(0.0, 0.0, 0.0);
		centralSlices.emplace_back(projector.projectionFourier);
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
	// Initialize
	readVolumes();
	createProjectors();
	projectCentralSlices();
	defineSampling();

	threadPool.resize(nThreads);
	std::vector<std::future<void>> futures;
	for(std::size_t i = 1; i < projectors.size(); ++i)
	{
		for (std::size_t j = 0; j < i; ++j)
		{
			futures.push_back(threadPool.push(
				[this, i, j] (std::size_t) 
				{ 
					this->alignPair(i, j); 
				} 
			));
		}
	}

	// Wait task completion
	std::for_each(
		futures.begin(), futures.end(), 
		std::mem_fn(&std::future<void>::wait)
	);

	// Write output
	std::lock_guard<std::mutex> lock(outputMetadataMutex);
	alignmentMd.write(fnOutMetadata);
}
