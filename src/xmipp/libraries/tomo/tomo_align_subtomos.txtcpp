/***************************************************************************
 *
 * Authors:     Jose Luis Vilas (jlvilas@cnb.csic.es)
 *              Oier Lauzirika  (olauzirika@cnb.csic.es)
 *
 * Spanish Research Council for Biotechnology, Madrid, Spain
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

#include "tomo_align_subtomos.h"
#include <sys/stat.h>
#include <core/metadata_extension.h>
#include "core/transformations.h"
#include <random>
#include <limits>
//#include "fftwT.h"
#include <CTPL/ctpl_stl.h>
#include <type_traits>
#include <chrono>

void ProgAlignSubtomos::defineParams()
{
	addUsageLine("This method carries out a subtomogram averaing. It means, given a set of subtomogram the program will estimate the average map.");
	addUsageLine("Two kinds of averaging can be performed:");
	addUsageLine("Reference: J.L. Vilas, et al. (202X)");
	addUsageLine("+* How the weighted average works:", true);
	addUsageLine("+ Each subtomogram is compared agains the reference with two metrics: A global correlation in real space, and a point-wise ");
	addUsageLine("+ phase correlation in Fourier Space. These two metrics are combined as in a new metric called aggregation function. ");
	addUsageLine("+ There are many kinds of aggregation functions, this algorithms considers the product of both correlations.");
	addUsageLine("+ Once the aggratation function of each subtomogram is estimated, they are normalized and used as weights");
	addSeeAlsoLine("resolution_fsc");

	addParamsLine("   --subtomos <input_file>            : Metadata with the list of subtomograms");
	addParamsLine("   [--ref <input_file=\"\">]    : Reference, this map use to be the STA result that will be refined by mean s of the smart STA of this method. If a map is not provided, then the algorithm will average all provided subtomos.");
	addParamsLine("   [--sta]         			 : (Optional) Use this flag to estimate the standard average instead of the weighted one.");
	addParamsLine("  --angularSampling <degrees>   		: Angular sampling rate in degrees.");
	addParamsLine("   [-o <output_folder=\"\">]          : Folder where the results will be stored.");
	addParamsLine("   [--sampling <Ts=1>]                : (Optional) Pixel size (Angstrom). If it is not provided by default will be 1 A/px.");
	addParamsLine("   [--threads <Nthreads=1>]           : (Optional) Number of threads to be used");

	addExampleLine("Resolution of two half maps half1.mrc and half2.mrc with a sampling rate of 2 A/px", false);
	addExampleLine("xmipp_tomo_subtomogram_averaging --half1 half1.mrc --half2 half2.mrc --sampling_rate 2 ");
	addExampleLine("Resolution of two half maps half1.mrc and half2.mrc with a sampling rate of 2 A/px and a mask mask.mrc", false);
	addExampleLine("xmipp_tomo_subtomogram_averaging --half1 half1.mrc --half2 half2.mrc --mask mask.mrc --sampling_rate 2 ");
}

void ProgAlignSubtomos::readParams()
{
	fnIn = getParam("--i");
	fnRef = getParam("--ref");
	doSTA = checkParam("--sta");
	angularSampling = getDoubleParam("--angularSampling");
	fnOut = getParam("-o");
	sampling = getDoubleParam("--sampling");
	Nthreads = getIntParam("--threads");
}

/*
template<typename T>
void ProgAlignSubtomos::readReference(MultidimArray<T> &ref)
{
	Image<T> refImg;
	refImg.read(fnRef);
	ref = refImg();

}


template<typename T>
void ProgAlignSubtomos::initialReference(MultidimArray<T> &ref, std::vector<FileName> &fnVec)
{
	auto numberOfsubtomos = fnVec.size();

	auto initSubtomos = std::max(std::round(0.1*numberOfsubtomos), 32);

	Image<T> subtomoImg;
	auto &subtomo = subtomoImg();

	subtomoImg.read(fnVec[0]);
	ref = subtomoImg();




	for (size_t idx = 1; idx<numberOfsubtomos; ++idx)
	{
		subtomoImg.read(fnVec[idx]);

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(ref)
		{
			DIRECT_MULTIDIM_ELEM(ref, n) += DIRECT_MULTIDIM_ELEM(subtomo, n);
		}
	}
}


void ProgAlignSubtomos::readSubtomos(std::vector<FileName> &fnVec, double &randomSubset)
{
	MetaDataVec md;
	FileName imageFilename;
	double rot, tilt, psi;

	auto numberOfsubtomos = md.size();

	for (std::size_t objId : md.ids())
	{
		// Read volume from MD
		md.getValue(MDL_IMAGE, imageFilename, objId);
		if (md.containsLabel(MDL_ANGLE_ROT) && md.containsLabel(MDL_ANGLE_TILT) && md.containsLabel(MDL_ANGLE_PSI))
		{
			md.getValue(MDL_ANGLE_ROT, rot, objId);
			md.getValue(MDL_ANGLE_TILT, tilt, objId);
			md.getValue(MDL_ANGLE_PSI, psi, objId);
		}
		fnVec.emplace_back();
		fnVec.back();
	}
}


template<typename T>
void ProgAlignSubtomos::alignSubtomos(MultidimArray<T> &ref, std::vector<FileName> &fnVec)
{
	size_t niters = 10;
	MetaDataVec mdAli;
	FileName fn;
	Image<double> imgSubtomo;
	auto &subtomo = imgSubtomo();

	for (size_t iter = 0; iter< niters; ++iter)
	{
		for (std::size_t idx = 0; idx < fnVec.size(); ++idx)
		{
			fn = fnVec[idx];

			imgSubtomo.read(fn);

			// Perform the volume alignment
			double rot, tilt, psi;
			const auto cost = twofoldAlign(ref, subtomo, rot, tilt, psi);

			// Write to metadata
			std::size_t id = mdAli.addObject();
			mdAli.setValue(MDL_IMAGE, fn, id);
			mdAli.setValue(MDL_ANGLE_ROT, rot, id);
			mdAli.setValue(MDL_ANGLE_TILT, tilt, id);
			mdAli.setValue(MDL_ANGLE_PSI, psi, id);
			mdAli.setValue(MDL_COST, cost, id);

			std::cout << idx << "\n";
		}

		FileName fnAli;
		fnAli = formatString("aliSubtomos_%s.xmd", iter);
		mdAli.write(fnAli);
	}
}
*/


void ProgAlignSubtomos::run()
{
	std::cout << "Starting ... " << std::endl;

	/*
	std::vector<FileName> fnVec;
	readSubtomos(fnVec);

	MultidimArray<double> ref;
	if (fnRef!="")
		readReference(ref);
	else
		initialReference(ref, fnVec);

	alignSubtomos(ref, fnVec);
	*/

}

