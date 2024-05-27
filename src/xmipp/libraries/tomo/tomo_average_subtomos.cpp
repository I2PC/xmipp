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

#include "tomo_average_subtomos.h"
#include <sys/stat.h>
#include <core/metadata_extension.h>
#include "core/transformations.h"
#include <limits>
#include <type_traits>
#include <chrono>

void ProgAverageSubtomos::defineParams()
{
	addUsageLine("This method carries out a subtomogram averaging. It means, given a set of subtomogram the program will estimate the average map.");
	addParamsLine("   -i <input_file>                    : Metadata with the list of subtomograms");
	addParamsLine("   -o <output_volume>          	 	 : Output volume.");
	addParamsLine("   [--notApplyAlignment]     		 : (Optional) Use this flag to estimate the standard average instead of the weighted one.");
	addParamsLine("   [--sampling <Ts=1>]                : (Optional) Pixel size (Angstrom). If it is not provided by default will be 1 A/px.");
	addParamsLine("   [--threads <Nthreads=1>]           : (Optional) Number of threads to be used");

	addExampleLine("Obtain the average of a set of subtomograms from a metadata file", false);
	addExampleLine("xmipp_tomo_average_subtomograms -i inputSubtomos.xmd -o average.mrc");
}

void ProgAverageSubtomos::readParams()
{
	fnSubtomos = getParam("-i");
	notApplyAlignment = checkParam("--notApplyAlignment");
	fnOut = getParam("-o");
	sampling = getDoubleParam("--sampling");
	Nthreads = getIntParam("--threads");
}


void ProgAverageSubtomos::averageSubtomograms(MetaDataVec &md)
{
	MultidimArray<double> ref;
	Image<double> subtomoImg;
	Matrix2D<double> eulerMat;
	auto &subtomo = subtomoImg();

	MultidimArray<double> subtomoAli;

	MetaDataVec mdAli;

	FileName fnSub;
	size_t idx = 0;
	for (const auto& row : md)
	{
		row.getValue(MDL_IMAGE, fnSub);
		subtomoImg.read(fnSub);
		subtomo.setXmippOrigin();

		if (ref.getDim() < 1)
		{
			ref.initZeros(subtomo);
		}

		if (notApplyAlignment)
		{
			ref += subtomo;
		}
		else
		{
			eulerMat.initIdentity(4);
			geo2TransformationMatrix(row, eulerMat);

			subtomoAli.resizeNoCopy(subtomo);
			applyGeometry(xmipp_transformation::BSPLINE3, subtomoAli, subtomo, eulerMat, xmipp_transformation::IS_NOT_INV, true, 0.);
			ref += subtomoAli;
		}

		idx++;
	}

	ref /= idx;

	subtomoImg() = std::move(ref);
	subtomoImg.write(fnOut);
}

void ProgAverageSubtomos::run()
{
	std::cout << "Starting ... " << std::endl;

	MetaDataVec mdSubtomos;
	mdSubtomos.read(fnSubtomos);
	averageSubtomograms(mdSubtomos);
}

