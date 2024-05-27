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
	addParamsLine("   [--notApplyAlignment]     		 : (Optional) Use this flag to estimate the standard average instead of the weighted one.");
	addParamsLine("   [-o <output_folder=\"\">]          : Folder where the results will be stored.");
	addParamsLine("   [--saveAligned]                    : Folder where the results will be stored.");
	addParamsLine("   [--sampling <Ts=1>]                : (Optional) Pixel size (Angstrom). If it is not provided by default will be 1 A/px.");
	addParamsLine("   [--threads <Nthreads=1>]           : (Optional) Number of threads to be used");

	addExampleLine("Obtain the average of a set of subtomograms from a metadata file", false);
	addExampleLine("xmipp_tomo_average_subtomograms -i inputSubtomos.xmd -o average.mrc");
}

void ProgAverageSubtomos::readParams()
{
	fnSubtomos = getParam("--subtomos");
	notapplyAlignment = checkParam("--notApplyAlignment");
	saveAligned = checkParam("--saveAligned");
	fnOut = getParam("-o");
	sampling = getDoubleParam("--sampling");
	Nthreads = getIntParam("--threads");
}


void ProgAverageSubtomos::averageSubtomograms(MetaDataVec &md, bool saveAligned=false)
{
	MultidimArray<double> &ref;
	Image<double> subtomoImg;
	auto &subtomo = subtomoImg();

	MultidimArray<double> subtomoAli;

	MetaDataVec mdAli;

	int status;
	std::string dirnameString;
	dirnameString = fnOut+"/alignedSubtomos";

	status = mkdir(dirnameString.c_str(), 0755);

	FileName fnSub;
	size_t idx = 0;
	for (const auto& row : md)
	{
		row.getValue(MDL_IMAGE, fnSub);
		subtomoImg.read(fnSub);

		subtomoAli.resizeNoCopy(subtomo);
		subtomo.setXmippOrigin();
		subtomoAli.setXmippOrigin();

		if (notapplyAlignment)
		{
			auto &subtomoAli = subtomo;
		}
		else
		{
			Matrix2D<double> eulerMat;
			eulerMat.initIdentity(4);
			geo2TransformationMatrix(row, eulerMat);

			applyGeometry(xmipp_transformation::BSPLINE3, subtomoAli, subtomo, eulerMat, xmipp_transformation::IS_NOT_INV, true, 0.);

			if (saveAligned)
			{
				FileName fnaligned = fnSub.getBaseName() + formatString("_aligned_%i.mrc", idx);
				auto fn = dirnameString + "/" + fnaligned;
				subtomoImg() = subtomoAli;
				subtomoImg.write(fn);

				MDRowVec rowAlign;
				rowAlign.setValue(MDL_IMAGE, fnaligned);
				mdAli.addRow(rowAlign);
			}
		}


		if (ref.getDim() < 1)
		{
			ref.initZeros(subtomo);
		}

		ref += subtomoAli;

		idx++;
	}

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(ref)
	{
		DIRECT_MULTIDIM_ELEM(ref, n) /= idx;
	}

	subtomoImg() = ref;
	subtomoImg.write(fnOut);

	if (saveAligned)
	{
		mdAli.write("alignedSubtomos.xmd");
	}

}

void ProgAverageSubtomos::run()
{
	std::cout << "Starting ... " << std::endl;

	MetaDataVec mdSubtomos;

	mdSubtomos.read(fnSubtomos);

	averageSubtomograms(mdSubtomos, saveAligned);

}

