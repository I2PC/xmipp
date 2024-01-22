/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
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

#include "tomo_resolution_subtomos.h"

#include "reconstruction/resolution_monogenic_signal.h"
#include <memory>
#include <core/bilib/kernel.h>
#include <core/metadata_extension.h>
#include <numeric>
#include <filesystem>


void ProgTomoResolutionSubtomos::readParams()
{
	fnTomo  = getParam("--tomo");
	fnHalf   = getParam("--half");
	fnCoor  = getParam("--coordinates");
	boxsize = getIntParam("--boxsize");
	lowRes = getDoubleParam("--lowRes");
	highRes = getDoubleParam("--highRes");
	resStep = getDoubleParam("--resStep");
	sampling = getDoubleParam("--sampling");
	fnOut   = getParam("-o");
	nthrs   = getIntParam("--threads");
}


void ProgTomoResolutionSubtomos::defineParams()
{
	addUsageLine("This program takes a tomogram and a set of coordinates and estimates the local resolution of the associated subtomograms to the set of coordinates."
			"The local resolution is estimated with MonoRes algorithm.");
	addParamsLine("  --tomo <vol_file=\"\">                   : Filename of the full or the first half tomogram (odd or even tomogram)");
	addParamsLine("  [--half <vol_file=\"\">]                 : Filename of the second half tomogram (even or odd tomogram)");
	addParamsLine("  --coordinates <vol_file=\"\">	          : Metadata (.xmd file) with the coordinates to be extracted from the tomogram");
	addParamsLine("  --sampling <s=1>                         : Sampling rate (A/px)");
	addParamsLine("  --boxsize <boxsize=100>                  : Particle box size in (voxels) of the subtomograms for which the local resolution will be estimated.");
	addParamsLine("  --lowRes <s=150>                         : Lowest resolution in (A) for the resolution range to be analyzed.");
	addParamsLine("  --highRes <s=1>                          : Highest resolution in (A) for the resolution range to be analyzed.");
	addParamsLine("  [--resStep <s=1>]                        : (Optional) The resolution is computed from low to high frequency in steps of this parameter in (A).");
	addParamsLine("  [--significance <s=0.95>]                : (Optional) The level of confidence for the hypothesis test between signal and noise.");
	addParamsLine("  -o <vol_file=\"\">                       : Path of the output directory. ");
	addParamsLine("  [--threads <s=4>]                        : Number of threads");
}


void ProgTomoResolutionSubtomos::createSphere(MultidimArray<int> &maskNormalize, int halfboxsize)
{
	maskNormalize.initZeros(1, boxsize, boxsize, boxsize);

	for (int k=0; k<boxsize; k++)
	{
		int k2 = (k-halfboxsize);
		k2 = k2*k2;
		for (int i=0; i<boxsize; i++)
		{
			int i2 = i-halfboxsize;
			int i2k2 = i2*i2 +k2 ;
			for (int j=0; j<boxsize; j++)
			{
				int j2 = (j- halfboxsize);
				if (sqrt(i2k2 + j2*j2)<halfboxsize)
					A3D_ELEM(maskNormalize, k, i, j) = 1;
			}
		}
	}
}


void ProgTomoResolutionSubtomos::extractSubtomos(const MultidimArray<double> &oddTomo, const MultidimArray<double> *evenTomo,
														MultidimArray<double> &subtomoOdd, MultidimArray<double> *subtomoEven,
														int halfboxsize, int xcoor, int ycoor, int zcoor, bool nextcoor)
{
	int xlim = xcoor+halfboxsize;
	int ylim = ycoor+halfboxsize;
	int zlim = zcoor+halfboxsize;

	auto xinit = xcoor - halfboxsize;
	auto yinit = ycoor - halfboxsize;
	auto zinit = zcoor - halfboxsize;

	if ((xlim>Xtom) || (ylim>Ytom) || (zlim>Ztom) || (xinit<0) || (yinit<0) || (zinit<0))
		nextcoor = true;

	if (evenTomo != nullptr && subtomoEven !=nullptr)
	{
		subtomoEven->initZeros(1, boxsize, boxsize, boxsize);
		for (int k=zinit; k<zlim; k++)
		{
			int kk = k - zcoor;
			for (int i=yinit; i<ylim; i++)
			{
				int ii = i-ycoor;
				for (int j=xinit; j<xlim; j++)
				{
					A3D_ELEM(subtomoOdd,  kk+halfboxsize, ii+halfboxsize, j+halfboxsize-xcoor) = A3D_ELEM(oddTomo, k, i, j);
					A3D_ELEM(*subtomoEven, kk+halfboxsize, ii+halfboxsize, j+halfboxsize-xcoor) = A3D_ELEM(*evenTomo, k, i, j);
				}
			}
		}
	}
	else
	{
		for (int k=zinit; k<zlim; k++)
		{
			int kk = k - zcoor;
			for (int i=yinit; i<ylim; i++)
			{
				int ii = i-ycoor;
				for (int j=xinit; j<xlim; j++)
				{
					A3D_ELEM(subtomoOdd,  kk+halfboxsize, ii+halfboxsize, j+halfboxsize-xcoor) = A3D_ELEM(oddTomo, k, i, j);
				}
			}
		}
	}
}


void ProgTomoResolutionSubtomos::setLocalResolutionSubtomo(const MultidimArray<double> &localResMap, MultidimArray<double> &tomo,
														int halfboxsize, int xcoor, int ycoor, int zcoor)
{
	int xlim = xcoor+halfboxsize;
	int ylim = ycoor+halfboxsize;
	int zlim = zcoor+halfboxsize;

	auto xinit = xcoor - halfboxsize;
	auto yinit = ycoor - halfboxsize;
	auto zinit = zcoor - halfboxsize;

	double nyquist = 2*sampling;

	for (int k=zinit; k<zlim; k++)
	{
		int kk = k - zcoor;
		for (int i=yinit; i<ylim; i++)
		{
			int ii = i-ycoor;
			for (int j=xinit; j<xlim; j++)
			{
				auto value = A3D_ELEM(localResMap,  kk+halfboxsize, ii+halfboxsize, j+halfboxsize-xcoor);
				if (value< nyquist)
				{
					A3D_ELEM(tomo, k, i, j) = lowRes;
				}
				else
				{
					A3D_ELEM(tomo, k, i, j) = value;
				}
			}
		}
	}
}

void ProgTomoResolutionSubtomos::fillingBackground()
{
	std::cout << "filling background" << std::endl;
}


void ProgTomoResolutionSubtomos::run()
{
	std::cout << "Starting ... "<< std::endl;
	std::cout << fnOut << std::endl;

	Image<double> tomoImg, halfTomoImg;
	auto &tomo = tomoImg();

	std::unique_ptr<MultidimArray<double>> halfTomo;
	std::unique_ptr<MultidimArray<double>> subtomoHalf;


	Image<double> subtomoImg, subtomoHalfImg;
	auto &subtomo = subtomoImg();

	tomoImg.read(fnTomo);
	Xtom = XSIZE(tomo);
	Ytom = YSIZE(tomo);
	Ztom = ZSIZE(tomo);

	if (fnHalf!="")
	{
		useHalves = true;
		halfTomoImg.read(fnHalf);
		halfTomo = std::make_unique<MultidimArray<double>>(halfTomoImg());
		subtomoHalf = std::make_unique<MultidimArray<double>>();

	}

	int halfboxsize = floor(0.5*boxsize);
	MultidimArray<int> sphericalMask;
	createSphere(sphericalMask, halfboxsize);

	MetaDataVec md;
	md.read(fnCoor);

	size_t particleid;
	MultidimArray<double> subtomoExtraction;

	int xcoor;
	int ycoor;
	int zcoor;

	size_t idx=1;

	Image<double> tomoLocalResolution;
	auto &monoTomoMap = tomoLocalResolution();
	MultidimArray<double> localResMap;
	monoTomoMap.resizeNoCopy(tomo);
	monoTomoMap.initConstant(lowRes);

	MetaDataVec mdout;
	MDRowVec rowout;

	ProgMonogenicSignalRes mono;

	for (const auto& row : md)
	{
		row.getValue(MDL_XCOOR, xcoor);
		row.getValue(MDL_YCOOR, ycoor);
		row.getValue(MDL_ZCOOR, zcoor);

		subtomo.initZeros(1, boxsize, boxsize, boxsize);
		subtomo.resetOrigin();
		bool nextcoor = false;
		extractSubtomos(tomo, halfTomo.get(), subtomo, subtomoHalf.get(), halfboxsize, xcoor, ycoor, zcoor, nextcoor);

		Image<double> sv;
		sv() = subtomo;
		FileName fn;
		fn = formatString("subtomo_%i.mrc", idx);
		sv.write(fn);

		if (nextcoor)
			continue;

		localResMap.initZeros(1, boxsize, boxsize, boxsize);
		mono.runMonoRes(subtomo, subtomoHalf.get(), sphericalMask, highRes, lowRes, resStep, "", sampling, 0.95, fnOut, true, true, nthrs, localResMap, false, false);

		localResMap.resetOrigin();
//		if (fnHalf!="")
//		{
//			mono.runMonoRes(subtomo, subtomoHalf, sphericalMask, highRes, lowRes, resStep, "", sampling, 0.95, fnOut, true, true, nthrs, localResMap, false);
//		}
//		else
//		{
//			//mono.runMonoRes(tomo, halfTomo.get(), sphericalMask, lowRes, highRes, resStep, "", sampling, 0.95, fnOut, true, true, nthrs, localResMap, false);
//			subtomoHalf
//			mono.runMonoRes(subtomo, nullptr, sphericalMask, highRes, lowRes, resStep, "", sampling, 0.95, fnOut, true, true, nthrs, localResMap, false);
//		}

		sv() = localResMap;
		fn = formatString("resmap_%i.mrc", idx);
		sv.write(fn);
		setLocalResolutionSubtomo(localResMap, monoTomoMap, halfboxsize, xcoor, ycoor, zcoor);
		idx++;
	}
	std::cout << " aaaas------------" << std::endl;

	tomoLocalResolution.write(fnOut+"/localResTomo.mrc");


	std::cout << "Local Resolution estimation finished succesfully!!" << std::endl;
}

