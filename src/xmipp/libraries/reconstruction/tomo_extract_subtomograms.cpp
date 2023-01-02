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

#include "tomo_extract_subtomograms.h"
#include <core/bilib/kernel.h>
#include <core/metadata_extension.h>
#include <numeric>
//#define DEBUG
//#define DEBUG_MASK
//#define TEST_FRINGES



void ProgTomoExtractSubtomograms::readParams()
{
	fnTom = getParam("--tomogram");
	fnCoor = getParam("--coordinates");
	boxsize = getIntParam("--boxsize");
	scaleFactor = getDoubleParam("--downsample");
	invertContrast = checkParam("--invertContrast");
	fnOut = getParam("-o");
	nthrs = getIntParam("--threads");
}


void ProgTomoExtractSubtomograms::defineParams()
{
	addUsageLine("This function takes a tomogram an extract a set of subtomogram from it. The coordinates of the subtomograms are speciffied in the metadata given by coordinates.");
	addParamsLine("  --tomogram <vol_file=\"\">   		: Filename of the tomogram containing the subtomograms to be extracted");
	addParamsLine("  --coordinates <vol_file=\"\">		: Metadata (.xmd file) with the coordidanates to be extracted from the tomogram");
	addParamsLine("  --boxsize <boxsize=100>			: Particle box size in voxels.");
	addParamsLine("  [--subtomo]						: Put this flag if the particles to be extracted are 3D particles (subtvolumes)");
	addParamsLine("  [--invertContrast]						: Put this flag if the particles to be extracted are 3D particles (subtvolumes)");
	addParamsLine("  [--downsample <scaleFactor=0.5>]	: Scale factor of the extracted subtomograms");
	addParamsLine("  -o <vol_file=\"\">  				: path to the output directory. ");
	addParamsLine("  [--threads <s=4>]               	: Number of threads");
}

void ProgTomoExtractSubtomograms::run()
{
	std::cout << "Starting ... "<< std::endl;

	MetaDataVec md;
	md.read(fnCoor);

	Image<double> tomImg;
	auto &tom = tomImg();
	tomImg.read(fnTom);

	size_t Xtom, Ytom, Ztom;
	Xtom = XSIZE(tom);
	Ytom = YSIZE(tom);
	Ztom = ZSIZE(tom);

	Image<double> subtomoImg;
	auto &subtomo = subtomoImg();

	int xcoor, ycoor, zcoor, xinit, yinit, zinit;
	size_t idx=1;

	int halfboxsize = floor(0.5*boxsize);

	for (const auto& row : md)
	{
		row.getValue(MDL_XCOOR, xcoor);
		row.getValue(MDL_YCOOR, ycoor);
		row.getValue(MDL_ZCOOR, zcoor);

		int xlim = xcoor+halfboxsize;
		int ylim = ycoor+halfboxsize;
		int zlim = zcoor+halfboxsize;

		xinit = xcoor - halfboxsize;
		yinit = ycoor - halfboxsize;
		zinit = zcoor - halfboxsize;

		std::cout << xcoor << " " << ycoor << " " << zcoor << std::endl;
		

		if ((xlim>Xtom) || (ylim>Ytom) || (zlim>Ztom) || (xinit<0) || (yinit<0) || (zinit<0))
			continue;

		subtomo.initZeros(1, boxsize, boxsize, boxsize);

		if (invertContrast)
		{
			for (int k=zinit; k<zlim; k++)
			{
				int kk = k - zcoor;
				for (int i=yinit; i<ylim; i++)
				{
					int ii = i-ycoor;
					for (int j=xinit; j<xlim; j++)
					{
						A3D_ELEM(subtomo, kk+halfboxsize, ii+halfboxsize, j+halfboxsize-xcoor) = -A3D_ELEM(tom, k, i, j);
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
						A3D_ELEM(subtomo, kk+halfboxsize, ii+halfboxsize, j+halfboxsize-xcoor) = A3D_ELEM(tom, k, i, j);
					}
				}
			}
		}
		
		FileName fn;
		fn = fnCoor.getBaseName() + formatString("-%i.mrc", idx);
		std::cout << fn << std::endl;
		std::cout << fnOut << std::endl;
		std::cout << fnOut+fn << std::endl;
		subtomoImg.write(fnOut+"/"+fn);

		++idx;
	}
}

