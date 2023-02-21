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

#include "tomo_extract_particlestacks.h"
#include <core/bilib/kernel.h>
#include <core/metadata_extension.h>
#include <numeric>
//#define DEBUG
//#define DEBUG_MASK
//#define TEST_FRINGES



void ProgTomoExtractParticleStacks::readParams()
{
	fnTs = getParam("--tiltseries");
	fnCoor = getParam("--coordinates");
	boxsize = getIntParam("--boxsize");
	invertContrast = checkParam("--invertContrast");
	scaleFactor = getDoubleParam("--downsample");
	fnOut = getParam("-o");
	nthrs = getIntParam("--threads");
}


void ProgTomoExtractParticleStacks::defineParams()
{
	addUsageLine("This function takes a tomogram an extract a set of subtomogram from it. The coordinates of the subtomograms are speciffied in the metadata given by coordinates.");
	addParamsLine("  --tiltseries <xmd_file=\"\">       : Metadata (.xmd file) with the coordidanates to be extracted from the tomogram");
	addParamsLine("  --coordinates <xmd_file=\"\">      : Metadata (.xmd file) with the coordidanates to be extracted from the tomogram");
	addParamsLine("  --boxsize <boxsize=100>            : Particle box size in voxels.");
	addParamsLine("  [--invertContrast]                 : Put this flag if the particles to be extracted are 3D particles (subtvolumes)");
	addParamsLine("  [--downsample <scaleFactor=0.5>]   : Scale factor of the extracted subtomograms");
	addParamsLine("  -o <mrc_file=\"\">                 : path to the output directory. ");
	addParamsLine("  [--threads <s=4>]                  : Number of threads");
}



void ProgTomoExtractParticleStacks::run()
{
	std::cout << "Starting ... "<< std::endl;

	MetaDataVec mdts, mdcoords, mdparticlestack;
	mdts.read(fnTs);
	mdcoords.read(fnCoor);

	int xcoor, ycoor, zcoor, xinit, yinit, zinit;
	size_t idx=1;

	FileName fnImg;

	Image<double> tiltImg;
	auto &ptrtiltImg = tiltImg();

	// The tilt series is stored as a stack of images;
	std::vector<MultidimArray<double> > tsImages(0);
	MultidimArray<double> tsImg;
	std::vector<double> tsTiltAngles(0), tsRotAngles(0);
	std::vector<FileName> tsNames(0);

	double tilt, rot;

	size_t Nimages = 0;

	for (const auto& row : mdts)
	{
		row.getValue(MDL_IMAGE, fnImg);
		row.getValue(MDL_ANGLE_TILT, tilt);
		row.getValue(MDL_ANGLE_ROT, rot);

		tiltImg.read(fnImg);

		tsImages.push_back(ptrtiltImg);

		tsTiltAngles.push_back(tilt);
		tsRotAngles.push_back(rot);
		tsNames.push_back(fnImg);
		Nimages +=1;
	}

	size_t Xts, Yts;
	Xts = XSIZE(ptrtiltImg);
	Yts = YSIZE(ptrtiltImg);


	int halfboxsize = floor(0.5*boxsize);
	Image<double> finalStack;
	auto &particlestack = finalStack();

	std::string tsid;
	size_t elem = 0; 

	for (const auto& row : mdcoords)
	{
		row.getValue(MDL_TSID, tsid);
		row.getValue(MDL_XCOOR, xcoor);
		row.getValue(MDL_YCOOR, ycoor);
		row.getValue(MDL_ZCOOR, zcoor);

		FileName fn;

		fn = fnCoor.getBaseName() + tsid + formatString("-%i.mrc", elem);

		particlestack.initZeros(Nimages, 1, boxsize, boxsize);

		for (size_t idx = 0; idx<Nimages; idx++)
		{
			tsImg = tsImages[idx];
			tilt = tsTiltAngles[idx]*PI/180;
			rot = tsRotAngles[idx];

			std::cout << tsNames[idx] << std::endl;

			double ct = cos(tilt);
			double st = sin(tilt);
	
			int x_2d, y_2d;
	
			x_2d = (int) (xcoor * ct) + 0.5*Xts; //- zcoor* st) + 0.5*Xts; 
			y_2d = (int) (ycoor+ 0.5*Yts);

			int xlim = x_2d + halfboxsize;
			int ylim = y_2d + halfboxsize;

			xinit = x_2d - halfboxsize;
			yinit = y_2d - halfboxsize;

			std::cout << tsTiltAngles[idx] << "  " << x_2d << " " << y_2d << std::endl;

			

			if ((xlim>Xts) || (ylim>Yts) || (xinit<0) || (yinit<0))
			{
				std::cout << "skipping " << std::endl;
				continue;
			}
				


			if (invertContrast)
			{
				for (int i=yinit; i<ylim; i++)
				{
					int ii = i-y_2d;
					for (int j=xinit; j<xlim; j++)
					{
						DIRECT_N_YX_ELEM(particlestack, idx, ii+halfboxsize, j+halfboxsize-x_2d) = -A2D_ELEM(tsImages[idx], i, j);
					}
				}
			}
			else
			{
				for (int i=yinit; i<ylim; i++)
				{
					int ii = i-y_2d;
					for (int j=xinit; j<xlim; j++)
					{
						DIRECT_N_YX_ELEM(particlestack, idx, ii+halfboxsize, j+halfboxsize-x_2d) = A2D_ELEM(tsImages[idx], i, j);
					}
				}
			}



			MDRowVec rowParticleStack;
			rowParticleStack.setValue(MDL_TSID, tsid);
			rowParticleStack.setValue(MDL_IMAGE, fn);
			rowParticleStack.setValue(MDL_ANGLE_TILT, tilt);
			rowParticleStack.setValue(MDL_ANGLE_ROT, rot);
			rowParticleStack.setValue(MDL_XCOOR, x_2d);
			rowParticleStack.setValue(MDL_YCOOR, y_2d);

			mdparticlestack.addRow(rowParticleStack);
		}



		finalStack.write(fnOut+"/"+fn);

		elem += 1;
	}



}

