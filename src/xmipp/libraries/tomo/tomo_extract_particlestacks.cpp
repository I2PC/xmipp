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
	normalize = checkParam("--normalize");
        swapXY = checkParam("--swapXY");
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
        addParamsLine("  [--swapXY]                         : Put this flag if the tomogram and the tilt series have the same dimensions but the X and Y coordinates are swaped");
	addParamsLine("  [--normalize]                      : Put this flag to normalize the background of the particle (zero mean and unit std)");
	addParamsLine("  [--downsample <scaleFactor=0.5>]   : Scale factor of the extracted subtomograms");
	addParamsLine("  -o <mrc_file=\"\">                 : path to the output directory. ");
	addParamsLine("  [--threads <s=4>]                  : Number of threads");
}

void ProgTomoExtractParticleStacks::createCircle(MultidimArray<double> &maskNormalize)
{
	int halfboxsize = 0.5* boxsize;
	if (normalize)
	{
		
		maskNormalize.initZeros(1, 1, boxsize, boxsize);

		for (int i=0; i<boxsize; i++)
		{
			int i2 = i-halfboxsize;
			int i2k2 = i2*i2;
			for (int j=0; j<boxsize; j++)
			{
				int j2 = (j- halfboxsize);
				if (sqrt(i2k2 + j2*j2)>halfboxsize)
					A2D_ELEM(maskNormalize, i, j) = 1;
			}
		}
	}
}

void ProgTomoExtractParticleStacks::run()
{
	std::cout << "Starting ... "<< std::endl;

	MetaDataVec mdts, mdcoords, mdparticlestack;
	mdts.read(fnTs);
	mdcoords.read(fnCoor);

	int xcoor, ycoor, zcoor, xinit, yinit;
	size_t idx=1;

	FileName fnImg;

	Image<double> tiltImg;
	auto &ptrtiltImg = tiltImg();

	// The tilt series is stored as a stack of images;
	std::vector<MultidimArray<double> > tsImages(0);
	MultidimArray<double> tsImg;
	std::vector<double> tsTiltAngles(0), tsRotAngles(0), tsShiftX(0), tsShiftY(0);
	std::vector<FileName> tsNames(0);

	double tilt, rot, tx, ty;

	size_t Nimages = 0;

	std::string tsid;

	for (const auto& row : mdts)
	{
		row.getValue(MDL_IMAGE, fnImg);
		row.getValue(MDL_ANGLE_TILT, tilt);
		row.getValue(MDL_ANGLE_ROT, rot);
		row.getValue(MDL_SHIFT_X, tx);
		row.getValue(MDL_SHIFT_Y, ty);
		row.getValue(MDL_TSID, tsid);

		tiltImg.read(fnImg);

		tsImages.push_back(ptrtiltImg);

		tsTiltAngles.push_back(tilt);
		tsRotAngles.push_back(rot);
		tsShiftX.push_back(tx);
		tsShiftY.push_back(ty);
		tsNames.push_back(fnImg);
		Nimages +=1;
	}

	size_t Xts, Yts;
	Xts = XSIZE(ptrtiltImg);
	Yts = YSIZE(ptrtiltImg);


	int halfboxsize = floor(0.5*boxsize);
	Image<double> finalStack;
	auto &particlestack = finalStack();

	MultidimArray<double> maskNormalize;
	if (normalize)
	{
		createCircle(maskNormalize);
	}

	size_t elem = 0; 

	FileName fnXmd;
	fnXmd = tsid + formatString(".xmd");

	std::cout << tsid << std::endl;

	for (const auto& row : mdcoords)
	{
		row.getValue(MDL_XCOOR, xcoor);
		row.getValue(MDL_YCOOR, ycoor);
		row.getValue(MDL_ZCOOR, zcoor);

		FileName fnMrc;

		fnMrc = tsid + formatString("-%i.mrcs", elem);
		
		std::vector<MultidimArray<double>> imgVec;
		
		particlestack.initZeros(Nimages, 1, boxsize, boxsize);

		for (size_t idx = 0; idx<Nimages; idx++)
		{
			tsImg = tsImages[idx];
			tilt = tsTiltAngles[idx]*PI/180;
			rot = tsRotAngles[idx]*PI/180;

			double ct = cos(tilt);
			double st = sin(tilt);

			double cr = cos(rot);
			double sr = sin(rot);

			tx = tsShiftX[idx];
			ty = tsShiftY[idx];
	
			/*
			First the piced coordinate, r, is projected on the aligned tilt sreies
			r' = Pr  Where r is the coordinates to be projected by the matrix P
			[x']   [ct   0   st][x]
			[y'] = [0    1    0][y]
			[z']   [0    0    0][z]
			Next is to undo the transformation This is Aligned->Unaligned
			If T is the transformation matrix unaligned->algined, we need T^-{1}
			Let us define a rotation matrix
			R=[cos(rot) -sin(rot)]
			  [ sin(rot) cos(rot)];

			The inverse of T is given by
			T^{-1} = [R' -R'*t;
			         0  0 1]);
			         
			 Where t are the shifts of T
			*/

			int x_2d, y_2d;

			// Projection
                        x_2d = (int) (xcoor * ct + zcoor* st);
                        y_2d = (int) (ycoor);
                        
                        std::cout << tilt*180/PI << "    " << x_2d << "   " << y_2d << "   " << rot*180/PI << std::endl;
                        
                        //Inverse transformation
                        double x_2d_prime =   cr*x_2d  + sr*y_2d - cr*tx  - sr*ty;
			double y_2d_prime =  -sr*x_2d  + cr*y_2d + sr*tx - cr*ty;
                        
                        if (swapXY)              
			{
			    x_2d = -x_2d_prime+0.5*Xts;
			    y_2d = -y_2d_prime+0.5*Yts;
			}
			else
			{
			    x_2d = -x_2d_prime+0.5*Yts;;
                            y_2d = -y_2d_prime+0.5*Xts;;
			}

			int xlim = x_2d + halfboxsize;
			int ylim = y_2d + halfboxsize;

			xinit = x_2d - halfboxsize;
			yinit = y_2d - halfboxsize;

			if ((xlim>Xts) || (ylim>Yts) || (xinit<0) || (yinit<0))
			{
				std::cout << "skipping " << std::endl;
				continue;
			}
			else
			{
			
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

			if (normalize)
			{
				for (size_t ti = 0; ti<Nimages; ti++)
				{
					double sumVal = 0, sumVal2 = 0;
				 	double counter = 0;

						long n = 0;
						for (int i=0; i<boxsize; i++)
						{
							for (int j=0; j<boxsize; j++)
							{
								if (DIRECT_MULTIDIM_ELEM(maskNormalize, n)>0)
								{
									double val = DIRECT_N_YX_ELEM(particlestack, ti, i, j);
									sumVal += val;
									sumVal2 += val*val;
									counter = counter + 1;
								}
								n++;			
								
							}
						}


					double mean, sigma2;
					mean = sumVal/counter;
					sigma2 = sqrt(sumVal2/counter - mean*mean);

					for (int i=0; i<boxsize; i++)
					{
						for (int j=0; j<boxsize; j++)
						{
							DIRECT_N_YX_ELEM(particlestack, ti, i, j) -= mean;
							DIRECT_N_YX_ELEM(particlestack, ti, i, j) /= sigma2;
						}
					}
				}
			}

			MDRowVec rowParticleStack;
			rowParticleStack.setValue(MDL_TSID, tsid);
			FileName idxstr;
			idxstr = formatString("%i@",idx+1);

			rowParticleStack.setValue(MDL_IMAGE, idxstr+fnMrc);
			rowParticleStack.setValue(MDL_ANGLE_TILT, tsTiltAngles[idx]);
			rowParticleStack.setValue(MDL_ANGLE_ROT, tsRotAngles[idx]);
			rowParticleStack.setValue(MDL_XCOOR, x_2d);
			rowParticleStack.setValue(MDL_YCOOR, y_2d);

			mdparticlestack.addRow(rowParticleStack);
			mdparticlestack.write(fnOut+"/"+fnXmd);
		}



		finalStack.write(fnOut+"/"+fnMrc);

		elem += 1;
	}



}

