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
#include <math.h>

void ProgTomoExtractParticleStacks::readParams()
{
	fnTs = getParam("--tiltseries");
	fnCoor = getParam("--coordinates");
	boxsize = getIntParam("--boxsize");
	sampling = getDoubleParam("--sampling");
	defocusPositive = checkParam("--defocusPositive");
	invertContrast = checkParam("--invertContrast");
	// scaleFactor = getDoubleParam("--downsample");
	normalize = checkParam("--normalize");
    swapXY = checkParam("--swapXY");
    setCTF = checkParam("--setCTF");
	fnOut = getParam("-o");
	nthrs = getIntParam("--threads");
}


void ProgTomoExtractParticleStacks::defineParams()
{
	addUsageLine("This function takes a tomogram an extract a set of subtomogram from it. The coordinates of the subtomograms are speciffied in the metadata given by coordinates.");
	addParamsLine("  --tiltseries <xmd_file=\"\">       : Metadata (.xmd file) with the tilt series");
	addParamsLine("  --coordinates <xmd_file=\"\">      : Metadata (.xmd file) with the coordidanates to be extracted from the tomogram");
	addParamsLine("  --boxsize <boxsize=100>            : Particle box size in voxels.");
	addParamsLine("  --sampling <s=1>                   : Sampling rate in (A).");
	addParamsLine("  [--defocusPositive]                : This flag must be put if the defocus increases or decreases along the z-axis. This is requires to set the local CTF");
	addParamsLine("  [--invertContrast]                 : Put this flag if the particles to be extracted are 3D particles (subtvolumes)");
    addParamsLine("  [--swapXY]                         : Put this flag if the tomogram and the tilt series have the same dimensions but the X and Y coordinates are swaped");
    addParamsLine("  [--setCTF]                         : Put this flag if the tilt series metadata has CTF parameters. The CTF per particle will be calculated and set in the final set of particles");
	addParamsLine("  [--normalize]                      : Put this flag to normalize the background of the particle (zero mean and unit std)");
	// addParamsLine("  [--downsample <scaleFactor=0.5>]   : Scale factor of the extracted subtomograms");
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

void ProgTomoExtractParticleStacks::getCoordinateOnTiltSeries(int xcoor, int ycoor, int zcoor, double &rot, double &tilt, double &tx, double &ty, int &x_2d, int &y_2d)
{
	double ct = cos(tilt);
	double st = sin(tilt);

	double cr = cos(rot);
	double sr = sin(rot);

	/*
	First the picked coordinate, r, is projected on the aligned tilt series
	r' = Pr  Where r is the coordinates to be projected by the matrix P
	[x']   [ct   0   st][x]
	[y'] = [0    1    0][y]
	[z']   [0    0    0][z]
	Next is to undo the transformation This is Aligned->Unaligned
	If T is the transformation matrix unaligned->aligned, we need T^-{1}
	Let us define a rotation matrix
	R=[cos(rot) -sin(rot)]
	  [ sin(rot) cos(rot)];

	The inverse of T is given by
	T^{-1} = [R' -R'*t;
	         0  0 1]);
	         
	 Where t are the shifts of T
	*/

	// Projection
	x_2d = (int) (xcoor * ct + zcoor* st);
	y_2d = (int) (ycoor);


    //Inverse transformation
    double x_2d_prime =   cr*x_2d  + sr*y_2d - cr*tx - sr*ty;
	double y_2d_prime =  -sr*x_2d  + cr*y_2d + sr*tx - cr*ty;

	/*
	std::cout << "xcoor = " << xcoor << std::endl;
	std::cout << "ycoor = " << ycoor << std::endl;
	std::cout << "zcoor = " << zcoor << std::endl;
	std::cout << "x_2d = " << x_2d << std::endl;
	std::cout << "y_2d = " << y_2d << std::endl;
	std::cout << "x_2d_prime = " << x_2d_prime << std::endl;
	std::cout << "y_2d_prime = " << y_2d_prime << std::endl;
	std::cout << "Xts = " << Xts << std::endl;
	std::cout << "Yts = " << Yts << std::endl;
	*/
        
    if (swapXY)
	{
    	//std::cout << "swapXY = true " << std::endl;
	    x_2d = -x_2d_prime+0.5*Xts;
	    y_2d = -y_2d_prime+0.5*Yts;
	}
	else
	{
		//std::cout << "swapXY = false " << std::endl;
		y_2d = y_2d_prime+0.5*Yts;
		x_2d = x_2d_prime+0.5*Xts;
	}
    /*
    std::cout << "x_2d = " << x_2d << std::endl;
    std::cout << "y_2d = " << y_2d << std::endl;
    std::cout << "===========" << std::endl;
    */
}

void ProgTomoExtractParticleStacks::readTiltSeriesInfo(std::string &tsid)
{
	std::cout << "Reading Tilt Series ... "<< std::endl;
	//TODO: Ensure there is only a single TSId
	MetaDataVec mdts;
	mdts.read(fnTs);

	Image<double> tiltImg;
	auto &ptrtiltImg = tiltImg();

	// The tilt series is stored as a stack of images;
	MultidimArray<double> tsImg;
	std::vector<FileName> tsNames(0);

	FileName fnImg;

	size_t Nimages = 0;
	double defU=0, defV=0, defAng=0, dose = 0;

	for (const auto& row : mdts)
	{
		row.getValue(MDL_IMAGE, fnImg);
		row.getValue(MDL_ANGLE_TILT, tilt);
		row.getValue(MDL_ANGLE_ROT, rot);

		if (setCTF)
		{
			row.getValue(MDL_CTF_DEFOCUSU, defU);
			row.getValue(MDL_CTF_DEFOCUSV, defV);
			row.getValue(MDL_CTF_DEFOCUS_ANGLE, defAng);
			row.getValue(MDL_DOSE, dose);
			tsDefU.push_back(defU);
			tsDefV.push_back(defV);
			tsDefAng.push_back(defAng);

			tsDose.push_back(dose);
		}


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

	Xts = XSIZE(ptrtiltImg);
	Yts = YSIZE(ptrtiltImg);
}



void ProgTomoExtractParticleStacks::run()
{
	std::cout << "Starting ... "<< std::endl;
	double signValue = 1;
	if (invertContrast)
	{
		signValue = -1;
	}

	MetaDataVec mdcoords, mdparticlestack;
	mdcoords.read(fnCoor);
	MetaDataVec mdts;

	int xcoor, ycoor, zcoor, xinit, yinit;
	size_t idx=1;

	mdts.read(fnTs);

	std::string tsid;
	readTiltSeriesInfo(tsid);

	double tilt, rot, tx, ty;

	size_t Nimages = 0;

	int halfboxsize = floor(0.5*boxsize);
	Image<double> finalStack;
	auto &particlestack = finalStack();

	MultidimArray<double> maskNormalize;
	if (normalize)
	{
		createCircle(maskNormalize);
	}

	size_t particleId = 0;
	double signDef = -1.0;


	FileName fnXmd;
	fnXmd = tsid + formatString(".xmd");

	for (const auto& row : mdcoords)
	{
		row.getValue(MDL_XCOOR, xcoor);
		row.getValue(MDL_YCOOR, ycoor);
		row.getValue(MDL_ZCOOR, zcoor);
		
		std::cout << "Coord ... "<< xcoor << " " << ycoor << " " << zcoor << std::endl;

		std::vector<MultidimArray<double>> imgVec;
		MultidimArray<double> singleImage;
		singleImage.initZeros(1,1, boxsize, boxsize);

		FileName fnMrc;
		fnMrc = tsid + formatString("-%i.mrcs", particleId);
		//std::cout << "-..................................." << std::endl;
		size_t imgNumber = 0;
		for (size_t idx = 0; idx<tsImages.size(); idx++)
		{
			//std::cout << "tilt = " << tsTiltAngles[idx] << std::endl;
			//std::cout << "tilt = " << tsTiltAngles[idx] << std::endl;
			auto tsImg = tsImages[idx];
			tilt = tsTiltAngles[idx]*PI/180;
			rot = tsRotAngles[idx]*PI/180;
			tx = tsShiftX[idx];
			ty = tsShiftY[idx];
			int x_2d, y_2d;
			getCoordinateOnTiltSeries(xcoor, ycoor, zcoor, rot, tilt, tx, ty, x_2d, y_2d);

			int xlim = x_2d + halfboxsize;
			int ylim = y_2d + halfboxsize;

			xinit = x_2d - halfboxsize;
			yinit = y_2d - halfboxsize;

			if ((xlim>Xts) || (ylim>Yts) || (xinit<0) || (yinit<0))
			{
				std::cout << "checki" << std::endl;
				continue;
			}

			double checkingMean = 0;
			for (int i=yinit; i<ylim; i++)
			{
				int ii = i-y_2d+halfboxsize;
				for (int j=xinit; j<xlim; j++)
				{
					double val = A2D_ELEM(tsImages[idx], i, j);
					checkingMean += val;
					A2D_ELEM(singleImage, ii, j+halfboxsize-x_2d) = signValue*val;
				}
			}


			std::cout << "checkingMean = " << checkingMean << std::endl;
			if (isnan(checkingMean))
			{
				std::cout << "checkingMean" << std::endl;
				continue;
			}

			if (normalize)
			{
				double sumVal = 0, sumVal2 = 0, counter = 0;

				long n = 0;

				for (int i=0; i<boxsize; i++)
				{
					for (int j=0; j<boxsize; j++)
					{
						if (DIRECT_MULTIDIM_ELEM(maskNormalize, n)>0)
						{
							double val = A2D_ELEM(singleImage, i, j);
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

				if (sigma2 == 0)
					continue;

				for (int i=0; i<boxsize; i++)
				{
					for (int j=0; j<boxsize; j++)
					{
						A2D_ELEM(singleImage, i, j) -= mean;
						A2D_ELEM(singleImage, i, j) /= sigma2;
					}
				}
			}

			imgVec.push_back(singleImage);
			MDRowVec rowParticleStack;
			rowParticleStack.setValue(MDL_TSID, tsid);
			FileName idxstr;
			idxstr = formatString("%i@",imgNumber+1);
			rowParticleStack.setValue(MDL_IMAGE, idxstr+fnMrc);
			rowParticleStack.setValue(MDL_PARTICLE_ID, particleId);
			rowParticleStack.setValue(MDL_ANGLE_TILT, tsTiltAngles[idx]);
			rowParticleStack.setValue(MDL_ANGLE_ROT, tsRotAngles[idx]);
			rowParticleStack.setValue(MDL_SHIFT_X, tsShiftX[idx]);
			rowParticleStack.setValue(MDL_SHIFT_Y, tsShiftY[idx]);
			rowParticleStack.setValue(MDL_DOSE, tsDose[idx]);

			double defU=0, defV=0, defAng=0;
			if (setCTF)
			{
//				if (defocusPositive)
//				{
//					signDef = 1.0;
//				}

				double Df = (xcoor * cos(tilt) + zcoor* sin(tilt))*sampling*sin(tilt);

				defU = tsDefU[idx]  + signDef*Df;
				defV = tsDefV[idx]  + signDef*Df;
			}
			rowParticleStack.setValue(MDL_CTF_DEFOCUSU, defU);
			rowParticleStack.setValue(MDL_CTF_DEFOCUSV, defV);
			rowParticleStack.setValue(MDL_CTF_DEFOCUS_ANGLE, tsDefAng[idx]);
			rowParticleStack.setValue(MDL_XCOOR, x_2d);
			rowParticleStack.setValue(MDL_YCOOR, y_2d);

			mdparticlestack.addRow(rowParticleStack);
			imgNumber++;
		}
		if (imgNumber>0)
		{
			mdparticlestack.write(fnOut+"/"+fnXmd);
			Nimages = imgVec.size();
			particlestack.initZeros(Nimages, 1, boxsize, boxsize);
			for (int k=0; k<Nimages; k++)
			{
				singleImage = imgVec[k];
				for (int i=0; i<boxsize; i++)
				{
					for (int j=0; j<boxsize; j++)
					{
						NZYX_ELEM(particlestack, k, 0, i, j) = A2D_ELEM(singleImage, i, j);
					}
				}
			}

			finalStack.write(fnOut+"/"+fnMrc);

			particleId += 1;
		}
	}
}

