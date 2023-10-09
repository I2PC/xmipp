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

#include "tomo_simulate_tilt_series.h"
#include <core/bilib/kernel.h>
#include "core/geometry.h"
#include "core/transformations.h"
#include "data/fourier_projection.h"
#include <core/metadata_extension.h>
#include <numeric>
#include <random>

void ProgTomoSimulateTiltseries::readParams()
{
	fnCoords = getParam("--coordinates");
	fnVol = getParam("--vol");
	xdim = getIntParam("--xdim");
	ydim = getIntParam("--ydim");
	minTilt = getDoubleParam("--minTilt");
	maxTilt = getDoubleParam("--maxTilt");
	tiltStep = getDoubleParam("--tiltStep");
	thickness = getIntParam("--thickness");
	fidSize = getDoubleParam("--fiducialSize");
	sampling = getDoubleParam("--sampling");
	sigmaNoise = getDoubleParam("--sigmaNoise");
	fnOut = getParam("-o");
}


void ProgTomoSimulateTiltseries::defineParams()
{
	addUsageLine("This function takes a tomogram an extract a set of subtomogram from it. The coordinates of the subtomograms are speciffied in the metadata given by coordinates.");
	addParamsLine("  --coordinates <xmd_file=\"\">        : Metadata (.xmd file) with the tilt series");
	addParamsLine("  --vol <xmd_file=\"\">                : Metadata (.xmd file) with the coordidanates to be extracted from the tomogram");
	addParamsLine("[  --xdim <xdim=1000>]                 : Particle box size in voxels.");
	addParamsLine("[  --ydim <ydim=1000> ]                : Particle box size in voxels.");
	addParamsLine("[  --minTilt <minTilt=-60.0>]          : Minimum tilt angle");
	addParamsLine("[  --maxTilt <maxTilt=60.0>]           : Maximum tilt angle");
	addParamsLine("[  --tiltStep <tiltStep=60.0>]         : Step of the angular sampling. Distance between two consequtive tilt angles");
	addParamsLine("[  --thickness <thickness=300>]        : Tomogram thickness in pixel");
	addParamsLine("[  --fiducialSize <fidSize=60.0>]      : Fiducial diameter in (nm)");
	addParamsLine("[  --sampling <s=1>]                   : Sampling rate in (A).");
	addParamsLine("[  --signaNoise <sigmaNoise=1>]        : Sampling rate in (A).");
	addParamsLine("  -o <mrc_file=\"\">                   : path to the output directory. ");
}


void ProgTomoSimulateTiltseries::createFiducial(MultidimArray<double> &fidImage, int boxsize)
{
	fidImage.initZeros(boxsize, boxsize);
	auto halfbox = round(0.5*boxsize);

	for (size_t j= 0; j<boxsize; j++)
	{
		auto j2 = (j-halfbox)*(j-halfbox);
		for (size_t i= 0; i<boxsize; i++)
		{
			auto i2 = (i-halfbox)*(i-halfbox);
			A2D_ELEM(fidImage, i, j) = sqrt(j2+i2);
		}
	}
}

void ProgTomoSimulateTiltseries::createCircle(MultidimArray<double> &circle, int boxsize)
{
	auto halfbox = round(0.5*boxsize);
	halfbox *=halfbox;

	size_t idx = 0;
	long n=0;
	for (size_t j= 0; j<boxsize; j++)
	{
		auto j2 = (j-halfbox)*(j-halfbox);
		for (size_t i= 0; i<boxsize; i++)
		{
			auto i2 = (i-halfbox)*(i-halfbox);
			if (i2+j2>halfbox)
			{
				DIRECT_MULTIDIM_ELEM(circle, n) = idx;
				n++;
			}
			idx++;
		}
	}
}


void ProgTomoSimulateTiltseries::run()
{
	std::cout << "Starting ... " << std::endl;

	Image<double> proteinImg;
	auto &ptrVol = proteinImg();

	MultidimArray<double> rotatedVol;
	rotatedVol.resizeNoCopy(ptrVol);

	proteinImg.read(fnVol);

	size_t boxsize, halfboxsize;
	boxsize = XSIZE(ptrVol);
	halfboxsize = round(0.5*boxsize);

	MultidimArray<double> fidImage;
	createFiducial(fidImage, boxsize);

	MetaDataVec mdCoords;
	int xcoor, ycoor, zcoor;
	double rot =0, tilt=0, psi=0, sx=0, sy=0;
	Matrix2D<double> eulerMat;
	Projection imgPrj;

	MultidimArray<double> &ptrProj = imgPrj();

	int BSplinedegree = 3;

    std::vector<double> tiltAngles;

    Image<double> tiltseriesImg;
    MultidimArray<double> &tiltseries = tiltseriesImg();

    int numberOfProjections;
    numberOfProjections = (maxTilt - minTilt)/tiltStep;
    tiltseries.initZeros(numberOfProjections, 1, ydim, xdim+2*boxsize);

    for (size_t idx = 0; idx<numberOfProjections; idx++)
    {
    	tiltAngles.push_back(minTilt+idx*tiltStep);
    }

    MetaDataVec mdTiltSeries;
    mdCoords.read(fnCoords);
    MDRowVec row, rowOut;

    std::cout << "Starting rows " << std::endl;
    int addCoords = 0;
	for (const auto& row : mdCoords)
	{
		row.getValue(MDL_XCOOR, xcoor);
		row.getValue(MDL_YCOOR, ycoor);
		row.getValue(MDL_ZCOOR, zcoor);

		xcoor = xcoor - 0.5*xdim;
		ycoor = ycoor - 0.5*ydim;
		zcoor = zcoor - 0.5*thickness;

		//std::cout << "Md xcoor " << xcoor << " Md ycoor " << ycoor << " Md zcoor" << zcoor << std::endl;

		if (row.containsLabel(MDL_ANGLE_ROT) && row.containsLabel(MDL_ANGLE_TILT) && row.containsLabel(MDL_ANGLE_PSI))
		{
			row.getValue(MDL_ANGLE_ROT, rot);
			row.getValue(MDL_ANGLE_TILT, tilt);
			row.getValue(MDL_ANGLE_PSI, psi);
		}
		else
		{
			eulerMat.initIdentity(4);
		}

		Euler_angles2matrix(rot, tilt, psi, eulerMat, true);

		applyGeometry(xmipp_transformation::BSPLINE3, rotatedVol, ptrVol, eulerMat, xmipp_transformation::IS_NOT_INV, true, 0.);
		rotatedVol.setXmippOrigin();
		//CenterFFT(rotatedVol,true);

		auto projector = FourierProjector(rotatedVol, 2.0, 0.5, BSplinedegree);

		for (size_t idx = 0; idx<tiltAngles.size(); idx++)
		{
			double tiltProj = tiltAngles[idx];
			if (addCoords<1)
			{
				rowOut.setValue(MDL_ANGLE_TILT, tiltProj);
				FileName fnTi;
				fnTi = formatString("%i@", idx+1);
				rowOut.setValue(MDL_IMAGE, fnTi + fnOut);
				mdTiltSeries.addRow(rowOut);
			}

			projectVolume(projector, imgPrj, ydim, xdim, 0.0, tiltProj, 0.0);
            Matrix1D<double> shifts(2);
            selfTranslate(xmipp_transformation::LINEAR,IMGMATRIX(imgPrj), shifts);

			//selfTranslate(xmipp_transformation::LINEAR, P(), roffset, xmipp_transformation::WRAP);

			tiltProj *= PI/180.0;
			double ct = cos(tiltProj);
			double st = sin(tiltProj);

			// Projection
			auto x_2d = (int) (xcoor * ct + zcoor* st);
			auto y_2d = (int) (ycoor);

			y_2d = y_2d+0.5*ydim;
			x_2d = x_2d+0.5*xdim;

			int xlim = x_2d + halfboxsize;
			int ylim = y_2d + halfboxsize;

			int xinit = x_2d - halfboxsize;
			int yinit = y_2d - halfboxsize;

			if ((xlim>xdim) || (ylim>ydim) || (xinit<0) || (yinit<0))
			{
				//std::cout << "Particle out of the tilt image" << std::endl;
				continue;
			}

			for (int i=0; i<boxsize; i++)
			{
				if (yinit+i<ydim)
				{
					for (int j=0; j<boxsize; j++)
					{
						if (xinit+j<xdim)
						{
							NZYX_ELEM(tiltseries, idx, 0, yinit+i, xinit+j) +=  DIRECT_A2D_ELEM(ptrProj, i, j);
						}
					}
				}
			}

		}

		addCoords++;
	}

	std::random_device rd{};
	std::mt19937 gen{rd()};

	// values near the mean are the most likely
	// standard deviation affects the dispersion of generated values from the mean
	std::normal_distribution d{0.0, sigmaNoise};

    // draw a sample from the normal distribution and round it to an integer
    auto randomNumber = [&d, &gen]{ return d(gen); };

	MultidimArray<double> tiltseriesOut;
	tiltseriesOut.initZeros(numberOfProjections, 1, ydim, xdim);
	for (size_t idx = 0; idx<tiltAngles.size(); idx++)
	{
		for (int i=0; i<ydim; i++)
		{
			for (int j=0; j<xdim; j++)
			{
				NZYX_ELEM(tiltseriesOut, idx, 0, i, j) =  NZYX_ELEM(tiltseries, idx, 0, i, j+boxsize)+randomNumber();
			}
		}
	}
	tiltseriesImg() = tiltseriesOut;
	tiltseriesImg.write(fnOut);
	FileName fnOutXmd;
	fnOutXmd = fnOut.removeLastExtension() + ".xmd";
	std::cout << fnOutXmd << std::endl;
	mdTiltSeries.write(fnOutXmd);
}

