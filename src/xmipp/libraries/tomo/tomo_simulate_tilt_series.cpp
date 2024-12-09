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
	sampling = getDoubleParam("--sampling");
	fnFid = getParam("--fiducialCoordinates");
	fidDiameter = getDoubleParam("--fiducialDiameter");
	sigmaNoise = getDoubleParam("--sigmaNoise");
	fnTsOut = getParam("--tiltseries");
	fnTomoOut = getParam("--tomogram");
}


void ProgTomoSimulateTiltseries::defineParams()
{
	addUsageLine("This function takes a tomogram an extract a set of subtomogram from it. The coordinates of the subtomograms are speciffied in the metadata given by coordinates.");
	addParamsLine("  --coordinates <xmd_file=\"\">           : Metadata (.xmd file) with the tilt series");
	addParamsLine("  --vol <xmd_file=\"\">                   : Volume (.mrc file) with the coordidanates to be extracted from the tomogram");
	addParamsLine("[  --xdim <xdim=1000>]                    : Particle box size in voxels.");
	addParamsLine("[  --ydim <ydim=1000> ]                   : Particle box size in voxels.");
	addParamsLine("[  --minTilt <minTilt=-60.0>]             : Minimum tilt angle");
	addParamsLine("[  --maxTilt <maxTilt=60.0>]              : Maximum tilt angle");
	addParamsLine("[  --tiltStep <tiltStep=60.0>]            : Step of the angular sampling. Distance between two consequtive tilt angles");
	addParamsLine("[  --thickness <thickness=300>]           : Tomogram thickness in pixel");
	addParamsLine("[  --sampling <s=1>]                      : Sampling rate in (A).");
	addParamsLine("[  --fiducialCoordinates <xmd_file=\"\">] : Metadata (.xmd file) with fiducial coordinates in the tomogram");
	addParamsLine("[  --fiducialDiameter <xmd_file=\"\">]    : Fiducial diameter in (A)");
	addParamsLine("[  --sigmaNoise <sigmaNoise=-1>]          : Standard deviation of the noise.");
	addParamsLine("  --tiltseries  <mrc_file=\"\">           : path to the output directory. ");
	addParamsLine("  --tomogram  <mrc_file=\"\">             : path to the output directory. ");
}


void ProgTomoSimulateTiltseries::createFiducial(MultidimArray<double> &fidImage, MultidimArray<double> &fidVol, int fidSize)
{
	fidImage.initZeros(fidSize, fidSize);
	fidVol.initZeros(fidSize, fidSize, fidSize);
	auto halfbox = round(0.5*fidSize);
	int fidSizeVoxels;

	auto fidSize2 = fidSize*fidSize;

	// We use 5*sigmaNoise to ensure the fiducial is visible with this noise level

	for (size_t i= 0; i<fidSize; i++)
	{
		auto i2 = (i-halfbox)*(i-halfbox);
		for (size_t j= 0; j<fidSize; j++)
		{
			auto j2 = (j-halfbox)*(j-halfbox);

			if ((j2 + i2) < fidSize2)
				A2D_ELEM(fidImage, i, j) = 5*sigmaNoise;
		}
	}

	for (size_t k= 0; k<fidSize; k++)
	{
		auto k2 = (k-halfbox)*(k-halfbox);
		for (size_t i= 0; i<fidSize; i++)
		{
			auto i2 = k2 + (i-halfbox)*(i-halfbox);
			for (size_t j= 0; j<fidSize; j++)
			{
				auto j2 = i2 + (j-halfbox)*(j-halfbox);

				if (i2 < fidSize2)
					A3D_ELEM(fidVol, k, i, j) = 5*sigmaNoise;
			}
		}
	}


}

void ProgTomoSimulateTiltseries::createSphere(MultidimArray<int> &mask, int boxsize)
{
	int halfbox = round(0.5*boxsize);
	auto halfbox2 =halfbox*halfbox;

	//std::cout << "halfbox = " << halfbox << std::endl;

	size_t idx = 0;
	long n=0;
	for (int k= 0; k<boxsize; k++)
	{
		int k2 = (k-halfbox)*(k-halfbox);
		//std::cout << "k2 = " << k2 << std::endl;
		for (int i= 0; i<boxsize; i++)
		{
			int i2 = (i-halfbox)*(i-halfbox);
			int i2k2 = i2 + k2;
			for (int j= 0; j<boxsize; j++)
			{
				int j2 = (j-halfbox)*(j-halfbox);

				if (i2k2+j2>halfbox2)
				{

					A3D_ELEM(mask, k, i, j) = 0;
				}
				else
				{
					A3D_ELEM(mask, k, i, j) = 1;
				}
				n++;
			}
		}
	}
}

void ProgTomoSimulateTiltseries::maskingRotatedSubtomo(MultidimArray<double> &subtomo, int boxsize)
{
	//subtomo.resetOrigin();
	auto halfbox = round(0.5*boxsize);
	auto halfbox2 =halfbox*halfbox;

	long n=0;
	for (size_t k= 0; k<boxsize; k++)
	{
		auto k2 = (k-halfbox)*(k-halfbox);
		for (size_t j= 0; j<boxsize; j++)
		{
			auto j2 = k2+(j-halfbox)*(j-halfbox);
			for (size_t i= 0; i<boxsize; i++)
			{
				auto i2 = (i-halfbox)*(i-halfbox);
				if (i2+j2>halfbox2)
				{
					DIRECT_MULTIDIM_ELEM(subtomo, n) = 0.0;

				}
				n++;
			}
		}
	}
}


void ProgTomoSimulateTiltseries::placeSubtomoInTomo(const MultidimArray<double> &subtomo, MultidimArray<double> &tomo,
													const int xcoord, const int ycoord, const int zcoord, const size_t boxsize)
{
	auto initzcoord = thickness/2 + zcoord-boxsize/2;
	auto initxcoord = xdim/2 + xcoord-boxsize/2;
	auto initycoord = ydim/2 + ycoord-boxsize/2;

	for (int k = initzcoord; k<(initzcoord+boxsize); k++)
	{
		for (int i = initycoord; i<(initycoord+boxsize); i++)
		{
			for (int j = initxcoord; j<(initxcoord+boxsize); j++)
			{
				DIRECT_A3D_ELEM(tomo, k, i, j) =  -DIRECT_A3D_ELEM(subtomo, k-initzcoord, i-initycoord, j-initxcoord);
			}
		}
	}
}


void ProgTomoSimulateTiltseries::run()
{
	std::cout << "Starting ... " << std::endl;

	// Reading the input map
	Image<double> proteinImg;
	auto &ptrVol = proteinImg();
	proteinImg.read(fnVol);

	MultidimArray<double> rotatedVol;	// This map is the rotated volume
	rotatedVol.resizeNoCopy(ptrVol);

	// Defining input coordinates and subtomogram orientation
	MetaDataVec mdCoords;
	int xcoor, ycoor, zcoor;
	double rot =0.0, tilt=0.0, psi=0, sx=0, sy=0;
	Matrix2D<double> eulerMat_proj, eulerMat_VolRotation, eulerMat;
	Projection imgPrj;

	MultidimArray<double> &ptrProj = imgPrj();

	int BSplinedegree = 3;
	MultidimArray<double> fidImage;
	MultidimArray<double> fidVol;

	if (fnFid != "")
	{
		int fidBoxsize;
		fidBoxsize = round(fidDiameter/sampling);
		createFiducial(fidImage, fidVol, fidBoxsize);
	}


    // Estimating the number of projections adn ouput tilt series
    std::vector<double> tiltAngles;
    Image<double> tiltseriesImg;
    MultidimArray<double> &tiltseries = tiltseriesImg();
    int numberOfProjections;
    numberOfProjections = (maxTilt - minTilt)/tiltStep + 1;

	size_t boxsize, halfboxsize;
	boxsize = XSIZE(ptrVol);
	halfboxsize = round(0.5*boxsize);
    tiltseries.initZeros(numberOfProjections, 1, ydim, xdim+2*boxsize);

    for (size_t idx = 0; idx<numberOfProjections; idx++)
    {
    	tiltAngles.push_back(minTilt+idx*tiltStep);
    }

    MultidimArray<int> mask;
    mask.resizeNoCopy(ptrVol);

    // Reading the coordinates and generating tilt series, tomograms and coordinates.
    MetaDataVec mdTiltSeries;
    mdCoords.read(fnCoords);
    MDRowVec row, rowOut;

    // This is the output tomogram
    Image<double> tomogram;
    auto &ptrTomo = tomogram();
    ptrTomo.initZeros(1, thickness, ydim, xdim);

    std::cout << "Starting rows " << std::endl;
    int addCoords = 0;

	for (const auto& row : mdCoords)
	{
		row.getValue(MDL_XCOOR, xcoor);
		row.getValue(MDL_YCOOR, ycoor);
		row.getValue(MDL_ZCOOR, zcoor);

		double theta, phi, xi;



		if (row.containsLabel(MDL_ANGLE_ROT) && row.containsLabel(MDL_ANGLE_TILT) && row.containsLabel(MDL_ANGLE_PSI))
		{
//			row.getValue(MDL_ANGLE_ROT, theta);
//			row.getValue(MDL_ANGLE_TILT, phi);
//			row.getValue(MDL_ANGLE_PSI, xi);

			row.getValue(MDL_ANGLE_PSI, theta);
			row.getValue(MDL_ANGLE_TILT, phi);
			row.getValue(MDL_ANGLE_ROT, xi);

//			theta *= -1;
//			phi  *= -1;
//			xi  *= -1;
		}
		else
		{
			double u = (double) rand()/RAND_MAX;
			double v = (double) rand()/RAND_MAX;
			double w = (double) rand()/RAND_MAX;

			theta = 360.0*u;
			phi = acos(2*v - 1.0)*180.0/PI;
			xi = 360.0*w;

			//eulerMat_proj.initIdentity(4);
		}

		std::cout << "theta = " << theta << "   " << "phi = " << phi << "   " << "xi = " << xi << std::endl;

		Euler_angles2matrix(theta, phi, xi, eulerMat_VolRotation, true);

		applyGeometry(xmipp_transformation::BSPLINE3, rotatedVol, ptrVol, eulerMat_VolRotation, xmipp_transformation::IS_NOT_INV, true, 0.);
		maskingRotatedSubtomo(rotatedVol, boxsize);

		rotatedVol.setXmippOrigin();
		//CenterFFT(rotatedVol,true);
		placeSubtomoInTomo(rotatedVol, ptrTomo, xcoor, ycoor, zcoor, boxsize);


		auto projector = FourierProjector(rotatedVol, 2.0, 0.5, BSplinedegree);

		for (size_t idx = 0; idx<tiltAngles.size(); idx++)
		{
			double tiltProj = tiltAngles[idx];
			if (addCoords<1)
			{
				rowOut.setValue(MDL_ANGLE_TILT, tiltProj);
				FileName fnTi;
				fnTi = formatString("%i@", idx+1);
				rowOut.setValue(MDL_IMAGE, fnTi + fnTsOut);
				mdTiltSeries.addRow(rowOut);
			}

			projectVolume(projector, imgPrj, boxsize, boxsize, 0.0, tiltProj, 0.0);

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

	MultidimArray<double> tiltseriesOut;
	tiltseriesOut.initZeros(numberOfProjections, 1, ydim, xdim);

	if (sigmaNoise>0)
	{
		std::random_device rd{};
		std::mt19937 gen{rd()};

		// values near the mean are the most likely
		// standard deviation affects the dispersion of generated values from the mean
		std::normal_distribution dts{0.0, sigmaNoise};
		std::normal_distribution dtom{0.0, sigmaNoise/boxsize};

		// draw a sample from the normal distribution and round it to an integer
		auto randomNumberTs = [&dts, &gen]{ return dts(gen); };
		auto randomNumberTom = [&dtom, &gen]{ return dtom(gen); };


		for (size_t idx = 0; idx<tiltAngles.size(); idx++)
		{
			for (int i=0; i<ydim; i++)
			{
				for (int j=0; j<xdim; j++)
				{
					NZYX_ELEM(tiltseriesOut, idx, 0, i, j) =  -NZYX_ELEM(tiltseries, idx, 0, i, j)+randomNumberTs();
				}
			}
		}

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(ptrTomo)
		{
			DIRECT_MULTIDIM_ELEM(ptrTomo, n) += randomNumberTom();
		}
	}
	else
	{

		for (size_t idx = 0; idx<tiltAngles.size(); idx++)
		{
			for (int i=0; i<ydim; i++)
			{
				for (int j=0; j<xdim; j++)
				{
					NZYX_ELEM(tiltseriesOut, idx, 0, i, j) =  -NZYX_ELEM(tiltseries, idx, 0, i, j);
				}
			}
		}
	}

	tiltseriesImg() = tiltseriesOut;
	tiltseriesImg.write(fnTsOut);
	FileName fnOutXmd;
	fnOutXmd = fnTsOut.removeLastExtension() + ".xmd";
	tomogram.write(fnTomoOut);
	std::cout << fnOutXmd << std::endl;
	mdTiltSeries.write(fnOutXmd);
}

