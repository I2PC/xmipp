/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
 * 			   Federico P. de Isidro Gómez		  fp.deisidro@cnb.csic.es
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
#include <numeric>
//#define DEBUG


void ProgTomoExtractSubtomograms::readParams()
{
	fnTom = getParam("--tomogram");
	fnCoor = getParam("--coordinates");
	boxsize = getIntParam("--boxsize");
	invertContrast = checkParam("--invertContrast");
	normalize = checkParam("--normalize");
	fnOut = getParam("-o");
	downsampleFactor = getDoubleParam("--downsample");
	nthrs = getIntParam("--threads");
	fixedBoxSize = checkParam("--fixedBoxSize");
}


void ProgTomoExtractSubtomograms::defineParams()
{
	addUsageLine("This function takes a tomogram an extract a set of subtomogram from it. The coordinates of the subtomograms are speciffied in the metadata given by coordinates.");
	addParamsLine("  --tomogram <vol_file=\"\">               : Filename of the tomogram containing the subtomograms to be extracted");
	addParamsLine("  --coordinates <vol_file=\"\">	          : Metadata (.xmd file) with the coordidanates to be extracted from the tomogram");
	addParamsLine("  --boxsize <boxsize=100>                  : Particle box size in voxels, of the particle without downsampling.");
	addParamsLine("  [--invertContrast]	                      : Set this flag to invert the contrast of the extracted subtomograms");
	addParamsLine("  [--normalize]                            : This flag will set the subtomograms to have zero mean and unit standard deviation.");
	addParamsLine("  [--downsample <downsampleFactor=1.0>]    : Scale factor of the extracted subtomograms. It must be greater than 1. A downsampling 2 reduces in a factor 2 the size of the subtomos.");
	addParamsLine("  [--fixedBoxSize]                         : If selected, programs calculates the extraction box to obtain a box size of the selected size when downsampling.");
	addParamsLine("  -o <vol_file=\"\">                       : Path of the output directory. ");
	addParamsLine("  [--threads <s=4>]                        : Number of threads");
}


void ProgTomoExtractSubtomograms::createSphere(int halfboxsize)
{
	long n=0;

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
				if (sqrt(i2k2 + j2*j2)>halfboxsize)
				{
					maskIdx.push_back(n);
				}
				n++;
			}
		}
	}
}


void ProgTomoExtractSubtomograms::normalizeSubtomo(MultidimArray<double> &subtomo, int halfboxsize)
{
		MultidimArray<double> maskNormalize;

		double sumVal = 0;
		double sumVal2 = 0;

		auto counter = maskIdx.size();
		for (size_t i=0; i<maskIdx.size(); i++)
		{
				double val = DIRECT_MULTIDIM_ELEM(subtomo, maskIdx[i]);
				sumVal += val;
				sumVal2 += val*val;

		}

		double mean;
		double sigma2;
		mean = sumVal/counter;
		sigma2 = sqrt(sumVal2/counter - mean*mean);

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(subtomo)
		{
			DIRECT_MULTIDIM_ELEM(subtomo, n) -= mean;
			DIRECT_MULTIDIM_ELEM(subtomo, n) /= sigma2;
		}
}

void ProgTomoExtractSubtomograms::writeSubtomo(int idx, int xcoor, int ycoor, int zcoor, size_t particleid)
{
	FileName fn;
	fn = fnCoor.getBaseName() + formatString("-%i.mrc", idx);

	#ifdef DEBUG
	std::cout << fn << std::endl;
	std::cout << fnOut << std::endl;
	std::cout << fnOut+fn << std::endl;
	#endif

	subtomoImg.write(fnOut+"/"+fn);

	if (particleid != -1)
	{
		rowout.setValue(MDL_PARTICLE_ID, particleid);
	}

	rowout.setValue(MDL_XCOOR, xcoor);
	rowout.setValue(MDL_YCOOR, ycoor);
	rowout.setValue(MDL_ZCOOR, zcoor);
	rowout.setValue(MDL_IMAGE, fn);
	mdout.addRow(rowout);

	fn = fnCoor.getBaseName() + "_extracted.xmd";
	mdout.write(fnOut+"/"+fn);
}

void ProgTomoExtractSubtomograms::extractSubtomoFixedSize(MultidimArray<double> &subtomoExtraction)
{
	// Downsampling
	FourierTransformer transformer1;
	FourierTransformer transformer2;

	MultidimArray< std::complex<double> > fftSubtomoExtraction;
	MultidimArray< std::complex<double> > fftSubtomo;

	fftSubtomo.initZeros(boxsize, boxsize, (boxsize/2)+1);
	transformer1.FourierTransform(subtomoExtraction, fftSubtomoExtraction, true);

	#ifdef DEBUG
	std::cout << "XSIZE(fftSubtomo) " << XSIZE(fftSubtomo) << std::endl;
	std::cout << "YSIZE(fftSubtomo) " << YSIZE(fftSubtomo) << std::endl;
	std::cout << "ZSIZE(fftSubtomo) " << ZSIZE(fftSubtomo) << std::endl;

	std::cout << "XSIZE(fftSubtomoExtraction) " << XSIZE(fftSubtomoExtraction) << std::endl;
	std::cout << "YSIZE(fftSubtomoExtraction) " << YSIZE(fftSubtomoExtraction) << std::endl;
	std::cout << "ZSIZE(fftSubtomoExtraction) " << ZSIZE(fftSubtomoExtraction) << std::endl;
	#endif

	if (downsampleFactor > 1)
	{
			std::cout << "1 "<< std::endl;

		downsample(fftSubtomoExtraction, fftSubtomo);
	}
	else  // downsampleFactor < 1
	{
					std::cout << "2 "<< std::endl;
		upsample(fftSubtomoExtraction, fftSubtomo);
	}
			std::cout << "3 "<< std::endl;

	subtomoExtraction.initZeros(1, boxsize, boxsize, boxsize);
	subtomoExtraction.resize(boxsize, boxsize, boxsize);
	// subtomoExtraction.initConstant(0);
				std::cout << "4 "<< std::endl;

	transformer2.inverseFourierTransform(fftSubtomo, subtomoExtraction);
				std::cout << "5 "<< std::endl;

}

void ProgTomoExtractSubtomograms::run()
{
	std::cout << "Starting ... "<< std::endl;

	md.read(fnCoor);

	Image<double> tomImg;
	auto &tom = tomImg();
	tomImg.read(fnTom);

	size_t Xtom;
	size_t Ytom;
	size_t Ztom;

	Xtom = XSIZE(tom);
	Ytom = YSIZE(tom);
	Ztom = ZSIZE(tom);

	auto &subtomo = subtomoImg();

	MultidimArray<double> subtomoExtraction;
	std::cout << "a "<< std::endl;

	int xcoor;
	int ycoor;
	int zcoor;
	int xinit;
	int yinit;
	int zinit;

	size_t idx=1;

	int halfboxsize = floor(0.5*boxsize);

	double invertSign = 1.0;

	if (invertContrast)
	{
		invertSign = -1;
	}

	if (normalize)
	{
		createSphere(halfboxsize);
	}
	std::cout << "b "<< std::endl;


	double dsFactorTolerance = 0.01;
	double dsFactorDiff = abs(downsampleFactor - 1);

	size_t boxSizeExtraction;
	if (fixedBoxSize && dsFactorDiff > dsFactorTolerance)
	{
		#ifdef DEBUG
		std::cout << "Entering fixed box size mode" << std::endl;
		#endif


		boxSizeExtraction = boxsize * downsampleFactor;
		halfboxsize = floor(0.5*boxSizeExtraction);
	}
	std::cout << "c "<< std::endl;

	for (const auto& row : md)
	{
		row.getValue(MDL_XCOOR, xcoor);
		row.getValue(MDL_YCOOR, ycoor);
		row.getValue(MDL_ZCOOR, zcoor);

		int xlim = xcoor + halfboxsize;
		int ylim = ycoor + halfboxsize;
		int zlim = zcoor + halfboxsize;

		xinit = xcoor - halfboxsize;
		yinit = ycoor - halfboxsize;
		zinit = zcoor - halfboxsize;

		if ((xlim>Xtom) || (ylim>Ytom) || (zlim>Ztom) || (xinit<0) || (yinit<0) || (zinit<0))
			continue;
		
			std::cout << "d "<< std::endl;


		if (fixedBoxSize && dsFactorDiff > dsFactorTolerance)
		{
			subtomo.initZeros(1, boxSizeExtraction, boxSizeExtraction, boxSizeExtraction);
		}
		else
		{
			subtomo.initZeros(1, boxsize, boxsize, boxsize);
		}
			std::cout << "e "<< std::endl;

		// Contrast inversion
		for (int k=zinit; k<zlim; k++)
		{
			int kk = k - zcoor;
			for (int i=yinit; i<ylim; i++)
			{
				int ii = i-ycoor;
				for (int j=xinit; j<xlim; j++)
				{
					A3D_ELEM(subtomo, kk+halfboxsize, ii+halfboxsize, j+halfboxsize-xcoor) = invertSign * A3D_ELEM(tom, k, i, j);
				}
			}
		}
			std::cout << "f "<< std::endl;

		if (fixedBoxSize && dsFactorDiff > dsFactorTolerance)
		{
			extractSubtomoFixedSize(subtomo);
		}
			std::cout << "g "<< std::endl;

		// Normalization
		if (normalize)
		{
			normalizeSubtomo(subtomo, halfboxsize);
		}

		size_t particleid = -1;
		if (row.containsLabel(MDL_PARTICLE_ID))
		{
			row.getValue(MDL_PARTICLE_ID, particleid);
		}

		writeSubtomo(idx, xcoor, ycoor, zcoor, particleid);

		++idx;
	}


	std::cout << "Subtomo substraction finished succesfully!!" << std::endl;
}


void ProgTomoExtractSubtomograms::upsample(const MultidimArray<std::complex<double>> &from, MultidimArray<std::complex<double>> &to)
{
	for (size_t k = 0; k < ZSIZE(from)/2; k++)
	{
		for (size_t j = 0; j < YSIZE(from)/2; j++)
		{
			for (size_t i = 0; i < XSIZE(from); i++)
			{
				// Origin cube
				DIRECT_A3D_ELEM(to, k, j, i) = DIRECT_A3D_ELEM(from, k, j, i);

				// Oposite cube
				DIRECT_A3D_ELEM(to,  ZSIZE(to) - ZSIZE(from)/2 + k, YSIZE(to) - YSIZE(from)/2 + j, i) =
				DIRECT_A3D_ELEM(from, ZSIZE(from)/2 + k, YSIZE(from)/2 + j, i);

				// Offset-Z cube
				DIRECT_A3D_ELEM(to,  ZSIZE(to) - ZSIZE(from)/2 + k, j, i) =
				DIRECT_A3D_ELEM(from, ZSIZE(from)/2 + k, j, i);

				// Offset-Y cube
				DIRECT_A3D_ELEM(to,  k, YSIZE(to) - YSIZE(from)/2 + j, i) =
				DIRECT_A3D_ELEM(from, k, YSIZE(from)/2 + j, i);
			}
		}
	}
}

void ProgTomoExtractSubtomograms::downsample(const MultidimArray<std::complex<double>> &from, MultidimArray<std::complex<double>> &to)
{
	for (size_t k = 0; k < ZSIZE(to)/2; k++)
	{
		for (size_t j = 0; j < YSIZE(to)/2; j++)
		{
			for (size_t i = 0; i < XSIZE(to); i++)
			{
				// Origin cube
				DIRECT_A3D_ELEM(to, k, j, i) = DIRECT_A3D_ELEM(from, k, j, i);

				// Oposite cube
				DIRECT_A3D_ELEM(to,  ZSIZE(to)/2 + k, YSIZE(to)/2 + j, i) =
				DIRECT_A3D_ELEM(from, ZSIZE(from) - ZSIZE(to)/2 + k, YSIZE(from) - YSIZE(to)/2 + j, i);

				// Offset-Z cube
				DIRECT_A3D_ELEM(to,  ZSIZE(to)/2 + k, j, i) =
				DIRECT_A3D_ELEM(from, ZSIZE(from) - ZSIZE(to)/2 + k, j, i);

				// Offset-Y cube
				DIRECT_A3D_ELEM(to,  k, YSIZE(to)/2 + j, i) =
				DIRECT_A3D_ELEM(from, k, YSIZE(from) - YSIZE(to)/2 + j, i);
			}
		}
	}
}