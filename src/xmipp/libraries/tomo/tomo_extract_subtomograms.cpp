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
	normalize = checkParam("--normalize");
	fnOut = getParam("-o");
	nthrs = getIntParam("--threads");
	downsample = checkParam("--downsample");
	downsampleFactor = getDoubleParam("--downsample");
}


void ProgTomoExtractSubtomograms::defineParams()
{
	addUsageLine("This function takes a tomogram an extract a set of subtomogram from it. The coordinates of the subtomograms are speciffied in the metadata given by coordinates.");
	addParamsLine("  --tomogram <vol_file=\"\">         : Filename of the tomogram containing the subtomograms to be extracted");
	addParamsLine("  --coordinates <vol_file=\"\">	    : Metadata (.xmd file) with the coordidanates to be extracted from the tomogram");
	addParamsLine("  --boxsize <boxsize=100>            : Particle box size in voxels.");
	addParamsLine("  [--subtomo]                        : Put this flag if the particles to be extracted are 3D particles (subtvolumes)");
	addParamsLine("  [--invertContrast]	                : Put this flag if the particles to be extracted are 3D particles (subtvolumes)");
	addParamsLine("  [--normalize]                      : Put this flag if the particles to be extracted are 3D particles (subtvolumes)");
	addParamsLine("  [--downsample <scaleFactor=0.5>]   : Scale factor of the extracted subtomograms");
	addParamsLine("  -o <vol_file=\"\">                 : path to the output directory. ");
	addParamsLine("  [--threads <s=4>]                  : Number of threads");
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

	MultidimArray<double> subtomoExtraction;

	int xcoor;
	int ycoor;
	int zcoor;
	int xinit;
	int yinit;
	int zinit;

	size_t idx=1;

	size_t boxSizeExtraction;

	if (downsample)
	{
		boxSizeExtraction = boxsize * downsampleFactor;
	}
	else
	{
		boxSizeExtraction = boxsize;
	}

	int halfboxsize = floor(0.5*boxsize);
	int halfBoxSizeExtraction = floor(0.5*boxSizeExtraction);

	MultidimArray<double> maskNormalize;
	if (normalize)
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
					if (sqrt(i2k2 + j2*j2)>halfboxsize)
						A3D_ELEM(maskNormalize, k, i, j) = 1;
				}
			}
		}
	}

   	FourierTransformer transformer1;
	FourierTransformer transformer2;
	for (const auto& row : md)
	{
		row.getValue(MDL_XCOOR, xcoor);
		row.getValue(MDL_YCOOR, ycoor);
		row.getValue(MDL_ZCOOR, zcoor);

		int xlim = xcoor+halfBoxSizeExtraction;
		int ylim = ycoor+halfBoxSizeExtraction;
		int zlim = zcoor+halfBoxSizeExtraction;

		xinit = xcoor - halfBoxSizeExtraction;
		yinit = ycoor - halfBoxSizeExtraction;
		zinit = zcoor - halfBoxSizeExtraction;		

		if ((xlim>Xtom) || (ylim>Ytom) || (zlim>Ztom) || (xinit<0) || (yinit<0) || (zinit<0))
			continue;

		
		subtomoExtraction.initZeros(1, boxSizeExtraction, boxSizeExtraction, boxSizeExtraction);

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
						A3D_ELEM(subtomoExtraction, kk+halfBoxSizeExtraction, ii+halfBoxSizeExtraction, j+halfBoxSizeExtraction-xcoor) = -A3D_ELEM(tom, k, i, j);
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
						A3D_ELEM(subtomoExtraction, kk+halfBoxSizeExtraction, ii+halfBoxSizeExtraction, j+halfBoxSizeExtraction-xcoor) = A3D_ELEM(tom, k, i, j);
					}
				}
			}
		}

		if (downsample)
		{

			MultidimArray< std::complex<double> > fftSubtomoExtraction;
			MultidimArray< std::complex<double> > fftSubtomo;

			fftSubtomo.initZeros(boxsize, boxsize, (boxsize/2)+1);
			transformer1.FourierTransform(subtomoExtraction, fftSubtomoExtraction, true);

			#ifdef DEBUG
			std::cout << "XSIZE(subtomoExtraction) " << XSIZE(subtomoExtraction) << std::endl;
			std::cout << "YSIZE(subtomoExtraction) " << YSIZE(subtomoExtraction) << std::endl;
			std::cout << "ZSIZE(subtomoExtraction) " << ZSIZE(subtomoExtraction) << std::endl;
			
			std::cout << "XSIZE(fftSubtomo) " << XSIZE(fftSubtomo) << std::endl;
			std::cout << "YSIZE(fftSubtomo) " << YSIZE(fftSubtomo) << std::endl;
			std::cout << "ZSIZE(fftSubtomo) " << ZSIZE(fftSubtomo) << std::endl;

			std::cout << "XSIZE(fftSubtomoExtraction) " << XSIZE(fftSubtomoExtraction) << std::endl;
			std::cout << "YSIZE(fftSubtomoExtraction) " << YSIZE(fftSubtomoExtraction) << std::endl;
			std::cout << "ZSIZE(fftSubtomoExtraction) " << ZSIZE(fftSubtomoExtraction) << std::endl;
			#endif

			if (downsampleFactor > 1)
			{
				for (size_t k = 0; k < ZSIZE(fftSubtomo)/2; k++)
				{
					for (size_t j = 0; j < YSIZE(fftSubtomo)/2; j++)
					{
						for (size_t i = 0; i < XSIZE(fftSubtomo); i++)
						{
							// Origin cube
							DIRECT_A3D_ELEM(fftSubtomo, k, j, i) = DIRECT_A3D_ELEM(fftSubtomoExtraction, k, j, i);

							// Oposite cube
							DIRECT_A3D_ELEM(fftSubtomo,  ZSIZE(fftSubtomo)/2 + k, YSIZE(fftSubtomo)/2 + j, i) = 
							DIRECT_A3D_ELEM(fftSubtomoExtraction, ZSIZE(fftSubtomoExtraction) - ZSIZE(fftSubtomo)/2 + k, YSIZE(fftSubtomoExtraction) - YSIZE(fftSubtomo)/2 + j, i);

							// Offset-Z cube
							DIRECT_A3D_ELEM(fftSubtomo,  ZSIZE(fftSubtomo)/2 + k, j, i) = 
							DIRECT_A3D_ELEM(fftSubtomoExtraction, ZSIZE(fftSubtomoExtraction) - ZSIZE(fftSubtomo)/2 + k, j, i);

							// Offset-Y cube
							DIRECT_A3D_ELEM(fftSubtomo,  k, YSIZE(fftSubtomo)/2 + j, i) = 
							DIRECT_A3D_ELEM(fftSubtomoExtraction, k, YSIZE(fftSubtomoExtraction) - YSIZE(fftSubtomo)/2 + j, i);
						}
					}
				}
			}
			else  // downsampleFactor < 1
			{
				for (size_t k = 0; k < ZSIZE(fftSubtomoExtraction)/2; k++)
				{
					for (size_t j = 0; j < YSIZE(fftSubtomoExtraction)/2; j++)
					{
						for (size_t i = 0; i < XSIZE(fftSubtomoExtraction); i++)
						{
							// Origin cube
							DIRECT_A3D_ELEM(fftSubtomo, k, j, i) = DIRECT_A3D_ELEM(fftSubtomoExtraction, k, j, i);

							// Oposite cube
							DIRECT_A3D_ELEM(fftSubtomo,  ZSIZE(fftSubtomo) - ZSIZE(fftSubtomoExtraction)/2 + k, YSIZE(fftSubtomo) - YSIZE(fftSubtomoExtraction)/2 + j, i) = 
							DIRECT_A3D_ELEM(fftSubtomoExtraction, ZSIZE(fftSubtomoExtraction)/2 + k, YSIZE(fftSubtomoExtraction)/2 + j, i);

							// Offset-Z cube
							DIRECT_A3D_ELEM(fftSubtomo,  ZSIZE(fftSubtomo) - ZSIZE(fftSubtomoExtraction)/2 + k, j, i) = 
							DIRECT_A3D_ELEM(fftSubtomoExtraction, ZSIZE(fftSubtomoExtraction)/2 + k, j, i);

							// Offset-Y cube
							DIRECT_A3D_ELEM(fftSubtomo,  k, YSIZE(fftSubtomo) - YSIZE(fftSubtomoExtraction)/2 + j, i) = 
							DIRECT_A3D_ELEM(fftSubtomoExtraction, k, YSIZE(fftSubtomoExtraction)/2 + j, i);
						}
					}
				}
			}
			subtomo.initZeros(boxsize, boxsize, boxsize);
			transformer2.inverseFourierTransform(fftSubtomo, subtomo);
		}
		else
		{
			subtomo = subtomoExtraction;  // *** reuse subtomoextraction (remove subtomo)
		}
		

		if (normalize)
		{
			double sumVal = 0, sumVal2 = 0;
			double counter = 0;
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(subtomo)
			{
				if (DIRECT_MULTIDIM_ELEM(maskNormalize, n)>0)
				{
					double val = DIRECT_MULTIDIM_ELEM(subtomo, n);
					sumVal += val;
					sumVal2 += val*val;
					counter = counter + 1;
				}
			}

			double mean, sigma2;
			mean = sumVal/counter;
			sigma2 = sqrt(sumVal2/counter - mean*mean);

			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(subtomo)
			{
				DIRECT_MULTIDIM_ELEM(subtomo, n) -= mean;
				DIRECT_MULTIDIM_ELEM(subtomo, n) /= sigma2;
			}
		}
		
		FileName fn;
		fn = fnCoor.getBaseName() + formatString("-%i.mrc", idx);

		#ifdef DEBUG
		std::cout << fn << std::endl;
		std::cout << fnOut << std::endl;
		std::cout << fnOut+fn << std::endl;
		#endif

		subtomoImg.write(fnOut+"/"+fn);

		++idx;
	}

	std::cout << "Subtomo substraction finished succesfully!!" << std::endl;
}

