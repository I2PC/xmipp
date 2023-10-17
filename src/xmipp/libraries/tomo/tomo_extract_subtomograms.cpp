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
//#define DEBUG_MASK
//#define TEST_FRINGES



void ProgTomoExtractSubtomograms::readParams()
{
	fnTom = getParam("--tomogram");
	fnCoor = getParam("--coordinates");
	boxsize = getIntParam("--boxsize");
	invertContrast = checkParam("--invertContrast");
	normalize = checkParam("--normalize");
	fnOut = getParam("-o");
	downsample = checkParam("--downsample");
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


void ProgTomoExtractSubtomograms::createSphere(MultidimArray<double> &maskNormalize, int halfboxsize)
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


void ProgTomoExtractSubtomograms::normalizeSubtomo(MultidimArray<double> &subtomo, int halfboxsize)
{
				MultidimArray<double> maskNormalize;
				createSphere(maskNormalize, halfboxsize);

				double sumVal = 0;
				double sumVal2 = 0;
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

void ProgTomoExtractSubtomograms::run()
{
	std::cout << "Starting ... "<< std::endl;

	md.read(fnCoor);

	Image<double> tomImg;
	auto &tom = tomImg();
	tomImg.read(fnTom);

	size_t Xtom, Ytom, Ztom;
	Xtom = XSIZE(tom);
	Ytom = YSIZE(tom);
	Ztom = ZSIZE(tom);

	auto &subtomo = subtomoImg();

	MultidimArray<double> subtomoExtraction;

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

	if (fixedBoxSize && downsample)
	{
		#ifdef DEBUG
		std::cout << "Entering fixed box size mode" << std::endl;
		#endif

		for (const auto& row : md)
		{
			#ifdef DEBUG
			std::cout << "Analyzing subtomo " << idx << std::endl;
			#endif

			row.getValue(MDL_XCOOR, xcoor);
			row.getValue(MDL_YCOOR, ycoor);
			row.getValue(MDL_ZCOOR, zcoor);

			size_t boxSizeExtraction;
			boxSizeExtraction = boxsize * downsampleFactor;

			int halfBoxSizeExtraction = floor(0.5*boxSizeExtraction);

			int xlim = xcoor + halfBoxSizeExtraction;
			int ylim = ycoor + halfBoxSizeExtraction;
			int zlim = zcoor + halfBoxSizeExtraction;

			xinit = xcoor - halfBoxSizeExtraction;
			yinit = ycoor - halfBoxSizeExtraction;
			zinit = zcoor - halfBoxSizeExtraction;

			#ifdef DEBUG
			std::cout << "boxsize " << boxsize << std::endl;
			std::cout << "downsampleFactor " << downsampleFactor << std::endl;
			std::cout << "boxSizeExtraction " << boxSizeExtraction << std::endl;

			std::cout << "xinit " << xinit << std::endl;
			std::cout << "yinit " << yinit << std::endl;
			std::cout << "zinit " << zinit << std::endl;
			
			std::cout << "xlim " << xlim << std::endl;
			std::cout << "ylim " << ylim << std::endl;
			std::cout << "Zlim " << zlim << std::endl;
			#endif

			if ((xlim>Xtom) || (ylim>Ytom) || (zlim>Ztom) || (xinit<0) || (yinit<0) || (zinit<0))
				continue;

			subtomoExtraction.initZeros(1, boxSizeExtraction, boxSizeExtraction, boxSizeExtraction);
			subtomo.initZeros(1, boxsize, boxsize, boxsize);

			// Contrast inversion
			for (int k=zinit; k<zlim; k++)
			{
				int kk = k - zcoor;
				for (int i=yinit; i<ylim; i++)
				{
					int ii = i-ycoor;
					for (int j=xinit; j<xlim; j++)
					{
						A3D_ELEM(subtomoExtraction, kk+halfBoxSizeExtraction, ii+halfBoxSizeExtraction, j+halfBoxSizeExtraction-xcoor) = invertSign * A3D_ELEM(tom, k, i, j);
					}
				}
			}


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
	}
	else
	{
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

			if ((xlim>Xtom) || (ylim>Ytom) || (zlim>Ztom) || (xinit<0) || (yinit<0) || (zinit<0))
				continue;

			
			subtomo.initZeros(1, boxsize, boxsize, boxsize);

			for (int k=zinit; k<zlim; k++)
			{
				int kk = k - zcoor;
				for (int i=yinit; i<ylim; i++)
				{
					int ii = i-ycoor;
					for (int j=xinit; j<xlim; j++)
					{
						A3D_ELEM(subtomo, kk+halfboxsize, ii+halfboxsize, j+halfboxsize-xcoor) = invertSign*A3D_ELEM(tom, k, i, j);
					}
				}
			}

			if (normalize)
			{
				normalizeSubtomo(subtomo, halfboxsize);
			}
			
			if (downsampleFactor>1.0)
			{
				int zdimOut, ydimOut, xdimOut;
				zdimOut = (int) boxsize/downsampleFactor;
				ydimOut = zdimOut;
				xdimOut = zdimOut;
				selfScaleToSizeFourier(zdimOut, ydimOut, xdimOut, subtomo, nthrs);
			}

			size_t particleid = -1;
			if (row.containsLabel(MDL_PARTICLE_ID))
			{
				row.getValue(MDL_PARTICLE_ID, particleid);
			}
			
			writeSubtomo(idx, xcoor, ycoor, zcoor, particleid);

			++idx;

		}
	}

	std::cout << "Subtomo substraction finished succesfully!!" << std::endl;
}

