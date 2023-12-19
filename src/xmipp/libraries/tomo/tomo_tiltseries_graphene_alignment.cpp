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

#include "tomo_tiltseries_graphene_alignment.h"



void ProgGrapheneAlignment::readParams()
{
	fnTs = getParam("--tiltseries");
	fnOut = getParam("-o");
	sampling = getDoubleParam("--sampling");
	grapheneParam = getDoubleParam("--latticeParam");
	nthrs = getIntParam("--threads");
}


void ProgGrapheneAlignment::defineParams()
{
	addUsageLine("This method align a tilt series that was acquired with a given grid geometric. Graphene is a particular hexagonal grid."
			    "The algorithm will try to align the diffraction peaks in Fourier Space.");
	addParamsLine("  --tiltseries <vol_file=\"\">                       : Xmd file with the list of images that define the tilt series");
	addParamsLine("  [--latticeParam <grapheneParam=1>]                 : Lattice param in Angstrom. This is the periodic distance that is repeated in the lattice");
	addParamsLine("  -o <vol_file=\"\">                                 : Output aligned tilt series");
	addParamsLine("  [--sampling <sampling=1>]   			            : Sampling rate (A/px)");
	addParamsLine("  [--threads <nthrs=4>]               	            : Number of threads to be used");
}


//TODO: This function is also in dose filter. Must be defined in a common module
void ProgGrapheneAlignment::readInputData(MetaDataVec &mdts)
{
	std::cout << "readInputData" << std::endl;
	if (fnTs.isMetaData())
		mdts.read(fnTs);
	else
	{
		REPORT_ERROR(ERR_ARG_INCORRECT, "The input must be a metadata with the image filenames");
		exit(0);
	}
}

void ProgGrapheneAlignment::getFourierShell(MultidimArray<std::complex<double>> &FTtiltImage, MultidimArray<std::complex<double>> &extractedShell, std::vector<long> &freqIdx, std::vector<double> &anglesVector)
{
	std::cout << "getFourierShell" << std::endl;
	extractedShell.initZeros(freqIdx.size());
	std::cout << "------------------------------------------" << std::endl;
	for (size_t i=0; i<freqIdx.size(); i++)
	{
		auto value = DIRECT_MULTIDIM_ELEM(FTtiltImage, freqIdx[i]);
		DIRECT_MULTIDIM_ELEM(extractedShell, i) = log(real(value*conj(value)));
		std::cout << anglesVector[i] << "," << log(real(value*conj(value))) << std::endl;
	}
	std::cout << "------------------------------------------" << std::endl;

//	MultidimArray<double> psd;
//	psd.resizeNoCopy(FTtiltImage);
//	psd.initZeros();
//
//	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FTtiltImage)
//	{
//		auto value = DIRECT_MULTIDIM_ELEM(FTtiltImage, n);
//		DIRECT_MULTIDIM_ELEM(psd,n) = real(value*conj(value));
//	}
//
//
//	Image<double> img;
//	img() = psd;
//	img.write("psd.mrc");
////
//
//	int angularSampling = 180;
//	double minAngle = std::min(anglesVector);
//	double maxAngle = std::max(anglesVector);
//	double rangeAngle = maxAngle - minAngle;
//
//	std::vector<double> profile1d;
//	std::vector<double> profile1dSize;
//
//	profile1d.initZeros(angularSampling);
//	profile1dSize.initZeros(angularSampling);
//
//	for(i=0; i< anglesVector.size() ; i++)
//	{
//		int rangedAngle = int(angularSampling * (anglesVector[i] - minAngle) / rangeAngle);
//		profile1d[rangedAngle] +=
//	}


}

void ProgGrapheneAlignment::indicesFourierShell(MultidimArray<std::complex<double>> &FTimg, MultidimArray<double> &tiltImage, std::vector<long> &freqIdx, std::vector<double> &anglesVector)
{
	std::cout << "indicesFourierShell" << std::endl;


	Matrix1D<double> freq_fourier_x, freq_fourier_y;

	// Initializing the frequency vectors
	freq_fourier_x.initZeros(XSIZE(FTimg));
	freq_fourier_y.initZeros(YSIZE(FTimg));

	// u is the frequency
	double u;
	size_t xvoldim, yvoldim;

	xvoldim = XSIZE(tiltImage);
	yvoldim = YSIZE(tiltImage);
	std::cout << "xdim = " << xvoldim << "  ydim = " << yvoldim << std::endl;

	// Defining frequency components. First element should be 0, it is set as the smallest number to avoid singularities
	VEC_ELEM(freq_fourier_y,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<YSIZE(FTimg); ++k){
		FFT_IDX2DIGFREQ(k,yvoldim, u);
		VEC_ELEM(freq_fourier_y, k) = u;
	}

	VEC_ELEM(freq_fourier_x,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<XSIZE(FTimg); ++k){
		FFT_IDX2DIGFREQ(k,xvoldim, u);
		VEC_ELEM(freq_fourier_x, k) = u;
	}


	MultidimArray<double> freqMap;
	freqMap.resizeNoCopy(FTimg);
	freqMap.initZeros();

	// Directional frequencies along each direction
	double uy, ux, uy2;
	long n=0;
	int idx = 0;

	double tolerance = 0.1;

	double lowLim = std::max(grapheneParam - tolerance, 2 *sampling);
	double lowerBound = sampling/(grapheneParam + tolerance);
	double upperBound = sampling/(lowLim);

	std::cout << "lowLim = " << lowLim << "   lowerBound = " << lowerBound << "   upperBound = " << upperBound  << std::endl;

	long count = 0;
	for(size_t i=0; i<YSIZE(FTimg); ++i)
	{
		uy = VEC_ELEM(freq_fourier_y, i);
		uy2 = uy*uy;
		for(size_t j=0; j<XSIZE(FTimg); ++j)
		{
			ux = VEC_ELEM(freq_fourier_x, j);
			u = sqrt(uy2 + ux*ux);

			if	((u<=upperBound) && (u>=lowerBound))
			{
				idx = (int) round(u * xvoldim);
				freqIdx.push_back(n);//DIRECT_MULTIDIM_ELEM(freqElems, idx) += 1;
				anglesVector.push_back(atan2(uy, ux) * 180.0 / PI);

				DIRECT_MULTIDIM_ELEM(freqMap,n) = u;
			}
			++n;
		}
	}

//	std::cout << "aa " << YSIZE(FTimg) << std::endl;
// 	for (size_t i = 0; i <YSIZE(FTimg); i++)
//	{
//		std::cout << VEC_ELEM(freq_fourier_y, i) << std::endl;
//	}
//
//	std::cout << "-----------------------" << std::endl;
//	for (size_t j = 0; j <XSIZE(FTimg); j++)
//	{
//		std::cout << VEC_ELEM(freq_fourier_x, j) << std::endl;
//	}

	 Image<double> img;
	 img() = freqMap;
	 img.write("freqMap.mrc");
}



void ProgGrapheneAlignment::smoothBorders(MultidimArray<double> &img, int N_smoothing)
{
	if (img.getDim() ==2)
	{
		int siz_y = YSIZE(img)*0.5;
		int siz_x = XSIZE(img)*0.5;

		int limit_distance_x = (siz_x-N_smoothing);
		int limit_distance_y = (siz_y-N_smoothing);

		long n=0;
		for(int i=0; i<YSIZE(img); ++i)
		{
			auto  uy = (i - siz_y);
			for(int j=0; j<XSIZE(img); ++j)
			{
				auto  ux = (j - siz_x);
				if (abs(ux)>=limit_distance_x)
				{
					DIRECT_MULTIDIM_ELEM(img, n) *= 0.5*(1+cos(PI*(limit_distance_x - abs(ux))/N_smoothing));
				}
				if (abs(uy)>=limit_distance_y)
				{
					DIRECT_MULTIDIM_ELEM(img, n) *= 0.5*(1+cos(PI*(limit_distance_y - abs(uy))/N_smoothing));
				}
				++n;
			}
		}
	}
	else
	{
		if (img.getDim() ==3)
		{
			int siz_z = ZSIZE(img)*0.5;
			int siz_y = YSIZE(img)*0.5;
			int siz_x = XSIZE(img)*0.5;


			int limit_distance_x = (siz_x-N_smoothing);
			int limit_distance_y = (siz_y-N_smoothing);
			int limit_distance_z = (siz_z-N_smoothing);

			long n=0;
			for(int k=0; k<ZSIZE(img); ++k)
			{
				auto uz = (k - siz_z);
				for(int i=0; i<YSIZE(img); ++i)
				{
					auto  uy = (i - siz_y);
					for(int j=0; j<XSIZE(img); ++j)
					{
						auto  ux = (j - siz_x);
						if (abs(ux)>=limit_distance_x)
						{
							DIRECT_MULTIDIM_ELEM(img, n) *= 0.5*(1+cos(PI*(limit_distance_x - abs(ux))/N_smoothing));
						}
						if (abs(uy)>=limit_distance_y)
						{
							DIRECT_MULTIDIM_ELEM(img, n) *= 0.5*(1+cos(PI*(limit_distance_y - abs(uy))/N_smoothing));
						}
						if (abs(uz)>=limit_distance_z)
						{
							DIRECT_MULTIDIM_ELEM(img, n) *= 0.5*(1+cos(PI*(limit_distance_z - abs(uz))/N_smoothing));
						}
						++n;
					}
				}
			}
		}
		else
		{
			REPORT_ERROR(ERR_ARG_INCORRECT, "The dimensions are not correct ");
			exit(0);
		}
	}



}


void ProgGrapheneAlignment::squareImageAndSmoothing(MultidimArray<double> &inImage, MultidimArray<double> &croppedImage, int N_smoothing)
{
	size_t xdim;
	size_t ydim;

	size_t xinit, yinit, xend, yend;

	xdim = XSIZE(inImage);
	ydim = YSIZE(inImage);

	if (xdim<ydim)
	{
		xinit=0;
		xend=xdim;
		yinit=round((ydim-xdim)*0.5);
		yend=yinit+xdim;
		croppedImage.initZeros(xdim, xdim);
	}
	else
	{
		yinit=0;
		yend=ydim;
		xinit=round((xdim-ydim)*0.5);
		xend=xinit+ydim;
		croppedImage.initZeros(ydim, ydim);
	}

	std::cout << "xend = " << xend << "   yend = " <<yend << "     xinit = " << xinit << "   yinit = " << yinit <<std::endl;

	for (int i=yinit; i<yend; i++)
	{
		for (int j=xinit; j<xend; j++)
		{
			auto imageVal = A2D_ELEM(inImage, i, j);

			if ((j-xinit)<N_smoothing)
			{
				imageVal *= 0.5*(1+cos(PI*(xinit + N_smoothing - j)/(N_smoothing)));
			}
			else if ((xend -j) < N_smoothing)
			{
					imageVal *= 0.5*(1+cos(PI*(xend +N_smoothing- j)/N_smoothing));
			}
			if ((i-yinit)<N_smoothing)
			{
				imageVal *= 0.5*(1+cos(PI*(yinit + N_smoothing - i)/N_smoothing));
			}
			else if ((yend - i)<(N_smoothing))
			{
				imageVal *= 0.5*(1+cos(PI*(yend + N_smoothing - i)/N_smoothing));
			}
			A2D_ELEM(croppedImage, i-yinit, j-xinit) = imageVal;
		}
	}
}


void ProgGrapheneAlignment::run()
{
	MetaDataVec mdts;

	readInputData(mdts);

	FourierTransformer transformer;
	FileName fnti, fnOutFrame;

	Image<double> tiltImage;
	MultidimArray<double> &ptrtiltImage = tiltImage();
	MultidimArray<std::complex<double>> extractedShell;
	MultidimArray<double> croppedImage;

	int mode = WRITE_OVERWRITE;

	std::vector<FileName> ti_fn;
	std::vector<double> ti_tilt;
	double tilt;

	bool firstImage = true;
	MultidimArray<std::complex<double>> fftImg;
	std::vector<long> freqIdx(0);
	std::vector<double> anglesVector(0);

	for (size_t objId : mdts.ids())
	{
		mdts.getValue(MDL_IMAGE, fnti, objId);
		mdts.getValue(MDL_ANGLE_TILT, tilt, objId);

		tiltImage.read(fnti);

		squareImageAndSmoothing(ptrtiltImage, croppedImage, 10);

		Image<double> saveImage;
		saveImage() = croppedImage;
		saveImage.write(fnOut);

		// Now do the Fourier transform and filter
		transformer.FourierTransform(croppedImage, fftImg, false);

		MultidimArray<double> psd;
		psd.resizeNoCopy(fftImg);
		psd.initZeros();

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fftImg)
		{
			auto value = DIRECT_MULTIDIM_ELEM(fftImg, n);
			DIRECT_MULTIDIM_ELEM(psd,n) = log(real(value*conj(value)));
		}


		Image<double> img;
		img() = psd;
		img.write("psd.mrc");


		if (firstImage)
		{
			firstImage = false;
			indicesFourierShell(fftImg, croppedImage, freqIdx, anglesVector);
			std::cout << " indicestFourierShell " << std::endl;
		}

		getFourierShell(fftImg, extractedShell, freqIdx, anglesVector);


		transformer.inverseFourierTransform();

//		FileName fnTsOut;
//		fnTsOut = fnOut;
//		tiltImage.write(fnTsOut, n+FIRST_IMAGE, true, mode);
//		mode = WRITE_APPEND;
//
//		++n;

//
//		ti_fn.push_back(fnti);
//		ti_tilt.push_back(tilt);
	}


}
