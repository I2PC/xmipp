/***************************************************************************
 *
 * Authors:     Jose Luis Vilas (joseluis.vilas-prieto@yale.edu)
 *                             or (jlvilas@cnb.csic.es)
 *              Hemant. D. Tagare (hemant.tagare@yale.edu)
 *
 * Yale University, New Haven, Connecticut, United States of America
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

#include "image_snr.h"
#include <core/metadata_extension.h>
#include <data/monogenic_signal.h>
#include <data/fourier_filter.h>
#include <core/metadata_extension.h>
#include <random>
#include <limits>
#include <CTPL/ctpl_stl.h>


void ProgEstimateSNR::defineParams()
{
	addUsageLine("Calculate Fourier Shell Occupancy - FSO curve - via directional FSC measurements.");
	addUsageLine("Following outputs are generated:");

	addParamsLine("   -i <input_file>                    : Input metadata with halves or first half map");
	addParamsLine("   [--half2 <input_file>   ]          : Input Half map 2");
	addParamsLine("   [-o <output_folder=\"\">]          : Folder where the results will be stored.");
	addParamsLine("   [--sampling <sampling=1>]          : (Optical) Pixel size (Angstrom). If it is not provided by default will be 1 A/px.");
    addParamsLine("   [--normalize]                      : (Optional) Put this flag to estimate the 3DFSC, and apply it as low pass filter to obtain a directionally filtered map. It mean to apply an anisotropic filter.");
	addParamsLine("   [--threads <Nthreads=1>]           : (Optional) Number of threads to be used");

	addExampleLine("Resolution of two half maps half1.mrc and half2.mrc with a sampling rate of 2 A/px", false);
	addExampleLine("xmipp_resolution_fso --half1 half1.mrc --half2 half2.mrc --sampling_rate 2 ");
	addExampleLine("Resolution of two half maps half1.mrc and half2.mrc with a sampling rate of 2 A/px and a mask mask.mrc", false);
	addExampleLine("xmipp_resolution_fso --half1 half1.mrc --half2 half2.mrc --mask mask.mrc --sampling_rate 2 ");
}


void ProgEstimateSNR::readParams()
{
	fnIn1 = getParam("-i");
	if (fnIn1.isMetaData())
	{
		std::cout << "Input will be read from the metadata file " << std::endl;
	}
	else
	{
		fnIn2 = getParam("--half2");
	}
	fnOut = getParam("-o");

	sampling = getDoubleParam("--sampling");
	normalize = checkParam("--normalize");
	
	Nthreads = getIntParam("--threads");
}


void ProgEstimateSNR::defineFrequencies(const MultidimArray< std::complex<double> > &mapfftV, const MultidimArray<double> &inputVol)
{
	// Initializing the frequency vectors
//	freq_fourier_z.initZeros(ZSIZE(mapfftV));
	freq_fourier_x.initZeros(XSIZE(mapfftV));
	freq_fourier_y.initZeros(YSIZE(mapfftV));

	// u is the frequency
	double u;

	// Defining frequency components. First element should be 0, it is set as the smallest number to avoid singularities

//	VEC_ELEM(freq_fourier_z,0) = std::numeric_limits<double>::min();
//	for(size_t k=1; k<ZSIZE(mapfftV); ++k){
//		FFT_IDX2DIGFREQ(k,ZSIZE(inputVol), u);
//		VEC_ELEM(freq_fourier_z, k) = u;
//	}

	VEC_ELEM(freq_fourier_y,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<YSIZE(mapfftV); ++k){
		FFT_IDX2DIGFREQ(k,YSIZE(inputVol), u);
		VEC_ELEM(freq_fourier_y, k) = u;
	}

	VEC_ELEM(freq_fourier_x,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<XSIZE(mapfftV); ++k){
		FFT_IDX2DIGFREQ(k,XSIZE(inputVol), u);
		VEC_ELEM(freq_fourier_x, k) = u;
	}

	//Initializing map with frequencies
	freqMap.resizeNoCopy(mapfftV);
	freqMap.initConstant(1.9);  //Nyquist is 2, we take 1.9 greater than Nyquist

	size_t xvoldim = XSIZE(inputVol);
	size_t yvoldim = YSIZE(inputVol);
//	size_t zvoldim = ZSIZE(inputVol);

	// Directional frequencies along each direction
	double uz, uy, ux, uz2, uz2y2;
	long n=0;
	int idx = 0;

	long Ncomps = 0;
		
//	for(size_t k=0; k<ZSIZE(mapfftV); ++k)
//	{
//		uz = VEC_ELEM(freq_fourier_z, k);
//		uz2 = uz*uz;
		for(size_t i=0; i<YSIZE(mapfftV); ++i)
		{
			uy = VEC_ELEM(freq_fourier_y, i);
			//uz2y2 = uz2 + uy*uy;
			uz2y2 = uy*uy;

			for(size_t j=0; j<XSIZE(mapfftV); ++j)
			{

				ux = VEC_ELEM(freq_fourier_x, j);
				double u = sqrt(uz2y2 + ux*ux);
				DIRECT_MULTIDIM_ELEM(freqMap,n) = u;
				++n;
			}
		}
//	}
	Image<double> svImg;
	svImg() = freqMap;
	svImg.write("freqs.mrc");

}



void ProgEstimateSNR::normalizeHalf(MultidimArray<double> &half)
{
	double sumH1=0, sum2H1=0;
	long counter = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(half)
	{
		double val1 = DIRECT_MULTIDIM_ELEM(half, n);

		sumH1 += val1;
		sum2H1 += val1*val1;
		counter++;
	}

	double mean1 = 0;
	double std1 = 0;

	mean1 = sumH1/counter;

	std1 = sqrt(sum2H1/counter - mean1*mean1);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(half)
	{
		DIRECT_MULTIDIM_ELEM(half, n) = (DIRECT_MULTIDIM_ELEM(half, n)-mean1)/std1;
	}
}


void ProgEstimateSNR::computeSNR(MultidimArray<std::complex<double>> &FT1, MultidimArray<std::complex<double>> &FT2, double &SNR, double &FC)
{
	MultidimArray<double> num, den1, den2;

	num.initZeros(xvoldim);
	den1 = num;
	den2 = num;

	double numScalar = 0, denScalar1 = 0, denScalar2 = 0;

	auto ZdimFT1=(int)ZSIZE(FT1);
	auto YdimFT1=(int)YSIZE(FT1);
	auto XdimFT1=(int)XSIZE(FT1);

	long n = 0;
	for (int k=0; k<ZdimFT1; k++)
	{
		for (int i=0; i<YdimFT1; i++)
		{
			for (int j=0; j<XdimFT1; j++)
			{
				double f = DIRECT_MULTIDIM_ELEM(freqMap,n);
				++n;

				if (f>0.5)
					continue;

				// Index of each frequency
				//auto idx = (int) round(f * xvoldim);

				// Fourier coefficients of both halves
				std::complex<double> &z1 = dAkij(FT1, k, i, j);
				std::complex<double> &z2 = dAkij(FT2, k, i, j);

				double absz1 = abs(z1);
				double absz2 = abs(z2);

				numScalar += real(conj(z1) * z2);
				denScalar1 += absz1*absz1;
				denScalar2 += absz2*absz2;
				//dAi(num,idx) += real(conj(z1) * z2);
				//dAi(den1,idx) += absz1*absz1;
				//dAi(den2,idx) += absz2*absz2;
			}
		}
	}

	FC = numScalar/sqrt((denScalar1*denScalar2)+1e-38);

	SNR = FC/(1-FC);
	std::cout <<"FSC= " << FC << "  SNR = " << SNR << std::endl;
}


void ProgEstimateSNR::prepareHalf(MultidimArray<double> &half, FileName &fn)
{
	std::cout << "Reading data..." << std::endl;

	FileName fnhalf;
	Image<double> img;
	auto &auxhalf = img();

	std::cout << "fn = " << fn << std::endl;

	img.read(fn);
	auxhalf.setXmippOrigin();

	if (normalize)
	{
		normalizeHalf(auxhalf);
	}

	half = auxhalf;
	CenterFFT(half, true);

}


void ProgEstimateSNR::computeHalves(FileName &fn1, FileName &fn2, double &FC, double SNR)
{
	MultidimArray<double> half1, half2;

	//This read the data and applies a fftshift
	std::cout << "prepare half 1" << std::endl;
	prepareHalf(half1, fn1);
	std::cout << "prepare half 2" << std::endl;
	prepareHalf(half2, fn2);

	xvoldim = std::min(XSIZE(half1), YSIZE(half1));

	FourierTransformer transformer2(FFTW_BACKWARD), transformer1(FFTW_BACKWARD);
	transformer1.setThreadsNumber(Nthreads);
	transformer2.setThreadsNumber(Nthreads);


	transformer1.FourierTransform(half1, FT1, false);
	transformer2.FourierTransform(half2, FT2, false);

	defineFrequencies(FT1, half1);

	computeSNR(FT1, FT2, SNR, FC);


	half1.clear();
	half2.clear();

	FT2.clear();
}

void ProgEstimateSNR::run()
	{
		std::cout << "Starting ... " << std::endl;

		FileName fn;
		fn = "/SNR.xmd";

		double FC;
		double SNR;

		MetaDataVec md, mdOut;
		if (fnIn1.isMetaData())
		{
			std::cout << "Reading metadata ..." << std::endl;
			md.read(fnIn1);
			size_t idx = 0;
			FileName fnH1, fnH2;
			double tilt;
			for (const auto& row : md)
			{
				idx++;
				row.getValue(MDL_HALF1, fnH1);
				row.getValue(MDL_HALF2, fnH2);
				row.getValue(MDL_ANGLE_TILT, tilt);

				MDRowVec rowOut;


				computeHalves(fnH1, fnH2, FC, SNR);
				rowOut.setValue(MDL_IDX, idx);
				rowOut.setValue(MDL_ANGLE_TILT, tilt);
				rowOut.setValue(MDL_RESOLUTION_FRC, FC);
				rowOut.setValue(MDL_RESOLUTION_SSNR,  SNR);

				mdOut.addRow(rowOut);

			}
		}
		else
		{
			std::cout << "Reading images ..." << std::endl;
			computeHalves(fnIn1, fnIn2, FC, SNR);

			// The global FSC is stored as a metadata
			size_t id;

			id=mdOut.addObject();

			mdOut.setValue(MDL_RESOLUTION_FRC, FC, id);
			mdOut.setValue(MDL_RESOLUTION_SSNR, SNR, id);


		}
		std::cout << "2fnOut+fn " << fnOut+fn << std::endl;

		mdOut.write(fnOut+fn);

		std::cout << "-------------Finished-------------" << std::endl;
}


