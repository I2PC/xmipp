/***************************************************************************
 *
 * Authors:     Jose Luis Vilas (jlvilas@cnb.csic.es)
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
#ifndef _MONOGENIC_HH
#define _MONOGENIC_HH

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <core/metadata.h>
#include <core/xmipp_image.h>
#include <data/sampling.h>
#include <core/xmipp_fft.h>
#include <core/xmipp_fftw.h>
#include <math.h>
#include <limits>
#include <complex>
#include <data/fourier_filter.h>
#include <data/filters.h>
//@{
/** Routines for working with monogenic signals
*/

class Monogenic
{
public:
	//Next set of functions are addressed to generate data for the unitary tests
	MultidimArray<double> createDataTest(size_t xdim, size_t ydim, size_t zdim, double wavelength,
			double mean, double sigma);

	MultidimArray<int> createMask(MultidimArray<double> &vol, int radius);

	void applyMask(MultidimArray<double> &vol, MultidimArray<int> &mask);

	MultidimArray< std::complex<double> > applyMaskFourier(MultidimArray< std::complex<double> > &vol, MultidimArray<double> &mask);

	void addNoise(MultidimArray<double> &vol, double mean, double stddev);

	//This function determines a multidimArray with the frequency values in Fourier space;
	MultidimArray<double> fourierFreqs_3D(const MultidimArray< std::complex<double> > &myfftV,
			const MultidimArray<double> &inputVol,
			Matrix1D<double> &freq_fourier_x,
			Matrix1D<double> &freq_fourier_y,
			Matrix1D<double> &freq_fourier_z);

	//This function determines a multidimArray with the frequency values in Fourier space;
	MultidimArray<double> fourierFreqs_2D(const MultidimArray< std::complex<double> > &myfftImg,
			const MultidimArray<double> &inputImg);

	//It computes the monogenic amplitude of an input volume;
	void monogenicAmplitude_3D(const MultidimArray<double> &inputVol, MultidimArray<double> &amplitude, int numberOfThreads);

	double averageInMultidimArray(const MultidimArray<double> &vol, MultidimArray<int> &mask);

	void statisticsInBinaryMask(const MultidimArray<double> &vol, MultidimArray<int> &mask, double &mean, double &sd);

	void statisticsInBinaryMask(const MultidimArray<double> &volS, const MultidimArray<double> &volN,
			MultidimArray<int> &mask, MultidimArray<int> &maskExcl, double &meanS, double &sdS,
			double &meanN, double &sdN, double &significance, double &thr95, double &NS, double &NN);

	void statisticsInOutBinaryMask(const MultidimArray<double> &volS,
			MultidimArray<int> &mask, MultidimArray<int> &maskExcl, double &meanS, double &sdS,
			double &meanN, double &sdN, double &significance, double &thr95, double &NS, double &NN);

	void monogenicAmplitude_3D_Fourier(const MultidimArray< std::complex<double> > &myfftV,
			MultidimArray<double> iu, MultidimArray<double> &amplitude, int numberOfThreads);

	void setLocalResolutionMap(const MultidimArray<double> &amplitudeMS,
			MultidimArray<int> &pMask, MultidimArray<double> &plocalResolutionMap,
			double &thresholdNoise, double &resolution, double &resolution_2);

	void setLocalResolutionMapAndFilter(const MultidimArray<double> &amplitudeMS,
			MultidimArray<int> &pMask, MultidimArray<double> &plocalResolutionMap,
			MultidimArray<double> &filteredMap, MultidimArray<double> &resolutionFiltered,
			double &thresholdNoise, double &resolution, double &resolution_2);

	void resolution2evalDir(int &fourier_idx, double min_step, double sampling, int volsize,
			double &resolution, double &last_resolution,
			int &last_fourier_idx,
			double &freq, double &freqL, double &freqH,
			bool &continueIter, bool &breakIter, bool &doNextIteration);

	void resolution2eval(int &count_res, double step,
			double &resolution, double &last_resolution,
			double &freq, double &freqL,
			int &last_fourier_idx,
			int &volsize,
			bool &continueIter,	bool &breakIter,
			double &sampling, double &minRes, double &maxRes,
			bool &doNextIteration, bool &automaticMode);

	void proteinRadiusVolumeAndShellStatistics(MultidimArray<int> &mask, double &radius,
			int &vol, MultidimArray<double> &radMap);
	void findCliffValue(MultidimArray<double> radMap, MultidimArray<double> &inputmap,
			double &radius,	double &radiuslimit, MultidimArray<int> &mask);

	bool TestmonogenicAmplitude_3D_Fourier();

	void amplitudeMonoSigDir3D_LPF(const MultidimArray< std::complex<double> > &myfftV,
			FourierTransformer &transformer_inv,
			MultidimArray< std::complex<double> > &fftVRiesz,
			MultidimArray< std::complex<double> > &fftVRiesz_aux, MultidimArray<double> &VRiesz,
			double freq, double freqH, double freqL, MultidimArray<double> &iu,
			Matrix1D<double> &freq_fourier_x, Matrix1D<double> &freq_fourier_y,
			Matrix1D<double> &freq_fourier_z, MultidimArray<double> &amplitude,
			int count, int dir, FileName fnDebug, int N_smoothing);

	void amplitudeMonoSig3D_LPF(const MultidimArray< std::complex<double> > &myfftV,
			FourierTransformer &transformer_inv,
			MultidimArray< std::complex<double> > &fftVRiesz,
			MultidimArray< std::complex<double> > &fftVRiesz_aux, MultidimArray<double> &VRiesz,
			double freq, double freqH, double freqL, MultidimArray<double> &iu,
			Matrix1D<double> &freq_fourier_x, Matrix1D<double> &freq_fourier_y,
			Matrix1D<double> &freq_fourier_z, MultidimArray<double> &amplitude,
			int count, FileName fnDebug);

	//Fast method: It computes the monogenic amplitude of an input volume;
	void monogenicAmplitude_3D(FourierTransformer &transformer, MultidimArray<double> &iu,
			Matrix1D<double> &freq_fourier_z, Matrix1D<double> &freq_fourier_y, Matrix1D<double> &freq_fourier_x,
			MultidimArray< std::complex<double> > &fftVRiesz, MultidimArray< std::complex<double> > &fftVRiesz_aux,
			MultidimArray<double> &VRiesz, const MultidimArray<double> &inputVol,
			MultidimArray< std::complex<double> > &myfftV, MultidimArray<double> &amplitude);

	//It computes the monogenic amplitude of an input image;
	void monogenicAmplitude_2D(const MultidimArray<double> &inputVol, MultidimArray<double> &amplitude, int numberOfThreads);

	//Fast method: It computes the monogenic amplitude of an input image;
	void monogenicAmplitude_2D(FourierTransformer &transformer, MultidimArray<double> &iu,
			Matrix1D<double> &freq_fourier_y, Matrix1D<double> &freq_fourier_x,
			MultidimArray< std::complex<double> > &fftVRiesz, MultidimArray< std::complex<double> > &fftVRiesz_aux,
			MultidimArray<double> &VRiesz, const MultidimArray<double> &inputVol,
			MultidimArray< std::complex<double> > &myfftV, MultidimArray<double> &amplitude);

	//It computes the monogenic amplitud of a HPF image
//	void monogenicAmplitude_2DHPF(FourierTransformer &transformer, MultidimArray<double> &iu,
//			double freqH, double freq,
//			Matrix1D<double> &freq_fourier_y, Matrix1D<double> &freq_fourier_x,
//			MultidimArray< std::complex<double> > &fftVRiesz,
//			MultidimArray< std::complex<double> > &fftVRiesz_aux,
//			MultidimArray<double> &VRiesz,
//			const MultidimArray<double> &inputImg,
//			MultidimArray< std::complex<double> > &myfftImg,
//			MultidimArray<double> &amplitude);
	void monogenicAmplitude_2DHPF(FourierTransformer &transformer, MultidimArray<double> &iu,
			double freqH, double freq, double freqL,
			Matrix1D<double> &freq_fourier_y, Matrix1D<double> &freq_fourier_x,
			MultidimArray< std::complex<double> > &fftVRiesz,
			MultidimArray< std::complex<double> > &fftVRiesz_aux,
			MultidimArray<double> &VRiesz,
			const MultidimArray<double> &inputImg,
			MultidimArray< std::complex<double> > &myfftImg,
			MultidimArray<double> &amplitude, int count, FileName fnDebug);
};

//@}
#endif
