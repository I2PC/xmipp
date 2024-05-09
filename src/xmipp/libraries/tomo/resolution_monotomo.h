/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
 *             Oier Lauzirika                         olauzirika@cnb.csic.es
 * 			   Carlos Oscar S. Sorzano            coss@cnb.csic.es (2016)
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

#ifndef _PROG_MONOTOMO
#define _PROG_MONOTOMO

#include <iostream>
#include <core/xmipp_program.h>
#include <core/xmipp_image.h>
#include <core/xmipp_fft.h>
#include <core/xmipp_fftw.h>
#include <math.h>
#include <limits>
#include <complex>
#include <data/fourier_filter.h>
#include <data/filters.h>
#include <string>
#include <data/aft.h>
#include <memory>


class ProgMonoTomo : public XmippProgram
{
public:
	 /** Filenames */
	FileName fnOut, fnVol, fnVol2, fnMeanVol;

	/** sampling rate, minimum resolution, and maximum resolution */
	double sampling, minRes, maxRes;

	/** Is the volume previously masked?*/
	int NVoxelsOriginalMask, Nvoxels, nthrs;
	size_t xdimFT, ydimFT, zdimFT, xdim, ydim, zdim;

	/** Step in digital frequency */
	double resStep, significance;

	/** The search for resolutions is linear or inverse**/
	bool noiseOnlyInHalves, automaticMode;

	std::vector<std::complex<float>> fourierSignal, fourierNoise;

public:

    void defineParams();
    void readParams();
    void produceSideInfo();

    /* Mogonogenid amplitud of a volume, given an input volume,
     * the monogenic amplitud is calculated and low pass filtered at frequency w1*/
    void amplitudeMonogenicSignal3D(const std::vector<std::complex<float>> &myfftV, float freq, float freqH, float freqL, MultidimArray<float> &amplitude, int count, FileName fnDebug);

    void firstMonoResEstimation(MultidimArray< std::complex<double> > &myfftV,
    		double freq, double freqH, double freqL, MultidimArray<double> &amplitude,
    		int count, FileName fnDebug, double &mean_Signal,
			double &mean_noise, double &thresholdFirstEstimation);

    void median3x3x3(MultidimArray<double> vol, MultidimArray<double> &filtered);

    //Computes the noise distribution inside a box with size boxsize, of a given map, and determines the percentile 95
    // which is stored in thresholdMatrix.
    void localNoise(MultidimArray<float> &noiseMap, Matrix2D<double> &noiseMatrix, int boxsize, Matrix2D<double> &thresholdMatrix);

    void postProcessingLocalResolutions(MultidimArray<float> &resolutionVol,
    		std::vector<float> &list);

    void resolution2eval(int &count_res, double step,
    								double &resolution, double &last_resolution,
    								double &freq, double &freqL,
    								int &last_fourier_idx,
    								bool &continueIter,	bool &breakIter);

    void smoothBorders(MultidimArray<float> &vol, MultidimArray<int> &pMask);

    void lowestResolutionbyPercentile(MultidimArray<float> &resolutionVol,
    		std::vector<float> &list, float &cut_value, float &resolutionThreshold);

    void getFilteringResolution(size_t idx, float freq, float lastResolution, float freqL, float &resolution);

    void gaussFilter(const MultidimArray<float> &vol, const float, MultidimArray<float> &VRiesz);

    void run();

public:
    Image<int> mask;
	std::vector<std::complex<float>> fftVRiesz, fftVRiesz_aux;
	Matrix2D<double> resolutionMatrix, maskMatrix;
	MultidimArray<float> VRiesz;
	std::unique_ptr<AFT<float>> forward_transformer, backward_transformer;
};
//@}
#endif
