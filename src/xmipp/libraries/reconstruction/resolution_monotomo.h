/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
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
#include <core/metadata.h>
#include <core/xmipp_fft.h>
#include <core/xmipp_fftw.h>
#include <math.h>
#include <limits>
#include <complex>
#include <data/fourier_filter.h>
#include <data/filters.h>
#include <string>
#include "symmetrize.h"

/**@defgroup Monogenic Resolution
   @ingroup ReconsLibrary */
//@{
/** SSNR parameters. */

class ProgMonoTomo : public XmippProgram
{
public:
	 /** Filenames */
	FileName fnOut, fnVol, fnVol2, fnMask, fnchim, fnSpatial,
	fnMeanVol, fnMaskOut, fnMd, fnFilt, fnmaskWedge;

	/** sampling rate, minimum resolution, and maximum resolution */
	double sampling, minRes, maxRes, R;

	/** Is the volume previously masked?*/
	int NVoxelsOriginalMask, Nvoxels, nthrs;

	/** Step in digital frequency */
	double freq_step, trimBound, significance;

	/** The search for resolutions is linear or inverse**/
	bool noiseOnlyInHalves, automaticMode;

public:

    void defineParams();
    void readParams();
    void produceSideInfo();

    /* Mogonogenid amplitud of a volume, given an input volume,
     * the monogenic amplitud is calculated and low pass filtered at frequency w1*/
    void amplitudeMonogenicSignal3D(MultidimArray< std::complex<double> > &myfftV,
    		double freq, double freqH, double freqL, MultidimArray<double> &amplitude,
    		int count, FileName fnDebug);
    void firstMonoResEstimation(MultidimArray< std::complex<double> > &myfftV,
    		double freq, double freqH, double freqL, MultidimArray<double> &amplitude,
    		int count, FileName fnDebug, double &mean_Signal,
			double &mean_noise, double &thresholdFirstEstimation);

    void median3x3x3(MultidimArray<double> vol, MultidimArray<double> &filtered);

    //Computes the noise distribution inside a box with size boxsize, of a given map, and determines the percentile 95
    // which is stored in thresholdMatrix.
    void localNoise(MultidimArray<double> &noiseMap, Matrix2D<double> &noiseMatrix, int boxsize, Matrix2D<double> &thresholdMatrix);

    void postProcessingLocalResolutions(MultidimArray<double> &resolutionVol,
    		std::vector<double> &list, MultidimArray<double> &resolutionChimera,
    		double &cut_value, MultidimArray<int> &pMask, double &resolutionThreshold);

    void resolution2eval(int &count_res, double step,
    								double &resolution, double &last_resolution,
    								double &freq, double &freqL,
    								int &last_fourier_idx,
    								bool &continueIter,	bool &breakIter,
    								bool &doNextIteration);

    void lowestResolutionbyPercentile(MultidimArray<double> &resolutionVol,
    								std::vector<double> &list,	double &cut_value, double &resolutionThreshold);

    void run();

public:
    Image<int> mask;
    MultidimArray<double> iu, VRiesz; // Inverse of the frequency
	MultidimArray< std::complex<double> > fftV, *fftN; // Fourier transform of the input volume
	FourierTransformer transformer_inv;
	MultidimArray< std::complex<double> > fftVRiesz, fftVRiesz_aux;
	FourierFilter lowPassFilter, FilterBand;
	bool halfMapsGiven;
	Image<double> Vfiltered, VresolutionFiltered;
	Matrix1D<double> freq_fourier_z, freq_fourier_y, freq_fourier_x;
	Matrix2D<double> resolutionMatrix, maskMatrix;
};
//@}
#endif
