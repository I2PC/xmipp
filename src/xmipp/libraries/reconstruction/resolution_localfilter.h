/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
 * 			   Carlos Oscar S. Sorzano            coss@cnb.csic.es (2018)
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

#ifndef _PROG_RESLOCALFILTER
#define _PROG_RESLOCALFILTER

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

/**@defgroup Resolution Local Filter
   @ingroup ReconsLibrary */
//@{
/** SSNR parameters. */

class ProgResLocalFilter : public XmippProgram
{
public:
	 /** Filenames */
	FileName fnOut, fnVol, fnRes, fnMask, fnchim, fnSpatial,
	fnMeanVol, fnMaskOut, fnMd, fnFilt;

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

    void postProcessingLocalResolutions(MultidimArray<double> &resolutionVol,
    		std::vector<double> &list, MultidimArray<double> &resolutionChimera,
    		double &cut_value, MultidimArray<int> &pMask, double &resolutionThreshold);

    void resolution2eval(int &count_res, double step,
    								double &resolution, double &last_resolution,
    								double &freq, double &freqL,
    								int &last_fourier_idx,
    								bool &continueIter,	bool &breakIter,
    								bool &doNextIteration);

    void run();

public:
    MultidimArray<double> iu; // Inverse of the frequency
	MultidimArray< std::complex<double> > fftV; // Fourier transform of the input volume
	FourierTransformer transformer_inv;
	Image<double> Vfiltered, VresolutionFiltered, resVol;
	Matrix1D<double> freq_fourier;
	double sigma;
};
//@}
#endif
