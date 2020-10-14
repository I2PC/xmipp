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

#ifndef _PROG_MONOGENIC_SIGNAL_RES
#define _PROG_MONOGENIC_SIGNAL_RES

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
#include <data/monogenic_signal.h>
#include <string>
#include "symmetrize.h"

/**@defgroup Monogenic Resolution
   @ingroup ReconsLibrary */
//@{
/** SSNR parameters. */

class ProgMonogenicSignalRes : public XmippProgram
{
public:
	 /** Filenames */
	FileName fnOut, fnVol, fnVol2, fnMask, fnMaskExl, fnchim, fnSpatial,
	fnMeanVol, fnMaskOut, fnMd;

	/** sampling rate, minimum resolution, and maximum resolution */
	double sampling, minRes, maxRes;

	/** Is the volume previously masked?*/
	long NVoxelsOriginalMask;
	int Nvoxels, nthrs;

	/** Step in digital frequency */
	double freq_step, significance;

	/** The search for resolutions is linear or inverse**/
	bool gaussian, noiseOnlyInHalves;

public:

    void defineParams();
    void readParams();
    void produceSideInfo();

    void excludeArea(MultidimArray<int> &pMask, MultidimArray<int> &pMaskExcl);

    /* Mogonogenid amplitud of a volume, given an input volume,
     * the monogenic amplitud is calculated and low pass filtered at frequency w1*/
    void amplitudeMonogenicSignal3D(MultidimArray< std::complex<double> > &myfftV,
    		double freq, double freqH, double freqL, MultidimArray<double> &amplitude,
    		int count, FileName fnDebug);

    void refiningMask(const MultidimArray< std::complex<double> > &myfftV,
			MultidimArray<double> &iu, int thrs, MultidimArray<int> &pMask);


    void postProcessingLocalResolutions(MultidimArray<double> &FilteredMap,
    		MultidimArray<double> &resolutionVol,
    		std::vector<double> &list, double &cut_value, MultidimArray<int> &pMask);

    void run();

private:
    Image<int> mask, maskExcl;
    MultidimArray<double> iu, VRiesz; // Inverse of the frequency
	MultidimArray< std::complex<double> > fftV, *fftN; // Fourier transform of the input volume
	FourierTransformer transformer_inv;
	MultidimArray< std::complex<double> > fftVRiesz, fftVRiesz_aux;
	FourierFilter lowPassFilter, FilterBand;
	bool halfMapsGiven;
	Image<double> Vfiltered, VresolutionFiltered;
	Matrix1D<double> freq_fourier;
	Matrix1D<double> freq_fourier_x, freq_fourier_y, freq_fourier_z;
	Matrix2D<double> resolutionMatrix, maskMatrix;
};
//@}
#endif
