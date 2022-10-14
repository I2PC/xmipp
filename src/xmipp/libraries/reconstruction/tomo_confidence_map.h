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

#ifndef _PROG_TOMO_CONFIDENCE_MAP
#define _PROG_TOMO_CONFIDENCE_MAP

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
#include "fftwT.h"

/**@defgroup Monogenic Resolution
   @ingroup ReconsLibrary */
//@{
/** SSNR parameters. */

class ProgTomoConfidecenceMap : public XmippProgram
{
public:
	 /** Filenames */
	FileName fnOut, fnVol, fnVol2, fnMask;

	bool locRes;

    size_t Xdim, Ydim, Zdim;

	/** sampling rate, minimum resolution, and maximum resolution */
	float sampling, lowRes, highRes, sigVal, fdr, step;

	/** Is the volume previously masked?*/
	int  nthrs;

public:

    void defineParams();
    void readParams();
    void readAndPrepareData();

	void confidenceMap(MultidimArray<float> &ignificanceMap, bool normalize);

    void normalizeTomogram(MultidimArray<float> &fullMap, MultidimArray<float> &noiseVarianceMap, MultidimArray<float> &noiseMeanMap);

    void estimateNoiseStatistics(const MultidimArray<float> &noiseMap, 
													 MultidimArray<float> &noiseVarianceMap, MultidimArray<float> &noiseMeanMap,
													 int boxsize, Matrix2D<float> &thresholdMatrix_mean, Matrix2D<float> &thresholdMatrix_std);

    void FDRcorrection();

	void ampMS(float &resolution, float &freq);


	template<typename T>
	std::vector<size_t> sort_indexes(const std::vector<T> &v);

	void computeSignificanceMap(MultidimArray<float> &fullMap, MultidimArray<float> &significanceMap,
													 Matrix2D<float> &thresholdMatrix_mean, Matrix2D<float> &thresholdMatrix_std);
    
	void amplitudeMonogenicSignal_float(MultidimArray<float> &significanceMap);

	void estimateLocalResolution(MultidimArray<float> &significanceMap);

	void updateResMap(MultidimArray<float> &resMap, MultidimArray<float> &significanceMap, float &resolution);

	void FDRcontrol(MultidimArray<float> &significanceMap);

	void filterNoiseAndMap(float &freq, float &tail, MultidimArray<double> &fm, MultidimArray<double> &nm);

	void chessBoardInterpolation(MultidimArray<float> &tomo);

    void run();

public:
	FFTwT<float> transformerFT;
    fftwf_plan plan;
    Image<int> mask;
	MultidimArray< float > fullMap, noiseMap;
};
//@}
#endif
