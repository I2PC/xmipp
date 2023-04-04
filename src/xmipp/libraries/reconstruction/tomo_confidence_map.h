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
#include <core/metadata_vec.h>
#include <math.h>
#include <limits>
#include <complex>
#include <data/fourier_filter.h>
#include <data/filters.h>
#include <string>
#include "core/linear_system_helper.h"
#include "fftwT.h"

/**@defgroup Monogenic Resolution
   @ingroup ReconsLibrary */
//@{
/** SSNR parameters. */

class ProgTomoConfidecenceMap : public XmippProgram
{
public:
	 /** Filenames */
	FileName fnOut;
	FileName fnTs;
	FileName fnOdd;
	FileName fnEven;
	FileName fnMask;

	bool locRes, medianFilterBool, applySmoothingBeforeConfidence, applySmoothingAfterConfidence;

    size_t Xdim, Ydim, Zdim;

	/** sampling rate, minimum resolution, and maximum resolution */
	float sampling;
	float lowRes;
	float highRes;
	float sigVal;
	float fdr;
	float step;
	float sigmaGauss;

	/** Is the volume previously masked?*/
	int  nthrs, boxsize;

public:

    void defineParams();
    
	void readParams();

	void confidenceMap(MultidimArray<float> &ignificanceMap, bool normalize, MultidimArray<float> &fullMap, MultidimArray<float> &noiseMap);

	void sortImages(MetaDataVec &md, std::vector<FileName> &vecfn, std::vector<double> &vectilt);

    void normalizeTomogram(MultidimArray<float> &fullMap, MultidimArray<float> &noiseVarianceMap, MultidimArray<float> &noiseMeanMap);

	void nosiseEstimation(WeightedLeastSquaresHelper &helperStd, WeightedLeastSquaresHelper &helperMean, 
												int lX, int lY, double hX, double hY, int Nx, int Ny, int boxsize, Matrix2D<double> &noiseStdMatrix, Matrix2D<double> &noiseMeanMatrix);

    void estimateNoiseStatistics(MultidimArray<float> &noiseMap, 
													 MultidimArray<float> &noiseVarianceMap, MultidimArray<float> &noiseMeanMap,
													 int boxsize, Matrix2D<float> &thresholdMatrix_mean, Matrix2D<float> &thresholdMatrix_std);

	void medianFilter(MultidimArray<float> &input_tomogram,
									       MultidimArray<float> &output_tomogram);

	template<typename T>
	std::vector<size_t> sort_indexes(const std::vector<T> &v);

	void computeSignificanceMap(MultidimArray<float> &fullMap, MultidimArray<float> &significanceMap,
													 Matrix2D<float> &thresholdMatrix_mean, Matrix2D<float> &thresholdMatrix_std);

	void defineFourierFilter(MultidimArray<std::complex<double>> &mapfftV);

	void convertToDouble(MultidimArray<float> &inTomo, MultidimArray<double> &outTomo);

	void convertToFloat(MultidimArray<double> &inTomo, MultidimArray<float> &outTomo);

	void nyquistFilter(MultidimArray<std::complex<double>> &fftImg);

	void medianFilter3D(MultidimArray<float> &input_tomogram,
									       MultidimArray<float> &output_tomogram);

	void medianFilter2D(MultidimArray<float> &input_img,
											MultidimArray<float> &output_img);

    void run();

public:
	FFTwT<float> transformerFT;
    //fftwf_plan plan;
    Image<int> mask;
	Matrix1D<float> freq_fourier_z, freq_fourier_y, freq_fourier_x;
	MultidimArray< float > fullMap, noiseMap, resMap;
	MultidimArray<double> fourierFilterShape;
};
//@}
#endif
