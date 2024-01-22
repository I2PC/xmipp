/***************************************************************************
 *
 * Authors:    Jose Luis Vilas,                                           jlvilas@cnb.csic.es
 *
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

#ifndef _PROG_TOMO_LOCAL_FILTER
#define _PROG_TOMO_LOCAL_FILTER

#include <iostream>
#include <core/xmipp_program.h>
#include <core/xmipp_image.h>
#include <core/xmipp_hdf5.h>
#include <core/xmipp_fft.h>
#include <core/xmipp_fftw.h>
#include <limits>
#include <complex>
#include <data/fourier_filter.h>
#include <data/filters.h>
#include <data/aft.h>
#include <string>
#include <memory>

class ProgTomoLocalFilter : public XmippProgram
{
public:
	 /** Filenames */
	FileName fnOut, fnVol, fnRes;

	size_t xdimFT, ydimFT, zdimFT;

	/** sampling rate, minimum resolution, and maximum resolution */
	float sampling, maxRes, minRes, K, maxFreq, minFreq, desv_Vorig, desvOutside_Vorig;
	int Nthread;
    std::vector<size_t> idxMap;
    std::vector<float> freqTomo;

    std::vector<std::complex<float>> fourierData;

public:

    void defineParams();
    void readParams();
    void produceSideInfo();

    /* Mogonogenid amplitud of a volume, given an input volume,
     * the monogenic amplitud is calculated and low pass filtered at frequency w1*/
    void lowPassFilterFunction(const MultidimArray< std::complex<double> > &myfftV,
    		double w, double wL, MultidimArray<double> &filteredVol, int count);

    void bandPassFilterFunction(const std::vector<std::complex<float>> &fourierData,
            float w, float wL, MultidimArray<float> &filteredVol, int count);

    void wideBandPassFilter(const MultidimArray< std::complex<double> > &myfftV,
                    double wmin, double wmax, double wL, MultidimArray<double> &filteredVol);

      void maxMinResolution(MultidimArray<float> &resVol,
			float &maxRes, float &minRes);

      void computeAvgStdev_within_binary_mask(const MultidimArray< float >&resVol,
      										const MultidimArray< float >&vol, float &stddev, bool outside=false );

    void localfiltering(std::vector<std::complex<float>>  &myfftV, MultidimArray<float> &localfilteredVol, float minRes, float maxRes, float step);


    void sameEnergy(MultidimArray< std::complex<double> > &myfftV,
			MultidimArray<double> &localfilteredVol,
			double &minFreq, double &maxFreq, double &step);

    void run();

public:
    MultidimArray<float> resVol;//, VsoftMask;
    MultidimArray<int> mask;
    MultidimArray<double> freqMap, sharpenedMap; // Inverse of the frequency
	MultidimArray< std::complex<double> > fftV, fftVfilter; // Fourier transform of the input volume
	FourierTransformer transformer, transformer_inv;
	std::unique_ptr<AFT<float>> forward_transformer, backward_transformer;
	FourierFilter FilterBand;
};
//@}
#endif
