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

#ifndef _PROG_GRAPHENE_ALIGNMENT
#define _PROG_GRAPHENE_ALIGNMENT

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
#include "core/metadata_vec.h"


class ProgGrapheneAlignment : public XmippProgram
{
public:
	 /** Filenames */
	FileName fnOut, fnTs;

	double sampling;

	int nthrs;

	double grapheneParam;


public:
	void readParams();

	void defineParams();

    void produceSideInfo();

    void squareImage(MultidimArray<double> &inImage, MultidimArray<double> &croppedImage);

    void squareImageAndSmoothing(MultidimArray<double> &inImage, MultidimArray<double> &croppedImage, int N_smoothing);

    void getFourierShell(MultidimArray<std::complex<double>> &FTtiltImage, MultidimArray<std::complex<double>> &extractedShell, std::vector<long> &freqIdx, std::vector<double> &anglesVector);

    void indicesFourierShell(MultidimArray<std::complex<double>> &FTimg, MultidimArray<double> &tiltImage, std::vector<long> &freqIdx, std::vector<double> &anglesVector);

    void smoothBorders(MultidimArray<double> &img, int N_smoothing);

    void readInputData(MetaDataVec &mdts);

    void run();
};
//@}
#endif
