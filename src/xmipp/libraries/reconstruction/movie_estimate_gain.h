/***************************************************************************
 *
 * Authors:        Victoria Peredo
 *                 Estrella Fernandez
 *                 Carlos Oscar S. Sorzano (coss@cnb.csic.es)
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
#ifndef _PROG_ESTIMATE_GAIN_HH
#define _PROG_ESTIMATE_GAIN_HH

#include "core/xmipp_program.h"
#include "core/xmipp_image.h"
#include "core/metadata.h"

/**@defgroup EstimateGainProgram Estimate gain from a movie
   @ingroup ReconsLibrary */
//@{

double computeTVColumns(MultidimArray<double> &I);
double computeTVRows(MultidimArray<double> &I);


class ProgMovieEstimateGain: public XmippProgram
{
public:
	FileName fnIn; // Set of input images
	FileName fnRoot; // Correction image
    FileName fnCorrected; // Corrected movie
	int Niter; // Number of iterations
	double sigma, maxSigma, sigmaStep;
	bool singleReference;
	int frameStep;
	FileName fnGain;
	bool applyGain;
public:
    void defineParams();
    void readParams();
    void show();
    void run();

    void produceSideInfo();
    void computeHistograms(const MultidimArray<double> &Iframe);
    void normalizeHistograms();
    void invertHistograms();

    void constructSmoothHistogramsByColumn(const double *listOfWeights, int width);
    void constructSmoothHistogramsByRow(const double *listOfWeights, int width);
    void transformGrayValuesColumn(const MultidimArray<double> &Iframe, MultidimArray<double> &IframeTransformedColumn);
    void transformGrayValuesRow(const MultidimArray<double> &Iframe, MultidimArray<double> &IframeTransformedRow);
    void computeTransformedHistograms(const MultidimArray<double> &Iframe);

    size_t selectBestSigmaByColumn(const MultidimArray<double> &Iframe);
    size_t selectBestSigmaByRow(const MultidimArray<double> &Iframe);



public:
	MetaData mdIn;
	MultidimArray<double> columnH,rowH, aSingleColumnH, aSingleRowH;
	MultidimArray<double> smoothColumnH, smoothRowH, sumObs;
	Image<double> ICorrection;

	std::vector<double> listOfSigmas;
	std::vector<double> listOfWidths;
	std::vector<double *> listOfWeights;
	int Xdim, Ydim;
};

//@}
#endif
