/***************************************************************************
 *
 * Authors:    Jose Luis Vilas (joseluis.vilas-prieto@yale.edu)
 *                             or (jlvilas@cnb.csic.es)
 *              Hemant. D. Tagare (hemant.tagare@yale.edu)
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

#ifndef _PROG_STA_DEBLUR
#define _PROG_STA_DEBLUR

#include <core/xmipp_program.h>
#include <core/matrix2d.h>
#include <core/xmipp_fftw.h>
#include <core/xmipp_filename.h>
#include <core/metadata_vec.h>
#include <data/fourier_filter.h>


class ProgSTADeblurring : public XmippProgram
{
private:
    // Filenames
    FileName fnSubtomos, fnRef, fnOut, fnMask;

    // Double Params
    double sampling, thrs, fscRes, pRecon;

    // Int params
    int Nthreads, niters;

    bool isStack = false;
       

    // Int params
    size_t xvoldim, yvoldim, zvoldim, fscshellNum;

	// Frequency vectors and frequency map
    MultidimArray<double> num, den1, den2, fsc, SNR, noisePower, mask;
    MultidimArray<double> refMap;
    size_t shelElems;

    FourierTransformer transformer;

	Matrix1D<double> freq_fourier_x;
	Matrix1D<double> freq_fourier_y;
	Matrix1D<double> freq_fourier_z;
    
    MultidimArray<double> freqMap;

	// Half maps
	MultidimArray< std::complex< double > > FTsubtomo, FTref;

	//Access indices
	MultidimArray<long> freqElems, cumpos, freqidx, arr2indx;


public:
        template<typename T>
        void createMask(MultidimArray<T> &vol, MultidimArray<T> &mask);

        void defineFrequencies(const MultidimArray< std::complex<double> > &mapfftV, const MultidimArray<double> &inputVol, MultidimArray<double> &freqMapIdx);

        void phaseCorrelationStep(MultidimArray<double> &refMap, MultidimArray<std::complex<double>> &FTref, MultidimArray<double> &mask);

        void weightsRealAndFourier(MultidimArray<double> &map, MetaDataVec &mdSubtomos, std::vector<double> &meanPhaseCorr,
											std::vector<double> &allCorrReal, size_t &Nsubtomos);

        //void normalizeSubtomos(MultidimArray<double> &refMap, MultidimArray<std::complex<double>> &FTref, MultidimArray<double> &mask);

        void rankingSubtomograms(std::vector<double> &meanPhaseCorr, 
											std::vector<double> &allCorrReal, std::vector<double> &wSubtomo);

        void weightedAverage(MetaDataVec &mdSubtomos, std::vector<double> &weightsVector, 
										std::vector<double> &allCorrReal, std::vector<double> &meanPhaseCorr, MultidimArray<double> &wAvg);

        //void detectMissingWedge(MultidimArray<double> &subtomo, bool fromAlignment);

        void applyTransformationMatrix(IMultidimArray<double> &subtomoOrig, MultidimArray<double> &subtomoRotated, Matrix2D<double> &eulermatrix,
												  MetaDataVecRow &mdAligned);

        void missingWedgeDetection(MultidimArray<double> &myfft, MultidimArray<double> &thresholdMissingWedge);

        void corr3(MultidimArray<double> &vol1, MultidimArray<double> &vol2, double &corrVal);

        template<typename T>
        void createReference(MultidimArray<T> &refMap);

        void defineReference(MultidimArray<double> &ptrRef);

        void generateProjections(FileName &fnVol, double &sampling_rate);

        void normalizeReference(MultidimArray<double> &map, MultidimArray<std::complex<double>> &FTmap);

        /* Defining the params and help of the algorithm */
        void defineParams();

        /* It reads the input parameters */
        void readParams();

        /* Run the program */
        void run();
};

#endif
