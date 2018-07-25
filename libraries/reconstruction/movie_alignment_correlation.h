/***************************************************************************
 *
 * Authors:    Carlos Oscar Sanchez Sorzano coss@cnb.csic.es
 *             David Strelak (davidstrelak@gmail.com)
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

#ifndef _PROG_MOVIE_ALIGNMENT_CORRELATION
#define _PROG_MOVIE_ALIGNMENT_CORRELATION

#include "data/filters.h"
#include "core/xmipp_fftw.h"
#include "reconstruction/movie_alignment_correlation_base.h"

/** Movie alignment correlation Parameters. */
template<typename T>
class ProgMovieAlignmentCorrelation: public AProgMovieAlignmentCorrelation<T> {

private:
    /**
     * After running this method, all relevant images from the movie are
     * loaded in 'frameFourier' and ready for further processing
     * @param movie input
     * @param dark correction to be used
     * @param gain correction to be used
     * @param targetOccupancy max frequency to be preserved in FT
     * @param lpf 1D profile of the low-pass filter
     */
    void loadData(const MetaData& movie, const Image<T>& dark,
            const Image<T>& gain, T targetOccupancy,
            const MultidimArray<T>& lpf);

    /**
     * Computes shifts of all images in the 'frameFourier'
     * Method uses one thread to calculate correlations.
     * @param N number of images to process
     * @param bX pair-wise shifts in X dimension
     * @param bY pair-wise shifts in Y dimension
     * @param A system matrix to be used
     */
    void computeShifts(size_t N, const Matrix1D<T>& bX, const Matrix1D<T>& bY,
            const Matrix2D<T>& A);

    /**
     * This method applies shifts stored in the metadata and computes 'average'
     * image
     * @param movie input
     * @param dark correction to be used
     * @param gain correction to be used
     * @param initialMic sum of the unaligned micrographs
     * @param Ninitial will store number of micrographs used for unaligned sum
     * @param averageMicrograph sum of the aligned micrographs
     * @param N will store number of micrographs used for aligned sum
     */
    void applyShiftsComputeAverage(const MetaData& movie, const Image<T>& dark,
            const Image<T>& gain, Image<T>& initialMic, size_t& Ninitial,
            Image<T>& averageMicrograph, size_t& N);
private:
    /**
     *  Fourier transforms of the input images, after cropping, gain and dark
     *  correction
     */
    std::vector<MultidimArray<std::complex<T> > *> frameFourier;
};

#endif
