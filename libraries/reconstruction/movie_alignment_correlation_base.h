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

/**
 * This class defines functionality common to all algorithms that are
 * aligning movies using cross-correlation.
 * Generally speaking, input images are loaded to memory (after gain and dark
 * correction), typically down-scaled, then cross-correlation between each
 * pair of the images is calculated. From these correlations, relative
 * positions are determined, and using least-square method the absolute shift
 * of each image is calculated.
 * Specific implementations should implement its virtual methods, while using
 * the common functionality.
 */
#ifndef _PROG_MOVIE_ALIGNMENT_CORRELATION_BASE
#define _PROG_MOVIE_ALIGNMENT_CORRELATION_BASE

#include "core/xmipp_program.h"
#include "core/metadata_extension.h"
#include "core/xmipp_fftw.h"
#include "data/point2D.h"
#include "data/point3D.h"
#include "data/rectangle.h"
#include "core/optional.h"

template<typename T>
struct AlignmentResult {
    size_t refFrame;
    // these are shifts from the reference frame in X/Y dimension,
    // i.e. if you want to compensate for the shift,
    // you have to shift in opposite direction (negate these values)
    std::vector<Point2D<T>> shifts;
};

template<typename T>
using FramePatch = Rectangle<Point3D<T>>;

template<typename T>
struct LocalAlignmentResult {
    AlignmentResult<T> globalHint;
    // these are shifts from the reference frame in X/Y dimension,
    // i.e. if you want to compensate for the shift,
    // you have to shift in opposite direction (negate these values)
    std::vector<std::pair<FramePatch<T>, Point2D<T>>> shifts;
};

template<typename T>
class AProgMovieAlignmentCorrelation: public XmippProgram {

    /** wrapping strategy constants */
#define OUTSIDE_WRAP 0
#define OUTSIDE_AVG 1
#define OUTSIDE_VALUE 2

public:
    /// Read argument from command line
    virtual void readParams();

    /// Show
    virtual void show();

    /// Define parameters
    virtual void defineParams();

    /// Run
    void run();

protected:
    /**
     * Method will create a 2D Low-Pass Filter from the 1D
     * profile, that can be used in Fourier domain
     * @param lpf 1D profile
     * @param xSize size of full image (space domain)
     * @param ySize size of full image (space/frequency domain)
     * @param targetOccupancy maximal frequency to be preserved
     * @param result resulting 2D filter. Must be of proper size, i.e.
     * xdim == xSize/2+1, ydim = ySize
     */
    void scaleLPF(const MultidimArray<T>& lpf, int xSize, int ySize,
            T targetOccupancy, MultidimArray<T>& result);

    /**
     * Method finds a reference image, i.e. an image which has smallest relative
     * shift to all other images.
     * @param N no of images
     * @param shiftX relative X shift of each image
     * @param shiftY relative Y shift of each image
     */
    int findReferenceImage(size_t N, const Matrix1D<T>& shiftX,
            const Matrix1D<T>& shiftY);

    /**
     * Method computes absolute shifts from relative shifts
     * @param bX relative shifts in X dim
     * @param bY relative shifts in Y dim
     * @param A system matrix to be used
     * @param shiftX absolute shifts in X dim
     * @param shiftY absolute shifts in Y dim
     */
    void solveEquationSystem(Matrix1D<T>& bX, Matrix1D<T>& bY, Matrix2D<T>& A,
            Matrix1D<T>& shiftX, Matrix1D<T>& shiftY);

    /**
     * Method to compute sum of shifts of some image in respect to a reference
     * image
     * @param iref index of the reference image
     * @param j index of the queried image
     * @param shiftX relative shifts in X dim
     * @param shiftY relative shifts in Y dim
     * @param totalShiftX resulting shift in X dim
     * @param totalShiftY resulting shift in Y dim
     */
    void computeTotalShift(int iref, int j, const Matrix1D<T> &shiftX,
            const Matrix1D<T> &shiftY, T &totalShiftX, T &totalShiftY);

    /**
     * Method computes an internal (down)scale factor of the micrographs
     * @param targetOccupancy max frequency (in Fourier domain) to preserve
     */
    T computeSizeFactor();

    MultidimArray<T> createLPF(T targetOccupancy, size_t xSize, size_t xFFTSize,
            size_t ySize);

    /**
     * Method to store relative shifts computed for the movie
     * @param alignment result to store
     * @param movie to be updated
     */
    void storeGlobalShifts(const AlignmentResult<T> &alignment,
            MetaData &movie);

private:
    /**
     * After running this method, all relevant images from the movie should
     * be loaded and ready for further processing
     * @param movie input
     * @param dark correction to be used
     * @param gain correction to be used
     * @param targetOccupancy max frequency to be preserved in FT
     * @param lpf 1D profile of the low-pass filter
     */
    virtual void loadData(const MetaData& movie, const Image<T>& dark,
            const Image<T>& gain, T targetOccupancy,
            const MultidimArray<T>& lpf) = 0;

    /**
     * After running this method, shifts (pair-wise) of all images loaded in
     * the 'loadData' * method should be determined
     * @param N number of images to process
     * @param bX pair-wise shifts in X dimension
     * @param bY pair-wise shifts in Y dimension
     * @param A system matrix to be filled
     */
    virtual void computeShifts(size_t N, const Matrix1D<T>& bX,
            const Matrix1D<T>& bY, const Matrix2D<T>& A) = 0;

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
    virtual void applyShiftsComputeAverage(const MetaData& movie,
            const Image<T>& dark, const Image<T>& gain, Image<T>& initialMic,
            size_t& Ninitial, Image<T>& averageMicrograph, size_t& N) = 0;

    virtual AlignmentResult<T> computeGlobalAlignment(const MetaData &movie,
            const Image<T> &dark, const Image<T> &gain) = 0;

    virtual LocalAlignmentResult<T> computeLocalAlignment(const MetaData &movie,
            const Image<T> &dark, const Image<T> &gain) = 0;

private:
    /**
     * Method computes an internal (down)scale factor of the micrographs
     * @param targetOccupancy max frequency (in Fourier domain) to preserve
     */
    void computeSizeFactor(T& targetOccupancy);



    /**
     * Method loads dark correction image
     * @param dark correction will be stored here
     */
    void loadDarkCorrection(Image<T>& dark);

    /**
     *  Method loads gain correction image
     *  @param gain correction will be stored here
     */
    void loadGainCorrection(Image<T>& gain);

    /**
     * Method to construct 1D low-pass filter profile
     * @param targetOccupancy max frequency to preserve
     * @param lpf filter will be stored here
     */
    void constructLPFold(T targetOccupancy, const MultidimArray<T>& lpf);

    /**
     * Method to construct 1D low-pass filter profile
     * @param targetOccupancy max frequency to preserve
     * @param lpf filter will be stored here
     */
    void constructLPF(T targetOccupancy, const MultidimArray<T>& lpf);

    /**
     * Internally, images are processed in smaller resolution. This function
     * sets proper variables
     * @param movie to obtain sizes from
     */
    void setNewDimensions(const MetaData& movie);

    /**
     * Loads movie from the file
     * @param movie where the input should be stored
     */
    void readMovie(MetaData& movie);

    /**
     * Sets all shifts in the movie to zero (0)
     * @param movie to update
     */
    void setZeroShift(MetaData& movie);

    /**
     * Method finds shifts of images in the micrograph and stores them
     * @param movie to be processed
     * @param dark correction
     * @param gain correction
     */
    int findShiftsAndStore(MetaData& movie, Image<T>& dark, Image<T>& gain);

    /**
     * Method to store all computed results to hard drive
     * @param initialMic sum of the unaligned images
     * @param Ninitial no of images in unaligned sum
     * @param averageMicrograph sum of the aligned images
     * @param N no of images in alined sum
     * @param movie to be stored
     * @param bestIref index of the reference image
     */
    void storeResults(Image<T>& initialMic, size_t Ninitial,
            Image<T>& averageMicrograph, size_t N, const MetaData& movie,
            int bestIref);

    /**
     * Method to correct indices of the images in the micrograph
     * @param movie to be used for correction
     */
    void correctLoopIndices(const MetaData& movie);

protected:
    // Target size of the frames
    int newXdim, newYdim;
    /** First and last frame (inclusive)*/
    int nfirst, nlast;
    /** Max shift */
    T maxShift;
    /*****************************/
    /** crop corner **/
    /*****************************/
    /** x left top corner **/
    int xLTcorner;
    /** y left top corner **/
    int yLTcorner;
    /** x right down corner **/
    int xDRcorner;
    /** y right down corner **/
    int yDRcorner;
    /** Aligned movie */
    FileName fnAligned;
    /** Aligned micrograph */
    FileName fnAvg;
    /** First and last frame*/
    int nfirstSum, nlastSum;
    /** Aligned micrograph */
    FileName fnInitialAvg;
    /** Binning factor */
    T bin;
    /** Bspline order */
    int BsplineOrder;
    /** Outside mode */
    int outsideMode;
    /** Outside value */
    T outsideValue;
    /** size factor between original size of the images and downscaled images) */
    T sizeFactor;
    bool processLocalShifts;

private:
    // Target sampling rate
    T newTs;
    /** Filename of movie metadata */
    FileName fnMovie;
    /** Correction images */
    FileName fnDark, fnGain;
    /** Sampling rate */
    T Ts;
    /** Max freq. */
    T maxFreq;
    /** Solver iterations */
    int solverIterations;
    /** Metadata with shifts */
    FileName fnOut;
    /** Do not calculate and use the input shifts */
    bool useInputShifts;

};
#endif
