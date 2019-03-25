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

#include "data/alignment_result.h"
#include "data/local_alignment_result.h"
#include "core/xmipp_program.h"
#include "core/metadata_extension.h"
#include "core/xmipp_fftw.h"
#include "core/optional.h"
#include "eq_system_solver.h"
#include "bspline_helper.h"
#include "data/point2D.h"

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
     * Compute alignment of the each frame from frame-to-frame shifts
     * @param bX frame-to-frame shift in X dim
     * @param bY frame-to-frame shift in Y dim
     * @param A system matrix to be used
     * @param refFrame reference frame
     * @param N no of frames
     * @param verbose level
     * @return respective global alignment
     */
    AlignmentResult<T> computeAlignment(
            Matrix1D<T> &bX, Matrix1D<T> &bY, Matrix2D<T> &A,
            const core::optional<size_t> &refFrame, size_t N, int verbose);

    /**
     * Method does a sanity check on the settings of the program,
     * reporting found issues to stderr.
     * @return success of the check
     */
    bool checkSettings();

    /**
     * Method finds a reference image, i.e. an image which has smallest relative
     * shift to all other images.
     * @param N no of images
     * @param shiftX relative X shift of each image
     * @param shiftY relative Y shift of each image
     * @return position of the reference frame
     */
    int findReferenceImage(size_t N, const Matrix1D<T>& shiftX,
            const Matrix1D<T>& shiftY);

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
     */
    T computeSizeFactor();

    /**
     * Method will create a 2D Low Pass Filter with given properties
     * @param targetOccupancy should be <0, 1.0>
     * @param xSize of the filter
     * @param ySize of the filter
     * @return requested LPF
     */
    MultidimArray<T> createLPF(T targetOccupancy, size_t xSize,
            size_t ySize);

    /**
     * Method loads a single frame from the movie
     * @param movie to load from
     * @param objId id of the image to load
     * @param out loaded frame
     */
    void loadFrame(const MetaData& movie, size_t objId,
            Image<T>& out);

    /**
     * Method loads a single frame from the movie and apply gain and dark
     * pixel correction
     * @param movie to load from
     * @param dark pixel correction
     * @param igain inverse gain correction
     * @param objId id of the image to load
     * @param out loaded frame
     */
    void loadFrame(const MetaData &movie, const Image<T> &dark,
            const Image<T> &igain, size_t objId,
            Image<T> &out);

    /**
     * Returns occupancy that can be used for filter generation
     */
    T getTargetOccupancy();

    /**
     * This method applies global shifts and can also produce 'average'
     * image (micrograph)
     * @param movie input
     * @param dark correction to be used
     * @param igain correction to be used
     * @param initialMic sum of the unaligned micrographs
     * @param Ninitial will store number of micrographs used for unaligned sum
     * @param averageMicrograph sum of the aligned micrographs
     * @param N will store number of micrographs used for aligned sum
     * @param globAlignment to apply
     */
    virtual void applyShiftsComputeAverage(const MetaData& movie,
            const Image<T>& dark, const Image<T>& igain, Image<T>& initialMic,
            size_t& Ninitial, Image<T>& averageMicrograph, size_t& N,
            const AlignmentResult<T> &globAlignment) = 0;

    /**
     * This method applies local shifts and can also produce 'average'
     * image (micrograph)
     * @param movie input
     * @param dark correction to be used
     * @param igain correction to be used
     * @param initialMic sum of the unaligned micrographs
     * @param Ninitial will store number of micrographs used for unaligned sum
     * @param averageMicrograph sum of the aligned micrographs
     * @param N will store number of micrographs used for aligned sum
     * @param alignment to apply
     */
    virtual void applyShiftsComputeAverage(
            const MetaData& movie, const Image<T>& dark, const Image<T>& igain,
            Image<T>& initialMic, size_t& Ninitial, Image<T>& averageMicrograph,
            size_t& N, const LocalAlignmentResult<T> &alignment) = 0;

    /**
     * This method computes global shift of the frames of the movie
     * @param movie to process
     * @param dark pixel correction
     * @param igain correction
     * @return global alignment of the movie
     */
    virtual AlignmentResult<T> computeGlobalAlignment(const MetaData &movie,
            const Image<T> &dark, const Image<T> &igain) = 0;

    /**
     * This method computes local shift of the frames of the movie
     * @param movie to process
     * @param dark pixel correction
     * @param igain correction
     * @return local alignment of the movie
     */
    virtual LocalAlignmentResult<T> computeLocalAlignment(const MetaData &movie,
            const Image<T> &dark, const Image<T> &igain,
            const AlignmentResult<T> &globAlignment) = 0;

    /**
     * This method releases all resources allocated so far
     */
    virtual void releaseAll() = 0;

    /**
     * Method to store all computed alignment to hard drive
     */
    void storeResults(const LocalAlignmentResult<T> &alignment);
private:

    /**
     * Method will create a 2D Low-Pass Filter from the 1D
     * profile, that can be used in Fourier domain
     * @param lpf 1D profile
     * @param xSize size of full image (space domain)
     * @param ySize size of full image (space/frequency domain)
     * @param targetOccupancy maximal frequency to be preserved, should be <0, 1.0>
     * @param result resulting 2D filter. Must be of proper size, i.e.
     * xdim == xSize/2+1, ydim = ySize
     */
    void scaleLPF(const MultidimArray<T>& lpf, int xSize, int ySize,
            T targetOccupancy, MultidimArray<T>& result);

    /**
     * Method to store global (frame) shifts computed for the movie
     * @param alignment result to store
     * @param movie to be updated
     */
    void storeGlobalShifts(const AlignmentResult<T> &alignment,
            MetaData &movie);

    /**
     * Method loads global shift from the given movie
     * @param movie where shifts are stored
     * @return global alignment as stored in the movie
     */
    AlignmentResult<T> loadGlobalShifts(MetaData &movie);

    /**
     * Method loads dark correction image
     * @param dark correction will be stored here
     */
    void loadDarkCorrection(Image<T>& dark);

    /**
     *  Method loads gain correction image
     *  @param gain correction will be stored here
     */
    void loadGainCorrection(Image<T>& igain);

    /**
     * Method to construct 1D low-pass filter profile
     * @param targetOccupancy max frequency to preserve
     * @param lpf filter will be stored here
     */
    void constructLPF(T targetOccupancy, const MultidimArray<T>& lpf);

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

    /**
     * Outputs global shift values to standard output
     * @param globAlignment to show
     */
    void printGlobalShift(const AlignmentResult<T> &globAlignment);

    /**
     * Returns sampling rate that user requested
     */
    T getRequestedSamplingRate();

protected:
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
    /** if true, local alignment should be performed */
    bool processLocalShifts;
    /** Solver iterations */
    int solverIterations;
    /** Metadata with shifts */
    FileName fnOut;
    /** Number of patches used for local alignment */
    std::pair<size_t, size_t> localAlignPatches;
    /** Control points used for local alignment */
    Dimensions localAlignmentControlPoints = Dimensions(0);


private:
    /** Filename of movie metadata */
    FileName fnMovie;
    /** Correction images */
    FileName fnDark, fnGain;
    /** Sampling rate */
    T Ts;
    /** Max freq. */
    T maxFreq;
    /** Do not calculate and use the input shifts */
    bool useInputShifts;

};
#endif
