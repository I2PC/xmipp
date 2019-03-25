/***************************************************************************
 *
 * Authors:    David Strelak (davidstrelak@gmail.com)
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

#ifndef MOVIE_ALIGNMENT_CORRELATION_GPU
#define MOVIE_ALIGNMENT_CORRELATION_GPU

#include <thread>
#include <mutex>
#include <future>
#include "reconstruction/movie_alignment_correlation_base.h"
#include "reconstruction_cuda/cuda_gpu_movie_alignment_correlation.h"
#include "reconstruction_cuda/cuda_gpu_geo_shift_transformer.h"
#include "reconstruction_cuda/cuda_gpu_geo_transformer.h"
#include "data/filters.h"
#include "data/fft_settings.h"
#include "data/bspline_grid.h"
#include "core/userSettings.h"
#include "reconstruction/bspline_helper.h"
#include "gpu.h"
#include "core/optional.h"

template<typename T>
class ProgMovieAlignmentCorrelationGPU: public AProgMovieAlignmentCorrelation<T> {
public:
    /// Read argument from command line
    void readParams();

    /// Show
    void show();

    /// Define parameters
    void defineParams();

private:

    /**
     * Inherited, see parent
     */
    void releaseAll() {
        delete[] movieRawData;
        movieRawData = nullptr;
    };

    /**
     * Estimates maximal size of the filter for given frame
     * Might be use to estimate memory requirements
     * @frame reference frame
     * @return max MB necessary for filter
     */
    int getMaxFilterSize(const Image<T> &frame);

    /**
     * Returns best settings for FFT on GPU. It is either
     * loaded from permanent storage, or GPU benchmark is ru
     * @param d dimension of the FFT
     * @param extraMem that should be left on GPU
     * @param crop if true, FFT can be of smaller size
     * @return best FFT setting
     */
    FFTSettings<T> getSettingsOrBenchmark(const Dimensions &d, size_t extraMem,
            bool crop);

    /**
     * Utility function for creating a key that can be used
     * for permanent storage
     * @param keyword to use
     * @param dim of the 'FFT problem'
     * @param crop the FFT?
     * @return a unique key
     */
    std::string const getKey(const std::string &keyword,
            const Dimensions &dim, bool crop) {
        std::stringstream ss;
        ss << gpu.value().UUID() << keyword << dim << crop;
        return ss.str();
    }

    /**
     * Method for obtaining the best FFT setting for given movie
     * @param movie to 'analyse'
     * @param optimize the sizes?
     * @return optimal FFT setting for the movie
     */
    FFTSettings<T> getMovieSettings(const MetaData &movie, bool optimize);

    /**
     * Method will align data of given size, using cross-correlation of
     * data obtained after application of the filter.
     * @param data where data (in spacial domain) are stored
     * consecutively. This memory will be reused!
     * @param in FFT setting of the input data.
     * @param correlation FFT setting
     * @param filter to be applied to each image
     * @param refFrame reference frame, if any
     * @param maxShift where the maximum correlation should be searched
     * @param framesInCorrrelationBuffer max number of frames that can be stored
     * in a single buffer on the GPU
     * @param verbose level
     * @return global alignment of each frame
     */
    AlignmentResult<T> align(T *data, const FFTSettings<T> &in, const FFTSettings<T> &correlation,
            MultidimArray<T> &filter, core::optional<size_t> &refFrame,
            size_t maxShift,
            size_t framesInCorrelationBuffer, int verbose);

    /**
     * Method computes shifts of each frame in respect to some reference frame
     * using cross-correlation on GPU
     * @param verbose level
     * @param maxShift where the maximum correlation should be searched
     * @param data where data (in frequency domain) are stored
     * consecutively. This memory will be reused!
     * @param settings for the correlations.
     * @param N original number of frames (notice that there are many more
     * correlations than original frames)!
     * @param scale between original frame size and correlation size
     * @param framesInCorrrelationBuffer max number of frames that can be stored
     * in a single buffer on the GPU
     * @param refFrame reference frame, if any
     * @return alignment of the data
     */
    AlignmentResult<T> computeShifts(int verbose, size_t maxShift, std::complex<T>* data,
            const FFTSettings<T> &settings, size_t N, std::pair<T, T> &scale,
            size_t framesInCorrelationBuffer,
            const core::optional<size_t>& refFrame);

    /**
     * Get best FFT settings for correlations of the original data
     * @param orig data
     * @param dowscale that should be applied for correlation
     * @return optimal FFT settings
     */
    FFTSettings<T> getCorrelationSettings(const FFTSettings<T> &orig,
            const std::pair<T, T> &downscale);

    /**
     * Get FFT settings for each patch used for local alignment
     * @param orig movie setting
     * @return optimal setting for each patch
     */
    FFTSettings<T> getPatchSettings(const FFTSettings<T> &orig);

    /**
     * Inherited, see parent
     */
    AlignmentResult<T> computeGlobalAlignment(const MetaData &movie,
            const Image<T> &dark,
            const Image<T> &igain);

    /**
     * Inherited, see parent
     */
    LocalAlignmentResult<T> computeLocalAlignment(const MetaData &movie,
            const Image<T> &dark, const Image<T> &igain,
            const AlignmentResult<T> &globAlignment);

    /**
     * Store setting for given dimensions to permanent storage
     * @param dim reference
     * @param s setting to store
     * @param applyCrop flag
     */
    void storeSizes(const Dimensions &dim, const FFTSettings<T> &s,
            bool applyCrop);

    /**
     * Returns best FFT setting for correlation of the given setting
     * @param s original setting
     * @param requested downscale used during correlation
     * @return FFT setting describing requested correlation
     */
    Dimensions getCorrelationHint(const FFTSettings<T> &s,
            const std::pair<T, T> &downscale);

    /**
     * Loads whole movie to the RAM
     * @param movie to load
     * @param dark pixel correction
     * @param igain correction
     */
    T* loadMovie(const MetaData& movie,
            const Image<T>& dark, const Image<T>& igain);

    /**
     * Loads setting for given dimensions from permanent storage
     * @param dim reference
     * @param applyCrop flag
     * @return stored setting, if any
     */
    core::optional<FFTSettings<T>> getStoredSizes(const Dimensions &dim,
            bool applyCrop);

    /**
     * Run benchmark to get the best FFT setting for given problem size
     * @param d dimension of the problem
     * @param extraMem to leave on GPU
     * @param crop flag
     */
    FFTSettings<T> runBenchmark(const Dimensions &d, size_t extraMem,
            bool crop);

    /**
     * Returns position of all 'local alignment patches' within a single frame
     * @param borders that should be left intact
     * @param movie size
     * @param patch size
     */
    std::vector<FramePatchMeta<T>> getPatchesLocation(const std::pair<T, T> &borders,
            const Dimensions &movie,
            const Dimensions &patch);

    /**
     * Imagine you align frames of the movie using global alignment
     * Some frames edges will overlap, i.e. there will be an are shared
     * by all frames, and edge area where at least one frame does not contribute.
     * This method computes the size of that area.
     * @param globAlignment to use
     * @param verbose level
     * @return no of pixels in X (Y) dimension where there might NOT be data from each frame
     */
    std::pair<T,T> getMovieBorders(const AlignmentResult<T> &globAlignment,
            int verbose);

    /**
     * Method returns a 'window'/'view' of each and all frames, aligned (to int positions)
     * using global alignment
     * @param allFrames, consecutive
     * @param patch defining the portion of each frame to load
     * @param globAlignment to compensate
     * @param movie dimension
     * @param result where data are stored
     */
    void getPatchData(const T *allFrames, const Rectangle<Point2D<T>> &patch,
            const AlignmentResult<T> &globAlignment,
            const Dimensions &movie, T *result);

    /**
     * Create local alignment from global alignment
     * @param movie to use
     * @param globAlignment to use
     */
    LocalAlignmentResult<T> localFromGlobal(
            const MetaData& movie,
            const AlignmentResult<T> &globAlignment);

    /**
     * Inherited, see parent
     */
    void applyShiftsComputeAverage(
            const MetaData& movie, const Image<T>& dark, const Image<T>& igain,
            Image<T>& initialMic, size_t& Ninitial, Image<T>& averageMicrograph,
            size_t& N, const AlignmentResult<T> &globAlignment);

    /**
     * Inherited, see parent
     * WARNING !!!
     * As a side effect, raw movie data might get corrupted. See the implementation.
     */
    void applyShiftsComputeAverage(
            const MetaData& movie, const Image<T>& dark, const Image<T>& igain,
            Image<T>& initialMic, size_t& Ninitial, Image<T>& averageMicrograph,
            size_t& N, const LocalAlignmentResult<T> &alignment);

    /**
     * Method returns requested downscale (<1) for the correlations used
     * for local alignment
     */
    std::pair<T,T> getLocalAlignmentCorrelationDownscale(
            const Dimensions &patchDim, T maxShift);

    /**
     * Method copies raw movie data according to the settings
     * @param settings new sizes of the movie
     * @param output where 'windowed' movie should be copied
     */
    void getCroppedMovie(const FFTSettings<T> &settings,
            T *output);

private:

    /** downscale to be used for local alignment correlation (<1) */
    std::pair<T,T> localCorrelationDownscale;

    /** No of frames used for averaging a single patch */
    int patchesAvg;


    /** Path to file where results of the benchmark might be stored */
    std::string storage;

    core::optional<GPU> gpu;

    /** contains the loaded movie, with consecutive data padded for in-place FFT */
    T *movieRawData = nullptr;

    /** contains dimensions of the movie stored in movieRawData */
    Dimensions rawMovieDim = Dimensions(0);

    /** mutex indicating when a thread can access data array used for alignment */
    mutable std::mutex alignDataMutex;

    /**
     * Keywords representing optimal settings of the algorithm.
     */
    std::string minMemoryStr = std::string("minMem");
    std::string optSizeXStr = std::string("optSizeX");
    std::string optSizeYStr = std::string("optSizeY");
    std::string optBatchSizeStr = std::string("optBatchSize");
};

#endif /* MOVIE_ALIGNMENT_CORRELATION_GPU */
