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

#pragma once

#include "reconstruction/movie_alignment_correlation_base.h"
#include "data/fft_settings.h"
#include "reconstruction_cuda/gpu.h"
#include "reconstruction_cuda/cuda_flexalign_scale.h"
#include "reconstruction_cuda/cuda_flexalign_correlate.h"

/**@defgroup ProgMovieAlignmentCorrelationGPU Movie Alignment Correlation GPU
   @ingroup ReconsCUDALibrary */
//@{
template<typename T>
class ProgMovieAlignmentCorrelationGPU: public AProgMovieAlignmentCorrelation<T> {
protected:
    void defineParams();
    void show();
    void readParams();

private:

    /**
     * Find good size for correlation of the frames
     * @param ref size of the frames
     * @return size of the frames such that FT is relatively fast
     */
    auto findGoodCorrelationSize(const Dimensions &ref);

    /**
     * Find good size for patches of the movie
     * @return size of the patch such that FT is relatively fast
     */
    auto findGoodPatchSize();

    /**
     * Find good size for cropping the movie frame
     * @param ref size of the movie
     * @return size of the movie such that FT is relatively fast
     */
    auto findGoodCropSize(const Dimensions &ref);
        
    /**
     * Get optimized size of the dimension for correlation
     * @param d full size of the signal
     * @return optimal size
     */
    auto getCorrelationHint(const Dimensions &d);

    /**
     * Returns position of all local alignment patches within a single frame
     * @param borders that should be left intact
     * @param patch size
     */
    std::vector<FramePatchMeta<T>> getPatchesLocation(const std::pair<T, T> &borders,
            const Dimensions &patch);

    /**
     * Returns a 'window' of all frames at specific position, taking into account the 
     * global shift and summing of frames
     * @param patch defining the portion of each frame to cut out
     * @param globAlignment to compensate
     * @param result where data are stored
     */
    void getPatchData(const Rectangle<Point2D<T>> &patch,
            const AlignmentResult<T> &globAlignment,
            T *result);

    /**
     * Returns the X and Y offset such that if you read from any frame after applying its global
     * shift, you will always read valid data
     * @param globAlignment to use
     * @param verbose level
     * @return no of pixels in X (Y) dimension where there might NOT be data from each frame
     */
    std::pair<T,T> getMovieBorders(const AlignmentResult<T> &globAlignment, int verbose);

    /**
     * This method optimizes and sets sizes and additional parameters for the local alignment
    */
    void LAOptimize();

    AlignmentResult<T> computeGlobalAlignment(const MetaData &movie,
            const Image<T> &dark,
            const Image<T> &igain) override;

    /**
     * Prepare filter for given scale and dimension. 
     * Filter is stored in CUDA managed memory
     * @param scale of the filter
     * @param dims of the filter
    */
    T* setFilter(float scale, const Dimensions &dims);

    /** Helper structure to avoid passing a lot of parameters */
    struct AlignmentContext
    { 
        int verbose;
        float maxShift;
        size_t N; // number of frames 
        std::pair<T, T> scale;
        core::optional<size_t> refFrame;
        Dimensions out = Dimensions(0);
        size_t alignmentBytes() const {
            return (N * (N-1) / 2) * 2 * sizeof(float); // 2D position for each correlation
        }
    };

    /**
     * Create local alignment from global alignment (each patch is shifted by respective global alignment)
     * @param globAlignment to use
     * @return local alignment equivalent to global alignment
     */
    LocalAlignmentResult<T> localFromGlobal(const AlignmentResult<T> &globAlignment);

    void applyShiftsComputeAverage(
            const MetaData& movieMD, const Image<T>& dark, const Image<T>& igain,
            Image<T>& initialMic, size_t& Ninitial, Image<T>& averageMicrograph,
            size_t& N, const AlignmentResult<T> &globAlignment) override;

    
    /**
     * @returns number of GPU streams that can be used for output generation without running out of memory
     **/
    auto getOutputStreamCount();

    /**
     * Store setting for given dimensions to permanent storage
     * @param orig size
     * @param opt optimized size
     * @param applyCrop flag
     */
    void storeSizes(const Dimensions &orig, const Dimensions &opt, bool applyCrop);

    /**
     * Loads setting for given dimensions from permanent storage
     * @param dim to look for
     * @param applyCrop flag
     * @return stored size, if any
     */
    std::optional<Dimensions> getStoredSizes(const Dimensions &dim, bool applyCrop);

    /**
     * This method optimizes and sets sizes and additional parameters for the global alignment
    */
    void GAOptimize();

    /**
     * Report used settings
     * @param scale of the correlation
     * @param SP 
     * @param CP
     * @param streams
     * @param threads
    */
    void report(float scale, 
        typename CUDAFlexAlignScale<T>::Params &SP, 
        typename CUDAFlexAlignCorrelate<T>::Params &CP, 
        int streams, int threads);

    /**
     * Crops raw movie frame according to the settings
     * @param settings new sizes of the frame
     * @param output where 'windowed' movie should be copied
     * @param src data
     */
    void getCroppedFrame(const Dimensions &settings, T *output, T *src);

    /**
     * Helper class holding movie data
    */
    class Movie final
    {
    public:
        /** 
         * Set dimension of the movie that will be stored during the runtime,
         * This will not allocate data for the movie
         * @see setFrameData()
         * @param dim of the raw movie, or after the binning
        */
        auto set(const Dimensions &dim) {
            mDim = dim;
            mFrames.reserve(dim.n());
            for (auto n = 0; n < dim.n(); ++n) {
                // create multidim arrays, but without data 
                mFrames.emplace_back(1, 1, dim.y(), dim.x(), nullptr);
            }
        }

        /**
         * @return dimension of the movie
        */
        auto &getDim() const {
            return mDim;
        }

        /**
         * @param index 0-based
         * @return frame data in SP
        */
        auto &getFrame(size_t index) const {
            return mFrames[index];
        }

        /**
         * Set storage for the frame data
         * @param index 0-based
         * @param ptr where frame is to be stored. Expected to be allocated by memory manager
        */
        auto setFrameData(size_t index, T *ptr) {
            mFrames[index].data = ptr;
        }

    private:
        std::vector<MultidimArray<T>> mFrames;
        Dimensions mDim = Dimensions(0);
    };

    /**
     * Loads specific frame to main memory
     * @param movieMD to use
     * @param dark pixel correction
     * @param igain correction
     * @param index index to load (starting at 0)
     * @return newly allocated memory (using memory manager) holding the frame
     */
    T* loadFrame(const MetaData& movieMD, const Image<T>& dark, const Image<T>& igain, size_t index);

    /**
     * Compute alignment from measured relative shifts
     * @param positions of the maxima. Should be [X,Y] pairs around center
     * @param context for of the alignment
     * @return computed alignment
    */
    auto computeAlignment(float *positions, const AlignmentContext &context);

    void releaseAll() override;

    LocalAlignmentResult<T> computeLocalAlignment(const MetaData &movie,
        const Image<T> &dark, const Image<T> &igain,
        const AlignmentResult<T> &globAlignment) override;

    /**
     * Utility function for creating a key that can be used for permanent storage
     * @param keyword to use
     * @param dim of the 'FFT problem'
     * @param crop the FFT?
     * @return a unique key
     */
    std::string const getKey(const std::string &keyword, const Dimensions &dim, bool crop) {
        std::stringstream ss;
        ss << version << " " << mGpu.value().getUUID() << keyword << dim << " " << crop;
        return ss.str();
    }        

    void applyShiftsComputeAverage(
        const MetaData& movie, const Image<T>& dark, const Image<T>& igain,
        Image<T>& initialMic, size_t& Ninitial, Image<T>& averageMicrograph,
        size_t& N, const LocalAlignmentResult<T> &alignment) override;

    // holding the loaded movie
    Movie movie;
    // number of GPU streams for global alignment
    int GAStreams = 1;
    // parameters for global alignment scaling
    typename CUDAFlexAlignScale<T>::Params GASP {
        .batch = 1, // always 1
    };
    // parameters for global alignment correlation
    typename CUDAFlexAlignCorrelate<T>::Params GACP; 
    // parameters for local alignment scaling
    typename CUDAFlexAlignScale<T>::Params LASP {
        .doBinning = false,
        .raw = Dimensions(0),
    };
    // parameters for local alignment correlation
    typename CUDAFlexAlignCorrelate<T>::Params LACP; 

    /** No of frames used for averaging a single patch */
    int patchesAvg;

    /** Path to file where results of the benchmark might be stored */
    std::string storage;

    core::optional<GPU> mGpu;

    /**
     * Keywords representing optimal settings of the algorithm.
     */
    std::string minMemoryStr = std::string("minMem");
    std::string optSizeXStr = std::string("optSizeX");
    std::string optSizeYStr = std::string("optSizeY");
    std::string optBatchSizeStr = std::string("optBatchSize");

    static constexpr auto version = "2.0";
};

//@}