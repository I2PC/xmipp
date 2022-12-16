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

#include "reconstruction/movie_alignment_correlation_base.h"
#include "data/fft_settings.h"
#include "reconstruction_cuda/gpu.h"

/**@defgroup ProgMovieAlignmentCorrelationGPU Movie Alignment Correlation GPU
   @ingroup ReconsCUDALibrary */
//@{
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
    struct PatchContext
    { // to neni jenom patch, mozna batch? Spis LES
        explicit PatchContext(LocalAlignmentResult<T> &r) : result(r){};
        LocalAlignmentResult<T> &result;
        size_t shiftsOffset;
        int verbose;
        size_t maxShift;
        size_t N;
        std::pair<T, T> scale;
        core::optional<size_t> refFrame;
        size_t centerSize;
        size_t framesInCorrelationBuffer;
        FFTSettings<T> correlationSettings = FFTSettings<T>(1);
        size_t corrElems() const {
            return (N * (N-1) / 2) * centerSize * centerSize;
        }
    };

    class Movie final
    {
    public:
        ~Movie();
        auto set(const Dimensions &dim, bool doBinning) {
            // dimension is either of the raw movie, or after the binning
            mDim = dim;
            mDoBinning = doBinning;
            mFrames.reserve(dim.n());
            for (auto n = 0; n < dim.n(); ++n) {
                // create multidim arrays, but without data 
                mFrames.emplace_back(1, 1, dim.y(), dim.x(), nullptr);
            }
        }

        auto &getDim() const {
            return mDim;
        }

        auto &getFrame(size_t i) const {
            return mFrames[i];
        }

        auto setFrameData(size_t i, T *ptr) {
            mFrames[i].data = ptr;
        }

    private:
        // internally, frames are stored separately
        // as we access them only one by one (i.e. no need for batches)
        // also there's no padding or cropping.
        // We store either raw, full size frames or binned frames
        std::vector<MultidimArray<T>> mFrames;
        Dimensions mDim = Dimensions(0);
        bool mDoBinning = false;
    };
    Movie movie;


    class GlobalAlignmentHelper final {
    public:
        auto findBatchesThreadsStreams(const GPU &gpu, ProgMovieAlignmentCorrelationGPU &instance);
        
        FFTSettings<T> movieSettings = FFTSettings<T>(0);
        FFTSettings<T> correlationSettings = FFTSettings<T>(0);
        size_t gpuStreams = 1;
        size_t cpuThreads = 2;
        size_t bufferSize; // for correlation

    friend std::ostream& operator<<(std::ostream &os, const GlobalAlignmentHelper &h) {
        os << "Settings for the movie: " << h.movieSettings << "\n";
        os << "GPU streams: " << h.gpuStreams << "; CPU threads: " << h.cpuThreads << "\n";
        os << "Settings for the correlation: " << h.correlationSettings << "\n";
        os << "Correlation buffer size: " << h.bufferSize;
        return os;
    }
        
    private:
        auto findGoodCropSize(const Dimensions &movie, const GPU &gpu, ProgMovieAlignmentCorrelationGPU &instance);
    };
    GlobalAlignmentHelper globalHelper;

    class LocalAlignmentHelper final {
    public:
        auto findBatchesThreadsStreams(const GPU &gpu, ProgMovieAlignmentCorrelationGPU &instance);
        
        FFTSettings<T> patchSettings = FFTSettings<T>(0);
        FFTSettings<T> correlationSettings = FFTSettings<T>(0);
        size_t cpuThreads = 4;
        size_t bufferSize; // for correlation

    friend std::ostream& operator<<(std::ostream &os, const LocalAlignmentHelper &h) {
        os << "Settings for the patches: " << h.patchSettings << "\n";
        os << "CPU threads: " << h.cpuThreads << "\n";
        os << "Settings for the correlation: " << h.correlationSettings << "\n";
        os << "Correlation buffer size: " << h.bufferSize;
        return os;
    }
        
    private:
        auto findGoodPatchSize(const Dimensions &hint, const GPU &gpu, ProgMovieAlignmentCorrelationGPU &instance);
        
    };
    LocalAlignmentHelper localHelper;


    auto findGoodCorrelationSize(const Dimensions &hint, const GPU &gpu);

    /**
     * Inherited, see parent
     */
    void releaseAll() override { /* nothing to do */  };

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
        ss << version << " " << gpu.value().getUUID() << keyword << dim << " " << crop;
        return ss.str();
    }

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

    auto computeShifts(T* correlations,
            PatchContext context); // pass by copy, this will be run asynchronously);

    /**
     * Get suggested size of the frame for correlation
     * @param s size for the input data
     * @return optimal size
     */
    auto getCorrelationHint(const Dimensions &orig);

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
     * @param orig size
     * @param opt optimized size
     * @param applyCrop flag
     */
    void storeSizes(const Dimensions &orig, const Dimensions &opt,
            bool applyCrop);

    /**
     * Loads specific range of frames to the RAM
     * @param movie to load
     * @param dark pixel correction
     * @param igain correction
     * @param index index to load (starting at 0)
     */
    T* loadFrame(const MetaData& movieMD,
        const Image<T>& dark, const Image<T>& igain, size_t index);

    /**
     * Loads setting for given dimensions from permanent storage
     * @param dim to look for
     * @param applyCrop flag
     * @return stored size, if any
     */
    std::optional<Dimensions> getStoredSizes(const Dimensions &dim,
            bool applyCrop);

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
     * @param patch defining the portion of each frame to load
     * @param globAlignment to compensate
     * @param result where data are stored
     */
    void getPatchData(const Rectangle<Point2D<T>> &patch,
            const AlignmentResult<T> &globAlignment,
            T *result);
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
     * Method copies raw movie data according to the settings
     * @param settings new sizes of the movie
     * @param output where 'windowed' movie should be copied
     * @param firstFrame to copy
     * @param noOfFrames to copy
     */
    void getCroppedFrame(const FFTSettings<T> &settings,
        T *output, T *src);


    /**
     * @param shift that we allow
     * @returns size of the (square) window where we can search for shift
     */
    size_t getCenterSize(size_t shift) {
        return std::ceil(shift * 2 + 1);
    }

    /**
     * @returns number of GPU streams that can be used for output generation without running out of memory
     **/
    auto getOutputStreamCount();


private:
    /** No of frames used for averaging a single patch */
    int patchesAvg;

    /** Path to file where results of the benchmark might be stored */
    std::string storage;

    core::optional<GPU> gpu;

    /**
     * Keywords representing optimal settings of the algorithm.
     */
    std::string minMemoryStr = std::string("minMem");
    std::string optSizeXStr = std::string("optSizeX");
    std::string optSizeYStr = std::string("optSizeY");
    std::string optBatchSizeStr = std::string("optBatchSize");

    static constexpr auto version = "1.2";
};

//@}
#endif /* MOVIE_ALIGNMENT_CORRELATION_GPU */
