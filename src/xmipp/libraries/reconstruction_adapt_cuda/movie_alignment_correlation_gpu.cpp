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

#include "reconstruction_adapt_cuda/movie_alignment_correlation_gpu.h"
#include "reconstruction_cuda/cuda_gpu_geo_transformer.h"
#include "core/userSettings.h"
#include "reconstruction_cuda/cuda_fft.h"
#include "reconstruction_adapt_cuda/basic_mem_manager.h"
#include <CTPL/ctpl_stl.h>



template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::defineParams() {
    AProgMovieAlignmentCorrelation<T>::defineParams();
    this->addParamsLine("  [--device <dev=0>]                 : GPU device to use. 0th by default");
    this->addParamsLine("  [--storage <fn=\"\">]              : Path to file that can be used to store results of the benchmark");
    this->addParamsLine("  [--patchesAvg <avg=3>]             : Number of near frames used for averaging a single patch");
    this->addExampleLine(
                "xmipp_cuda_movie_alignment_correlation -i movie.xmd --oaligned alignedMovie.stk --oavg alignedMicrograph.mrc --device 0");
    this->addSeeAlsoLine("xmipp_movie_alignment_correlation");
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::show() {
    AProgMovieAlignmentCorrelation<T>::show();
    std::cout << "Device:              " << mGpu.value().device() << " (" << mGpu.value().getUUID() << ")" << "\n";
    std::cout << "Benchmark storage    " << (storage.empty() ? "Default" : storage) << "\n";
    std::cout << "Patches avg:         " << patchesAvg << "\n";
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::readParams() {
    AProgMovieAlignmentCorrelation<T>::readParams();

    // read GPU
    int device = this->getIntParam("--device");
    if (device < 0) REPORT_ERROR(ERR_ARG_INCORRECT, "Invalid GPU device");
    mGpu = core::optional<GPU>(device);
    mGpu.value().set();

    // read permanent storage
    storage = this->getParam("--storage");

    // read patch averaging
    patchesAvg = this->getIntParam("--patchesAvg");
    if (patchesAvg < 1) REPORT_ERROR(ERR_ARG_INCORRECT, "Patch averaging has to be at least 1 (one).");
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::GlobalAlignmentHelper::findGoodCropSize(const Dimensions &ref, const GPU &gpu, ProgMovieAlignmentCorrelationGPU &instance) {
    const bool crop = true;
    if (auto optDim = instance.getStoredSizes(ref, crop); optDim) {
        return optDim.value().copyForN(ref.n());
    }
    std::cout << "Benchmarking cuFFT ..." << std::endl;
    auto hint = FFTSettings<T>(ref.createSingle()); // movie frame is big enought to give us an idea
    auto candidate = std::unique_ptr<FFTSettings<T>>(CudaFFT<T>::findOptimal(gpu, hint, 0, hint.sDim().x() == hint.sDim().y(), 10, crop, true));
    if (!candidate) {
        REPORT_ERROR(ERR_GPU_MEMORY, "Insufficient GPU memory for processing a single frame of the movie.");
    }
    instance.storeSizes(ref, candidate->sDim(), crop);
    return candidate->sDim().copyForN(ref.n());
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::findGoodCorrelationSize(const Dimensions &ref, const GPU &gpu) {
    const bool crop = false;
    if (auto optDim = this->getStoredSizes(ref, crop); optDim) {
        return optDim.value().copyForN(ref.n());
    }
    std::cout << "Benchmarking cuFFT ..." << std::endl;
    auto hint = FFTSettings<T>(ref, 1, false, false);
    auto candidate = std::unique_ptr<FFTSettings<T>>(CudaFFT<T>::findOptimal(gpu, hint, 0, hint.sDim().x() == hint.sDim().y(), 20, crop, true));
    if (!candidate) {
        REPORT_ERROR(ERR_GPU_MEMORY, "Insufficient GPU memory for processing a correlations of the movie.");
    }
    this->storeSizes(ref, candidate->sDim(), crop);
    return candidate->sDim().copyForN(ref.n());
}

template<typename T>
auto  ProgMovieAlignmentCorrelationGPU<T>::findGoodPatchSize() {
    const bool crop = false;
    const auto reqPatchSize = this->getRequestedPatchSize();
    auto ref = Dimensions(reqPatchSize.first, reqPatchSize.second, 1, this->getMovieSize().n());
    if (auto optDim = this->getStoredSizes(ref, crop); optDim) {
        return optDim.value().copyForN(ref.n());
    }
    std::cout << "Benchmarking cuFFT ..." << std::endl;
    auto hint = FFTSettings<T>(ref);
    auto candidate = std::unique_ptr<FFTSettings<T>>(CudaFFT<T>::findOptimal(mGpu.value(), hint, 0, hint.sDim().x() == hint.sDim().y(), 20, crop, true));
    if (!candidate) {
        REPORT_ERROR(ERR_GPU_MEMORY, "Insufficient GPU memory for processing a correlations of the movie.");
    }
    this->storeSizes(ref, candidate->sDim(), crop);
    return candidate->sDim();
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getCorrelationHint(const Dimensions &d) {
    auto getNearestEven = [this] (size_t v, T minScale, size_t shift) { // scale is less than 1
        size_t size = std::ceil(getCenterSize(shift) / 2.f) * 2; // to get even size
        while ((size / (float)v) < minScale) {
            size += 2;
        }
        return size;
    };
    const T requestedScale = this->getScaleFactor();
    // hint, possibly bigger then requested, so that it fits max shift window
    Dimensions hint(getNearestEven(d.x(), requestedScale, static_cast<size_t>(this->maxShift)),
            getNearestEven(d.y(), requestedScale, static_cast<size_t>(this->maxShift)),
            d.z(), d.n());
    return hint;
}

template<typename T>
std::vector<FramePatchMeta<T>> ProgMovieAlignmentCorrelationGPU<T>::getPatchesLocation(
        const std::pair<T, T> &borders, const Dimensions &patch) {
    size_t patchesX = this->localAlignPatches.first;
    size_t patchesY = this->localAlignPatches.second;
    T windowXSize = this->getMovieSize().x() - 2 * borders.first;
    T windowYSize = this->getMovieSize().y() - 2 * borders.second;
    T corrX = std::ceil(((patchesX * patch.x()) - windowXSize) / (T) (patchesX - 1));
    T corrY = std::ceil(((patchesY * patch.y()) - windowYSize) / (T) (patchesY - 1));
    T stepX = (T)patch.x() - corrX;
    T stepY = (T)patch.y() - corrY;
    std::vector<FramePatchMeta<T>> result;
    for (size_t y = 0; y < patchesY; ++y) {
        for (size_t x = 0; x < patchesX; ++x) {
            T tlx = borders.first + x * stepX; // Top Left
            T tly = borders.second + y * stepY;
            T brx = tlx + patch.x() - 1; // Bottom Right
            T bry = tly + patch.y() - 1; // -1 for indexing
            Point2D<T> tl(tlx, tly);
            Point2D<T> br(brx, bry);
            Rectangle<Point2D<T>> r(tl, br);
            result.emplace_back(FramePatchMeta<T> { .rec = r, .id_x = x, .id_y = y });
        }
    }
    return result;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::getPatchData(const Rectangle<Point2D<T>> &patch, 
        const AlignmentResult<T> &globAlignment, T *result) {
    const auto &movieDim = movie.getDim();
    const int n = static_cast<int>(movieDim.n());
    const auto patchSize = patch.getSize();
    const auto bufferBytes = patchSize.x * sizeof(T);
    auto buffer = std::make_unique<T[]>(patchSize.x); // faster than memory manager
    for (int t = 0; t < n; ++t) {// for each patch
        for (int y = 0; y < patchSize.y; ++y) { // for each row
            bool copy = true;
            // while averaging odd num of frames, use copy equally from previous and following frames
            // otherwise prefer following frames
            for (int f = std::max(0, t - ((patchesAvg - 1) / 2)); f <= std::min(n - 1, t + (patchesAvg / 2)); ++f) {
                const auto *frame = movie.getFrame(f).data;
                const auto xShift = static_cast<int>(std::round(globAlignment.shifts[f].x));
                const auto yShift = static_cast<int>(std::round(globAlignment.shifts[f].y));
                // notice we don't test any boundaries - it should always be within the range of the frame
                // see implementation of patch position generation and frame border computation
                const int srcY = patch.tl.y + y + yShift;
                const int srcX = patch.tl.x + xShift;
                auto *src = frame + srcY * movieDim.x() + srcX;
                if (copy) {
                    memcpy(buffer.get(), src, bufferBytes);
                } else {
                    for (int x = 0; x < patchSize.x; ++x) {
                        buffer[x] += src[x];
                    }
                }
                copy = false;
            }
            const int patchOffset = t * patchSize.x * patchSize.y;
            const int destIndex = patchOffset + y * patchSize.x;
            memcpy(result + destIndex, buffer.get(), bufferBytes); // write result
        }
    }
}

template<typename T>
std::pair<T,T> ProgMovieAlignmentCorrelationGPU<T>::getMovieBorders(
        const AlignmentResult<T> &globAlignment, int verbose) {
    T minX = std::numeric_limits<T>::max();
    T maxX = std::numeric_limits<T>::min();
    T minY = std::numeric_limits<T>::max();
    T maxY = std::numeric_limits<T>::min();
    for (const auto& s : globAlignment.shifts) {
        minX = std::min(std::floor(s.x), minX);
        maxX = std::max(std::ceil(s.x), maxX);
        minY = std::min(std::floor(s.y), minY);
        maxY = std::max(std::ceil(s.y), maxY);
    }
    auto res = std::make_pair(std::abs(maxX - minX), std::abs(maxY - minY));
    if (verbose > 1) {
        std::cout << "Movie borders: x=" << res.first << " y=" << res.second << std::endl;
    }
    return res;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::LAOptimize() {
    const auto pSize = findGoodPatchSize();
    LASP.movie = pSize;
    const auto cSize = this->findGoodCorrelationSize(this->getCorrelationHint(pSize), mGpu.value());
    LASP.out = LACP.dim = cSize;
    const auto maxBytes = static_cast<size_t>(static_cast<float>(mGpu.value().lastFreeBytes()) * 0.9f); // leave some buffer in case of memory fragmentation
    auto getMemReq = [&LASP=LASP, &LACP=LACP, &gpu=mGpu.value()]() {
        auto scale = CUDAFlexAlignScale<T>(LASP, gpu).estimateBytes();
        auto correlate = CUDAFlexAlignCorrelate<T>(LACP, gpu).estimateBytes();
        return scale + correlate;
    };
    auto cond = [&LASP=LASP]() {
        // we want only no. of batches that can process the patches without extra invocations
        return 0 == LASP.movie.n() % LASP.batch;
    };

    // we're gonna run both scaling and correlation in two streams to overlap memory transfers and computations
    // more streams do not make sense because we're limited by the transfers
    size_t bufferDivider = 1;
    size_t corrBatchDivider = 0;
    const size_t correlations = cSize.n() * (cSize.n() - 1) / 2;
    do {
        corrBatchDivider++;
        auto corrBatch = correlations / corrBatchDivider;
        LACP.batch = corrBatch;
        if ((pSize.n() / bufferDivider) > corrBatch) {
            bufferDivider += (bufferDivider == 1) ? 2 : 1; // we use two buffers, so we need the same memory for batch == 1 and == 2
        }
        LACP.bufferSize = pSize.n() / bufferDivider;
        for (auto scaleBatch = pSize.n(); scaleBatch > 0; --scaleBatch) {
            LASP.batch = scaleBatch;
            if (cond() && getMemReq() <= maxBytes) {
                return;
            }
        }
    } while (true);
}

template<typename T>
T* ProgMovieAlignmentCorrelationGPU<T>::setFilter(float scale, const Dimensions &dims) {
    MultidimArray<T> tmp = this->createLPF(this->getPixelResolution(scale), LASP.out);
    auto *filter = reinterpret_cast<T*>(BasicMemManager::instance().get(tmp.nzyxdim *sizeof(T), MemType::CUDA_MANAGED));
    memcpy(filter, tmp.data, tmp.nzyxdim *sizeof(T));
    return filter;
}


template<typename T>
LocalAlignmentResult<T> ProgMovieAlignmentCorrelationGPU<T>::computeLocalAlignment(
        const MetaData &movieMD, const Image<T> &dark, const Image<T> &igain,
        const AlignmentResult<T> &globAlignment) {
    LAOptimize();
    const auto movieSize = this->getMovieSize();
    if ((movieSize.x() < LASP.movie.x())
        || (movieSize.y() < LASP.movie.y())) {
        REPORT_ERROR(ERR_PARAM_INCORRECT, "Movie is too small for local alignment.");
    }

    auto borders = getMovieBorders(globAlignment, this->verbose > 1);
    auto patchesLocation = this->getPatchesLocation(borders, LASP.movie);
    auto actualScale = static_cast<float>(LASP.out.x()) / static_cast<float>(LASP.movie.x()); // assuming we use square patches

    GPU streams[] = {GPU(mGpu.value().device(), 1), GPU(mGpu.value().device(), 2)};
    auto loadPool = ctpl::thread_pool(std::min(this->localAlignPatches.first, this->localAlignPatches.second));

    if (this->verbose) {
        std::cout << "Actual scale factor (X): " << actualScale << "\n";
        std::cout << "Size of the patch      : " << LASP.movie << "\n";
        std::cout << "Size of the correlation: " << LASP.out << "\n";
        std::cout << "Correlation batch      : " << LACP.batch << "\n";
        std::cout << "GPU streams            : " << sizeof(streams) / sizeof(streams[0]) << "\n";
        std::cout << "CPU threads            : " << loadPool.size() << "\n";
    }

    // prepare memory
    auto *filter = setFilter(actualScale, LASP.out);
    std::vector<T*> corrPositions(loadPool.size()); // initializes to nullptrs
    std::vector<T*> patches(loadPool.size()); // initializes to nullptrs
    std::vector<std::complex<T>*> scalledPatches(loadPool.size()); // initializes to nullptrs
    
    // prepare control structures
    const AlignmentContext context {
        .verbose = this->verbose,
        .maxShift = this->maxShift * actualScale,
        .N = LASP.movie.n(),
        .scale = std::make_pair(LASP.movie.x() / (T) LASP.out.x(), LASP.movie.y() / (T) LASP.out.y()),
        .refFrame = globAlignment.refFrame,
        .out = LASP.out,
    };
    std::mutex mutex[2]; // one for each step, as they need to happen 'atomically'
    streams[0].set(); streams[1].set();
    auto scaler = CUDAFlexAlignScale<T>(LASP, streams[0]);
    scaler.init();
    auto correlater = CUDAFlexAlignCorrelate<T>(LACP, streams[1]);
    correlater.init();

    // prepare result
    LocalAlignmentResult<T> result { globalHint:globAlignment, movieDim:movieSize};
    result.shifts.reserve(patchesLocation.size() * movieSize.n());
    std::vector<std::future<void>> futures;
    futures.reserve(patchesLocation.size());
    
    // helper lambdas
    auto alloc = [](size_t bytes) {
        // it's faster to allocate CPU memory and then pin it, because registering can run in parallel
        auto *ptr = BasicMemManager::instance().get(bytes, MemType::CPU_PAGE_ALIGNED);
        GPU::pinMemory(ptr, bytes);
        return ptr;
    };
    
    // process each patch
    for (auto &p : patchesLocation) {
        // prefill some info about patch
        const auto shiftsOffset = result.shifts.size();
        for (size_t i = 0;i < movieSize.n();++i) {
            // keep this consistent with data loading
            int globShiftX = std::round(globAlignment.shifts.at(i).x);
            int globShiftY = std::round(globAlignment.shifts.at(i).y);
            p.id_t = i;
            // total shift (i.e. global shift + local shift) will be computed later on
            result.shifts.emplace_back(p, Point2D<T>(globShiftX, globShiftY));
        }
        // parallel routine
        auto routine = [&, p, shiftsOffset](int thrId) { // p and shiftsOffset by copy to avoid race condition
            // alllocate memory for patch data
            if (nullptr == patches.at(thrId)) {
                patches[thrId] = reinterpret_cast<T*>(alloc(scaler.getMovieSettings().sBytesBatch()));
                scalledPatches[thrId] = reinterpret_cast<std::complex<T>*>(alloc(scaler.getOutputSettings().fBytes()));
            }
            auto *data = patches.at(thrId);

            // allocate memory for positions of the correlation maxima
            if (nullptr == corrPositions.at(thrId)) {
                auto bytes = context.alignmentBytes();
                corrPositions[thrId] = reinterpret_cast<T*>(alloc(bytes));
                memset(corrPositions[thrId], 0, bytes); // otherwise valgrind complains
            }
            auto &correlations = corrPositions.at(thrId);

            // get data
            getPatchData(p.rec, globAlignment, data);

            // downscale patches, result is in FD
            for (auto i = 0; i < LASP.movie.n(); i += LASP.batch) {
                std::unique_lock<std::mutex> lock(mutex[0]);
                scaler.run(nullptr, 
                    data + i * scaler.getMovieSettings().sDim().sizeSingle(), 
                    scalledPatches[thrId] + i * scaler.getOutputSettings().fDim().sizeSingle(), 
                    filter);
            }
            scaler.synch(); 

            // compute correlations
            {
                std::unique_lock<std::mutex> lock(mutex[1]);
                correlater.run(scalledPatches[thrId], correlations, context.maxShift);
                correlater.synch();
            }

            // compute resulting shifts
            auto res = computeShifts(correlations, context);
            for (size_t i = 0;i < context.N;++i) {
                // update total shift (i.e. global shift + local shift)
                result.shifts[i + shiftsOffset].second += res.shifts[i];
            }
        };
        futures.emplace_back(loadPool.push(routine));
    }

    // wait till everything is done
    for (auto &f : futures) { f.get(); }

    // clean the memory
    for (auto *ptr : corrPositions) { 
        mGpu.value().unpinMemory(ptr);
        BasicMemManager::instance().give(ptr); }
    for (auto *ptr : patches) { 
        mGpu.value().unpinMemory(ptr);
        BasicMemManager::instance().give(ptr);
    }
    for (auto *ptr : scalledPatches) { 
        mGpu.value().unpinMemory(ptr);
        BasicMemManager::instance().give(ptr);
    }
    BasicMemManager::instance().give(filter);

    // compute coefficients for BSpline
    auto coeffs = BSplineHelper::computeBSplineCoeffs(movieSize, result,
            this->localAlignmentControlPoints, this->localAlignPatches,
            this->verbose, this->solverIterations);
    result.bsplineRep = core::optional<BSplineGrid<T>>(
            BSplineGrid<T>(this->localAlignmentControlPoints, coeffs.first, coeffs.second));

    return result;
}

template<typename T>
LocalAlignmentResult<T> ProgMovieAlignmentCorrelationGPU<T>::localFromGlobal(
        const MetaData& movie,
        const AlignmentResult<T> &globAlignment) {
    // FIXME this is copy from the computeLocalAlignment. Consider refactoring it
    LAOptimize();
    auto movieSize = this->getMovieSize();
    auto borders = getMovieBorders(globAlignment, this->verbose > 1);
    auto patchesLocation = this->getPatchesLocation(borders, LASP.movie);
    LocalAlignmentResult<T> result { globalHint:globAlignment, movieDim:movieSize };
    // get alignment for all patches
    for (auto &&p : patchesLocation) {
        // process it
        for (size_t i = 0; i < movieSize.n(); ++i) {
            FramePatchMeta<T> tmp = p;
            tmp.id_t = i;
            result.shifts.emplace_back(tmp, Point2D<T>(globAlignment.shifts.at(i).x, globAlignment.shifts.at(i).y));
        }
    }

    auto coeffs = BSplineHelper::computeBSplineCoeffs(movieSize, result,
            this->localAlignmentControlPoints, this->localAlignPatches,
            this->verbose, this->solverIterations);
    result.bsplineRep = core::optional<BSplineGrid<T>>(
            BSplineGrid<T>(this->localAlignmentControlPoints, coeffs.first, coeffs.second));

    return result;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::applyShiftsComputeAverage(
        const MetaData& movie, const Image<T>& dark, const Image<T>& igain,
        Image<T>& initialMic, size_t& Ninitial, Image<T>& averageMicrograph,
        size_t& N, const AlignmentResult<T> &globAlignment) {
    applyShiftsComputeAverage(movie, dark, igain, initialMic, Ninitial, averageMicrograph,
            N, localFromGlobal(movie, globAlignment));
}

template <typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getOutputStreamCount()
{
    mGpu.value().updateMemoryInfo();
    auto maxStreams = [this]()
    {
        auto count = 4;
        // upper estimation is 2 full frames of GPU data per stream
        while (2 * count * movie.getDim().xy() * sizeof(T) > this->mGpu.value().lastFreeBytes())
        {
            count--;
        }
        return std::max(count, 1);
    }();
    if (this->verbose > 1)
    {
        std::cout << "GPU streams used for output generation: " << maxStreams << "\n";
    }
    return maxStreams;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::applyShiftsComputeAverage(
        const MetaData& movieMD, const Image<T>& dark, const Image<T>& igain,
        Image<T>& initialMic, size_t& Ninitial, Image<T>& averageMicrograph,
        size_t& N, const LocalAlignmentResult<T> &alignment) {
    Ninitial = N = 0;
    if ( ! alignment.bsplineRep) {
        REPORT_ERROR(ERR_VALUE_INCORRECT,
            "Missing BSpline representation. This should not happen. Please contact developers.");
    }

    struct AuxData {
        MultidimArray<T> shiftedFrame;
        MultidimArray<T> reducedFrame;
        GeoTransformer<T> transformer;
        MultidimArray<double> croppedFrameD;
        MultidimArray<double> reducedFrameD;
        GPU stream;
        T *hIn;
        T *hOut;
    };

    auto coeffs = std::make_pair(alignment.bsplineRep.value().getCoeffsX(),
        alignment.bsplineRep.value().getCoeffsY());

    // prepare data for each thread
    ctpl::thread_pool pool = ctpl::thread_pool(getOutputStreamCount());
    auto aux = std::vector<AuxData>(pool.size());
    auto futures = std::vector<std::future<void>>();
    for (auto i = 0; i < pool.size(); ++i) {
        aux[i].stream = GPU(mGpu.value().device(), i + 1);
        aux[i].hIn = reinterpret_cast<T*>(BasicMemManager::instance().get(movie.getDim().xy() * sizeof(T), MemType::CUDA_HOST));
    }

    int frameIndex = -1;
    std::mutex mutex;
    FOR_ALL_OBJECTS_IN_METADATA(movieMD)
    {
        frameIndex++;
        if ((frameIndex >= this->nfirstSum) && (frameIndex <= this->nlastSum)) {
            // user might want to align frames 3..10, but sum only 4..6
            // by deducting the first frame that was aligned, we get proper offset to the stored memory
            auto routine = [&, frameIndex](int threadId) { // all by reference, frameIndex by copy
                int frameOffset = frameIndex - this->nfirst;
                auto &a = aux[threadId];
                a.stream.set();
                auto &frame = movie.getFrame(frameIndex);

                if ( ! this->fnInitialAvg.isEmpty()) {
                    std::unique_lock<std::mutex> lock(mutex);
                    if (0 == initialMic().yxdim)
                        initialMic() = frame;
                    else
                        initialMic() += frame;
                    Ninitial++;
                }

                if ( ! this->fnAligned.isEmpty() || ! this->fnAvg.isEmpty()) {
                    if (nullptr == a.hOut) {
                        a.hOut = reinterpret_cast<T*>(BasicMemManager::instance().get(frame.yxdim * sizeof(T), MemType::CUDA_HOST));
                    }
                    auto shiftedFrame = MultidimArray<T>(1, 1, frame.ydim, frame.xdim, a.hOut);
                    a.transformer.initLazyForBSpline(frame.xdim, frame.ydim, alignment.movieDim.n(),
                            this->localAlignmentControlPoints.x(), this->localAlignmentControlPoints.y(), this->localAlignmentControlPoints.n(), a.stream);
                    a.transformer.applyBSplineTransform(3, shiftedFrame, frame, coeffs, frameOffset);

                    a.stream.synch(); // make sure that data is fetched from GPU
                    if (this->fnAligned != "") {
                        Image<T> tmp(shiftedFrame);
                        std::unique_lock<std::mutex> lock(mutex);
                        tmp.write(this->fnAligned, frameOffset + 1, true,
                                WRITE_REPLACE);
                    }
                    if (this->fnAvg != "") {
                        std::unique_lock<std::mutex> lock(mutex);
                        if (0 == averageMicrograph().yxdim)
                            averageMicrograph() = shiftedFrame;
                        else
                            averageMicrograph() += shiftedFrame;
                        N++;
                    }
                }
            };
            futures.emplace_back(pool.push(routine));
        }
    }
    for (auto &t : futures) {
        t.get();
    }
    for (auto i = 0; i < pool.size(); ++i) {
        BasicMemManager::instance().give(aux[i].hIn);
        BasicMemManager::instance().give(aux[i].hOut);
    }
}


template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::storeSizes(const Dimensions &orig,
        const Dimensions &opt, bool applyCrop) {
    auto single = orig.copyForN(1);
    UserSettings::get(storage).insert(*this,
            getKey(optSizeXStr, single, applyCrop), opt.x());
    UserSettings::get(storage).insert(*this,
            getKey(optSizeYStr, single, applyCrop), opt.y());
    UserSettings::get(storage).store(); // write changes immediately
}

template<typename T>
std::optional<Dimensions> ProgMovieAlignmentCorrelationGPU<T>::getStoredSizes(
        const Dimensions &dim, bool applyCrop) {
    size_t x, y, batch;
    auto single = dim.copyForN(1);
    bool res = true;
    res = res && UserSettings::get(storage).find(*this,
                    getKey(optSizeXStr, single, applyCrop), x);
    res = res && UserSettings::get(storage).find(*this,
                    getKey(optSizeYStr, single, applyCrop), y);
    if (res) {
        return std::optional(Dimensions(x, y, 1, dim.n()));
    } else {
        return {};
    }
}


template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::GlobalAlignmentHelper::findBatchesThreadsStreams(const GPU &gpu, ProgMovieAlignmentCorrelationGPU &instance) {
    const auto rSize = instance.getMovieSizeRaw();
    const auto doBinning = instance.applyBinning();
    const auto mSize = doBinning ? instance.getMovieSize() : findGoodCropSize(rSize, gpu, instance); // FIXME rename to findGoodMovieSize
    const auto refSize = doBinning ? mSize : rSize;
    auto correlation = instance.getCorrelationHint(refSize); 
    auto cSize = instance.findGoodCorrelationSize(correlation, gpu);
    const auto maxBytes = gpu.lastFreeBytes() * 0.9f; // leave some buffer in case of memory fragmentation
    auto getMemReq = [this, &gpu, &rSize, doBinning]() {
        typename CUDAFlexAlignScale<T>::Params p {
        .doBinning = doBinning,
        .raw = rSize,
        .movie = movieSettings.sDim(),
        .out = correlationSettings.sDim(),
        .batch = 1,
        };
        return CUDAFlexAlignScale<T>(p, gpu).estimateBytes();
    };
    auto cond = [&mSize, this]() {
        // we want only no. of batches that can process the movie without extra invocations
        return (0 == movieSettings.sDim().n() % movieSettings.batch()) && (cpuThreads * movieSettings.batch()) <= movieSettings.sDim().n();
    };
    auto set = [&mSize, &cSize, this](size_t batch, size_t streams, size_t threads) {
        movieSettings = FFTSettings<T>(mSize, batch);
        correlationSettings = FFTSettings<T>(cSize);
        gpuStreams = streams;
        cpuThreads = threads;
    };
    // two streams to overlap memory transfers and computations, 4 threads to make sure they are fully utilized
    // more streams do not make sense because we're limited by the transfers
    // bigger batch leads to more time wasted on memory allocation - it gets importand if you have lower number of frames
    set(1, 2, 4);
    if ((getMemReq() >= maxBytes) && cond()) {
        set(1, 1, 2);
    }
    if (getMemReq() >= maxBytes) {
        REPORT_ERROR(ERR_GPU_MEMORY, "Insufficient GPU memory for processing global alignment.");
    }

    auto M = [&mSize, this]() {
        auto noOfBuffers = (bufferSize == mSize.n()) ? 1 : 2;
        auto buffers = correlationSettings.fBytesSingle() * bufferSize * noOfBuffers;
        auto plan = CudaFFT<T>().estimateTotalBytes(correlationSettings);
        return plan + buffers;
    };
    size_t bufferDivider = 1;
    size_t batchDivider = 0;
    do {
        batchDivider++;
        auto batch = cSize.n() / batchDivider; // number of correlations for FT
        bufferSize = mSize.n() / bufferDivider; // number of input frames in a single buffer
        if (bufferSize > batch) {
            bufferDivider += (bufferDivider == 1) ? 2 : 1; // we use two buffers, so we need the same memory for batch == 1 and == 2
        }
        correlationSettings = FFTSettings<T>(cSize, batch, false, false);
    } while (M() >= maxBytes);
}


template<typename T>
AlignmentResult<T> ProgMovieAlignmentCorrelationGPU<T>::computeGlobalAlignment(
        const MetaData &movieMD, const Image<T> &dark, const Image<T> &igain) {
    // prepare storage for the movie
    movie.set(this->getMovieSize(), this->applyBinning());


    globalHelper.findBatchesThreadsStreams(mGpu.value(), *this);

    auto &movieSettings = globalHelper.movieSettings;
    auto &correlationSettings = globalHelper.correlationSettings;
    T actualScale = correlationSettings.sDim().x() / (T)movieSettings.sDim().x();

    // prepare filter
    MultidimArray<T> filterTmp = this->createLPF(this->getPixelResolution(actualScale), correlationSettings.sDim());
    T* filterData = reinterpret_cast<T*>(BasicMemManager::instance().get(filterTmp.nzyxdim *sizeof(T), MemType::CUDA_MANAGED));
    memcpy(filterData, filterTmp.data, filterTmp.nzyxdim *sizeof(T));
    
    if (this->verbose) {
        std::cout << "Requested scale factor: " << this->getScaleFactor() << std::endl;
        std::cout << "Actual scale factor (X): " << actualScale << std::endl;
        std::cout << globalHelper << std::endl;
    }

    // create a buffer for correlations in FD
    auto *scaledFrames = reinterpret_cast<std::complex<T>*>(BasicMemManager::instance().get(correlationSettings.fBytesSingle() * movieSettings.sDim().n(), MemType::CPU_PAGE_ALIGNED));

    auto cpuPool = ctpl::thread_pool(globalHelper.cpuThreads);
    auto gpuPool = ctpl::thread_pool(globalHelper.gpuStreams);
    auto auxData = std::vector<CUDAFlexAlignScale<T>>();
    auxData.reserve(gpuPool.size());

    std::vector<GPU> streams(gpuPool.size());
    for (auto i = 0; i < streams.size(); ++i) {
        streams.at(i) = GPU(mGpu.value().device(), i + 1);
        typename CUDAFlexAlignScale<T>::Params p;
        p.movie = movieSettings.sDim();
        p.out = correlationSettings.sDim();
        p.raw = this->getMovieSizeRaw();
        p.doBinning = this->applyBinning();
        auxData.emplace_back(p, streams[i]);
        auto routine = [&auxData, &streams, i](int stream) {
            streams[i].set();
            auxData[i].init();
        };
        gpuPool.push(routine);
    }

    for (auto i = 0; i < movieSettings.sDim().n(); i += movieSettings.batch()) {
        auto routine = [&](int thrId, size_t first)
        {
            auto *rawFrame = loadFrame(movieMD, dark, igain, first);
            auto *frame = reinterpret_cast<T *>(BasicMemManager::instance().get(movieSettings.sBytesSingle(), MemType::CUDA_HOST));
            if (this->applyBinning()) {
                movie.setFrameData(first, frame); // we want to store the binned frame
            } else {
                movie.setFrameData(first, rawFrame); // we want to store the raw frame
                getCroppedFrame(movieSettings, frame, rawFrame);
            }
            gpuPool.push([&](int stream) { 
                auxData[stream].run(rawFrame, frame, scaledFrames + first * correlationSettings.fDim().sizeSingle(), filterData);
                streams[stream].synch(); 
            }).get();
            BasicMemManager::instance().give(this->applyBinning() ? rawFrame : frame);
        };
        cpuPool.push(routine, i);
    }
    cpuPool.stop(true);
    gpuPool.stop(true);

    auxData.clear(); // to release unnecessary data

    BasicMemManager::instance().release();


    // auto result = computeShifts(this->verbose, this->maxShift, scaledFrames, correlationSettings,
    //     movieSettings.sDim().n(),
    //     scale, globalHelper.bufferSize, {});

    const AlignmentContext context {
        .verbose = this->verbose,
        .maxShift = static_cast<float>(this->maxShift),
        .N = movieSettings.sDim().n(),
        .scale = std::make_pair(movie.getDim().x() / (T) correlationSettings.sDim().x(),
            movie.getDim().y() / (T) correlationSettings.sDim().y()),
        .refFrame = std::nullopt,
        .out = correlationSettings.sDim(),
    };

    auto *correlations = reinterpret_cast<T*>(BasicMemManager::instance().get(context.alignmentBytes(), MemType::CUDA_HOST));
    
// CorrelationData<T> corrAuxData;
    // corrAuxData.alloc(correlationSettings, globalHelper.bufferSize, streams[1]);
    typename CUDAFlexAlignCorrelate<T>::Params p {
        .dim = correlationSettings.sDim().copyForN(context.N),
        .bufferSize = globalHelper.bufferSize,
        .batch = correlationSettings.batch(),
    };
    auto tmp = CUDAFlexAlignCorrelate<T>(p, streams[1]);
    tmp.init();
    tmp.run(scaledFrames, correlations, context.maxShift / context.scale.first);
    // result is a centered correlation function with (hopefully) a cross
    // indicating the requested shift
streams[1].synch();
// corrAuxData.release();
    auto result = computeShifts(correlations, context);
    BasicMemManager::instance().give(correlations);
    BasicMemManager::instance().give(filterData);
    BasicMemManager::instance().give(scaledFrames);
    BasicMemManager::instance().release(MemType::CUDA);
    return result;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::getCroppedFrame(const FFTSettings<T> &settings,
        T *dest, T *src) {
    for (size_t y = 0; y < settings.sDim().y(); ++y) {
        memcpy(dest + (settings.sDim().x() * y),
                src + (movie.getDim().x() * y),
                settings.sDim().x() * sizeof(T));
    }
}

template<typename T>
ProgMovieAlignmentCorrelationGPU<T>::Movie::~Movie() {
    for (auto &f : mFrames) {
        BasicMemManager::instance().give(f.data);
    }
    mFrames.clear();
}

template<typename T>
T* ProgMovieAlignmentCorrelationGPU<T>::loadFrame(const MetaData& movieMD,
        const Image<T>& dark, const Image<T>& igain, size_t index) {
    const auto &movieDim = this->getMovieSizeRaw();
    int frameIndex = -1;
    size_t counter = 0;
    for (size_t objId : movieMD.ids())
    {
        // get to correct index
        frameIndex++;
        if (frameIndex < this->nfirst) continue;
        if (frameIndex > this->nlast) break;

        if (counter == index) {
            // load frame
            auto *ptr = reinterpret_cast<T*>(BasicMemManager::instance().get(movieDim.xy() * sizeof(T), MemType::CPU_PAGE_ALIGNED));
            auto mda = MultidimArray<T>(1, 1, movieDim.y(), movieDim.x(), ptr);
            Image<T> frame(mda);
            AProgMovieAlignmentCorrelation<T>::loadFrame(movieMD, dark, igain, objId, frame);
            return ptr;
        }
        counter++;
    }
    return nullptr;
}

template <typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::computeShifts(
    T *correlations, const AlignmentContext &context)
{ // pass by copy, this will be run asynchronously)
    // N is number of images, n is number of correlations
    // compute correlations (each frame with following ones)

    // result is a centered correlation function with (hopefully) a cross
    // indicating the requested shift

        auto noOfCorrelations = context.N * (context.N - 1) / 2;
        // we are done with the input data, so release it
        Matrix2D<T> A(noOfCorrelations, context.N - 1);
        Matrix1D<T> bX(noOfCorrelations), bY(noOfCorrelations);

        // find the actual shift (max peak) for each pair of frames
        // and create a set or equations
        size_t idx = 0;

        for (size_t i = 0; i < context.N - 1; ++i) {
            for (size_t j = i + 1; j < context.N; ++j) {
                auto index = static_cast<size_t>(correlations[idx]);
                bX(idx) = correlations[2*idx] - (context.out.x() / 2.0);
                bY(idx) = correlations[2*idx+1] - (context.out.y() / 2.0);
                bX(idx) *= context.scale.first; // scale to expected size
                bY(idx) *= context.scale.second;
                if (context.verbose > 1) {
                    std::cerr << "Frame " << i << " to Frame " << j << " -> ("
                            << bX(idx) << "," << bY(idx) << ")" << std::endl;
                }
                for (int ij = i; ij < j; ij++) {
                    A(idx, ij) = 1;
                }
                idx++;
            }
        }
    return this->computeAlignment(bX, bY, A, context.refFrame, context.N, context.verbose);
}

// explicit specialization
template class ProgMovieAlignmentCorrelationGPU<float> ;
