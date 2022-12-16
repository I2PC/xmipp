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
#include "core/utils/memory_utils.h"
#include <thread>
#include "reconstruction_cuda/cuda_gpu_movie_alignment_correlation.h"
#include "reconstruction_cuda/cuda_gpu_geo_transformer.h"
#include "data/filters.h"
#include "core/userSettings.h"
#include "reconstruction_cuda/cuda_fft.h"
#include "core/utils/time_utils.h"
#include "reconstruction_adapt_cuda/basic_mem_manager.h"
#include "core/xmipp_image_generic.h"
#include <CTPL/ctpl_stl.h>

#include "reconstruction_cuda/cuda_flexalign_scale.h"

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
    std::cout << "Device:              " << gpu.value().device() << " (" << gpu.value().getUUID() << ")" << "\n";
    std::cout << "Benchmark storage    " << (storage.empty() ? "Default" : storage) << "\n";
    std::cout << "Patches avg:         " << patchesAvg << "\n";
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::readParams() {
    AProgMovieAlignmentCorrelation<T>::readParams();

    // read GPU
    int device = this->getIntParam("--device");
    if (device < 0)
        REPORT_ERROR(ERR_ARG_INCORRECT,
            "Invalid GPU device");
    gpu = core::optional<GPU>(device);
    gpu.value().set();

    // read permanent storage
    storage = this->getParam("--storage");

    // read patch averaging
    patchesAvg = this->getIntParam("--patchesAvg");
    if (patchesAvg < 1)
        REPORT_ERROR(ERR_ARG_INCORRECT,
            "Patch averaging has to be at least one.");
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::GlobalAlignmentHelper::findGoodCropSize(const Dimensions &movie, const GPU &gpu, ProgMovieAlignmentCorrelationGPU &instance) {
    const bool crop = true;
    auto optDim = instance.getStoredSizes(movie, crop);
    if (optDim) {
        return optDim.value().copyForN(movie.n());
    }
    std::cout << "Benchmarking cuFFT ..." << std::endl;
    auto hint = FFTSettings<T>(movie.createSingle()); // movie frame is big enought to give us an idea
    auto candidate = std::unique_ptr<FFTSettings<T>>(CudaFFT<T>::findOptimal(gpu, hint, 0, hint.sDim().x() == hint.sDim().y(), 10, crop, true));
    if (!candidate) {
        REPORT_ERROR(ERR_GPU_MEMORY, "Insufficient GPU memory for processing a single frame of the movie.");
    }
    instance.storeSizes(movie, candidate->sDim(), crop);
    return candidate->sDim().copyForN(movie.n());
}

template<typename T>
auto  ProgMovieAlignmentCorrelationGPU<T>::findGoodCorrelationSize(const Dimensions &hint, const GPU &gpu) {
    const bool crop = false;
    auto optDim = this->getStoredSizes(hint, crop);
    if (optDim) {
        return optDim.value().copyForN(hint.n());
    }
    std::cout << "Benchmarking cuFFT ..." << std::endl;
    auto settings = FFTSettings<T>(hint.copyForN((std::ceil(sqrt(hint.n() * 2))))); // test just number of frames, to get an idea (it's faster)
    auto candidate = std::unique_ptr<FFTSettings<T>>(CudaFFT<T>::findOptimal(gpu, settings, 0, settings.sDim().x() == settings.sDim().y(), 20, crop, true));
    if (!candidate) {
        REPORT_ERROR(ERR_GPU_MEMORY, "Insufficient GPU memory for processing a correlations of the movie.");
    }
    this->storeSizes(hint, candidate->sDim(), crop);
    return candidate->sDim().copyForN(hint.n());
}

template<typename T>
auto  ProgMovieAlignmentCorrelationGPU<T>::LocalAlignmentHelper::findGoodPatchSize(const Dimensions &hint, const GPU &gpu, ProgMovieAlignmentCorrelationGPU &instance) {
    const bool crop = false;
    auto optDim = instance.getStoredSizes(hint, crop);
    if (optDim) {
        return optDim.value().copyForN(hint.n());
    }
    std::cout << "Benchmarking cuFFT ..." << std::endl;
    auto settings = FFTSettings<T>(hint);
    auto candidate = std::unique_ptr<FFTSettings<T>>(CudaFFT<T>::findOptimal(gpu, settings, 0, settings.sDim().x() == settings.sDim().y(), 20, crop, true));
    if (!candidate) {
        REPORT_ERROR(ERR_GPU_MEMORY, "Insufficient GPU memory for processing a correlations of the movie.");
    }
    instance.storeSizes(hint, candidate->sDim(), crop);
    return candidate->sDim();
}


template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getCorrelationHint(
        const Dimensions &d) {
    auto getNearestEven = [this] (size_t v, T minScale, size_t shift) { // scale is less than 1
        size_t size = std::ceil(getCenterSize(shift) / 2.f) * 2; // to get even size
        while ((size / (float)v) < minScale) {
            size += 2;
        }
        return size;
    };
    const T requestedScale = this->getScaleFactor();
    // hint, possibly bigger then requested, so that it fits max shift window
    Dimensions hint(getNearestEven(d.x(), requestedScale, this->maxShift),
            getNearestEven(d.y(), requestedScale, this->maxShift),
            d.z(), (d.n() * (d.n() - 1)) / 2); // number of correlations);
    return hint;
}

template<typename T>
std::vector<FramePatchMeta<T>> ProgMovieAlignmentCorrelationGPU<T>::getPatchesLocation(
        const std::pair<T, T> &borders,
        const Dimensions &movie, const Dimensions &patch) {
    size_t patchesX = this->localAlignPatches.first;
    size_t patchesY = this->localAlignPatches.second;
    T windowXSize = movie.x() - 2 * borders.first;
    T windowYSize = movie.y() - 2 * borders.second;
    T corrX = std::ceil(
            ((patchesX * patch.x()) - windowXSize) / (T) (patchesX - 1));
    T corrY = std::ceil(
            ((patchesY * patch.y()) - windowYSize) / (T) (patchesY - 1));
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
            result.emplace_back(
                    FramePatchMeta<T> { .rec = r, .id_x = x, .id_y =
                    y });
        }
    }
    return result;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::getPatchData(const Rectangle<Point2D<T>> &patch, 
        const AlignmentResult<T> &globAlignment, T *result) {
    const auto &movieDim = movie.getDim();
    const int n = movieDim.n();
    const auto patchSize = patch.getSize();
    const auto bufferBytes = patchSize.x * sizeof(T);
    auto *buffer = new T[static_cast<int>(patchSize.x)]; // faster than memory manager
    for (int t = 0; t < n; ++t) {// for each patch
            for (int y = 0; y < patchSize.y; ++y) { // for each row
                bool copy = true;
                // while averaging odd num of frames, use copy equally from previous and following frames
                // otherwise prefer following frames
                for (int f = std::max(0, t - ((patchesAvg - 1) / 2)); f <= std::min(n - 1, t + (patchesAvg / 2)); ++f) {
                    const auto *frame = movie.getFrame(f).data;
                    const int xShift = std::round(globAlignment.shifts[f].x);
                    const int yShift = std::round(globAlignment.shifts[f].y);
                    // notice we don't test any access - it should always be within the boundaries of the frame
                    // see implementation of patch position generation and frame border computation
                    const int srcY = patch.tl.y + y + yShift;
                    const int srcX = patch.tl.x + xShift;
                    auto *src = frame + srcY * movieDim.x() + srcX;
                    if (copy) {
                        memcpy(buffer, src, bufferBytes);
                    } else {
                        for (int x = 0; x < patchSize.x; ++x) {
                            buffer[x] += src[x];
                        }
                    }
                    copy = false;
                }
                const int patchOffset = t * patchSize.x * patchSize.y;
                const int destIndex = patchOffset + y * patchSize.x;
                memcpy(result + destIndex, buffer, bufferBytes); // write result
            }
    }
    delete[] buffer;
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
        std::cout << "Movie borders: x=" << res.first << " y=" << res.second
                << std::endl;
    }
    return res;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::LocalAlignmentHelper::findBatchesThreadsStreams(const GPU &gpu, ProgMovieAlignmentCorrelationGPU &instance) {
    const auto reqPatchSize = instance.getRequestedPatchSize();
    auto pSize = findGoodPatchSize(Dimensions(reqPatchSize.first, reqPatchSize.second, 1, instance.getMovieSize().n()), gpu, instance);
    auto correlation = instance.getCorrelationHint(pSize);
    auto cSize = instance.findGoodCorrelationSize(correlation, gpu);
    const auto maxBytes = gpu.lastFreeBytes() * 0.9f; // leave some buffer in case of memory fragmentation
    auto getMemReq = [&pSize, &gpu, this]() {
        // for scale 
        typename CUDAFlexAlignScale<T>::Params p {
        .doBinning = false,
        .raw = Dimensions(0),
        .movie = patchSettings.sDim(),
        .movieBatch = patchSettings.batch(),
        .out = correlationSettings.sDim(),
        .outBatch = correlationSettings.batch(),
        };
        auto scale = CUDAFlexAlignScale<T>(p, gpu).estimateBytes();
        // for correlation
        auto noOfBuffers = (bufferSize == pSize.n()) ? 1 : 2;
        auto buffers = correlationSettings.fBytesSingle() * bufferSize * noOfBuffers;
        auto plan = CudaFFT<T>().estimateTotalBytes(correlationSettings);
        return scale + plan + buffers;
    };
    auto cond = [this]() {
        // we want only no. of batches that can process the patches without extra invocations
        return 0 == patchSettings.sDim().n() % patchSettings.batch();
    };

    // we're gonna run both scaling and correlation in two streams to overlap memory transfers and computations
    // more streams do not make sense because we're limited by the transfers
    size_t bufferDivider = 1;
    size_t corrBatchDivider = 0;
    do {
        corrBatchDivider++;
        auto corrBatch = cSize.n() / corrBatchDivider; // number of correlations for FT
        bufferSize = pSize.n() / bufferDivider; // number of patches in a single buffer
        if (bufferSize > corrBatch) {
            bufferDivider += (bufferDivider == 1) ? 2 : 1; // we use two buffers, so we need the same memory for batch == 1 and == 2
        }
        correlationSettings = FFTSettings<T>(cSize, corrBatch, false, false);
        for (auto scaleBatch = pSize.n(); scaleBatch > 0; --scaleBatch) {
            patchSettings = FFTSettings<T>(pSize, scaleBatch);
            if (cond() && getMemReq() <= maxBytes) {
                return;
            }
        }
    } while (true);
}


template<typename T>
LocalAlignmentResult<T> ProgMovieAlignmentCorrelationGPU<T>::computeLocalAlignment(
        const MetaData &movieMD, const Image<T> &dark, const Image<T> &igain,
        const AlignmentResult<T> &globAlignment) {
    using memoryUtils::MB;

    localHelper.findBatchesThreadsStreams(gpu.value(), *this);
    auto movieSize = this->getMovieSize();
    auto &patchSettings = localHelper.patchSettings;
    auto &correlationSettings = localHelper.correlationSettings;
    auto borders = getMovieBorders(globAlignment, this->verbose > 1);
    auto patchesLocation = this->getPatchesLocation(borders, movieSize,
            patchSettings.sDim());
    T actualScale = correlationSettings.sDim().x() / (T)patchSettings.sDim().x(); // assuming we use square patches

    if (this->verbose) {
        std::cout << "Actual scale factor (X): " << actualScale << std::endl;
        std::cout << localHelper << std::endl;
    }


    if ((movieSize.x() < patchSettings.sDim().x())
        || (movieSize.y() < patchSettings.sDim().y())) {
        REPORT_ERROR(ERR_PARAM_INCORRECT, "Movie is too small for local alignment.");
    }

    // prepare filter
    // FIXME DS make sure that the resulting filter is correct, even if we do non-uniform scaling
    MultidimArray<T> filterTmp = this->createLPF(this->getPixelResolution(actualScale), correlationSettings.sDim());
    T* filterData = reinterpret_cast<T*>(BasicMemManager::instance().get(filterTmp.nzyxdim *sizeof(T), MemType::CUDA_MANAGED));
    memcpy(filterData, filterTmp.data, filterTmp.nzyxdim *sizeof(T));

    // prepare result
    LocalAlignmentResult<T> result { globalHint:globAlignment, movieDim:movieSize};
    result.shifts.reserve(patchesLocation.size() * movieSize.n());
    auto refFrame = core::optional<size_t>(globAlignment.refFrame);

    auto createContext = [&, this](auto &p) {
        static std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex); // we need to lock this part to ensure serial access to result.shifts
        auto context = PatchContext(result);
        context.verbose = this->verbose;
        context.maxShift = this->maxShift;
        context.shiftsOffset = result.shifts.size();
        context.N = patchSettings.sDim().n();
        context.scale = std::make_pair(patchSettings.sDim().x() / (T) correlationSettings.sDim().x(),
            patchSettings.sDim().y() / (T) correlationSettings.sDim().y());
        context.refFrame = refFrame;
        context.centerSize = getCenterSize(this->maxShift);
        context.framesInCorrelationBuffer = localHelper.bufferSize;
        context.correlationSettings = correlationSettings;
        // prefill some info about patch
        for (size_t i = 0;i < movieSize.n();++i) {
            FramePatchMeta<T> tmp = p;
            // keep consistent with data loading
            int globShiftX = std::round(globAlignment.shifts.at(i).x);
            int globShiftY = std::round(globAlignment.shifts.at(i).y);
            tmp.id_t = i;
            // total shift (i.e. global shift + local shift) will be computed later on
            result.shifts.emplace_back(tmp, Point2D<T>(globShiftX, globShiftY));
        }
        return context;
    };

    auto loadPool = ctpl::thread_pool(localHelper.cpuThreads);
    std::vector<T*> corrBuffers(loadPool.size()); // initializes to nullptrs
    std::vector<T*> patchData(loadPool.size()); // initializes to nullptrs
    std::vector<std::complex<T>*> scalledPatches(loadPool.size()); // initializes to nullptrs
    std::vector<std::future<void>> futures;
    futures.reserve(patchesLocation.size());
    GPU streams[2] = {GPU(gpu.value().device(), 1), GPU(gpu.value().device(), 2)};
    streams[0].set();
    streams[1].set();

    std::mutex mutex[3];

    typename CUDAFlexAlignScale<T>::Params p {
        .doBinning = false,
        .raw = Dimensions(0),
        .movie = patchSettings.sDim(),
        .movieBatch = patchSettings.batch(),
        .out = correlationSettings.sDim(),
        .outBatch = correlationSettings.batch(),
    };
    auto auxData = CUDAFlexAlignScale<T>(p, streams[0]);
    auxData.init();
    CorrelationData<T> corrAuxData;
    corrAuxData.alloc(correlationSettings, localHelper.bufferSize, streams[1]);

    // use additional thread that would load the data at the background
    // get alignment for all patches and resulting correlations
    for (auto &&p : patchesLocation) {
        auto routine = [&](int thrId) {
            auto context = createContext(p);

            auto alloc = [&](size_t bytes) {
                 // it's faster to allocate CPU memory and then pin it, because registering can run in parallel
                auto *ptr = BasicMemManager::instance().get(bytes, MemType::CPU_PAGE_ALIGNED);
                gpu.value().pinMemory(ptr, bytes);
                return ptr;
            };

            // alllocate and clear patch data
            if (nullptr == patchData.at(thrId)) {
                patchData[thrId] = reinterpret_cast<T*>(alloc(patchSettings.sBytes()));
                scalledPatches[thrId] = reinterpret_cast<std::complex<T>*>(alloc(correlationSettings.fBytes()));
            }
            auto *data = patchData.at(thrId);

            // allocate and clear correlation data
            if (nullptr == corrBuffers.at(thrId)) {
                corrBuffers[thrId] = reinterpret_cast<T*>(alloc(context.corrElems() * sizeof(T)));
            }
            auto *correlations = corrBuffers.at(thrId);
            memset(correlations, 0, context.corrElems() * sizeof(T));

            // get data
                // std::unique_lock<std::mutex> lock(mutex[2]);
            // memset(data, 0, patchSettings.sBytes()); // for version 6
            getPatchData(p.rec, globAlignment, data);

            // convert to FFT, downscale them and compute correlations
                for (auto i = 0; i < patchSettings.sDim().n(); i += patchSettings.batch()) {
                    std::unique_lock<std::mutex> lock(mutex[0]);
                    auxData.run(nullptr, data + i * patchSettings.sElemsBatch(), scalledPatches[thrId] + i * correlationSettings.fDim().sizeSingle(), filterData);
                }
                streams[0].synch();

                // performFFTAndScale<T>(data, patchSettings.sDim().n(), patchSettings.sDim().x(), patchSettings.sDim().y(), patchSettings.batch(),
                //         correlationSettings.fDim().x(), correlationSettings.sDim().y(), filter);
                {
                    std::unique_lock<std::mutex> lock(mutex[1]);
                computeCorrelations(context.maxShift / context.scale.first, context.N, correlationSettings, scalledPatches[thrId],
                        context.framesInCorrelationBuffer,
                        correlations, corrAuxData, streams[1]);
                }
                streams[1].synch();
            // }).get(); // wait till done - i.e. correlations are computed and on CPU

            // compute resulting shifts
            computeShifts(correlations, context);
        };
        futures.emplace_back(loadPool.push(routine));
    }
    // wait for the last processing thread
    for (auto &f : futures) { f.get(); }

    for (auto *ptr : corrBuffers) { 
        gpu.value().unpinMemory(ptr);
        BasicMemManager::instance().give(ptr); }
    for (auto *ptr : patchData) { 
        gpu.value().unpinMemory(ptr);
        BasicMemManager::instance().give(ptr);
    }
    for (auto *ptr : scalledPatches) { 
        gpu.value().unpinMemory(ptr);
        BasicMemManager::instance().give(ptr);
    }
    BasicMemManager::instance().give(filterData);
    corrAuxData.release();

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
    localHelper.findBatchesThreadsStreams(gpu.value(), *this);
    auto movieSize = this->getMovieSize();
    auto &patchSettings = localHelper.patchSettings;
    auto &correlationSettings = localHelper.correlationSettings;
    auto borders = getMovieBorders(globAlignment, this->verbose > 1);
    auto patchesLocation = this->getPatchesLocation(borders, movieSize,
            patchSettings.sDim());
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
    gpu.value().updateMemoryInfo();
    auto maxStreams = [this]()
    {
        auto count = 4;
        // upper estimation is 2 full frames of GPU data per stream
        while (2 * count * movie.getDim().xy() * sizeof(T) > this->gpu.value().lastFreeBytes())
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
        aux[i].stream = GPU(gpu.value().device(), i + 1);
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
        .movieBatch = 1,
        .out = correlationSettings.sDim(),
        .outBatch = 1,
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
        correlationSettings = FFTSettings<T>(cSize, batch);
    } while (M() >= maxBytes);
}


template<typename T>
AlignmentResult<T> ProgMovieAlignmentCorrelationGPU<T>::computeGlobalAlignment(
        const MetaData &movieMD, const Image<T> &dark, const Image<T> &igain) {
    // prepare storage for the movie
    movie.set(this->getMovieSize(), this->applyBinning());


    globalHelper.findBatchesThreadsStreams(gpu.value(), *this);

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
        streams.at(i) = GPU(gpu.value().device(), i + 1);
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

    auto scale = std::make_pair(movie.getDim().x() / (T) correlationSettings.sDim().x(),
        movie.getDim().y() / (T) correlationSettings.sDim().y());

    auto result = computeShifts(this->verbose, this->maxShift, scaledFrames, correlationSettings,
        movieSettings.sDim().n(),
        scale, globalHelper.bufferSize, {});

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
    T *correlations,
    PatchContext context)
{ // pass by copy, this will be run asynchronously)
    // N is number of images, n is number of correlations
    // compute correlations (each frame with following ones)

    // result is a centered correlation function with (hopefully) a cross
    // indicating the requested shift

    // auto routine = [this](int, PatchContext context, T* correlations) {
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
                bX(idx) = correlations[2*idx] - (context.correlationSettings.sDim().x() / 2.0);
                bY(idx) = correlations[2*idx+1] - (context.correlationSettings.sDim().y() / 2.0);
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
    auto result = this->computeAlignment(bX, bY, A, context.refFrame, context.N, context.verbose);
    for (size_t i = 0;i < context.N;++i) {
        // update total shift (i.e. global shift + local shift)
        context.result.shifts[i + context.shiftsOffset].second += result.shifts[i];
    }

}

template<typename T>
AlignmentResult<T> ProgMovieAlignmentCorrelationGPU<T>::computeShifts(int verbose,
        size_t maxShift,
        std::complex<T>* data, const FFTSettings<T>& settings, size_t N,
        std::pair<T, T>& scale,
        size_t framesInCorrelationBuffer,
        const core::optional<size_t>& refFrame) {
    // N is number of images, n is number of correlations
    // compute correlations (each frame with following ones)
    size_t centerSize = getCenterSize(maxShift);
    auto *correlations = new T[N*(N-1)/2 * centerSize * centerSize]();
    computeCorrelations(maxShift / scale.first, N, data, settings.fDim().x(),
            settings.sDim().x(),
            settings.fDim().y(), framesInCorrelationBuffer,
            settings.batch(), correlations);
    // result is a centered correlation function with (hopefully) a cross
    // indicating the requested shift

    // we are done with the input data, so release it
    Matrix2D<T> A(N * (N - 1) / 2, N - 1);
    Matrix1D<T> bX(N * (N - 1) / 2), bY(N * (N - 1) / 2);

    // find the actual shift (max peak) for each pair of frames
    // and create a set or equations
    size_t idx = 0;
    MultidimArray<T> Mcorr(centerSize, centerSize);
    T* origData = Mcorr.data;

    for (size_t i = 0; i < N - 1; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            auto index = static_cast<size_t>(correlations[idx]);
            auto max = 0.0;
            bX(idx) = correlations[2*idx] - (settings.sDim().x() / 2.0);//index % settings.sDim().x() - (settings.sDim().x() / 2.0);
            bY(idx) = correlations[2*idx+1] - (settings.sDim().y() / 2.0);//index / settings.sDim().x() - (settings.sDim().y() / 2.0);
            // size_t offset = idx * centerSize * centerSize;
            // Mcorr.data = correlations + offset;
            // Mcorr.setXmippOrigin();
            // size_t index;
            // auto max = myBestShift(Mcorr, bX(idx), bY(idx), NULL,
            //         maxShift / scale.first, index);
            // printf("%f [%f, %f] at index %ld (after conversion: [%f, %f]\n",max, bX(idx), bY(idx), index, bX(idx) * scale.first, bY(idx) * scale.second);
            bX(idx) *= scale.first; // scale to expected size
            bY(idx) *= scale.second;
            if (verbose > 1) {
                std::cerr << "Frame " << i << " to Frame " << j << " -> ("
                        << bX(idx) << "," << bY(idx) << ")" << std::endl;
            }
            for (int ij = i; ij < j; ij++) {
                A(idx, ij) = 1;
            }
            idx++;
        }
    }
    Mcorr.data = origData;
    delete[] correlations;

    // now get the estimated shift (from the equation system)
    // from each frame to successing frame
    AlignmentResult<T> result = this->computeAlignment(bX, bY, A, refFrame, N, verbose);
    return result;
}

// explicit specialization
template class ProgMovieAlignmentCorrelationGPU<float> ;
