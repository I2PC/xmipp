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

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::defineParams() {
    AProgMovieAlignmentCorrelation<T>::defineParams();
    this->addParamsLine("  [--device <dev=0>]                 : GPU device to use. 0th by default");
    this->addParamsLine("  [--storage <fn=\"\">]              : Path to file that can be used to store results of the benchmark");
    this->addParamsLine("  [--patchesAvg <avg=3>]             : Number of near frames used for averaging a single patch");
    this->addParamsLine("  [--skipAutotuning]                 : Skip autotuning of the cuFFT library");
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
    std::cout << "Autotuning:          " << (skipAutotuning ? "off" : "on") << std::endl;
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

    skipAutotuning = this->checkParam("--skipAutotuning");

    // read patch averaging
    patchesAvg = this->getIntParam("--patchesAvg");
    if (patchesAvg < 1)
        REPORT_ERROR(ERR_ARG_INCORRECT,
            "Patch averaging has to be at least one.");
}

template<typename T>
FFTSettings<T> ProgMovieAlignmentCorrelationGPU<T>::getSettingsOrBenchmark(
        const Dimensions &d, size_t extraBytes, bool crop) {
    auto optSetting = getStoredSizes(d, crop);
    FFTSettings<T> result =
            optSetting ?
                    optSetting.value() : runBenchmark(d, extraBytes, crop);
    if (!optSetting) {
        storeSizes(d, result, crop);
    }
    return result;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::GlobalAlignmentHelper::findGoodCropSize(const Dimensions &movie, const GPU &gpu, ProgMovieAlignmentCorrelationGPU &instance) {
    const bool crop = true;
    auto optDim = instance.getStoredSizesNew(movie, crop);
    if (optDim) {
        return optDim.value().copyForN(movie.n());
    }
    std::cout << "Benchmarking cuFFT ..." << std::endl;
    auto hint = FFTSettings<T>(movie.createSingle()); // movie frame is big enought to give us an idea
    auto candidate = std::unique_ptr<FFTSettings<T>>(CudaFFT<T>::findOptimal(gpu, hint, 0, hint.sDim().x() == hint.sDim().y(), 10, crop, true));
    if (!candidate) {
        REPORT_ERROR(ERR_GPU_MEMORY, "Insufficient GPU memory for processing a single frame of the movie.");
    }
    instance.storeSizesNew(movie, candidate->sDim(), crop);
    return candidate->sDim().copyForN(movie.n());
}

template<typename T>
auto  ProgMovieAlignmentCorrelationGPU<T>::GlobalAlignmentHelper::findGoodCorrelationSize(const Dimensions &hint, const GPU &gpu, ProgMovieAlignmentCorrelationGPU &instance) {
    const bool crop = false;
    auto optDim = instance.getStoredSizesNew(hint, crop);
    if (optDim) {
        return optDim.value().copyForN(hint.n());
    }
    std::cout << "Benchmarking cuFFT ..." << std::endl;
    auto settings = FFTSettings<T>(hint.copyForN((std::ceil(sqrt(hint.n() * 2))))); // test just number of frames, to get an idea (it's faster)
    auto candidate = std::unique_ptr<FFTSettings<T>>(CudaFFT<T>::findOptimal(gpu, settings, 0, settings.sDim().x() == settings.sDim().y(), 20, crop, true));
    if (!candidate) {
        REPORT_ERROR(ERR_GPU_MEMORY, "Insufficient GPU memory for processing a correlations of the movie.");
    }
    instance.storeSizesNew(hint, candidate->sDim(), crop);
    return candidate->sDim().copyForN(hint.n());
}

template<typename T>
auto  ProgMovieAlignmentCorrelationGPU<T>::LocalAlignmentHelper::findGoodCorrelationSize(const Dimensions &hint, const GPU &gpu, ProgMovieAlignmentCorrelationGPU &instance) {
    const bool crop = false;
    auto optDim = instance.getStoredSizesNew(hint, crop);
    if (optDim) {
        return optDim.value().copyForN(hint.n());
    }
    std::cout << "Benchmarking cuFFT ..." << std::endl;
    auto settings = FFTSettings<T>(hint.copyForN((std::ceil(sqrt(hint.n() * 2))))); // test just number of frames, to get an idea (it's faster)
    auto candidate = std::unique_ptr<FFTSettings<T>>(CudaFFT<T>::findOptimal(gpu, settings, 0, settings.sDim().x() == settings.sDim().y(), 20, crop, true));
    if (!candidate) {
        REPORT_ERROR(ERR_GPU_MEMORY, "Insufficient GPU memory for processing a correlations of the movie.");
    }
    instance.storeSizesNew(hint, candidate->sDim(), crop);
    return candidate->sDim().copyForN(hint.n());
}


template<typename T>
auto  ProgMovieAlignmentCorrelationGPU<T>::LocalAlignmentHelper::findGoodPatchSize(const Dimensions &hint, const GPU &gpu, ProgMovieAlignmentCorrelationGPU &instance) {
    const bool crop = false;
    auto optDim = instance.getStoredSizesNew(hint, crop);
    if (optDim) {
        return optDim.value().copyForN(hint.n());
    }
    std::cout << "Benchmarking cuFFT ..." << std::endl;
    auto settings = FFTSettings<T>(hint);
    auto candidate = std::unique_ptr<FFTSettings<T>>(CudaFFT<T>::findOptimal(gpu, settings, 0, settings.sDim().x() == settings.sDim().y(), 20, crop, true));
    if (!candidate) {
        REPORT_ERROR(ERR_GPU_MEMORY, "Insufficient GPU memory for processing a correlations of the movie.");
    }
    instance.storeSizesNew(hint, candidate->sDim(), crop);
    return candidate->sDim();
}

template<typename T>
FFTSettings<T> ProgMovieAlignmentCorrelationGPU<T>::getMovieSettings(
        const MetaData &movie, bool optimize) {
    gpu.value().updateMemoryInfo();
    auto dim = this->getMovieSize();

    if (optimize) {
        size_t maxFilterBytes = getMaxFilterBytes(dim);
        return getSettingsOrBenchmark(dim, maxFilterBytes, true);
    } else {
        return FFTSettings<T>(dim, 1, false);
    }
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
FFTSettings<T> ProgMovieAlignmentCorrelationGPU<T>::getCorrelationSettings(
        const FFTSettings<T> &s) {
    gpu.value().updateMemoryInfo();
    auto getNearestEven = [this] (size_t v, T minScale, size_t shift) { // scale is less than 1
        size_t size = std::ceil(getCenterSize(shift) / 2.f) * 2; // to get even size
        while ((size / (float)v) < minScale) {
            size += 2;
        }
        return size;
    };
    const T requestedScale = this->getScaleFactor();
    // hint, possibly bigger then requested, so that it fits max shift window
    Dimensions hint(getNearestEven(s.sDim().x(), requestedScale, this->maxShift),
            getNearestEven(s.sDim().y(), requestedScale, this->maxShift),
            s.sDim().z(),
            (s.sDim().n() * (s.sDim().n() - 1)) / 2); // number of correlations);

    // divide available memory to 3 parts (2 buffers + 1 FFT)
    size_t correlationBufferBytes = gpu.value().lastFreeBytes() / 3;

    return getSettingsOrBenchmark(hint, 2 * correlationBufferBytes, false);
}

template<typename T>
FFTSettings<T> ProgMovieAlignmentCorrelationGPU<T>::getPatchSettings(
        const FFTSettings<T> &orig) {
    gpu.value().updateMemoryInfo();
    const auto reqSize = this->getRequestedPatchSize();
    Dimensions hint(reqSize.first, reqSize.second,
            orig.sDim().z(), orig.sDim().n());
    // divide available memory to 3 parts (2 buffers + 1 FFT)
    size_t correlationBufferBytes = gpu.value().lastFreeBytes() / 3;

    return getSettingsOrBenchmark(hint, 2 * correlationBufferBytes, false);
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
    auto &movieDim = movie.getFullDim();
    size_t n = movieDim.n();
    auto patchSize = patch.getSize();
    auto copyPatchData = [&](size_t srcFrameIdx, size_t t, bool add) {
        auto *fullFrame = this->movie.getFullFrame(srcFrameIdx).data;
        size_t patchOffset = t * patchSize.x * patchSize.y;
        // keep the shift consistent while adding local shift
        int xShift = std::round(globAlignment.shifts.at(srcFrameIdx).x);
        int yShift = std::round(globAlignment.shifts.at(srcFrameIdx).y);
        for (size_t y = 0; y < patchSize.y; ++y) {
            size_t srcY = patch.tl.y + y;
            if (yShift < 0) {
                srcY -= (size_t)std::abs(yShift); // assuming shift is smaller than offset
            } else {
                srcY += yShift;
            }
            size_t srcIndex = (srcY * movieDim.x()) + (size_t)patch.tl.x;
            if (xShift < 0) {
                srcIndex -= (size_t)std::abs(xShift);
            } else {
                srcIndex += xShift;
            }
            size_t destIndex = patchOffset + y * patchSize.x;
            if (add) {
                for (size_t x = 0; x < patchSize.x; ++x) {
                    result[destIndex + x] += fullFrame[srcIndex + x];
                }
            } else {
                memcpy(result + destIndex, fullFrame + srcIndex, patchSize.x * sizeof(T));
            }
        }
    };
    for (int t = 0; t < n; ++t) {
        // copy the data from specific frame
        copyPatchData(t, t, false);
        // add data from frames with lower indices
        // while averaging odd num of frames, use copy equally from previous and following frames
        // otherwise prefer following frames
        for (int b = 1; b <= ((patchesAvg - 1) / 2); ++b) {
            if (t >= b) {
                copyPatchData(t - b, t, true);
            }
        }
        // add data from frames with higher indices
        for (int f = 1; f <= (patchesAvg / 2); ++f) {
            if ((t + f) < n) {
                copyPatchData(t + f, t, true);
            }
        }
    }
}


template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::getPatchDataV1(const Rectangle<Point2D<T>> &patch, 
        const AlignmentResult<T> &globAlignment, T *result) {
    const auto &movieDim = movie.getFullDim();
    const int n = movieDim.n();
    const auto patchSize = patch.getSize();
    const auto bufferBytes = patchSize.x * sizeof(T);
    auto *buffer = reinterpret_cast<T*>(BasicMemManager::instance().get(bufferBytes, MemType::CPU));
    for (int t = 0; t < n; ++t) {// for each patch
        for (size_t y = 0; y < patchSize.y; ++y) { // for each row
            memset(buffer, 0, bufferBytes); // must be set to 0, because we might 'read' outside of the frames due to shift
            // while averaging odd num of frames, use copy equally from previous and following frames
            // otherwise prefer following frames
            for (int f = std::max(0, t - ((patchesAvg - 1) / 2)); f <= std::min(n - 1, t + (patchesAvg / 2)); ++f) {
                const auto *fullFrame = this->movie.getFullFrame(f).data;
                const int xShift = std::round(globAlignment.shifts[f].x);
                const int yShift = std::round(globAlignment.shifts[f].y);
                const int srcY = patch.tl.y + y + yShift;
                // if (srcY < 0 || srcY >= static_cast<int>(movieDim.y())) {
                //     continue;
                // }
                for (int x = 0; x < patchSize.x; ++x) {
                    const int srcX = patch.tl.x + x + xShift;
                    // if (srcX >= 0 && srcX < static_cast<int>(movieDim.x())) {
                        buffer[x] += fullFrame[static_cast<size_t>(srcY) * movieDim.x() + static_cast<size_t>(srcX)];
                    // }
                }
                
            }
            const size_t patchOffset = t * patchSize.x * patchSize.y;
            const size_t destIndex = patchOffset + y * patchSize.x;
            memcpy(result + destIndex, buffer, bufferBytes); // write result
        }
        // 136ms (115ms bez kontroly v X cyklu)
    }
    BasicMemManager::instance().give(buffer);
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::getPatchDataV2(const Rectangle<Point2D<T>> &patch, 
        const AlignmentResult<T> &globAlignment, T *result) {
    const auto &movieDim = movie.getFullDim();
    const int n = movieDim.n();
    const auto patchSize = patch.getSize();
    const auto bufferBytes = patchSize.x * sizeof(T);
    auto *buffer = reinterpret_cast<T*>(BasicMemManager::instance().get(bufferBytes, MemType::CPU));
    for (int t = 0; t < n; ++t) {// for each patch
        // while averaging odd num of frames, use copy equally from previous and following frames
        // otherwise prefer following frames
        for (int f = std::max(0, t - ((patchesAvg - 1) / 2)); f <= std::min(n - 1, t + (patchesAvg / 2)); ++f) {
            const auto *fullFrame = this->movie.getFullFrame(f).data;
            const int xShift = std::round(globAlignment.shifts.at(f).x);
            const int yShift = std::round(globAlignment.shifts.at(f).y);
            for (size_t y = 0; y < patchSize.y; ++y) { // for each row
                memset(buffer, 0, bufferBytes); // must be set to 0, because we might 'read' outside of the frames due to shift
                const int srcY = patch.tl.y + y + yShift;
                if (srcY < 0 || srcY >= static_cast<int>(movieDim.y())) {
                    continue;
                }
                for (int x = 0; x < patchSize.x; ++x) {
                    const int srcX = patch.tl.x + x + xShift;
                    if (srcX >= 0 && srcX < static_cast<int>(movieDim.x())) {
                        buffer[x] += fullFrame[static_cast<size_t>(srcY) * movieDim.x() + static_cast<size_t>(srcX)];
                    }
                }
                
                const size_t patchOffset = t * patchSize.x * patchSize.y;
                const size_t destIndex = patchOffset + y * patchSize.x;
                memcpy(result + destIndex, buffer, bufferBytes); // write result
            }
        }
    }
    BasicMemManager::instance().give(buffer);
    // 140 ms
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::getPatchDataV3(const Rectangle<Point2D<T>> &patch, 
        const AlignmentResult<T> &globAlignment, T *result) {
    const auto &movieDim = movie.getFullDim();
    const int n = movieDim.n();
    const auto patchSize = patch.getSize();
    const auto bufferBytes = patchSize.x * sizeof(T);
    // auto *buffer = reinterpret_cast<T*>(BasicMemManager::instance().get(bufferBytes, MemType::CPU));
    for (int f = 0; f < n; ++f) {// for each frame
        const auto *fullFrame = this->movie.getFullFrame(f).data;
        const int xShift = std::round(globAlignment.shifts.at(f).x);
        const int yShift = std::round(globAlignment.shifts.at(f).y);
        for (size_t y = 0; y < patchSize.y; ++y) { // for each row
            const int srcY = patch.tl.y + y + yShift;
            if (srcY < 0 || srcY >= static_cast<int>(movieDim.y())) {
                continue;
            }
            for (int x = 0; x < patchSize.x; ++x) {
                const int srcX = patch.tl.x + x + xShift;
                if (srcX >= 0 && srcX < static_cast<int>(movieDim.x())) {
                    for (int t = std::max(0, f - ((patchesAvg - 1) / 2)); t <= std::min(n - 1, f + (patchesAvg / 2)); ++t) {
                        const size_t patchOffset = t * patchSize.x * patchSize.y;
                        const size_t destIndex = patchOffset + y * patchSize.x;
                        result[destIndex + x] += fullFrame[static_cast<size_t>(srcY) * movieDim.x() + static_cast<size_t>(srcX)];
                    }
                }
            }
        // while averaging odd num of frames, use copy equally from previous and following frames
        // otherwise prefer following frames
                // memset(buffer, 0, bufferBytes); // must be set to 0, because we might 'read' outside of the frames due to shift
                // }
                
                // memcpy(result + destIndex, buffer, bufferBytes); // write result
            // }
        }
    }
    // BasicMemManager::instance().give(buffer);
    // 238 ms
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::getPatchDataV4(const Rectangle<Point2D<T>> &patch, 
        const AlignmentResult<T> &globAlignment, T *result) {
    const auto &movieDim = movie.getFullDim();
    const int n = movieDim.n();
    const auto patchSize = patch.getSize();
    const auto bufferBytes = patchSize.x * sizeof(T);
    static ctpl::thread_pool pool = ctpl::thread_pool(2);
    std::vector<T*> buffers;
    for (int i = 0; i < pool.size(); ++i) { buffers.push_back(new T[static_cast<int>(patchSize.x)]); }
    for (int t = 0; t < n; ++t) {// for each patch
        auto routine = [&](int thrId, int firstY, int lastY) {

            for (int y = firstY; y < lastY; ++y) { // for each row
                const int patchOffset = t * patchSize.x * patchSize.y;
                const int destIndex = patchOffset + y * patchSize.x;
            // auto *buffer = buffers[thrId];
            auto *buffer = result + destIndex;
                // memset(buffer, 0, bufferBytes); // must be set to 0, because we might 'read' outside of the frames due to shift
                // while averaging odd num of frames, use copy equally from previous and following frames
                // otherwise prefer following frames
                bool copy = true;
                for (int f = std::max(0, t - ((patchesAvg - 1) / 2)); f <= std::min(n - 1, t + (patchesAvg / 2)); ++f) {
                    const auto *fullFrame = this->movie.getFullFrame(f).data;
                    const int xShift = std::round(globAlignment.shifts[f].x);
                    const int yShift = std::round(globAlignment.shifts[f].y);
                    const int srcY = patch.tl.y + y + yShift;
                    // if (srcY < 0 || srcY >= static_cast<int>(movieDim.y())) {
                    //     continue;
                    // }
                    const int srcX = patch.tl.x + xShift;
                    auto *src = fullFrame + srcY * movieDim.x() + srcX;
                    if (copy) {
                        memcpy(buffer, src, bufferBytes);
                    } else {
                    for (int x = 0; x < patchSize.x; ++x) {
                        // if (srcX >= 0 && srcX < static_cast<int>(movieDim.x())) {
                            buffer[x] += src[x];
                        // }
                    }
                    }
                    copy = false;
                    
                }
                // memcpy(result + destIndex, buffer, bufferBytes); // write result
            }
        };
        // routine(0, 0, patchSize.y / 2);
        // routine(0, patchSize.y / 2, patchSize.y);
        auto f1 = pool.push(routine, 0, patchSize.y / 2);
        auto f2 = pool.push(routine, patchSize.y / 2, patchSize.y);
        f1.get(); f2.get();
        // 58ms
    }
    // pool.stop(true);
    for (auto &b : buffers) {delete[] b;};
    // BasicMemManager::instance().give(buffer);
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::getPatchDataV5(const Rectangle<Point2D<T>> &patch, 
        const AlignmentResult<T> &globAlignment, T *result) {
    const auto &movieDim = movie.getFullDim();
    const int n = movieDim.n();
    const auto patchSize = patch.getSize();
    const auto bufferBytes = patchSize.x * sizeof(T);
    auto *buffer = reinterpret_cast<T*>(BasicMemManager::instance().get(bufferBytes, MemType::CPU));
    for (int t = 0; t < n; ++t) {// for each patch
        // while averaging odd num of frames, use copy equally from previous and following frames
        // otherwise prefer following frames
        for (int f = std::max(0, t - ((patchesAvg - 1) / 2)); f <= std::min(n - 1, t + (patchesAvg / 2)); ++f) {
            const auto *fullFrame = this->movie.getFullFrame(f).data;
            const int xShift = std::round(globAlignment.shifts.at(f).x);
            const int yShift = std::round(globAlignment.shifts.at(f).y);
            const int srcX = patch.tl.x + xShift;
            for (size_t y = 0; y < patchSize.y; ++y) { // for each row
                memset(buffer, 0, bufferBytes); // must be set to 0, because we might 'read' outside of the frames due to shift
                const int srcY = patch.tl.y + y + yShift;
                // // if (srcY < 0 || srcY >= static_cast<int>(movieDim.y())) {
                //     continue;
                // }
                for (int x = 0; x < patchSize.x; ++x) {
                    // if (srcX >= 0 && srcX < static_cast<int>(movieDim.x())) {
                        buffer[x] += fullFrame[srcY * movieDim.x() + srcX + x];
                    // }
                }
                
                const size_t patchOffset = t * patchSize.x * patchSize.y;
                const size_t destIndex = patchOffset + y * patchSize.x;
                memcpy(result + destIndex, buffer, bufferBytes); // write result
            }
        }
    }
    BasicMemManager::instance().give(buffer);
    // 97 ms
}

// template<typename T>
// void ProgMovieAlignmentCorrelationGPU<T>::getPatchDataV6(const Rectangle<Point2D<T>> &patch, 
//         const AlignmentResult<T> &globAlignment, T *result) {
//     const auto &movieDim = movie.getFullDim();
//     const int n = movieDim.n();
//     const auto patchSize = patch.getSize();
//     const auto bufferBytes = patchSize.x * sizeof(T);
//     std::vector<T[]> buffers(patchesAvg);
//     for (auto &b : buffers) { b = new T[static_cast<int>(patchSize.x);]}
//     // auto *buffer = reinterpret_cast<T*>(BasicMemManager::instance().get(bufferBytes, MemType::CPU));
//     for (int f = 0; f < n; ++f) {// for each frame
//         const auto *fullFrame = this->movie.getFullFrame(f).data;
//         const int xShift = std::round(globAlignment.shifts[f].x);
//         const int yShift = std::round(globAlignment.shifts[f].y);
//         const int srcX = patch.tl.x + xShift;
//             // if (srcY < 0 || srcY >= static_cast<int>(movieDim.y())) {
//             //     continue;
//             // }
//         for (size_t y = 0; y < patchSize.y; ++y) { // for each row
            
//             for (int t = std::max(0, f - ((patchesAvg - 1) / 2)); t <= std::min(n - 1, f + (patchesAvg / 2)); ++t) {
//                 const int srcY = patch.tl.y + y + yShift;
//                 for (int x = 0; x < patchSize.x; ++x) {
//                     const size_t patchOffset = t * patchSize.x * patchSize.y;
//                     const size_t destIndex = patchOffset + y * patchSize.x;
//                     result[destIndex + x] += fullFrame[srcY * movieDim.x() + srcX + x];
//                 }
//             }
//         // while averaging odd num of frames, use copy equally from previous and following frames
//         // otherwise prefer following frames
//                 // memset(buffer, 0, bufferBytes); // must be set to 0, because we might 'read' outside of the frames due to shift
//                 // }
                
//                 // memcpy(result + destIndex, buffer, bufferBytes); // write result
//             // }
//         }
//     }
//     // BasicMemManager::instance().give(buffer);
//     // 77 ms
// }

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::getPatchDataV7(const Rectangle<Point2D<T>> &patch, 
        const AlignmentResult<T> &globAlignment, T *result) {
    const auto &movieDim = movie.getFullDim();
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
                    const auto *fullFrame = this->movie.getFullFrame(f).data;
                    const int xShift = std::round(globAlignment.shifts[f].x);
                    const int yShift = std::round(globAlignment.shifts[f].y);
                    // notice we don't test any access - it should always be within the boundaries of the frame
                    // see implementation of patch position generation and frame border computation
                    const int srcY = patch.tl.y + y + yShift;
                    const int srcX = patch.tl.x + xShift;
                    auto *src = fullFrame + srcY * movieDim.x() + srcX;
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
void ProgMovieAlignmentCorrelationGPU<T>::storeSizes(const Dimensions &dim,
        const FFTSettings<T> &s, bool applyCrop) {
    UserSettings::get(storage).insert(*this,
            getKey(optSizeXStr, dim, applyCrop), s.sDim().x());
    UserSettings::get(storage).insert(*this,
            getKey(optSizeYStr, dim, applyCrop), s.sDim().y());
    UserSettings::get(storage).insert(*this,
            getKey(optBatchSizeStr, dim, applyCrop), s.batch());
    UserSettings::get(storage).insert(*this,
            getKey(minMemoryStr, dim, applyCrop), memoryUtils::MB(gpu.value().lastFreeBytes()));
    UserSettings::get(storage).store(); // write changes immediately
}

template<typename T>
core::optional<FFTSettings<T>> ProgMovieAlignmentCorrelationGPU<T>::getStoredSizes(
        const Dimensions &dim, bool applyCrop) {
    size_t x, y, batch, neededMB;
    bool res = true;
    res = res
            && UserSettings::get(storage).find(*this,
                    getKey(optSizeXStr, dim, applyCrop), x);
    res = res
            && UserSettings::get(storage).find(*this,
                    getKey(optSizeYStr, dim, applyCrop), y);
    res = res
            && UserSettings::get(storage).find(*this,
                    getKey(optBatchSizeStr, dim, applyCrop), batch);
    res = res
            && UserSettings::get(storage).find(*this,
                    getKey(minMemoryStr, dim, applyCrop), neededMB);
    // check available memory
    res = res && (neededMB <= memoryUtils::MB(gpu.value().lastFreeBytes()));
    if (res) {
        return core::optional<FFTSettings<T>>(
                FFTSettings<T>(x, y, 1, dim.n(), batch, false));
    } else {
        return core::optional<FFTSettings<T>>();
    }
}


template<typename T>
FFTSettings<T> ProgMovieAlignmentCorrelationGPU<T>::runBenchmark(const Dimensions &d,
        size_t extraBytes, bool crop) {
    // FIXME DS remove tmp
    auto tmp1 = FFTSettings<T>(d, d.n(), false);
    FFTSettings<T> tmp(0);
    if (skipAutotuning) {
        tmp = CudaFFT<T>::findMaxBatch(tmp1, gpu.value().lastFreeBytes() - extraBytes);
    } else {
        if (this->verbose) std::cerr << "Benchmarking cuFFT ..." << std::endl;
        // take additional memory requirement into account
        // FIXME DS make sure that result is smaller than available data
        tmp =  CudaFFT<T>::findOptimalSizeOrMaxBatch(gpu.value(), tmp1,
                extraBytes, d.x() == d.y(), crop ? 10 : 20, // allow max 10% change for cropping, 20 for 'padding'
                crop, this->verbose);
    }
    auto goodBatch = tmp.batch();
    if (goodBatch < d.n()) { // in case we cannot process whole batch at once, make reasonable chunks
        goodBatch = d.n() / std::ceil(d.n() / (float)tmp.batch());
    }
    return FFTSettings<T>(tmp.sDim().x(), tmp.sDim().y(), tmp.sDim().z(), tmp.sDim().n(), goodBatch, false);
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
    const auto movie = instance.getMovieSize();
    const auto reqPatchSize = instance.getRequestedPatchSize();
    auto pSize = findGoodPatchSize(Dimensions(reqPatchSize.first, reqPatchSize.second, 1, movie.n()), gpu, instance);
    instance.setNoOfPaches(movie, pSize);
    auto correlation = instance.getCorrelationHint(pSize);
    auto cSize = findGoodCorrelationSize(correlation, gpu, instance);
    const auto maxBytes = gpu.lastFreeBytes() * 0.9f; // leave some buffer in case of memory fragmentation
    auto getMemReq = [&pSize, this]() {
        // for scale
        auto scale = GlobAlignmentData<T>::estimateBytes(patchSettings, correlationSettings);
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
        std::cout << "No. of patches: " << this->localAlignPatches.first << " x " << this->localAlignPatches.second << std::endl;
        std::cout << "Actual scale factor (X): " << actualScale << std::endl;
        std::cout << localHelper << std::endl;
    }
    if (this->localAlignPatches.first <= this->localAlignmentControlPoints.x()
        || this->localAlignPatches.second <= this->localAlignmentControlPoints.y()) {
            throw std::logic_error("More control points than patches. Decrease the number of control points.");
    }

    if ((movieSize.x() < patchSettings.sDim().x())
        || (movieSize.y() < patchSettings.sDim().y())) {
        REPORT_ERROR(ERR_PARAM_INCORRECT, "Movie is too small for local alignment.");
    }

    // load movie to memory
    if ( ! movie.hasFullMovie()) {
        loadMovie(movieMD, dark, igain);
    }

    // prepare filter
    // FIXME DS make sure that the resulting filter is correct, even if we do non-uniform scaling
    MultidimArray<T> filterTmp = this->createLPF(this->getPixelResolution(actualScale), correlationSettings.sDim());
    T* filterData = reinterpret_cast<T*>(BasicMemManager::instance().get(filterTmp.nzyxdim *sizeof(T), MemType::CUDA_MANAGED));
    memcpy(filterData, filterTmp.data, filterTmp.nzyxdim *sizeof(T));
    auto filter = MultidimArray<T>(1, 1, filterTmp.ydim, filterTmp.xdim, filterData);

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

    std::vector<T*> corrBuffers(loadPool.size()); // initializes to nullptrs
    std::vector<T*> patchData(loadPool.size()); // initializes to nullptrs
    std::vector<std::complex<T>*> scalledPatches(loadPool.size()); // initializes to nullptrs
    std::vector<std::future<void>> futures;
    futures.reserve(patchesLocation.size());
    GPU streams[2] = {GPU(gpu.value().device(), 1), GPU(gpu.value().device(), 2)};
    streams[0].set();
    streams[1].set();

    std::mutex mutex[3];

    GlobAlignmentData<T> auxData;
    auxData.alloc(patchSettings.createBatch(), correlationSettings, streams[0]);
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
            getPatchDataV7(p.rec, globAlignment, data);

            // convert to FFT, downscale them and compute correlations
            // GPUPool.push([&](int){
                for (auto i = 0; i < patchSettings.sDim().n(); i += patchSettings.batch()) {
                    std::unique_lock<std::mutex> lock(mutex[0]);
                    performFFTAndScale(data + i * patchSettings.sElemsBatch(), patchSettings.createBatch(),
                                                scalledPatches[thrId] + i * correlationSettings.fDim().sizeSingle(), correlationSettings,
                                                filter, streams[0], auxData);
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
    auxData.release();

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
    auto movieSettings = getMovieSettings(movie, false);
    LocalAlignmentResult<T> result { globalHint:globAlignment, movieDim:movieSettings.sDim() };
    auto patchSettings = this->getPatchSettings(movieSettings);
    this->setNoOfPaches(movieSettings.sDim(), patchSettings.sDim());
    auto borders = getMovieBorders(globAlignment, 0);
    auto patchesLocation = this->getPatchesLocation(borders, movieSettings.sDim(),
            patchSettings.sDim());
    // get alignment for all patches
    for (auto &&p : patchesLocation) {
        // process it
        for (size_t i = 0; i < movieSettings.sDim().n(); ++i) {
            FramePatchMeta<T> tmp = p;
            tmp.id_t = i;
            result.shifts.emplace_back(tmp, Point2D<T>(globAlignment.shifts.at(i).x, globAlignment.shifts.at(i).y));
        }
    }

    auto coeffs = BSplineHelper::computeBSplineCoeffs(movieSettings.sDim(), result,
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
        while (2 * count * movie.getFullDim().xy() * sizeof(T) > this->gpu.value().lastFreeBytes())
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
        aux[i].hIn = reinterpret_cast<T*>(BasicMemManager::instance().get(movie.getFullDim().xy() * sizeof(T), MemType::CUDA_HOST));
    }

    const T binning = this->getOutputBinning();
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
                auto *data = movie.getFullFrame(frameIndex).data;
                auto croppedFrame = MultidimArray(1, 1, movie.getFullDim().y(), movie.getFullDim().x(), a.hIn);
                memcpy(croppedFrame.data, data, croppedFrame.yxdim * sizeof(T));

                if (binning > 0) {
                    typeCast(croppedFrame, a.croppedFrameD);
                    auto scale = [binning](auto dim) {
                    return static_cast<int>(
                        std::floor(static_cast<T>(dim) / binning));
                    };
                    scaleToSizeFourier(1, scale(croppedFrame.ydim),
                                    scale(croppedFrame.xdim),
                                    a.croppedFrameD, a.reducedFrameD);

                    typeCast(a.reducedFrameD, a.reducedFrame);
                    // we need to construct cropped frame again with reduced size, but with the original memory block
                    croppedFrame = MultidimArray(1, 1, a.reducedFrame.ydim, a.reducedFrame.xdim, a.hIn);
                    memcpy(croppedFrame.data, a.reducedFrame.data, a.reducedFrame.yxdim * sizeof(T));
                }

                if ( ! this->fnInitialAvg.isEmpty()) {
                    std::unique_lock<std::mutex> lock(mutex);
                    if (0 == initialMic().yxdim)
                        initialMic() = croppedFrame;
                    else
                        initialMic() += croppedFrame;
                    Ninitial++;
                }

                if ( ! this->fnAligned.isEmpty() || ! this->fnAvg.isEmpty()) {
                    if (nullptr == a.hOut) {
                        a.hOut = reinterpret_cast<T*>(BasicMemManager::instance().get(croppedFrame.yxdim * sizeof(T), MemType::CUDA_HOST));
                    }
                    auto shiftedFrame = MultidimArray<T>(1, 1, croppedFrame.ydim, croppedFrame.xdim, a.hOut);
                    a.transformer.initLazyForBSpline(croppedFrame.xdim, croppedFrame.ydim, alignment.movieDim.n(),
                            this->localAlignmentControlPoints.x(), this->localAlignmentControlPoints.y(), this->localAlignmentControlPoints.n(), a.stream);
                    a.transformer.applyBSplineTransform(this->BsplineOrder, shiftedFrame, croppedFrame, coeffs, frameOffset);

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
void ProgMovieAlignmentCorrelationGPU<T>::storeSizesNew(const Dimensions &orig,
        const Dimensions &opt, bool applyCrop) {
    auto single = orig.copyForN(1);
    UserSettings::get(storage).insert(*this,
            getKey(optSizeXStr, single, applyCrop), opt.x());
    UserSettings::get(storage).insert(*this,
            getKey(optSizeYStr, single, applyCrop), opt.y());
    UserSettings::get(storage).store(); // write changes immediately
}

template<typename T>
std::optional<Dimensions> ProgMovieAlignmentCorrelationGPU<T>::getStoredSizesNew(
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
auto ProgMovieAlignmentCorrelationGPU<T>::GlobalAlignmentHelper::findBatchesThreadsStreams(const Dimensions &movie, const GPU &gpu, ProgMovieAlignmentCorrelationGPU &instance) {
    auto mSize = findGoodCropSize(movie, gpu, instance);
    auto correlation = instance.getCorrelationHint(movie); 
    auto cSize = findGoodCorrelationSize(correlation, gpu, instance);
    const auto maxBytes = gpu.lastFreeBytes() * 0.9f; // leave some buffer in case of memory fragmentation
    auto getMemReq = [this]() {
        return GlobAlignmentData<T>::estimateBytes(movieSettings, correlationSettings) * gpuStreams;
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
    
    using memoryUtils::MB;
    const auto movieSize = this->getMovieSize();
    movie.setFullDim(movieSize); // this will also reserve enough space in the movie vector
    globalHelper.findBatchesThreadsStreams(movieSize, gpu.value(), *this);

    auto &movieSettings = globalHelper.movieSettings;
    auto &correlationSettings = globalHelper.correlationSettings;
    T actualScale = correlationSettings.sDim().x() / (T)movieSettings.sDim().x();

    // prepare filter
    MultidimArray<T> filterTmp = this->createLPF(this->getPixelResolution(actualScale), correlationSettings.sDim());
    T* filterData = reinterpret_cast<T*>(BasicMemManager::instance().get(filterTmp.nzyxdim *sizeof(T), MemType::CUDA_MANAGED));
    memcpy(filterData, filterTmp.data, filterTmp.nzyxdim *sizeof(T));
    auto filter = MultidimArray<T>(1, 1, filterTmp.ydim, filterTmp.xdim, filterData);
    
    if (this->verbose) {
        std::cout << "Requested scale factor: " << this->getScaleFactor() << std::endl;
        std::cout << "Actual scale factor (X): " << actualScale << std::endl;
        std::cout << globalHelper << std::endl;
    }

    // create a buffer for correlations in FD
    auto *scaledFrames = reinterpret_cast<std::complex<T>*>(BasicMemManager::instance().get(correlationSettings.fBytes(), MemType::CPU_PAGE_ALIGNED));

    auto cpuPool = ctpl::thread_pool(globalHelper.cpuThreads);
    auto gpuPool = ctpl::thread_pool(globalHelper.gpuStreams);
    std::vector<T*> croppedFrames(cpuPool.size());
    std::vector<GlobAlignmentData<T>> auxData(gpuPool.size());

    std::vector<GPU> streams(gpuPool.size());
    for (auto i = 0; i < streams.size(); ++i) {
        streams.at(i) = GPU(gpu.value().device(), i + 1);
        auto routine = [&movieSettings, &correlationSettings, &auxData, &streams, i](int stream) {
            streams.at(i).set();
            auxData.at(i).alloc(movieSettings.createBatch(), correlationSettings, streams.at(i));
        };
        gpuPool.push(routine);
    }

    for (auto i = 0; i < movieSettings.sDim().n(); i += movieSettings.batch()) {
        auto routine = [&](int thrId, size_t first, size_t count)
        {
            loadFrames(movieMD, dark, igain, first, count);
            if (nullptr == croppedFrames[thrId])
            {
                croppedFrames[thrId] = reinterpret_cast<T *>(BasicMemManager::instance().get(movieSettings.sBytesBatch(), MemType::CUDA_HOST));
            }
            auto *cFrames = croppedFrames[thrId];
            getCroppedFrames(movieSettings, cFrames, first, count);
            gpuPool.push([&](int stream)
                         { 
                            performFFTAndScale(croppedFrames[thrId], movieSettings.createBatch(),
                                              scaledFrames + first * correlationSettings.fDim().sizeSingle(), correlationSettings,
                                              filter, streams[stream], auxData[stream]);
                                              streams[stream].synch(); })
                .get();
        };
        cpuPool.push(routine, i, movieSettings.batch());
    }
    cpuPool.stop(true);
    gpuPool.stop(true);
    for (auto *ptr : croppedFrames) {
        BasicMemManager::instance().give(ptr);
    }
    for (auto &d : auxData) {
        d.release();
    }
    BasicMemManager::instance().release();
    BasicMemManager::instance().release();

    auto scale = std::make_pair(movieSettings.sDim().x() / (T) correlationSettings.sDim().x(),
        movieSettings.sDim().y() / (T) correlationSettings.sDim().y());

    auto result = computeShifts(this->verbose, this->maxShift, scaledFrames, correlationSettings,
        movieSettings.sDim().n(),
        scale, globalHelper.bufferSize, {});

    BasicMemManager::instance().give(filterData);
    BasicMemManager::instance().give(scaledFrames);
    BasicMemManager::instance().release(MemType::CUDA);
    return result;
}

template<typename T>
AlignmentResult<T> ProgMovieAlignmentCorrelationGPU<T>::align(T *data,
        const FFTSettings<T> &in, const FFTSettings<T> &correlation,
        MultidimArray<T> &filter,
        core::optional<size_t> &refFrame,
        size_t maxShift, size_t framesInCorrelationBuffer, int verbose) {
    assert(nullptr != data);
    size_t N = in.sDim().n();
    // scale and transform to FFT on GPU
    performFFTAndScale<T>(data, N, in.sDim().x(), in.sDim().y(), in.batch(),
            correlation.fDim().x(), correlation.fDim().y(), filter);

    auto scale = std::make_pair(in.sDim().x() / (T) correlation.sDim().x(),
            in.sDim().y() / (T) correlation.sDim().y());

    return computeShifts(verbose, maxShift, (std::complex<T>*) data, correlation,
            in.sDim().n(),
            scale, framesInCorrelationBuffer, refFrame);
}

// template<typename T>
// void ProgMovieAlignmentCorrelationGPU<T>::getCroppedMovie(const FFTSettings<T> &settings,
//         T *output) {
//     for (size_t n = 0; n < settings.sDim().n(); ++n) {
//         T *src = movie.getFullFrame(n).data; // points to first float in the image
//         T *dest = output + (n * settings.sDim().xy()); // points to first float in the image
//         for (size_t y = 0; y < settings.sDim().y(); ++y) {
//             memcpy(dest + (settings.sDim().x() * y),
//                     src + (movie.getFullDim().x() * y),
//                     settings.sDim().x() * sizeof(T));
//         }
//     }
// }

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::getCroppedFrames(const FFTSettings<T> &settings,
        T *output, size_t firstFrame, size_t noOfFrames) {
    for (size_t n = 0; n < noOfFrames; ++n) {
        T *src = movie.getFullFrame(n + firstFrame).data; // points to first float in the image
        T *dest = output + (n * settings.sDim().xy()); // points to first float in the image
        for (size_t y = 0; y < settings.sDim().y(); ++y) {
            memcpy(dest + (settings.sDim().x() * y),
                    src + (movie.getFullDim().x() * y),
                    settings.sDim().x() * sizeof(T));
        }
    }
}

template<typename T>
MultidimArray<T> &ProgMovieAlignmentCorrelationGPU<T>::Movie::allocate(size_t x, size_t y) {
    auto *ptr = memoryUtils::page_aligned_alloc<T>(x * y, false);
    return mFullFrames.emplace_back(1, 1, y, x, ptr);
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::Movie::releaseFullFrames() {
    for (auto f = 0; f < mFullFrames.size(); ++f) {
        releaseFrame(f);
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::Movie::releaseFrame(size_t index) {
    auto &f = mFullFrames[index];
    BasicMemManager::instance().give(f.data);
    f.data = nullptr;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::loadMovie(const MetaData& movieMD,
        const Image<T>& dark, const Image<T>& igain) {
    const auto movieSize = this->getMovieSize();
    movie.setFullDim(movieSize); // this will also reserve enough space in the movie vector

    ctpl::thread_pool pool = ctpl::thread_pool(2);
    for (auto i = 0; i < movieSize.n(); ++i) {
        pool.push([&movieMD, &dark, &igain, this](int thrId, size_t first, size_t count){
            loadFrames(movieMD, dark, igain, first, count);
        }, i, 1);
    }
    pool.stop(true);
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::loadFrames(const MetaData& movieMD,
        const Image<T>& dark, const Image<T>& igain, size_t first, size_t count) {
    auto &movieDim = movie.getFullDim();
    int frameIndex = -1;
    size_t counter = 0;
    for (size_t objId : movieMD.ids())
    {
        // get to correct index
        frameIndex++;
        if (frameIndex < this->nfirst) continue;
        if (frameIndex > this->nlast) break;

        if ((counter >= first) && counter < (first + count)) {
            // load image
            auto *ptr = reinterpret_cast<T*>(BasicMemManager::instance().get(movieDim.xy() * sizeof(T), MemType::CPU_PAGE_ALIGNED));
            auto &dest = movie.getFullFrame(frameIndex);
            dest.data = ptr;
            Image<T> frame(dest);
            this->loadFrame(movieMD, dark, igain, objId, frame);
        }
        counter++;
    }
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

        // auto LES = [bX, bY, context, A, this](int) mutable {
            // now get the estimated shift (from the equation system)
            // from each frame to successing frame
            auto result = this->computeAlignment(bX, bY, A, context.refFrame, context.N, context.verbose);
            // prefill some info about patch
            for (size_t i = 0;i < context.N;++i) {
                // update total shift (i.e. global shift + local shift)
                context.result.shifts[i + context.shiftsOffset].second += result.shifts[i];
            }
        // };
    // };
}


template<typename T>
T myBestShift(MultidimArray<T> &Mcorr,
               T &shiftX, T &shiftY, const MultidimArray<int> *mask, int maxShift, size_t &index)
{
    int imax = INT_MIN;
    int jmax;
    int i_actual;
    int j_actual;
    double xmax;
    double ymax;
    double avecorr;
    double stdcorr;
    double dummy;
    bool neighbourhood = true;

    /*
     Warning: for masks with a small number of non-zero pixels, this routine is NOT reliable...
     Anyway, maybe using a mask is not a good idea at al...
     */

    // Adjust statistics within shiftmask to average 0 and stddev 1

        // Mcorr.statisticsAdjust((T)0, (T)1);

    // Look for maximum shift
    index = 0;
    if (maxShift==-1)
    	Mcorr.maxIndex(imax, jmax);
    else
    {
    	int maxShift2=maxShift*maxShift;
    	auto bestCorr=std::numeric_limits<T>::lowest();
    	for (int i=-maxShift; i<=maxShift; i++)
    		for (int j=-maxShift; j<=maxShift; j++)
    		{
                index++;
    			if (i*i+j*j>maxShift2) // continue if the Euclidean distance is too far
    				continue;
    			else if (A2D_ELEM(Mcorr, i, j)>bestCorr)
    			{
    				imax=i;
    				jmax=j;
    				bestCorr=A2D_ELEM(Mcorr, imax, jmax);
    			}
    		}
    }
    auto max = A2D_ELEM(Mcorr, imax, jmax);
    index = (imax - STARTINGY(Mcorr)) * Mcorr.xdim + (jmax - STARTINGX(Mcorr));
    shiftX = jmax;
    shiftY = imax;
    return max;

    
    // Estimate n_max around the maximum
    int n_max = -1;
    while (neighbourhood)
    {
        n_max++;
        for (int i = -n_max; i <= n_max && neighbourhood; i++)
        {
            i_actual = i + imax;
            if (i_actual < STARTINGY(Mcorr) || i_actual > FINISHINGY(Mcorr))
            {
                neighbourhood = false;
                break;
            }
            for (int j = -n_max; j <= n_max && neighbourhood; j++)
            {
                j_actual = j + jmax;
                if (j_actual < STARTINGX(Mcorr) || j_actual > FINISHINGX(Mcorr))
                {
                    neighbourhood = false;
                    break;
                }
                else if (max / 1.414 > A2D_ELEM(Mcorr, i_actual, j_actual))
                {
                    neighbourhood = false;
                    break;
                }
            }
        }
    }

    // We have the neighbourhood => looking for the gravity centre
    xmax = ymax = 0.;
    double sumcorr = 0.;
    if (imax-n_max<STARTINGY(Mcorr))
        n_max=std::min(imax-STARTINGY(Mcorr),n_max);
    if (imax+n_max>FINISHINGY(Mcorr))
        n_max=std::min(FINISHINGY(Mcorr)-imax,n_max);
    if (jmax-n_max<STARTINGY(Mcorr))
        n_max=std::min(jmax-STARTINGX(Mcorr),n_max);
    if (jmax+n_max>FINISHINGY(Mcorr))
        n_max=std::min(FINISHINGX(Mcorr)-jmax,n_max);
    for (int i = -n_max; i <= n_max; i++)
    {
        i_actual = i + imax;
        for (int j = -n_max; j <= n_max; j++)
        {
            j_actual = j + jmax;
            double val = A2D_ELEM(Mcorr, i_actual, j_actual);
            ymax += i_actual * val;
            xmax += j_actual * val;
            sumcorr += val;
        }
    }
    if (sumcorr != 0)
    {
        shiftX = xmax / sumcorr;
        shiftY = ymax / sumcorr;
    }
    return max;
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

template<typename T>
size_t ProgMovieAlignmentCorrelationGPU<T>::getMaxFilterBytes(
        const Dimensions &dim) {
    size_t maxXPow2 = std::ceil(log(dim.x()) / log(2));
    size_t maxX = std::pow(2, maxXPow2);
    size_t maxFFTX = maxX / 2 + 1;
    size_t maxYPow2 = std::ceil(log(dim.y()) / log(2));
    size_t maxY = std::pow(2, maxYPow2);
    size_t bytes = maxFFTX * maxY * sizeof(T);
    return bytes;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::releaseAll()
{
    BasicMemManager::instance().release();
    movie.releaseFullFrames();
};

// explicit specialization
template class ProgMovieAlignmentCorrelationGPU<float> ;
