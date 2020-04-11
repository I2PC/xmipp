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
    auto tmp = GPU(device);
    tmp.set();
    gpu = core::optional<GPU>(tmp);

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
FFTSettings<T> ProgMovieAlignmentCorrelationGPU<T>::getMovieSettings(
        const MetaData &movie, bool optimize) {
    Image<T> frame;
    int noOfImgs = this->nlast - this->nfirst + 1;
    this->loadFrame(movie, movie.firstObject(), frame);
    Dimensions dim(frame.data.xdim, frame.data.ydim, 1, noOfImgs);

    if (optimize) {
        size_t maxFilterBytes = getMaxFilterBytes(frame);
        return getSettingsOrBenchmark(dim, maxFilterBytes, true);
    } else {
        return FFTSettings<T>(dim, 1, false);
    }
}

template<typename T>
Dimensions ProgMovieAlignmentCorrelationGPU<T>::getCorrelationHint(
        const FFTSettings<T> &s,
        const std::pair<T, T> &downscale) {
    // we need odd size of the input, to be able to
    // compute FFT more efficiently (and e.g. perform shift by multiplication)
    auto scaleEven = [] (size_t v, T downscale) {
        return (int(v * downscale) / 2) * 2;
    };
    Dimensions result(scaleEven(s.dim.x(), downscale.first),
            scaleEven(s.dim.y(), downscale.second), s.dim.z(),
            (s.dim.n() * (s.dim.n() - 1)) / 2); // number of correlations);
    return result;
}

template<typename T>
FFTSettings<T> ProgMovieAlignmentCorrelationGPU<T>::getCorrelationSettings(
        const FFTSettings<T> &orig,
        const std::pair<T, T> &downscale) {
    auto hint = getCorrelationHint(orig, downscale);
    // divide available memory to 3 parts (2 buffers + 1 FFT)
    size_t correlationBufferBytes = gpu.value().lastFreeBytes() / 3;

    return getSettingsOrBenchmark(hint, 2 * correlationBufferBytes, false);
}

template<typename T>
FFTSettings<T> ProgMovieAlignmentCorrelationGPU<T>::getPatchSettings(
        const FFTSettings<T> &orig) {
    const auto reqSize = this->getRequestedPatchSize();
    Dimensions hint(reqSize.first, reqSize.second,
            orig.dim.z(), orig.dim.n());
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
void ProgMovieAlignmentCorrelationGPU<T>::getPatchData(const T *allFrames,
        const Rectangle<Point2D<T>> &patch, const AlignmentResult<T> &globAlignment,
        const Dimensions &movie, T *result) {
    size_t n = movie.n();
    auto patchSize = patch.getSize();
    auto copyPatchData = [&](size_t srcFrameIdx, size_t t, bool add) {
        size_t frameOffset = srcFrameIdx * movie.x() * movie.y();
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
            size_t srcIndex = frameOffset + (srcY * movie.x()) + (size_t)patch.tl.x;
            if (xShift < 0) {
                srcIndex -= (size_t)std::abs(xShift);
            } else {
                srcIndex += xShift;
            }
            size_t destIndex = patchOffset + y * patchSize.x;
            if (add) {
                for (size_t x = 0; x < patchSize.x; ++x) {
                    result[destIndex + x] += allFrames[srcIndex + x];
                }
            } else {
                memcpy(result + destIndex, allFrames + srcIndex, patchSize.x * sizeof(T));
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
void ProgMovieAlignmentCorrelationGPU<T>::storeSizes(const Dimensions &dim,
        const FFTSettings<T> &s, bool applyCrop) {
    UserSettings::get(storage).insert(*this,
            getKey(optSizeXStr, dim, applyCrop), s.dim.x());
    UserSettings::get(storage).insert(*this,
            getKey(optSizeYStr, dim, applyCrop), s.dim.y());
    UserSettings::get(storage).insert(*this,
            getKey(optBatchSizeStr, dim, applyCrop), s.batch);
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
    gpu.value().updateMemoryInfo();
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
    auto tmp1 = FFTSettingsNew<T>(d, d.n(), false);
    FFTSettingsNew<T> tmp(0);
    if (skipAutotuning) {
        tmp = CudaFFT<T>::findMaxBatch(tmp1, gpu.value().lastFreeBytes() - extraBytes);
    } else {
        if (this->verbose) std::cerr << "Benchmarking cuFFT ..." << std::endl;
        // take additional memory requirement into account
        tmp =  CudaFFT<T>::findOptimalSizeOrMaxBatch(gpu.value(), tmp1,
                extraBytes, d.x() == d.y(), crop ? 10 : 20, // allow max 10% change for cropping, 20 for 'padding'
                crop, this->verbose);
    }
    return FFTSettings<T>(tmp.sDim().x(), tmp.sDim().y(), tmp.sDim().z(), tmp.sDim().n(), tmp.batch(), false);
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
std::pair<T,T> ProgMovieAlignmentCorrelationGPU<T>::getLocalAlignmentCorrelationDownscale(
        const Dimensions &patchDim, T maxShift) {
    T minX = ((maxShift * 2) + 1) / patchDim.x();
    T minY = ((maxShift * 2) + 1) / patchDim.y();
    T idealScale = this->getScaleFactor();
    return std::make_pair(
            std::max(minX, idealScale),
            std::max(minY, idealScale));
}

template<typename T>
LocalAlignmentResult<T> ProgMovieAlignmentCorrelationGPU<T>::computeLocalAlignment(
        const MetaData &movie, const Image<T> &dark, const Image<T> &igain,
        const AlignmentResult<T> &globAlignment) {
    using memoryUtils::MB;
    auto movieSettings = this->getMovieSettings(movie, false);
    auto patchSettings = this->getPatchSettings(movieSettings);
    this->setNoOfPaches(movieSettings.dim, patchSettings.dim);
    auto correlationSettings = this->getCorrelationSettings(patchSettings,
            getLocalAlignmentCorrelationDownscale(patchSettings.dim, this->maxShift));
    auto borders = getMovieBorders(globAlignment, this->verbose > 1);
    auto patchesLocation = this->getPatchesLocation(borders, movieSettings.dim,
            patchSettings.dim);
    T actualScale = correlationSettings.dim.x() / (T)patchSettings.dim.x();
    if (this->verbose > 1) {
        std::cout << "No. of patches: " << this->localAlignPatches.first << " x " << this->localAlignPatches.second << std::endl;
        std::cout << "Actual scale factor: " << actualScale << std::endl;
        std::cout << "Settings for the patches: " << patchSettings << std::endl;
        std::cout << "Settings for the correlation: " << correlationSettings << std::endl;
    }

    if ((movieSettings.dim.x() < patchSettings.dim.x())
        || (movieSettings.dim.y() < patchSettings.dim.y())) {
        REPORT_ERROR(ERR_PARAM_INCORRECT, "Movie is too small for local alignment.");
    }

    // load movie to memory
    if (nullptr == movieRawData) {
        movieRawData = loadMovie(movie, dark, igain);
    }
    // we need to work with full-size movie, with no cropping
    assert(movieSettings.dim == rawMovieDim);

    // prepare filter
    // FIXME DS make sure that the resulting filter is correct, even if we do non-uniform scaling
    MultidimArray<T> filter = this->createLPF(this->getPixelResolution(actualScale), correlationSettings.dim);


    // compute max of frames in buffer
    T corrSizeMB = MB<T>((size_t) correlationSettings.x_freq
            * correlationSettings.dim.y()
            * sizeof(std::complex<T>));
    size_t framesInBuffer = std::ceil(MB(gpu.value().lastFreeBytes() / 3) / corrSizeMB);

    // prepare result
    LocalAlignmentResult<T> result { globalHint:globAlignment, movieDim:movieSettings.dim};
    result.shifts.reserve(patchesLocation.size() * movieSettings.dim.n());
    auto refFrame = core::optional<size_t>(globAlignment.refFrame);

    // allocate additional memory for the patches
    // we reuse the data, so we need enough space for the patches data
    // and for the resulting correlations, which cannot be bigger than (padded) input data
    size_t patchesElements = std::max(
        patchSettings.elemsFreq(),
        patchSettings.elemsSpacial());
    T *patchesData1 = new T[patchesElements];
    T *patchesData2 = new T[patchesElements];

    std::thread* processing_thread = nullptr;

    auto wait_and_delete = [](std::thread*& thread) {
        if (thread) {
            thread->join();
            delete thread;
            thread = nullptr;
        }
    };

    // use additional thread that would load the data at the background
    // get alignment for all patches and resulting correlations
    for (auto &&p : patchesLocation) {
        // get data
        memset(patchesData1, 0, patchesElements * sizeof(T));
        getPatchData(movieRawData, p.rec, globAlignment, movieSettings.dim,
                patchesData1);
        // don't swap buffers while some thread is accessing its content
        wait_and_delete(processing_thread);

        // swap buffers
        auto tmp = patchesData2;
        patchesData2 = patchesData1;
        patchesData1 = tmp;
        // run processing thread on the background

        processing_thread = new std::thread([&, p]() {
            // make sure to set proper GPU
            this->gpu.value().set();

            if (this->verbose > 1) {
                std::cout << "\nProcessing patch " << p.id_x << " " << p.id_y << std::endl;
            }
            // get alignment
            auto alignment = align(patchesData2, patchSettings,
                    correlationSettings, filter, refFrame,
                    this->maxShift, framesInBuffer, this->verbose);
            // process it
            for (size_t i = 0;i < movieSettings.dim.n();++i) {
                FramePatchMeta<T> tmp = p;
                // keep consistent with data loading
                int globShiftX = std::round(globAlignment.shifts.at(i).x);
                int globShiftY = std::round(globAlignment.shifts.at(i).y);
                tmp.id_t = i;
                // total shift is global shift + local shift
                result.shifts.emplace_back(tmp, Point2D<T>(globShiftX, globShiftY)
                        + alignment.shifts.at(i));
            }
        });

    }
    // wait for the last processing thread
    wait_and_delete(processing_thread);

    delete[] patchesData1;
    delete[] patchesData2;

    auto coeffs = BSplineHelper::computeBSplineCoeffs(movieSettings.dim, result,
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
    LocalAlignmentResult<T> result { globalHint:globAlignment, movieDim:movieSettings.dim };
    auto patchSettings = this->getPatchSettings(movieSettings);
    auto borders = getMovieBorders(globAlignment, 0);
    auto patchesLocation = this->getPatchesLocation(borders, movieSettings.dim,
            patchSettings.dim);
    // get alignment for all patches
    for (auto &&p : patchesLocation) {
        // process it
        for (size_t i = 0; i < movieSettings.dim.n(); ++i) {
            FramePatchMeta<T> tmp = p;
            tmp.id_t = i;
            result.shifts.emplace_back(tmp, Point2D<T>(globAlignment.shifts.at(i).x, globAlignment.shifts.at(i).y));
        }
    }

    auto coeffs = BSplineHelper::computeBSplineCoeffs(movieSettings.dim, result,
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

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::applyShiftsComputeAverage(
        const MetaData& movie, const Image<T>& dark, const Image<T>& igain,
        Image<T>& initialMic, size_t& Ninitial, Image<T>& averageMicrograph,
        size_t& N, const LocalAlignmentResult<T> &alignment) {
    // Apply shifts and compute average
    Image<T> croppedFrame(rawMovieDim.x(), rawMovieDim.y());
    T *croppedFrameData = croppedFrame.data.data;
    Image<T> reducedFrame, shiftedFrame;
    int frameIndex = -1;
    Ninitial = N = 0;
    GeoTransformer<T> transformer;
    if ( ! alignment.bsplineRep) {
        REPORT_ERROR(ERR_VALUE_INCORRECT,
            "Missing BSpline representation. This should not happen. Please contact developers.");
    }

    auto coeffs = std::make_pair(alignment.bsplineRep.value().getCoeffsX(),
        alignment.bsplineRep.value().getCoeffsY());

    const T binning = this->getOutputBinning();
    FOR_ALL_OBJECTS_IN_METADATA(movie)
    {
        frameIndex++;
        if ((frameIndex >= this->nfirstSum) && (frameIndex <= this->nlastSum)) {
            // user might want to align frames 3..10, but sum only 4..6
            // by deducting the first frame that was aligned, we get proper offset to the stored memory
            int frameOffset = frameIndex - this->nfirst;
            // load frame
            // we can point to proper part of the already loaded movie
            croppedFrame.data.data = movieRawData + (frameOffset * rawMovieDim.xy());

            if (binning > 0) {
                // FIXME add templates to respective functions/classes to avoid type casting
                /**
                 * WARNING
                 * As a side effect, raw movie data will get corrupted
                 */
                Image<double> croppedFrameDouble;
                Image<double> reducedFrameDouble;
                typeCast(croppedFrame(), croppedFrameDouble());

                scaleToSizeFourier(1, floor(YSIZE(croppedFrame()) / binning),
                        floor(XSIZE(croppedFrame()) / binning),
                        croppedFrameDouble(), reducedFrameDouble());

                typeCast(reducedFrameDouble(), reducedFrame());

                croppedFrame() = reducedFrame();
            }

            if ( ! this->fnInitialAvg.isEmpty()) {
                if (frameIndex == this->nfirstSum)
                    initialMic() = croppedFrame();
                else
                    initialMic() += croppedFrame();
                Ninitial++;
            }

            if (this->fnAligned != "" || this->fnAvg != "") {
                transformer.initLazyForBSpline(croppedFrame.data.xdim, croppedFrame.data.ydim, alignment.movieDim.n(),
                        this->localAlignmentControlPoints.x(), this->localAlignmentControlPoints.y(), this->localAlignmentControlPoints.n());
                transformer.applyBSplineTransform(this->BsplineOrder, shiftedFrame(), croppedFrame(), coeffs, frameOffset);


                if (this->fnAligned != "")
                    shiftedFrame.write(this->fnAligned, frameOffset + 1, true,
                            WRITE_REPLACE);
                if (this->fnAvg != "") {
                    if (frameIndex == this->nfirstSum)
                        averageMicrograph() = shiftedFrame();
                    else
                        averageMicrograph() += shiftedFrame();
                    N++;
                }
            }
            if (this->verbose > 1) {
                std::cout << "Frame " << std::to_string(frameIndex) << " processed." << std::endl;
            }
        }
    }
    // assign original data to avoid memory leak
    croppedFrame.data.data = croppedFrameData;
}

template<typename T>
AlignmentResult<T> ProgMovieAlignmentCorrelationGPU<T>::computeGlobalAlignment(
        const MetaData &movie, const Image<T> &dark, const Image<T> &igain) {
    using memoryUtils::MB;
    auto movieSettings = this->getMovieSettings(movie, true);
    T requestedScale = this->getScaleFactor();
    auto correlationSetting = this->getCorrelationSettings(movieSettings,
            std::make_pair(requestedScale, requestedScale));
    T actualScale = correlationSetting.dim.x() / (T)movieSettings.dim.x();

    MultidimArray<T> filter = this->createLPF(this->getPixelResolution(actualScale), correlationSetting.dim);
    if (this->verbose) {
        std::cout << "Requested scale factor: " << requestedScale << std::endl;
        std::cout << "Actual scale factor: " << actualScale << std::endl;
        std::cout << "Settings for the movie: " << movieSettings << std::endl;
        std::cout << "Settings for the correlation: " << correlationSetting << std::endl;
    }


    T corrSizeMB = ((size_t) correlationSetting.x_freq
            * correlationSetting.dim.y()
            * sizeof(std::complex<T>)) / ((T) 1024 * 1024);
    size_t framesInBuffer = std::ceil((MB(gpu.value().lastFreeBytes() / 3)) / corrSizeMB);

    auto reference = core::optional<size_t>();


    // load movie to memory
    if (nullptr == movieRawData) {
        movieRawData = loadMovie(movie, dark, igain);
    }
    size_t elems = std::max(movieSettings.elemsFreq(), movieSettings.elemsSpacial());
    T *data = new T[elems];
    getCroppedMovie(movieSettings, data);

    auto result = align(data, movieSettings, correlationSetting,
                    filter, reference,
            this->maxShift, framesInBuffer, this->verbose);
    delete[] data;
    return result;
}

template<typename T>
AlignmentResult<T> ProgMovieAlignmentCorrelationGPU<T>::align(T *data,
        const FFTSettings<T> &in, const FFTSettings<T> &correlation,
        MultidimArray<T> &filter,
        core::optional<size_t> &refFrame,
        size_t maxShift, size_t framesInCorrelationBuffer, int verbose) {
    assert(nullptr != data);
    size_t N = in.dim.n();
    // scale and transform to FFT on GPU
    performFFTAndScale<T>(data, N, in.dim.x(), in.dim.y(), in.batch,
            correlation.x_freq, correlation.dim.y(), filter);

    auto scale = std::make_pair(in.dim.x() / (T) correlation.dim.x(),
            in.dim.y() / (T) correlation.dim.y());

    return computeShifts(verbose, maxShift, (std::complex<T>*) data, correlation,
            in.dim.n(),
            scale, framesInCorrelationBuffer, refFrame);
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::getCroppedMovie(const FFTSettings<T> &settings,
        T *output) {
    for (size_t n = 0; n < settings.dim.n(); ++n) {
        T *src = movieRawData + (n * rawMovieDim.xy()); // points to first float in the image
        T *dest = output + (n * settings.dim.xy()); // points to first float in the image
        for (size_t y = 0; y < settings.dim.y(); ++y) {
            memcpy(dest + (settings.dim.x() * y),
                    src + (rawMovieDim.x() * y),
                    settings.dim.x() * sizeof(T));
        }
    }
}

template<typename T>
T* ProgMovieAlignmentCorrelationGPU<T>::loadMovie(const MetaData& movie,
        const Image<T>& dark, const Image<T>& igain) {
    T* imgs = nullptr;
    Image<T> frame;

    int movieImgIndex = -1;
    FOR_ALL_OBJECTS_IN_METADATA(movie)
    {
        // update variables
        movieImgIndex++;
        if (movieImgIndex < this->nfirst) continue;
        if (movieImgIndex > this->nlast) break;

        // load image
        this->loadFrame(movie, dark, igain, __iter.objId, frame);

        if (nullptr == imgs) {
            rawMovieDim = Dimensions(frame().xdim, frame().ydim, 1,
                    this->nlast - this->nfirst + 1);
            auto settings = FFTSettings<T>(rawMovieDim, 1, false);
            imgs = new T[std::max(settings.elemsFreq(), settings.elemsSpacial())]();
        }

        // copy all frames to memory, consecutively. There will be a space behind
        // in case we need to reuse the memory for FT
        T* dest = imgs
                + ((movieImgIndex - this->nfirst) * rawMovieDim.xy()); // points to first float in the image
        memcpy(dest, frame.data.data, rawMovieDim.xy() * sizeof(T));
    }
    return imgs;
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
    T* correlations;
    size_t centerSize = std::ceil(maxShift * 2 + 1);
    computeCorrelations(centerSize, N, data, settings.x_freq,
            settings.dim.x(),
            settings.dim.y(), framesInCorrelationBuffer,
            settings.batch, correlations);
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
            size_t offset = idx * centerSize * centerSize;
            Mcorr.data = correlations + offset;
            Mcorr.setXmippOrigin();
            bestShift(Mcorr, bX(idx), bY(idx), NULL,
                    maxShift / scale.first);
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
        const Image<T> &frame) {
    size_t maxXPow2 = std::ceil(log(frame.data.xdim) / log(2));
    size_t maxX = std::pow(2, maxXPow2);
    size_t maxFFTX = maxX / 2 + 1;
    size_t maxYPow2 = std::ceil(log(frame.data.ydim) / log(2));
    size_t maxY = std::pow(2, maxYPow2);
    size_t bytes = maxFFTX * maxY * sizeof(T);
    return bytes;
}

// explicit specialization
template class ProgMovieAlignmentCorrelationGPU<float> ;
