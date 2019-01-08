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
    this->addExampleLine(
                "xmipp_cuda_movie_alignment_correlation -i movie.xmd --oaligned alignedMovie.stk --oavg alignedMicrograph.mrc --device 0");
    this->addSeeAlsoLine("xmipp_movie_alignment_correlation");
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::show() {
    AProgMovieAlignmentCorrelation<T>::show();
    std::cout << "gpu set: " << gpu.has_value() << std::endl;
    std::cout << "Device:              " << gpu.value().device() << " (" << gpu.value().UUID() << ")" << std::endl;
    std::cout << "Benchmark storage    " << (storage.empty() ? "Default" : storage) << std::endl;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::readParams() {
    AProgMovieAlignmentCorrelation<T>::readParams();
    int device = this->getIntParam("--device");
    gpu = std::move(core::optional<GPU>(GPU(device)));
    std::cout << "gpu set: " << (gpu.has_value() ? "yes" : "no") << std::endl;
    storage = this->getParam("--storage");
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getSettingsOrBenchmark(
        const Dimensions &d, size_t extraMem, bool crop) {
    auto optSetting = getStoredSizes(d, crop);
    FFTSettings<T> result =
            optSetting ?
                    optSetting.value() : runBenchmark(d, extraMem, crop);
    if (!optSetting) {
        storeSizes(d, result, crop);
    }
    return result;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getMovieSettings(
        const MetaData &movie, bool optimize) {
    Image<T> frame;
    int noOfImgs = this->nlast - this->nfirst + 1;
    loadFrame(movie, movie.firstObject(), frame);
    Dimensions dim(frame.data.xdim, frame.data.ydim, 1, noOfImgs);
    int maxFilterSize = getMaxFilterSize(frame);

    if (optimize) {
        return getSettingsOrBenchmark(dim, maxFilterSize, true);
    } else {
        return FFTSettings<T>(dim.x, dim.y, dim.z, dim.n, 1, false);
    }
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getCorrelationHint(
        const FFTSettings<T> &s,
        const std::pair<T, T> &downscale) {
    // we need odd size of the input, to be able to
    // compute FFT more efficiently (and e.g. perform shift by multiplication)
    auto scaleEven = [] (size_t v, T downscale) {
        return (int(v * downscale) / 2) * 2;
    };
    Dimensions result(scaleEven(s.dim.x, downscale.first),
            scaleEven(s.dim.y, downscale.second), s.dim.z,
            (s.dim.n * (s.dim.n - 1)) / 2); // number of correlations);
    return result;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getCorrelationSettings(
        const FFTSettings<T> &orig,
        const std::pair<T, T> &downscale) {
    auto hint = getCorrelationHint(orig, downscale);
    size_t correlationBufferSizeMB = gpu.value().lastFreeMem() / 3; // divide available memory to 3 parts (2 buffers + 1 FFT)

    return getSettingsOrBenchmark(hint, 2 * correlationBufferSizeMB, false);
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getPatchSettings(
        const FFTSettings<T> &orig) {
    Dimensions hint(512, 512, // this should be a trade-off between speed and present signal
            // but check the speed to make sure
            orig.dim.z, orig.dim.n);
    size_t correlationBufferSizeMB = gpu.value().lastFreeMem() / 3; // divide available memory to 3 parts (2 buffers + 1 FFT)

    return getSettingsOrBenchmark(hint, 2 * correlationBufferSizeMB, false);
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getPatchesLocation(
        const std::pair<T, T> &borders,
        const FFTSettings<T> &movie, const FFTSettings<T> &patch) {
    size_t patchesX = localAlignPatches.first;
    size_t patchesY = localAlignPatches.second;
    size_t windowXSize = movie.dim.x - 2 * borders.first;
    size_t windowYSize = movie.dim.y - 2 * borders.second;
    T corrX = std::ceil(
            ((patchesX * patch.dim.x) - windowXSize) / (T) (patchesX - 1));
    T corrY = std::ceil(
            ((patchesY * patch.dim.y) - windowYSize) / (T) (patchesY - 1));
    size_t stepX = patch.dim.x - corrX;
    size_t stepY = patch.dim.y - corrY;
    std::vector<FramePatchMeta<T>> result;
    for (size_t y = 0; y < patchesY; ++y) {
        for (size_t x = 0; x < patchesX; ++x) {
            T tlx = borders.first + x * stepX; // Top Left
            T tly = borders.second + y * stepY;
            T brx = tlx + patch.dim.x - 1; // Bottom Right
            T bry = tly + patch.dim.y - 1; // -1 for indexing
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
        const Rec2D<T> &patch, const AlignmentResult<T> &globAlignment,
        const FFTSettings<T> &movie, T *result) {
    size_t n = movie.dim.n;
    auto patchSize = patch.getSize();
    auto copyPatchData = [&](size_t srcFrameIdx, size_t t, bool add) {
        size_t frameOffset = srcFrameIdx * movie.dim.x * movie.dim.y;
        size_t patchOffset = t * patchSize.x * patchSize.y;
        int xShift = std::round(globAlignment.shifts.at(srcFrameIdx).x);
        int yShift = std::round(globAlignment.shifts.at(srcFrameIdx).y);
        for (size_t y = 0; y < patchSize.y; ++y) {
            size_t srcY = patch.tl.y + y;
            if (yShift < 0) {
                srcY -= (size_t)std::abs(yShift); // assuming shift is smaller than offset
            } else {
                srcY += yShift;
            }
            size_t srcIndex = frameOffset + (srcY * movie.dim.x) + (size_t)patch.tl.x;
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
    for (size_t t = 0; t < n; ++t) {
        copyPatchData(t, t, false);
        if (t > 0) copyPatchData(t - 1, t, true);
        if ((t + 1) < n) copyPatchData(t + 1, t, true);
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::storeSizes(const Dimensions &dim,
        const FFTSettings<T> &s, bool applyCrop) {
    UserSettings::get(storage).insert(*this,
            getKey(optSizeXStr, dim, applyCrop), s.dim.x);
    UserSettings::get(storage).insert(*this,
            getKey(optSizeYStr, dim, applyCrop), s.dim.y);
    UserSettings::get(storage).insert(*this,
            getKey(optBatchSizeStr, dim, applyCrop), s.batch);
    UserSettings::get(storage).insert(*this,
            getKey(minMemoryStr, dim, applyCrop), gpu.value().lastFreeMem());
    UserSettings::get(storage).store(); // write changes immediately
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getStoredSizes(
        const Dimensions &dim, bool applyCrop) {
    size_t x, y, batch, neededMem;
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
                    getKey(minMemoryStr, dim, applyCrop), neededMem);
    res = res && neededMem <= getFreeMem(gpu.value().device());
    if (res) {
        return core::optional<FFTSettings<T>>(
                FFTSettings<T>(x, y, 1, dim.n, batch, true));
    } else {
        return core::optional<FFTSettings<T>>();
    }
}


template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::runBenchmark(const Dimensions &d,
        size_t extraMem, bool crop) {
    if (this->verbose) std::cerr << "Benchmarking cuFFT ..." << std::endl;
    // take additional memory requirement into account
    int x, y, batch;
    getBestFFTSize(d.n, d.x, d.y, batch, crop, x, y, extraMem, this->verbose,
            gpu.value().device(), d.x == d.y, 10); // allow max 10% change

    return FFTSettings<T>(x, y, 1, d.n, batch, true);
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getMovieBorders(
        const AlignmentResult<T> &globAlignment, bool verbose) {
    T minX = std::numeric_limits<T>::max();
    T maxX = std::numeric_limits<T>::min();
    T minY = std::numeric_limits<T>::max();
    T maxY = std::numeric_limits<T>::min();
    for (const auto& s : globAlignment.shifts) {
        std::cout << s.x << " " << s.y << std::endl;
        minX = std::min(std::floor(s.x), minX);
        maxX = std::max(std::ceil(s.x), maxX);
        minY = std::min(std::floor(s.y), minY);
        maxY = std::max(std::ceil(s.y), maxY);
    }
    auto res = std::make_pair(std::abs(maxX - minX), std::abs(maxY - minY));
    if (verbose) {
        std::cout << "Movie borders: x=" << res.first << " y=" << res.second
                << std::endl;
    }
    return res;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::computeBSplineCoefs(const Dimensions &movieSize,
        const LocalAlignmentResult<T> &alignment,
        const Dimensions &controlPoints, const std::pair<size_t, size_t> &noOfPatches) {
    // get coefficients fo the BSpline that can represent the shifts (formula  from the paper)
    int lX = controlPoints.x;
    int lY = controlPoints.y;
    int lT = controlPoints.n;
    int noOfPatchesXY = noOfPatches.first * noOfPatches.second;
    Matrix2D<T>A(noOfPatchesXY*movieSize.n, lX * lY * lT);
    Matrix1D<T>bX(noOfPatchesXY*movieSize.n);
    Matrix1D<T>bY(noOfPatchesXY*movieSize.n);
    T hX = (lX == 3) ? movieSize.x : (movieSize.x / (T)(lX-3));
    T hY = (lY == 3) ? movieSize.y : (movieSize.y / (T)(lY-3));
    T hT = (lT == 3) ? movieSize.n : (movieSize.n / (T)(lT-3));

    for (auto &&r : alignment.shifts) {
        auto meta = r.first;
        auto shift = r.second;
        int tileIdxT = meta.id_t;
        int tileCenterT = tileIdxT * 1 + 0 + 0;
        int tileIdxX = meta.id_x;
        int tileIdxY = meta.id_y;
        int tileCenterX = meta.rec.getCenter().x;
        int tileCenterY = meta.rec.getCenter().y;
        int i = (tileIdxY * noOfPatches.first) + tileIdxX;

        for (int j = 0; j < (lT * lY * lX); ++j) {
            int controlIdxT = (j / (lY * lX)) - 1;
            int XY = j % (lY * lX);
            int controlIdxY = (XY / lX) -1;
            int controlIdxX = (XY % lX) -1;
            // note: if control point is not in the tile vicinity, val == 0 and can be skipped
            T val = Bspline03((tileCenterX / (T)hX) - controlIdxX) *
                    Bspline03((tileCenterY / (T)hY) - controlIdxY) *
                    Bspline03((tileCenterT / (T)hT) - controlIdxT);
            MAT_ELEM(A,tileIdxT*noOfPatchesXY + i,j) = val;
        }
        VEC_ELEM(bX,tileIdxT*noOfPatchesXY + i) = -shift.x; // we want the BSPline describing opposite transformation,
        VEC_ELEM(bY,tileIdxT*noOfPatchesXY + i) = -shift.y; // so that we can use it to compensate for the shift
    }

    // solve the equation system for the spline coefficients
    Matrix1D<T> coefsX, coefsY;
    this->solveEquationSystem(bX, bY, A, coefsX, coefsY);
    return std::make_pair(coefsX, coefsY);
}

template<typename T>
LocalAlignmentResult<T> ProgMovieAlignmentCorrelationGPU<T>::computeLocalAlignment(
        const MetaData &movie, const Image<T> &dark, const Image<T> &gain,
        const AlignmentResult<T> &globAlignment) {
//    auto gpu = GPU(device);
    auto movieSettings = this->getMovieSettings(movie, false);
    auto patchSettings = this->getPatchSettings(movieSettings);
    auto correlationSettings = this->getCorrelationSettings(patchSettings,
            std::make_pair(1, 1));
    auto borders = getMovieBorders(globAlignment, this->verbose);
    auto patchesLocation = this->getPatchesLocation(borders, movieSettings,
            patchSettings);
    if (this->verbose) {
        std::cout << "Settings for the patches: " << patchSettings << std::endl;
    }
    if (this->verbose) {
        std::cout << "Settings for the patches: " << correlationSettings << std::endl;
    }

    // load movie to memory
    T* movieData = loadMovie(movie, movieSettings, dark, gain);

    // allocate additional memory for the patches
    size_t patchesElements = correlationSettings.dim.n
            * correlationSettings.dim.y
            * std::max(correlationSettings.dim.x,
                    correlationSettings.x_freq * 2);
    T *patchesData = new T[patchesElements];

    // prepare filter
    MultidimArray<T> filter = this->createLPF(this->getTargetOccupancy(), correlationSettings.dim.x,
            correlationSettings.x_freq, correlationSettings.dim.y);

    // compute max of frames in buffer
    T corrSizeMB = ((size_t) correlationSettings.x_freq
            * correlationSettings.dim.y
            * sizeof(std::complex<T>))
            / ((T) 1024 * 1024);
    size_t framesInBuffer = std::ceil((gpu.value().lastFreeMem() / 3) / corrSizeMB);

    // prepare result
    LocalAlignmentResult<T> result { .globalHint = globAlignment };
    result.shifts.reserve(patchesLocation.size() * movieSettings.dim.n);
    auto refFrame = core::optional<size_t>(globAlignment.refFrame);

    // get alignment for all patches
    for (auto &&p : patchesLocation) {
        std::cout << "Processing patch " << p.id_x << " " << p.id_y << std::endl;
        // get data
        memset(patchesData, 0, patchesElements * sizeof(T));
        getPatchData(movieData, p.rec, globAlignment, movieSettings,
                patchesData);
        // get alignment
        auto alignment = align(patchesData, patchSettings, correlationSettings,
                filter, refFrame,
                this->maxShift, framesInBuffer, false);
        // process it
        for (size_t i = 0; i < movieSettings.dim.n; ++i) {
            FramePatchMeta<T> tmp = p;
            tmp.id_t = i;
            // total shift is global shift + local shift
            result.shifts.emplace_back(tmp, globAlignment.shifts.at(i) + alignment.shifts.at(i));
        }
        std::cout << std::endl;
    }

    delete[] movieData;
    delete[] patchesData;
    return result;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::localFromGlobal(
        const MetaData& movie,
        const AlignmentResult<T> &globAlignment) {
//    auto gpu = GPU(device);
    auto movieSettings = getMovieSettings(movie, false);
    LocalAlignmentResult<T> result { .globalHint = globAlignment };
    auto patchSettings = this->getPatchSettings(movieSettings);
    auto borders = getMovieBorders(globAlignment);
    auto patchesLocation = this->getPatchesLocation(borders, movieSettings,
            patchSettings);
    // get alignment for all patches
    for (auto &&p : patchesLocation) {
        // process it
        for (size_t i = 0; i < movieSettings.dim.n; ++i) {
            FramePatchMeta<T> tmp = p;
            tmp.id_t = i;
            result.shifts.emplace_back(tmp, Point2D<T>(globAlignment.shifts.at(i).x, globAlignment.shifts.at(i).y));
        }
    }
    return result;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::applyShiftsComputeAverage(
        const MetaData& movie, const Image<T>& dark, const Image<T>& gain,
        Image<T>& initialMic, size_t& Ninitial, Image<T>& averageMicrograph,
        size_t& N, const AlignmentResult<T> &globAlignment) {
    applyShiftsComputeAverage(movie, dark, gain, initialMic, Ninitial, averageMicrograph,
            N, localFromGlobal(movie, globAlignment));
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::applyShiftsComputeAverage(
        const MetaData& movie, const Image<T>& dark, const Image<T>& gain,
        Image<T>& initialMic, size_t& Ninitial, Image<T>& averageMicrograph,
        size_t& N, const LocalAlignmentResult<T> &alignment) {
    // Apply shifts and compute average
    Image<T> frame, croppedFrame, reducedFrame, shiftedFrame;
    FileName fnFrame;
    int j = 0;
    int n = 0;
    Ninitial = N = 0;
    GeoTransformer<T> transformer;
//    auto gpu = GPU(device);
    auto movieSettings = getMovieSettings(movie, false);
    auto coefs = computeBSplineCoefs(movieSettings.dim, alignment, localAlignmentControlPoints, localAlignPatches);
    FOR_ALL_OBJECTS_IN_METADATA(movie)
    {
        if (n >= this->nfirstSum && n <= this->nlastSum) {
            movie.getValue(MDL_IMAGE, fnFrame, __iter.objId);

            // load frame
            frame.read(fnFrame);
            if (XSIZE(dark()) > 0) frame() -= dark();
            if (XSIZE(gain()) > 0) frame() *= gain();
            if (this->yDRcorner != -1) {
                frame().window(croppedFrame(), this->yLTcorner, this->xLTcorner,
                        this->yDRcorner, this->xDRcorner);
            } else croppedFrame() = frame();

            if (this->fnInitialAvg != "") {
                throw std::invalid_argument("fnInitialAvg is currently not supported. "
                        "If you need it, please contact developers or use xmipp_image_statistics program");
            }

            if (this->fnAligned != "" || this->fnAvg != "") {
                transformer.initLazyForBSpline(frame.data.xdim, frame.data.ydim, movieSettings.dim.n,
                    localAlignmentControlPoints.x, localAlignmentControlPoints.y, localAlignmentControlPoints.n);
                transformer.applyBSplineTransform(this->BsplineOrder, shiftedFrame(), croppedFrame(), coefs, j);

                if (this->bin > 0) {
                    // FIXME add templates to respective functions/classes to avoid type casting
                    Image<double> shiftedFrameDouble;
                    Image<double> reducedFrameDouble;
                    typeCast(shiftedFrame(), shiftedFrameDouble());

                    scaleToSizeFourier(1, floor(YSIZE(shiftedFrame()) / this->bin),
                            floor(XSIZE(shiftedFrame()) / this->bin),
                            shiftedFrameDouble(), reducedFrameDouble());

                    typeCast(reducedFrameDouble(), reducedFrame());

                    shiftedFrame() = reducedFrame();
                }

                if (this->fnAligned != "")
                    shiftedFrame.write(this->fnAligned, j + 1, true,
                            WRITE_REPLACE);
                if (this->fnAvg != "") {
                    if (j == 0)
                        averageMicrograph() = shiftedFrame();
                    else
                        averageMicrograph() += shiftedFrame();
                    N++;
                }
            }
            std::cout << fnFrame << " processed." << std::endl;
            j++;
        }
        n++;
    }
}

//
//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::applyShiftsComputeAverage(
//        const MetaData& movie, const Image<T>& dark, const Image<T>& gain,
//        const AlignmentResult<T> &globAlignment) {
//
//
//
//
//    auto cPoints = localAlignmentControlPoints;
//    auto patches = localAlignPatches;
//
//    auto coefs = computeBSplineCoefs(movieSettings.dim, result, cPoints, patches); // FIXME move to final function
//
//
//    // Apply shifts and compute average
//    Image<T> frame, croppedFrame, reducedFrame, shiftedFrame;
//    Image<T> averageMicrograph;
//    Matrix1D<T> shift(2);
//    FileName fnFrame;
//    int j = 0;
//    int n = 0;
//    size_t Ninitial = 0;
//    size_t N = 0;
//    GeoTransformer<T> transformer;
//    FOR_ALL_OBJECTS_IN_METADATA(movie)
//    {
//        if (n >= this->nfirstSum && n <= this->nlastSum) {
//            movie.getValue(MDL_IMAGE, fnFrame, __iter.objId);
//            frame.read(fnFrame);
//            if (XSIZE(dark()) > 0)
//                frame() -= dark();
//            if (XSIZE(gain()) > 0)
//                frame() *= gain();
//            croppedFrame() = frame();
//            transformer.initLazyForBSpline(frame.data.xdim, frame.data.ydim, movieSettings.dim.n,
//                    cPoints.x, cPoints.y, cPoints.n);
//            std::cout << "processing frame " << j << std::endl;
//            transformer.applyBSplineTransform(this->BsplineOrder, shiftedFrame(), croppedFrame(), coefs, j);
//                    if (j == 0) {
//                        averageMicrograph() = shiftedFrame();
//                        std::cout << "initializing shifted frame" << std::endl;
//                    } else {
//                        averageMicrograph() += shiftedFrame();
//                        std::cout << "adding shifted frame" << std::endl;
//                    }
//                    N++;
//            j++;
//        }
//        n++;
//    }
//    averageMicrograph.write("avg_test.vol");
//}
//
//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::applyShiftsComputeAverage(
//        const MetaData& movie, const Image<T>& dark, const Image<T>& gain,
//    std::pair<Matrix1D<T>, Matrix1D<T>> &coefs) {
//    // Apply shifts and compute average
//    Image<T> frame, croppedFrame, reducedFrame, shiftedFrame;
//    Image<T> averageMicrograph;
//    Matrix1D<T> shift(2);
//    FileName fnFrame;
//    int j = 0;
//    int n = 0;
//    size_t Ninitial = 0;
//    size_t N = 0;
//    GeoTransformer<T> transformer;
//    FOR_ALL_OBJECTS_IN_METADATA(movie)
//    {
//        if (n >= this->nfirstSum && n <= this->nlastSum) {
//            movie.getValue(MDL_IMAGE, fnFrame, __iter.objId);
//            frame.read(fnFrame);
//            if (XSIZE(dark()) > 0)
//                frame() -= dark();
//            if (XSIZE(gain()) > 0)
//                frame() *= gain();
//            croppedFrame() = frame();
//            transformer.initLazyForBSpline(frame.data.xdim, frame.data.ydim, 50,
//                    localAlignmentControlPoints.x, localAlignmentControlPoints.y, localAlignmentControlPoints.n);
//            std::cout << "processing frame " << j << std::endl;
//            transformer.applyBSplineTransform(this->BsplineOrder, shiftedFrame(), croppedFrame(), coefs, j);
//                    if (j == 0) {
//                        averageMicrograph() = shiftedFrame();
//                        std::cout << "initializing shifted frame" << std::endl;
//                    } else {
//                        averageMicrograph() += shiftedFrame();
//                        std::cout << "adding shifted frame" << std::endl;
//                    }
//                    N++;
//            j++;
//        }
//        n++;
//    }
//    averageMicrograph.write("avg_test.vol");
//}


template<typename T>
AlignmentResult<T> ProgMovieAlignmentCorrelationGPU<T>::computeGlobalAlignment(
        const MetaData &movie, const Image<T> &dark, const Image<T> &gain) {
//    auto gpu = GPU(device);
    auto movieSettings = this->getMovieSettings(movie, true);
    T sizeFactor = this->computeSizeFactor();
    if (this->verbose) {
        std::cout << "Settings for the movie: " << movieSettings << std::endl;
    }
    auto correlationSetting = this->getCorrelationSettings(movieSettings,
            std::make_pair(sizeFactor, sizeFactor));
    if (this->verbose) {
        std::cout << "Settings for the correlation: " << correlationSetting << std::endl;
    }

    MultidimArray<T> filter = this->createLPF(this->getTargetOccupancy(), correlationSetting.dim.x,
            correlationSetting.x_freq, correlationSetting.dim.y);

    T corrSizeMB = ((size_t) correlationSetting.x_freq
            * correlationSetting.dim.y
            * sizeof(std::complex<T>)) / ((T) 1024 * 1024);
    size_t framesInBuffer = std::ceil((gpu.value().lastFreeMem() / 3) / corrSizeMB);

    auto reference = core::optional<size_t>();

    T* data = loadMovie(movie, movieSettings, dark, gain);
    auto result = align(data, movieSettings, correlationSetting,
                    filter, reference,
            this->maxShift, framesInBuffer, this->verbose);
    delete[] data;
    return result;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::align(T *data,
        const FFTSettings<T> &in, const FFTSettings<T> &downscale,
        MultidimArray<T> &filter,
        core::optional<size_t> &refFrame,
        size_t maxShift, size_t framesInCorrelationBuffer, bool verbose) {
    assert(nullptr != data);
    size_t N = in.dim.n;
    // scale and transform to FFT on GPU
    performFFTAndScale<T>(data, N, in.dim.x, in.dim.y, in.batch,
            downscale.x_freq, downscale.dim.y, filter);

    auto scale = std::make_pair(in.dim.x / (T) downscale.dim.x,
            in.dim.y / (T) downscale.dim.y);

    return computeShifts(verbose, maxShift, (std::complex<T>*) data, downscale,
            in.dim.n,
            scale, framesInCorrelationBuffer, refFrame);
}

template<typename T>
T* ProgMovieAlignmentCorrelationGPU<T>::loadMovie(const MetaData& movie,
        const FFTSettings<T> &settings, const Image<T>& dark,
        const Image<T>& gain) {
    // allocate enough memory for the images. Since it will be reused, it has to be big
    // enough to store either all FFTs or all input images
    T* imgs = new T[std::max(settings.bytesFreq(), settings.bytesSpacial())]();
    Image<T> frame;

    int movieImgIndex = -1;
    FOR_ALL_OBJECTS_IN_METADATA(movie)
    {
        // update variables
        movieImgIndex++;
        if (movieImgIndex < this->nfirst) continue;
        if (movieImgIndex > this->nlast) break;

        // load image
        loadFrame(movie, __iter.objId, frame);
        if (XSIZE(dark()) > 0) frame() -= dark();
        if (XSIZE(gain()) > 0) frame() *= gain();

        // copy line by line, adding offset at the end of each line
        // result is the same image, padded in the X and Y dimensions
        T* dest = imgs
                + ((movieImgIndex - this->nfirst) * settings.dim.x
                        * settings.dim.y); // points to first float in the image
        for (size_t i = 0; i < settings.dim.y; ++i) {
            memcpy(dest + (settings.dim.x * i),
                    frame.data.data + i * frame.data.xdim,
                    settings.dim.x * sizeof(T));
        }
    }
    return imgs;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::computeShifts(
        Matrix1D<T> &bX, Matrix1D<T> &bY, Matrix2D<T> &A,
        const core::optional<size_t> &refFrame, size_t N) {
    // now get the estimated shift (from the equation system)
    // from each frame to successing frame
    Matrix1D<T> shiftX, shiftY;
    this->solveEquationSystem(bX, bY, A, shiftX, shiftY);
    // prepare result
    AlignmentResult<T> result {.refFrame = refFrame ?
                    refFrame.value() :
                    this->findReferenceImage(N, shiftX, shiftY)};
    result.shifts.reserve(N);
    // compute total shift in respect to reference frame
    for (size_t i = 0; i < N; ++i) {
        T x, y;
        this->computeTotalShift(result.refFrame, i, shiftX, shiftY, x, y);
        result.shifts.emplace_back(x, y);
    }
    return result;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::computeShifts(bool verbose,
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
            settings.dim.x,
            settings.dim.y, framesInCorrelationBuffer,
            settings.batch, correlations);
    // result is a centered correlation function with (hopefully) a cross
    // indicating the requested shift

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
            if (verbose) {
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
    AlignmentResult<T> result = computeShifts(bX, bY, A, refFrame, N);
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::applyShiftsComputeAverage(
//        const MetaData& movie, const Image<T>& dark, const Image<T>& gain,
//        Image<T>& initialMic, size_t& Ninitial, Image<T>& averageMicrograph,
//        size_t& N) {
//    // Apply shifts and compute average
//    Image<T> frame, croppedFrame, reducedFrame, shiftedFrame;
//    Matrix1D<T> shift(2);
//    FileName fnFrame;
//    int j = 0;
//    int n = 0;
//    Ninitial = N = 0;
//    GeoShiftTransformer<T> transformer;
//    FOR_ALL_OBJECTS_IN_METADATA(movie)
//    {
//        if (n >= this->nfirstSum && n <= this->nlastSum) {
//            movie.getValue(MDL_IMAGE, fnFrame, __iter.objId);
//            movie.getValue(MDL_SHIFT_X, XX(shift), __iter.objId);
//            movie.getValue(MDL_SHIFT_Y, YY(shift), __iter.objId);
//
//            std::cout << fnFrame << " shiftX=" << XX(shift) << " shiftY="
//                    << YY(shift) << std::endl;
//            frame.read(fnFrame);
//            if (XSIZE(dark()) > 0)
//                frame() -= dark();
//            if (XSIZE(gain()) > 0)
//                frame() *= gain();
//            if (this->yDRcorner != -1)
//                frame().window(croppedFrame(), this->yLTcorner, this->xLTcorner,
//                        this->yDRcorner, this->xDRcorner);
//            else
//                croppedFrame() = frame();
//            if (this->bin > 0) {
//                // FIXME add templates to respective functions/classes to avoid type casting
//                Image<double> croppedFrameDouble;
//                Image<double> reducedFrameDouble;
//                typeCast(croppedFrame(), croppedFrameDouble());
//
//                scaleToSizeFourier(1, floor(YSIZE(croppedFrame()) / this->bin),
//                        floor(XSIZE(croppedFrame()) / this->bin),
//                        croppedFrameDouble(), reducedFrameDouble());
//
//                typeCast(reducedFrameDouble(), reducedFrame());
//
//                shift /= this->bin;
//                croppedFrame() = reducedFrame();
//            }
//
//            if (this->fnInitialAvg != "") {
//                if (j == 0)
//                    initialMic() = croppedFrame();
//                else
//                    initialMic() += croppedFrame();
//                Ninitial++;
//            }
//
//            if (this->fnAligned != "" || this->fnAvg != "") {
//                if (this->outsideMode == OUTSIDE_WRAP) {
////                    Matrix2D<T> tmp;
////                    translation2DMatrix(shift, tmp, true);
//                    transformer.initLazy(croppedFrame().xdim,
//                            croppedFrame().ydim, 1, device);
//                    transformer.applyShift(shiftedFrame(), croppedFrame(), XX(shift), YY(shift));
////                    transformer.applyGeometry(this->BsplineOrder,
////                            shiftedFrame(), croppedFrame(), tmp, IS_INV, WRAP);
//                } else if (this->outsideMode == OUTSIDE_VALUE)
//                    translate(this->BsplineOrder, shiftedFrame(),
//                            croppedFrame(), shift, DONT_WRAP,
//                            this->outsideValue);
//                else
//                    translate(this->BsplineOrder, shiftedFrame(),
//                            croppedFrame(), shift, DONT_WRAP,
//                            (T) croppedFrame().computeAvg());
//                if (this->fnAligned != "")
//                    shiftedFrame.write(this->fnAligned, j + 1, true,
//                            WRITE_REPLACE);
//                if (this->fnAvg != "") {
//                    if (j == 0)
//                        averageMicrograph() = shiftedFrame();
//                    else
//                        averageMicrograph() += shiftedFrame();
//                    N++;
//                }
//            }
//            j++;
//        }
//        n++;
//    }
//}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::loadFrame(const MetaData& movie,
        size_t objId, Image<T>& out) {
    FileName fnFrame;
    movie.getValue(MDL_IMAGE, fnFrame, objId);
    if (-1 != this->yDRcorner) {
        Image<T> tmp;
        tmp.read(fnFrame);
        tmp().window(out(), this->yLTcorner, this->xLTcorner, this->yDRcorner,
                this->xDRcorner);
    } else {
        out.read(fnFrame);
    }
}

template<typename T>
int ProgMovieAlignmentCorrelationGPU<T>::getMaxFilterSize(
        const Image<T> &frame) {
    size_t maxXPow2 = std::ceil(log(frame.data.xdim) / log(2));
    size_t maxX = std::pow(2, maxXPow2);
    size_t maxFFTX = maxX / 2 + 1;
    size_t maxYPow2 = std::ceil(log(frame.data.ydim) / log(2));
    size_t maxY = std::pow(2, maxYPow2);
    size_t bytes = maxFFTX * maxY * sizeof(T);
    return bytes / (1024 * 1024);
}

//template<typename T>
//T* ProgMovieAlignmentCorrelationGPU<T>::loadToRAM(const MetaData& movie,
//        int noOfImgs, const Image<T>& dark, const Image<T>& gain,
//        bool cropInput) {
//    // allocate enough memory for the images. Since it will be reused, it has to be big
//    // enough to store either all FFTs or all input images
//    T* imgs = new T[noOfImgs * inputOptSizeY
//            * std::max(inputOptSizeX, inputOptSizeFFTX * 2)]();
//    Image<T> frame;
//
//    int movieImgIndex = -1;
//    FOR_ALL_OBJECTS_IN_METADATA(movie)
//    {
//        // update variables
//        movieImgIndex++;
//        if (movieImgIndex < this->nfirst) continue;
//        if (movieImgIndex > this->nlast) break;
//
//        // load image
//        loadFrame(movie, __iter.objId, frame);
//        if (XSIZE(dark()) > 0) frame() -= dark();
//        if (XSIZE(gain()) > 0) frame() *= gain();
//
//        // copy line by line, adding offset at the end of each line
//        // result is the same image, padded in the X and Y dimensions
//        T* dest = imgs
//                + ((movieImgIndex - this->nfirst) * inputOptSizeX
//                        * inputOptSizeY); // points to first float in the image
//        for (size_t i = 0; i < inputOptSizeY; ++i) {
//            memcpy(dest + (inputOptSizeX * i),
//                    frame.data.data + i * frame.data.xdim,
//                    inputOptSizeX * sizeof(T));
//        }
//    }
//    return imgs;
//}

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::setSizes(Image<T> &frame,
//        int noOfImgs) {
//
//    std::string UUID = getUUID(device);
//
//    int maxFilterSize = getMaxFilterSize(frame);
//    size_t availableMemMB = getFreeMem(device);
//    correlationBufferSizeMB = availableMemMB / 3; // divide available memory to 3 parts (2 buffers + 1 FFT)
//
//    if (! getStoredSizes(frame, noOfImgs, UUID)) {
//        runBenchmark(frame, noOfImgs, UUID);
//        storeSizes(frame, noOfImgs, UUID);
//    }
//
//    T corrSizeMB = ((size_t) croppedOptSizeFFTX * croppedOptSizeY
//            * sizeof(std::complex<T>)) / (1024 * 1024.);
//    correlationBufferImgs = std::ceil(correlationBufferSizeMB / corrSizeMB);
//}

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::runBenchmark(Image<T> &frame,
//        int noOfImgs, std::string &uuid) {
//    // get best sizes
//    int maxFilterSize = getMaxFilterSize(frame);
//    if (this->verbose)
//        std::cerr << "Benchmarking cuFFT ..." << std::endl;
//
//    size_t noOfCorrelations = (noOfImgs * (noOfImgs - 1)) / 2;
//
//    // we also need enough memory for filter
//    getBestFFTSize(noOfImgs, frame.data.xdim, frame.data.ydim, inputOptBatchSize,
//            true,
//            inputOptSizeX, inputOptSizeY, maxFilterSize, this->verbose, device,
//            frame().xdim == frame().ydim, 10); // allow max 10% change
//
//    inputOptSizeFFTX = inputOptSizeX / 2 + 1;
//
//    getBestFFTSize(noOfCorrelations, this->newXdim, this->newYdim,
//            croppedOptBatchSize, false, croppedOptSizeX, croppedOptSizeY,
//            correlationBufferSizeMB * 2, this->verbose, device,
//            this->newXdim == this->newYdim, 10);
//
//    croppedOptSizeFFTX = croppedOptSizeX / 2 + 1;
//}


//template<typename T>
//bool ProgMovieAlignmentCorrelationGPU<T>::getStoredSizes(Image<T> &frame,
//        int noOfImgs, std::string &uuid) {
//    bool res = true;
////    size_t neededMem;
////    res = res && UserSettings::get(storage).find(*this,
////        getKey(uuid, inputOptSizeXStr, frame, noOfImgs, true), inputOptSizeX);
////    res = res && UserSettings::get(storage).find(*this,
////        getKey(uuid, inputOptSizeYStr, frame, noOfImgs, true), inputOptSizeY);
////    res = res && UserSettings::get(storage).find(*this,
////        getKey(uuid, inputOptBatchSizeStr, frame, noOfImgs, true), inputOptBatchSize);
////    inputOptSizeFFTX =  inputOptSizeX / 2 + 1;
////    res = res && UserSettings::get(storage).find(*this,
////        getKey(uuid, availableMemoryStr, frame, noOfImgs, true), neededMem);
////    res = res && neededMem <= getFreeMem(device);
////
////    res = res && UserSettings::get(storage).find(*this,
////        getKey(uuid, croppedOptSizeXStr, this->newXdim, this->newYdim, noOfImgs, false), croppedOptSizeX);
////    res = res && UserSettings::get(storage).find(*this,
////        getKey(uuid, croppedOptSizeYStr, this->newXdim, this->newYdim, noOfImgs, false), croppedOptSizeY);
////    res = res && UserSettings::get(storage).find(*this,
////        getKey(uuid, croppedOptBatchSizeStr, this->newXdim, this->newYdim, noOfImgs, false),
////        croppedOptBatchSize);
////    croppedOptSizeFFTX =  croppedOptSizeX / 2 + 1;
////    res = res && UserSettings::get(storage).find(*this,
////        getKey(uuid, availableMemoryStr, this->newXdim, this->newYdim, noOfImgs, false), neededMem);
////    res = res && neededMem <= getFreeMem(device);
//
//    return res;
//}

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::storeSizes(Image<T> &frame,
//        int noOfImgs, std::string &uuid) {
////    UserSettings::get(storage).insert(*this,
////        getKey(uuid, inputOptSizeXStr, frame, noOfImgs, true), inputOptSizeX);
////    UserSettings::get(storage).insert(*this,
////        getKey(uuid, inputOptSizeYStr, frame, noOfImgs, true), inputOptSizeY);
////    UserSettings::get(storage).insert(*this,
////        getKey(uuid, inputOptBatchSizeStr, frame, noOfImgs, true),
////        inputOptBatchSize);
////    UserSettings::get(storage).insert(*this,
////        getKey(uuid, availableMemoryStr, frame, noOfImgs, true), getFreeMem(device));
////
////    UserSettings::get(storage).insert(*this,
////        getKey(uuid, croppedOptSizeXStr, this->newXdim, this->newYdim, noOfImgs, false),
////        croppedOptSizeX);
////    UserSettings::get(storage).insert(*this,
////        getKey(uuid, croppedOptSizeYStr, this->newXdim, this->newYdim, noOfImgs, false),
////        croppedOptSizeY);
////    UserSettings::get(storage).insert(*this,
////        getKey(uuid, croppedOptBatchSizeStr,
////        this->newXdim, this->newYdim, noOfImgs, false),
////        croppedOptBatchSize);
////    UserSettings::get(storage).insert(*this,
////        getKey(uuid, availableMemoryStr, this->newXdim, this->newYdim, noOfImgs, false),
////        getFreeMem(device));
//}

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::testFFT() {
//
//    double delta = 0.00001;
//    size_t x, y;
//    x = y = 2304;
//    size_t order = 10000;
//
//    srand(42);
//
//    Image<double> inputDouble(x, y); // keep sync with values
//    Image<float> inputFloat(x, y); // keep sync with values
//    size_t pixels = inputDouble.data.xdim * inputDouble.data.ydim;
//    for (size_t y = 0; y < inputDouble.data.ydim; ++y) {
//        for (size_t x = 0; x < inputDouble.data.xdim; ++x) {
//            size_t index = y * inputDouble.data.xdim + x;
//            double value = rand() / (RAND_MAX / 2000.);
//            inputDouble.data.data[index] = value;
//            inputFloat.data.data[index] = (float) value;
//        }
//    }
//
//    // CPU part
//
//    MultidimArray<std::complex<double> > tmpFFTCpu;
//    FourierTransformer transformer;
//
//    transformer.FourierTransform(inputDouble(), tmpFFTCpu, true);
//
//    // store results to drive
//    Image<double> fftCPU(tmpFFTCpu.xdim, tmpFFTCpu.ydim);
//    size_t fftPixels = fftCPU.data.yxdim;
//    for (size_t i = 0; i < fftPixels; i++) {
//        fftCPU.data.data[i] = tmpFFTCpu.data[i].real();
//    }
//    fftCPU.write("testFFTCpu.vol");
//
//    // GPU part
//
//    GpuMultidimArrayAtGpu<float> gpuIn(inputFloat.data.xdim,
//            inputFloat.data.ydim);
//    gpuIn.copyToGpu(inputFloat.data.data);
//    GpuMultidimArrayAtGpu<std::complex<float> > gpuFFT;
//    mycufftHandle handle;
//    gpuIn.fft(gpuFFT, handle);
//
//    fftPixels = gpuFFT.yxdim;
//    std::complex<float>* tmpFFTGpu = new std::complex<float>[fftPixels];
//    gpuFFT.copyToCpu(tmpFFTGpu);
//
//    // store results to drive
//    Image<float> fftGPU(gpuFFT.Xdim, gpuFFT.Ydim);
//    float norm = inputFloat.data.yxdim;
//    for (size_t i = 0; i < fftPixels; i++) {
//        fftGPU.data.data[i] = tmpFFTGpu[i].real() / norm;
//    }
//    fftGPU.write("testFFTGpu.vol");
//
//    ////////////////////////////////////////
//
//    if (fftCPU.data.xdim != fftGPU.data.xdim) {
//        printf("wrong size: X cpu %lu X gpu %lu\n", fftCPU.data.xdim,
//                fftGPU.data.xdim);
//    }
//    if (fftCPU.data.ydim != fftGPU.data.ydim) {
//        printf("wrong size: Y cpu %lu Y gpu %lu\n", fftCPU.data.xdim,
//                fftGPU.data.xdim);
//    }
//
//    for (size_t i = 0; i < fftCPU.data.yxdim; ++i) {
//        float cpuReal = tmpFFTCpu.data[i].real();
//        float cpuImag = tmpFFTCpu.data[i].imag();
//        float gpuReal = tmpFFTGpu[i].real() / norm;
//        float gpuImag = tmpFFTGpu[i].imag() / norm;
//        if ((std::abs(cpuReal - gpuReal) > delta)
//                || (std::abs(cpuImag - gpuImag) > delta)) {
//            printf("ERROR FFT: %lu cpu (%f, %f) gpu (%f, %f)\n", i, cpuReal,
//                    cpuImag, gpuReal, gpuImag);
//        }
//    }
//
//    delete[] tmpFFTGpu;
//
//}

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::testFilterAndScale() {
//    double delta = 0.00001;
//    size_t xIn, yIn, xOut, yOut, xOutFFT;
//    xIn = yIn = 4096;
//    xOut = yOut = 2275;
//    xOutFFT = xOut / 2 + 1;
//
//    size_t fftPixels = xOutFFT * yOut;
//    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[fftPixels];
//    float* filter = new float[fftPixels];
//    for (size_t i = 0; i < fftPixels; ++i) {
//        filter[i] = (rand() * 100) / (float) RAND_MAX;
//    }
//
//    srand(42);
//
//    Image<double> inputDouble(xIn, yIn); // keep sync with values
//    Image<float> inputFloat(xIn, yIn); // keep sync with values
//    size_t pixels = inputDouble.data.xdim * inputDouble.data.ydim;
//    for (size_t y = 0; y < inputDouble.data.ydim; ++y) {
//        for (size_t x = 0; x < inputDouble.data.xdim; ++x) {
//            size_t index = y * inputDouble.data.xdim + x;
//            double value = rand() > (RAND_MAX / 2) ? -1 : 1; // ((int)(1000 * (double)rand() / (RAND_MAX))) / 1000.f;
//            inputDouble.data.data[index] = value;
//            inputFloat.data.data[index] = (float) value;
//        }
//    }
////	inputDouble(0,0) = 1;
////	inputFloat(0,0) = 1;
//    Image<double> outputDouble(xOut, yOut);
//    Image<double> reducedFrame;
//
//    // CPU part
//
//    scaleToSizeFourier(1, yOut, xOut, inputDouble(), reducedFrame());
////	inputDouble().printStats();
////	printf("\n");
////	reducedFrame().printStats();
////	printf("\n");
//    // Now do the Fourier transform and filter
//    MultidimArray<std::complex<double> > *tmpFFTCpuOut = new MultidimArray<
//            std::complex<double> >;
//    MultidimArray<std::complex<double> > *tmpFFTCpuOutFull = new MultidimArray<
//            std::complex<double> >;
//    FourierTransformer transformer;
//
//    transformer.FourierTransform(inputDouble(), *tmpFFTCpuOutFull);
////	std::cout << *tmpFFTCpuOutFull<< std::endl;
//
//    transformer.FourierTransform(reducedFrame(), *tmpFFTCpuOut, true);
//    for (size_t nn = 0; nn < fftPixels; ++nn) {
//        double wlpf = filter[nn];
//        DIRECT_MULTIDIM_ELEM(*tmpFFTCpuOut,nn) *= wlpf;
//    }
//
//    // store results to drive
//    Image<double> fftCPU(tmpFFTCpuOut->xdim, tmpFFTCpuOut->ydim);
//    fftPixels = tmpFFTCpuOut->yxdim;
//    for (size_t i = 0; i < fftPixels; i++) {
//        fftCPU.data.data[i] = tmpFFTCpuOut->data[i].real();
//        if (fftCPU.data.data[i] > 10)
//            fftCPU.data.data[i] = 0;
//    }
//    fftCPU.write("testFFTCpuScaledFiltered.vol");
//
//    // GPU part
//
//    float* d_filter = loadToGPU(filter, fftPixels);
//
//    GpuMultidimArrayAtGpu<float> gpuIn(inputFloat.data.xdim,
//            inputFloat.data.ydim);
//    gpuIn.copyToGpu(inputFloat.data.data);
//    GpuMultidimArrayAtGpu<std::complex<float> > gpuFFT;
//    mycufftHandle handle;
//
////    processInput(gpuIn, gpuFFT, handle, xIn, yIn, 1, xOutFFT, yOut, d_filter,
////            tmpFFTGpuOut); // FIXME test
//
//    // store results to drive
//    Image<float> fftGPU(xOutFFT, yOut);
//    float norm = inputFloat.data.yxdim;
//    for (size_t i = 0; i < fftPixels; i++) {
//        fftGPU.data.data[i] = tmpFFTGpuOut[i].real() / norm;
//        if (fftGPU.data.data[i] > 10)
//            fftGPU.data.data[i] = 0;
//    }
//    fftGPU.write("testFFTGpuScaledFiltered.vol");
//
//    ////////////////////////////////////////
//
//    if (fftCPU.data.xdim != fftGPU.data.xdim) {
//        printf("wrong size: X cpu %lu X gpu %lu\n", fftCPU.data.xdim,
//                fftGPU.data.xdim);
//    }
//    if (fftCPU.data.ydim != fftGPU.data.ydim) {
//        printf("wrong size: Y cpu %lu Y gpu %lu\n", fftCPU.data.xdim,
//                fftGPU.data.xdim);
//    }
//    if (tmpFFTCpuOut->ydim != yOut) {
//        printf("wrong size tmpFFTCpuOut: Y cpu %lu Y gpu %lu\n",
//                tmpFFTCpuOut->ydim, yOut);
//    }
//    if (tmpFFTCpuOut->xdim != xOutFFT) {
//        printf("wrong size tmpFFTCpuOut: X cpu %lu X gpu %lu\n",
//                tmpFFTCpuOut->xdim, xOutFFT);
//    }
//
//    for (size_t i = 0; i < fftCPU.data.yxdim; ++i) {
//        float cpuReal = tmpFFTCpuOut->data[i].real();
//        float cpuImag = tmpFFTCpuOut->data[i].imag();
//        float gpuReal = tmpFFTGpuOut[i].real() / norm;
//        float gpuImag = tmpFFTGpuOut[i].imag() / norm;
//        if ((std::abs(cpuReal - gpuReal) > delta)
//                || (std::abs(cpuImag - gpuImag) > delta)) {
//            printf("ERROR FILTER: %lu cpu (%f, %f) gpu (%f, %f)\n", i, cpuReal,
//                    cpuImag, gpuReal, gpuImag);
//        }
//    }
//    delete[] tmpFFTGpuOut;
//}

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::testScalingGpuOO() {
//    double delta = 0.000001;
//    size_t xIn, yIn, xOut, yOut, xOutFFT, xInFFT;
//    xIn = yIn = 9;
//    xOut = yOut = 5;
//    xOutFFT = xOut / 2 + 1; // == 3
//    xInFFT = xIn / 2 + 1; // == 5
//
//    std::complex<float>* tmpFFTGpuIn = new std::complex<float>[yIn * xInFFT];
//    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[yOut * xOutFFT];
//    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
//    for (size_t y = 0; y < yIn; ++y) {
//        for (size_t x = 0; x < xInFFT; ++x) {
//            size_t index = y * xInFFT + x;
//            tmpFFTGpuIn[index] = std::complex<float>(y, x);
//        }
//    }
//
//    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
//    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
//    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);
//
//    tmpFFTCpuOutExpected[3] = std::complex<double>(1, 0);
//    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 1);
//    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 2);
//
//    tmpFFTCpuOutExpected[6] = std::complex<double>(2, 0);
//    tmpFFTCpuOutExpected[7] = std::complex<double>(2, 1);
//    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 2);
//
//    tmpFFTCpuOutExpected[9] = std::complex<double>(7, 0);
//    tmpFFTCpuOutExpected[10] = std::complex<double>(7, 1);
//    tmpFFTCpuOutExpected[11] = std::complex<double>(7, 2);
//
//    tmpFFTCpuOutExpected[12] = std::complex<double>(8, 0);
//    tmpFFTCpuOutExpected[13] = std::complex<double>(8, 1);
//    tmpFFTCpuOutExpected[14] = std::complex<double>(8, 2);
//
////    applyFilterAndCrop<float>(tmpFFTGpuIn, tmpFFTGpuOut, 1, xInFFT, yIn,
////            xOutFFT, yOut, NULL); // FIXME test
//
//    ////////////////////////////////////////
//
//    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
//        float cpuReal = tmpFFTGpuOut[i].real();
//        float cpuImag = tmpFFTGpuOut[i].imag();
//        float expReal = tmpFFTCpuOutExpected[i].real();
//        float expImag = tmpFFTCpuOutExpected[i].imag();
//        if ((std::abs(cpuReal - expReal) > delta)
//                || (std::abs(cpuImag - expImag) > delta)) {
//            printf("ERROR SCALE GPU OO: %lu gpu (%f, %f) exp (%f, %f)\n", i,
//                    cpuReal, cpuImag, expReal, expImag);
//        }
//    }
//}

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::testScalingGpuEO() {
//    double delta = 0.000001;
//    size_t xIn, yIn, xOut, yOut, xOutFFT, xInFFT;
//    xIn = yIn = 10;
//    xOut = yOut = 5;
//    xOutFFT = xOut / 2 + 1; // == 3
//    xInFFT = xIn / 2 + 1; // == 6
//
//    std::complex<float>* tmpFFTGpuIn = new std::complex<float>[yIn * xInFFT];
//    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[yOut * xOutFFT];
//    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
//    for (size_t y = 0; y < yIn; ++y) {
//        for (size_t x = 0; x < xInFFT; ++x) {
//            size_t index = y * xInFFT + x;
//            tmpFFTGpuIn[index] = std::complex<float>(y, x);
//        }
//    }
//
//    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
//    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
//    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);
//
//    tmpFFTCpuOutExpected[3] = std::complex<double>(1, 0);
//    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 1);
//    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 2);
//
//    tmpFFTCpuOutExpected[6] = std::complex<double>(2, 0);
//    tmpFFTCpuOutExpected[7] = std::complex<double>(2, 1);
//    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 2);
//
//    tmpFFTCpuOutExpected[9] = std::complex<double>(8, 0);
//    tmpFFTCpuOutExpected[10] = std::complex<double>(8, 1);
//    tmpFFTCpuOutExpected[11] = std::complex<double>(8, 2);
//
//    tmpFFTCpuOutExpected[12] = std::complex<double>(9, 0);
//    tmpFFTCpuOutExpected[13] = std::complex<double>(9, 1);
//    tmpFFTCpuOutExpected[14] = std::complex<double>(9, 2);
//
////    applyFilterAndCrop<float>(tmpFFTGpuIn, tmpFFTGpuOut, 1, xInFFT, yIn,
////            xOutFFT, yOut, NULL); // FIXME test
//
//    ////////////////////////////////////////
//
//    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
//        float cpuReal = tmpFFTGpuOut[i].real();
//        float cpuImag = tmpFFTGpuOut[i].imag();
//        float expReal = tmpFFTCpuOutExpected[i].real();
//        float expImag = tmpFFTCpuOutExpected[i].imag();
//        if ((std::abs(cpuReal - expReal) > delta)
//                || (std::abs(cpuImag - expImag) > delta)) {
//            printf("ERROR SCALE GPU EO: %lu gpu (%f, %f) exp (%f, %f)\n", i,
//                    cpuReal, cpuImag, expReal, expImag);
//        }
//    }
//}

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::testScalingGpuOE() {
//    double delta = 0.000001;
//    size_t xIn, yIn, xOut, yOut, xOutFFT, xInFFT;
//    xIn = yIn = 9;
//    xOut = yOut = 6;
//    xOutFFT = xOut / 2 + 1; // == 4
//    xInFFT = xIn / 2 + 1; // == 5
//
//    std::complex<float>* tmpFFTGpuIn = new std::complex<float>[yIn * xInFFT];
//    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[yOut * xOutFFT];
//    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
//    for (size_t y = 0; y < yIn; ++y) {
//        for (size_t x = 0; x < xInFFT; ++x) {
//            size_t index = y * xInFFT + x;
//            tmpFFTGpuIn[index] = std::complex<float>(y, x);
//        }
//    }
//    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
//    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
//    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);
//    tmpFFTCpuOutExpected[3] = std::complex<double>(0, 3);
//
//    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 0);
//    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 1);
//    tmpFFTCpuOutExpected[6] = std::complex<double>(1, 2);
//    tmpFFTCpuOutExpected[7] = std::complex<double>(1, 3);
//
//    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 0);
//    tmpFFTCpuOutExpected[9] = std::complex<double>(2, 1);
//    tmpFFTCpuOutExpected[10] = std::complex<double>(2, 2);
//    tmpFFTCpuOutExpected[11] = std::complex<double>(2, 3);
//
//    tmpFFTCpuOutExpected[12] = std::complex<double>(3, 0);
//    tmpFFTCpuOutExpected[13] = std::complex<double>(3, 1);
//    tmpFFTCpuOutExpected[14] = std::complex<double>(3, 2);
//    tmpFFTCpuOutExpected[15] = std::complex<double>(3, 3);
//
//    tmpFFTCpuOutExpected[16] = std::complex<double>(7, 0);
//    tmpFFTCpuOutExpected[17] = std::complex<double>(7, 1);
//    tmpFFTCpuOutExpected[18] = std::complex<double>(7, 2);
//    tmpFFTCpuOutExpected[19] = std::complex<double>(7, 3);
//
//    tmpFFTCpuOutExpected[20] = std::complex<double>(8, 0);
//    tmpFFTCpuOutExpected[21] = std::complex<double>(8, 1);
//    tmpFFTCpuOutExpected[22] = std::complex<double>(8, 2);
//    tmpFFTCpuOutExpected[23] = std::complex<double>(8, 3);
//
////    applyFilterAndCrop<float>(tmpFFTGpuIn, tmpFFTGpuOut, 1, xInFFT, yIn,
////            xOutFFT, yOut, NULL); // FIXME test
//
//    ////////////////////////////////////////
//
//    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
//        float cpuReal = tmpFFTGpuOut[i].real();
//        float cpuImag = tmpFFTGpuOut[i].imag();
//        float expReal = tmpFFTCpuOutExpected[i].real();
//        float expImag = tmpFFTCpuOutExpected[i].imag();
//        if ((std::abs(cpuReal - expReal) > delta)
//                || (std::abs(cpuImag - expImag) > delta)) {
//            printf("ERROR SCALE GPU OE: %lu gpu (%f, %f) exp (%f, %f)\n", i,
//                    cpuReal, cpuImag, expReal, expImag);
//        }
//    }
//}

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::testScalingGpuEE() {
//    double delta = 0.000001;
//    size_t xIn, yIn, xOut, yOut, xOutFFT, xInFFT;
//    xIn = yIn = 10;
//    xOut = yOut = 6;
//    xOutFFT = xOut / 2 + 1; // == 4
//    xInFFT = xIn / 2 + 1; // == 6
//
//    std::complex<float>* tmpFFTGpuIn = new std::complex<float>[yIn * xInFFT];
//    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[yOut * xOutFFT];
//    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
//    for (size_t y = 0; y < yIn; ++y) {
//        for (size_t x = 0; x < xInFFT; ++x) {
//            size_t index = y * xInFFT + x;
//            tmpFFTGpuIn[index] = std::complex<float>(y, x);
//        }
//    }
//
//    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
//    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
//    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);
//    tmpFFTCpuOutExpected[3] = std::complex<double>(0, 3);
//
//    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 0);
//    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 1);
//    tmpFFTCpuOutExpected[6] = std::complex<double>(1, 2);
//    tmpFFTCpuOutExpected[7] = std::complex<double>(1, 3);
//
//    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 0);
//    tmpFFTCpuOutExpected[9] = std::complex<double>(2, 1);
//    tmpFFTCpuOutExpected[10] = std::complex<double>(2, 2);
//    tmpFFTCpuOutExpected[11] = std::complex<double>(2, 3);
//
//    tmpFFTCpuOutExpected[12] = std::complex<double>(3, 0);
//    tmpFFTCpuOutExpected[13] = std::complex<double>(3, 1);
//    tmpFFTCpuOutExpected[14] = std::complex<double>(3, 2);
//    tmpFFTCpuOutExpected[15] = std::complex<double>(3, 3);
//
//    tmpFFTCpuOutExpected[16] = std::complex<double>(8, 0);
//    tmpFFTCpuOutExpected[17] = std::complex<double>(8, 1);
//    tmpFFTCpuOutExpected[18] = std::complex<double>(8, 2);
//    tmpFFTCpuOutExpected[19] = std::complex<double>(8, 3);
//
//    tmpFFTCpuOutExpected[20] = std::complex<double>(9, 0);
//    tmpFFTCpuOutExpected[21] = std::complex<double>(9, 1);
//    tmpFFTCpuOutExpected[22] = std::complex<double>(9, 2);
//    tmpFFTCpuOutExpected[23] = std::complex<double>(9, 3);
//
////    applyFilterAndCrop<float>(tmpFFTGpuIn, tmpFFTGpuOut, 1, xInFFT, yIn,
////            xOutFFT, yOut, NULL); // FIXME test
//
//    ////////////////////////////////////////
//
//    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
//        float cpuReal = tmpFFTGpuOut[i].real();
//        float cpuImag = tmpFFTGpuOut[i].imag();
//        float expReal = tmpFFTCpuOutExpected[i].real();
//        float expImag = tmpFFTCpuOutExpected[i].imag();
//        if ((std::abs(cpuReal - expReal) > delta)
//                || (std::abs(cpuImag - expImag) > delta)) {
//            printf("ERROR SCALE GPU EE: %lu gpu (%f, %f) exp (%f, %f)\n", i,
//                    cpuReal, cpuImag, expReal, expImag);
//        }
//    }
//}

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::testScalingCpuOO() {
//    double delta = 0.000001;
//    size_t xIn, yIn, xOut, yOut, xOutFFT, xInFFT;
//    xIn = yIn = 9;
//    xOut = yOut = 5;
//    xOutFFT = xOut / 2 + 1; // == 3
//
//    Image<double> inputDouble(xIn, yIn);
//    Image<double> outputDouble(xOut, yOut);
//    MultidimArray<std::complex<double> > tmpFFTCpuIn(yIn, xIn / 2 + 1);
//    MultidimArray<std::complex<double> > tmpFFTCpuOut(yOut, xOutFFT);
//    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
//    for (size_t y = 0; y < tmpFFTCpuIn.ydim; ++y) {
//        for (size_t x = 0; x < tmpFFTCpuIn.xdim; ++x) {
//            size_t index = y * tmpFFTCpuIn.xdim + x;
//            tmpFFTCpuIn.data[index] = std::complex<double>(y, x);
//        }
//    }
//
//    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
//    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
//    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);
//
//    tmpFFTCpuOutExpected[3] = std::complex<double>(1, 0);
//    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 1);
//    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 2);
//
//    tmpFFTCpuOutExpected[6] = std::complex<double>(2, 0);
//    tmpFFTCpuOutExpected[7] = std::complex<double>(2, 1);
//    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 2);
//
//    tmpFFTCpuOutExpected[9] = std::complex<double>(7, 0);
//    tmpFFTCpuOutExpected[10] = std::complex<double>(7, 1);
//    tmpFFTCpuOutExpected[11] = std::complex<double>(7, 2);
//
//    tmpFFTCpuOutExpected[12] = std::complex<double>(8, 0);
//    tmpFFTCpuOutExpected[13] = std::complex<double>(8, 1);
//    tmpFFTCpuOutExpected[14] = std::complex<double>(8, 2);
//
//    scaleToSizeFourier(inputDouble(), outputDouble(), tmpFFTCpuIn,
//            tmpFFTCpuOut);
//
//    ////////////////////////////////////////
//
//    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
//        float cpuReal = tmpFFTCpuOut.data[i].real();
//        float cpuImag = tmpFFTCpuOut.data[i].imag();
//        float expReal = tmpFFTCpuOutExpected[i].real();
//        float expImag = tmpFFTCpuOutExpected[i].imag();
//        if ((std::abs(cpuReal - expReal) > delta)
//                || (std::abs(cpuImag - expImag) > delta)) {
//            printf("ERROR SCALE CPU OO: %lu cpu (%f, %f) exp (%f, %f)\n", i,
//                    cpuReal, cpuImag, expReal, expImag);
//        }
//    }
//}

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::testScalingCpuEO() {
//    double delta = 0.000001;
//    size_t xIn, yIn, xOut, yOut, xOutFFT;
//    xIn = yIn = 10;
//    xOut = yOut = 5;
//    xOutFFT = xOut / 2 + 1; // == 3
//
//    Image<double> inputDouble(xIn, yIn);
//    Image<double> outputDouble(xOut, yOut);
//    MultidimArray<std::complex<double> > tmpFFTCpuIn(yIn, xIn / 2 + 1);
//    MultidimArray<std::complex<double> > tmpFFTCpuOut(yOut, xOutFFT);
//    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
//    for (size_t y = 0; y < tmpFFTCpuIn.ydim; ++y) {
//        for (size_t x = 0; x < tmpFFTCpuIn.xdim; ++x) {
//            size_t index = y * tmpFFTCpuIn.xdim + x;
//            tmpFFTCpuIn.data[index] = std::complex<double>(y, x);
//        }
//    }
//
//    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
//    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
//    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);
//
//    tmpFFTCpuOutExpected[3] = std::complex<double>(1, 0);
//    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 1);
//    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 2);
//
//    tmpFFTCpuOutExpected[6] = std::complex<double>(2, 0);
//    tmpFFTCpuOutExpected[7] = std::complex<double>(2, 1);
//    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 2);
//
//    tmpFFTCpuOutExpected[9] = std::complex<double>(8, 0);
//    tmpFFTCpuOutExpected[10] = std::complex<double>(8, 1);
//    tmpFFTCpuOutExpected[11] = std::complex<double>(8, 2);
//
//    tmpFFTCpuOutExpected[12] = std::complex<double>(9, 0);
//    tmpFFTCpuOutExpected[13] = std::complex<double>(9, 1);
//    tmpFFTCpuOutExpected[14] = std::complex<double>(9, 2);
//
//    scaleToSizeFourier(inputDouble(), outputDouble(), tmpFFTCpuIn,
//            tmpFFTCpuOut);
//
//    ////////////////////////////////////////
//
//    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
//        float cpuReal = tmpFFTCpuOut.data[i].real();
//        float cpuImag = tmpFFTCpuOut.data[i].imag();
//        float expReal = tmpFFTCpuOutExpected[i].real();
//        float expImag = tmpFFTCpuOutExpected[i].imag();
//        if ((std::abs(cpuReal - expReal) > delta)
//                || (std::abs(cpuImag - expImag) > delta)) {
//            printf("ERROR SCALE CPU EO: %lu cpu (%f, %f) exp (%f, %f)\n", i,
//                    cpuReal, cpuImag, expReal, expImag);
//        }
//    }
//}

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::testScalingCpuOE() {
//    double delta = 0.000001;
//    size_t xIn, yIn, xOut, yOut, xOutFFT;
//    xIn = yIn = 9;
//    xOut = yOut = 6;
//    xOutFFT = xOut / 2 + 1; // == 4
//
//    Image<double> inputDouble(xIn, yIn);
//    Image<double> outputDouble(xOut, yOut);
//    MultidimArray<std::complex<double> > tmpFFTCpuIn(yIn, xIn / 2 + 1);
//    MultidimArray<std::complex<double> > tmpFFTCpuOut(yOut, xOutFFT);
//    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
//    for (size_t y = 0; y < tmpFFTCpuIn.ydim; ++y) {
//        for (size_t x = 0; x < tmpFFTCpuIn.xdim; ++x) {
//            size_t index = y * tmpFFTCpuIn.xdim + x;
//            tmpFFTCpuIn.data[index] = std::complex<double>(y, x);
//        }
//    }
//
//    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
//    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
//    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);
//    tmpFFTCpuOutExpected[3] = std::complex<double>(0, 3);
//
//    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 0);
//    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 1);
//    tmpFFTCpuOutExpected[6] = std::complex<double>(1, 2);
//    tmpFFTCpuOutExpected[7] = std::complex<double>(1, 3);
//
//    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 0);
//    tmpFFTCpuOutExpected[9] = std::complex<double>(2, 1);
//    tmpFFTCpuOutExpected[10] = std::complex<double>(2, 2);
//    tmpFFTCpuOutExpected[11] = std::complex<double>(2, 3);
//
//    tmpFFTCpuOutExpected[12] = std::complex<double>(3, 0);
//    tmpFFTCpuOutExpected[13] = std::complex<double>(3, 1);
//    tmpFFTCpuOutExpected[14] = std::complex<double>(3, 2);
//    tmpFFTCpuOutExpected[15] = std::complex<double>(3, 3);
//
//    tmpFFTCpuOutExpected[16] = std::complex<double>(7, 0);
//    tmpFFTCpuOutExpected[17] = std::complex<double>(7, 1);
//    tmpFFTCpuOutExpected[18] = std::complex<double>(7, 2);
//    tmpFFTCpuOutExpected[19] = std::complex<double>(7, 3);
//
//    tmpFFTCpuOutExpected[20] = std::complex<double>(8, 0);
//    tmpFFTCpuOutExpected[21] = std::complex<double>(8, 1);
//    tmpFFTCpuOutExpected[22] = std::complex<double>(8, 2);
//    tmpFFTCpuOutExpected[23] = std::complex<double>(8, 3);
//
//    scaleToSizeFourier(inputDouble(), outputDouble(), tmpFFTCpuIn,
//            tmpFFTCpuOut);
//
//    ////////////////////////////////////////
//
//    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
//        float cpuReal = tmpFFTCpuOut.data[i].real();
//        float cpuImag = tmpFFTCpuOut.data[i].imag();
//        float expReal = tmpFFTCpuOutExpected[i].real();
//        float expImag = tmpFFTCpuOutExpected[i].imag();
//        if ((std::abs(cpuReal - expReal) > delta)
//                || (std::abs(cpuImag - expImag) > delta)) {
//            printf("ERROR SCALE CPU OE: %lu cpu (%f, %f) exp (%f, %f)\n", i,
//                    cpuReal, cpuImag, expReal, expImag);
//        }
//    }
//}

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::testScalingCpuEE() {
//    double delta = 0.000001;
//    size_t xIn, yIn, xOut, yOut, xOutFFT;
//    xIn = yIn = 10;
//    xOut = yOut = 6;
//    xOutFFT = xOut / 2 + 1; // == 4
//
//    Image<double> inputDouble(xIn, yIn);
//    Image<double> outputDouble(xOut, yOut);
//    MultidimArray<std::complex<double> > tmpFFTCpuIn(yIn, xIn / 2 + 1);
//    MultidimArray<std::complex<double> > tmpFFTCpuOut(yOut, xOutFFT);
//    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
//    for (size_t y = 0; y < tmpFFTCpuIn.ydim; ++y) {
//        for (size_t x = 0; x < tmpFFTCpuIn.xdim; ++x) {
//            size_t index = y * tmpFFTCpuIn.xdim + x;
//            tmpFFTCpuIn.data[index] = std::complex<double>(y, x);
//        }
//    }
//
//    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
//    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
//    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);
//    tmpFFTCpuOutExpected[3] = std::complex<double>(0, 3);
//
//    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 0);
//    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 1);
//    tmpFFTCpuOutExpected[6] = std::complex<double>(1, 2);
//    tmpFFTCpuOutExpected[7] = std::complex<double>(1, 3);
//
//    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 0);
//    tmpFFTCpuOutExpected[9] = std::complex<double>(2, 1);
//    tmpFFTCpuOutExpected[10] = std::complex<double>(2, 2);
//    tmpFFTCpuOutExpected[11] = std::complex<double>(2, 3);
//
//    tmpFFTCpuOutExpected[12] = std::complex<double>(3, 0);
//    tmpFFTCpuOutExpected[13] = std::complex<double>(3, 1);
//    tmpFFTCpuOutExpected[14] = std::complex<double>(3, 2);
//    tmpFFTCpuOutExpected[15] = std::complex<double>(3, 3);
//
//    tmpFFTCpuOutExpected[16] = std::complex<double>(8, 0);
//    tmpFFTCpuOutExpected[17] = std::complex<double>(8, 1);
//    tmpFFTCpuOutExpected[18] = std::complex<double>(8, 2);
//    tmpFFTCpuOutExpected[19] = std::complex<double>(8, 3);
//
//    tmpFFTCpuOutExpected[20] = std::complex<double>(9, 0);
//    tmpFFTCpuOutExpected[21] = std::complex<double>(9, 1);
//    tmpFFTCpuOutExpected[22] = std::complex<double>(9, 2);
//    tmpFFTCpuOutExpected[23] = std::complex<double>(9, 3);
//
//    scaleToSizeFourier(inputDouble(), outputDouble(), tmpFFTCpuIn,
//            tmpFFTCpuOut);
//
//    ////////////////////////////////////////
//
//    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
//        float cpuReal = tmpFFTCpuOut.data[i].real();
//        float cpuImag = tmpFFTCpuOut.data[i].imag();
//        float expReal = tmpFFTCpuOutExpected[i].real();
//        float expImag = tmpFFTCpuOutExpected[i].imag();
//        if ((std::abs(cpuReal - expReal) > delta)
//                || (std::abs(cpuImag - expImag) > delta)) {
//            printf("ERROR SCALE CPU EE: %lu cpu (%f, %f) exp (%f, %f)\n", i,
//                    cpuReal, cpuImag, expReal, expImag);
//        }
//    }
//}

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::testFFTAndScale() {
//    double delta = 0.00001;
//    size_t xIn, yIn, xOut, yOut, xOutFFT;
//    xIn = yIn = 4096;
//    xOut = yOut = 2276;
//    xOutFFT = xOut / 2 + 1;
//    size_t order = 10000;
//    size_t fftPixels = xOutFFT * yOut;
//
//    srand(42);
//
//    Image<double> inputDouble(xIn, yIn); // keep sync with values
//    Image<float> inputFloat(xIn, yIn); // keep sync with values
//    size_t pixels = inputDouble.data.xdim * inputDouble.data.ydim;
//    for (size_t y = 0; y < inputDouble.data.ydim; ++y) {
//        for (size_t x = 0; x < inputDouble.data.xdim; ++x) {
//            size_t index = y * inputDouble.data.xdim + x;
//            double value = rand() / (RAND_MAX / 2000.);
//            inputDouble.data.data[index] = value;
//            inputFloat.data.data[index] = (float) value;
//        }
//    }
//
//    float* filter = new float[fftPixels];
//    for (size_t i = 0; i < fftPixels; ++i) {
//        filter[i] = rand() / (float) RAND_MAX;
//    }
//
//    // CPU part
//    Image<double> outputDouble(xOut, yOut);
//    MultidimArray<std::complex<double> > tmpFFTCpuIn;
//    MultidimArray<std::complex<double> > tmpFFTCpuOut(yOut, xOutFFT);
//    FourierTransformer transformer;
//
//    transformer.FourierTransform(inputDouble(), tmpFFTCpuIn, true);
//    scaleToSizeFourier(inputDouble(), outputDouble(), tmpFFTCpuIn,
//            tmpFFTCpuOut);
//
//    for (size_t nn = 0; nn < fftPixels; ++nn) {
//        double wlpf = filter[nn];
//        DIRECT_MULTIDIM_ELEM(tmpFFTCpuOut,nn) *= wlpf;
//    }
//
//    // store results to drive
//    Image<double> fftCPU(tmpFFTCpuOut.xdim, tmpFFTCpuOut.ydim);
//    for (size_t i = 0; i < fftPixels; i++) {
//        fftCPU.data.data[i] = tmpFFTCpuOut.data[i].real();
//    }
//    fftCPU.write("testFFTCpuScaled.vol");
//
//    // GPU part
//
//    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[fftPixels];
//    float* d_filter = loadToGPU(filter, fftPixels);
//
//    GpuMultidimArrayAtGpu<float> gpuIn(inputFloat.data.xdim,
//            inputFloat.data.ydim);
//    gpuIn.copyToGpu(inputFloat.data.data);
//    GpuMultidimArrayAtGpu<std::complex<float> > gpuFFT;
//    mycufftHandle handle;
//
////    processInput(gpuIn, gpuFFT, handle, xIn, yIn, 1, xOutFFT, yOut, d_filter,
////            tmpFFTGpuOut); FIXME test
//
//    // store results to drive
//    Image<float> fftGPU(xOutFFT, yOut);
//    float norm = inputFloat.data.yxdim;
//    for (size_t i = 0; i < fftPixels; i++) {
//        fftGPU.data.data[i] = tmpFFTGpuOut[i].real() / norm;
//    }
//    fftGPU.write("testFFTGpuScaled.vol");
//
//    ////////////////////////////////////////
//
//    if (fftCPU.data.xdim != fftGPU.data.xdim) {
//        printf("wrong size: X cpu %lu X gpu %lu\n", fftCPU.data.xdim,
//                fftGPU.data.xdim);
//    }
//    if (fftCPU.data.ydim != fftGPU.data.ydim) {
//        printf("wrong size: Y cpu %lu Y gpu %lu\n", fftCPU.data.xdim,
//                fftGPU.data.xdim);
//    }
//
//    for (size_t i = 0; i < fftCPU.data.yxdim; ++i) {
//        float cpuReal = tmpFFTCpuOut.data[i].real();
//        float cpuImag = tmpFFTCpuOut.data[i].imag();
//        float gpuReal = tmpFFTGpuOut[i].real() / norm;
//        float gpuImag = tmpFFTGpuOut[i].imag() / norm;
//        if ((std::abs(cpuReal - gpuReal) > delta)
//                || (std::abs(cpuImag - gpuImag) > delta)) {
//            printf("ERROR SCALE: %lu cpu (%f, %f) gpu (%f, %f)\n", i, cpuReal,
//                    cpuImag, gpuReal, gpuImag);
//        }
//    }
//    delete[] tmpFFTGpuOut;
//}

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::loadData(const MetaData& movie,
//        const Image<T>& dark, const Image<T>& gain, T targetOccupancy,
//        const MultidimArray<T>& lpf) {
//
//    setDevice(device);
//
//    bool cropInput = (this->yDRcorner != -1);
//    int noOfImgs = this->nlast - this->nfirst + 1;
//
//    // get frame info
//    Image<T> frame;
//    loadFrame(movie, movie.firstObject(), frame);
//    setSizes(frame, noOfImgs);
//    // prepare filter
//    MultidimArray<T> filter;
//    filter.initZeros(croppedOptSizeY, croppedOptSizeFFTX);
//    this->scaleLPF(lpf, croppedOptSizeX, croppedOptSizeY, targetOccupancy,
//            filter);
//
//    // load all frames to RAM
//    // reuse memory
//    frameFourier = (std::complex<T>*)loadToRAM(movie, noOfImgs, dark, gain, cropInput);
//    // scale and transform to FFT on GPU
//    performFFTAndScale((T*)frameFourier, noOfImgs, inputOptSizeX,
//            inputOptSizeY, inputOptBatchSize, croppedOptSizeFFTX,
//            croppedOptSizeY, filter);
//}

//template<typename T>
//void ProgMovieAlignmentCorrelationGPU<T>::computeShifts(size_t N,
//        const Matrix1D<T>& bX, const Matrix1D<T>& bY, const Matrix2D<T>& A) {
//    setDevice(device);
//
//    T* correlations;
//    size_t centerSize = std::ceil(this->maxShift * 2 + 1);
//    computeCorrelations(centerSize, N, frameFourier, croppedOptSizeFFTX,
//            croppedOptSizeX, croppedOptSizeY, correlationBufferImgs,
//            croppedOptBatchSize, correlations);
//
//    // since we are using different size of FFT, we need to scale results to
//    // 'expected' size
//    T localSizeFactorX = this->sizeFactor
//            / (croppedOptSizeX / (T) inputOptSizeX);
//    T localSizeFactorY = this->sizeFactor
//            / (croppedOptSizeY / (T) inputOptSizeY);
//
//    int idx = 0;
//    MultidimArray<T> Mcorr(centerSize, centerSize);
//    for (size_t i = 0; i < N - 1; ++i) {
//        for (size_t j = i + 1; j < N; ++j) {
//            size_t offset = idx * centerSize * centerSize;
//            Mcorr.data = correlations + offset;
//            Mcorr.setXmippOrigin();
//            bestShift(Mcorr, bX(idx), bY(idx), NULL, this->maxShift);
//            bX(idx) *= localSizeFactorX; // scale to expected size
//            bY(idx) *= localSizeFactorY;
//            if (this->verbose)
//                std::cerr << "Frame " << i + this->nfirst << " to Frame "
//                        << j + this->nfirst << " -> ("
//                        << bX(idx) / this->sizeFactor << ","
//                        << bY(idx) / this->sizeFactor << ")" << std::endl;
//            for (int ij = i; ij < j; ij++)
//                A(idx, ij) = 1;
//
//            idx++;
//        }
//    }
//    Mcorr.data = NULL;
//    delete[] frameFourier;
//}

// explicit specialization
template class ProgMovieAlignmentCorrelationGPU<float> ;
