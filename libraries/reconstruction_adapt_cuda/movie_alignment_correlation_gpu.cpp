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
    std::cout << "Device:              " << device << std::endl;
    std::cout << "Benchmark storage    " << (storage.empty() ? "Default" : storage) << std::endl;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::readParams() {
    AProgMovieAlignmentCorrelation<T>::readParams();
    device = this->getIntParam("--device");
    storage = this->getParam("--storage");
}

template<typename T>
core::optional<FFTSettings<T>> ProgMovieAlignmentCorrelationGPU<T>::getStoredMovieSettings(
        Image<T> &frame, int noOfImgs, std::string &uuid) {
    size_t x, y, batch, neededMem;
    bool res = true;
    res = res
            && UserSettings::get(storage).find(*this,
                    getKey(uuid, inputOptSizeXStr, frame, noOfImgs, true), x);
    res = res
            && UserSettings::get(storage).find(*this,
                    getKey(uuid, inputOptSizeYStr, frame, noOfImgs, true), y);
    res = res
            && UserSettings::get(storage).find(*this,
                    getKey(uuid, inputOptBatchSizeStr, frame, noOfImgs, true),
                    batch);
    res = res
            && UserSettings::get(storage).find(*this,
                    getKey(uuid, availableMemoryStr, frame, noOfImgs, true),
                    neededMem);
    res = res && neededMem <= getFreeMem(device);
    if (res) {
        return core::optional<FFTSettings<T>>(
                FFTSettings<T>(x, y, 1, noOfImgs, batch, true));
    } else {
        return core::optional<FFTSettings<T>>();
    }
}

template<typename T>
core::optional<FFTSettings<T>> ProgMovieAlignmentCorrelationGPU<T>::getStoredCropSettings(
        const FFTSettings<T> &hint, const std::string &uuid) {
    size_t x, y, batch, neededMem;
    bool res = true;
    res = res
            && UserSettings::get(storage).find(*this,
                    getKey(uuid, croppedOptSizeXStr, hint.x_spacial, hint.y,
                            hint.n, false), x);
    res = res
            && UserSettings::get(storage).find(*this,
                    getKey(uuid, croppedOptSizeYStr, hint.x_spacial, hint.y,
                            hint.n, false), y);
    res = res
            && UserSettings::get(storage).find(*this,
                    getKey(uuid, croppedOptBatchSizeStr, hint.x_spacial, hint.y,
                            hint.n, false), batch);
    res = res
            && UserSettings::get(storage).find(*this,
                    getKey(uuid, availableMemoryStr, hint.x_spacial, hint.y,
                            hint.n, false), neededMem);
    res = res && neededMem <= getFreeMem(device);
    if (res) {
        return core::optional<FFTSettings<T>>(
                FFTSettings<T>(x, y, 1, (hint.n * hint.n - 1) / 2, batch, true));
    } else {
        return core::optional<FFTSettings<T>>();
    }
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getMovieSettings(
        const MetaData &movie, const GPU &gpu) {
    int noOfImgs = this->nlast - this->nfirst + 1;
    // get frame size
    Image<T> frame;
    loadFrame(movie, movie.firstObject(), frame);
    std::string UUID = getUUID(device);
    auto optSetting = getStoredMovieSettings(frame, noOfImgs, UUID);
    FFTSettings<T> result =
            optSetting ?
                    optSetting.value() :
                    runMovieBenchmark(frame, noOfImgs, UUID);
    if (!optSetting) {
        storeSizes(result, gpu);
    }
    return result;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getCropHint(const FFTSettings<T> &s,
        const std::pair<T, T> &downscale) {
    // we need odd size of the input, to be able to
    // compute FFT more efficiently (and e.g. perform shift by multiplication)
    auto scaleEven = [] (size_t v, T downscale) {
        return (int(v * downscale) / 2) * 2;
    };
    FFTSettings<T> result(scaleEven(s.x_spacial, downscale.first),
            scaleEven(s.y, downscale.second), s.z, s.n, s.batch, s.isInPlace);
    return result;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::getCropSettings(
        const FFTSettings<T> &orig, const GPU &gpu,
        const std::pair<T, T> &downscale) {
    auto hint = getCropHint(orig, downscale);
    auto optSetting = getStoredCropSettings(hint, gpu.UUID);
    FFTSettings<T> result =
            optSetting ?
                    optSetting.value() :
                    runCropBenchmark(hint, gpu);
    if (!optSetting) {
        storeSizes(result, gpu);
    }
    return result;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::runMovieBenchmark(
        const Image<T> &frame,
        int noOfImgs, const std::string &uuid) {
    if (this->verbose) std::cerr
            << "Benchmarking cuFFT (optimal frame size) ..." << std::endl;
    // take additional memory requirement into account
    int maxFilterSize = getMaxFilterSize(frame);
    int x, y, batch;
    getBestFFTSize(noOfImgs, frame.data.xdim, frame.data.ydim, batch, true, x,
            y, maxFilterSize, this->verbose, device,
            frame().xdim == frame().ydim, 10); // allow max 10% change

    return FFTSettings<T>(x, y, 1, noOfImgs, batch, true);
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::storeSizes(const FFTSettings<T> &s,
        const GPU &gpu) {
    UserSettings::get(storage).insert(*this,
            getKey(gpu.UUID, inputOptSizeXStr, s.x_spacial, s.y, s.n, true),
            s.x_spacial);
    UserSettings::get(storage).insert(*this,
            getKey(gpu.UUID, inputOptSizeYStr, s.x_spacial, s.y, s.n, true),
            s.y);
    UserSettings::get(storage).insert(*this,
            getKey(gpu.UUID, inputOptBatchSizeStr, s.x_spacial, s.y, s.n, true),
            s.batch);
    UserSettings::get(storage).insert(*this,
            getKey(gpu.UUID, availableMemoryStr, s.x_spacial, s.y, s.n, true),
            gpu.lastFreeMem);
    UserSettings::get(storage).store(); // write changes
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::runCropBenchmark(
        const FFTSettings<T> &hint, const GPU &gpu) {
    if (this->verbose) std::cerr
            << "Benchmarking cuFFT (optimal cropped size) ..." << std::endl;
    // take additional memory requirement into account
    int x, y, batch;
    size_t noOfCorrelations = (hint.n * (hint.n - 1)) / 2;
    size_t correlationBufferSizeMB = gpu.lastFreeMem / 3; // divide available memory to 3 parts (2 buffers + 1 FFT)
    std::cout << correlationBufferSizeMB << std::endl;
    std::cout << gpu.lastFreeMem << std::endl;
    getBestFFTSize(noOfCorrelations, hint.x_spacial, hint.y, batch, false, x, y,
            correlationBufferSizeMB * 2, this->verbose, gpu.device,
            hint.x_spacial == hint.y, 10); // allow max 10% change

    return FFTSettings<T>(x, y, 1, noOfCorrelations, batch, true);
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::computeGlobalAlignment(
        const MetaData &movie, const Image<T> &dark, const Image<T> &gain) {
    auto gpu = GPU(device);
    auto movieSettings = this->getMovieSettings(movie, gpu);
    T sizeFactor = this->computeSizeFactor();
    auto cropSettings = this->getCropSettings(movieSettings, gpu,
            std::make_pair(sizeFactor, sizeFactor));
    std::cout << movieSettings << std::endl << cropSettings << std::endl;

    MultidimArray<T> filter = this->createLPF(0.9, cropSettings.x_spacial,
            cropSettings.x_freq,
            cropSettings.y);
    Image<T> tmp(filter.xdim, filter.ydim);
    tmp.data.data = filter.data;
    tmp.write("filter_new.vol");
    tmp.data.data = NULL;
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::align(T* data,
        const FFTSettings<T> &in, const FFTSettings<T> &downscale,
        MultidimArray<T> &filter,
        core::optional<size_t> &refFrame,
        size_t maxShift, size_t framesInCorrelationBuffer) {
    assert(nullptr != data);
    size_t N = in.n;
    // scale and transform to FFT on GPU
    performFFTAndScale<T>(data, N, in.x_spacial, in.y, in.batch,
            downscale.x_freq, downscale.y, filter);

    auto scale = std::make_pair(in.x_spacial / (T) downscale.x_spacial,
            in.y / (T) downscale.y);

    return computeShifts(maxShift, (std::complex<T>*) data, downscale, scale,
            framesInCorrelationBuffer, refFrame);
}

template<typename T>
auto ProgMovieAlignmentCorrelationGPU<T>::computeShifts(size_t maxShift,
        std::complex<T>* data, const FFTSettings<T>& settings,
        std::pair<T, T>& scale,
        size_t framesInCorrelationBuffer,
        const core::optional<size_t>& refFrame) {
    // compute correlations (each frame with following ones)
    T* correlations;
    size_t centerSize = std::ceil(maxShift * 2 + 1);
    computeCorrelations(centerSize, settings.n, data, settings.x_freq,
            settings.x_spacial, settings.y, framesInCorrelationBuffer,
            settings.batch, correlations);
    // result is a centered correlation function with (hopefully) a cross
    // indicating the requested shift

    size_t N = settings.n;
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
            if (this->verbose) {
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

    // now get the estimated shift (from the equation system)
    // from each frame to successing frame
    Matrix1D<T> shiftX, shiftY;
    this->solveEquationSystem(bX, bY, A, shiftX, shiftY);

    // prepare result
    AlignmentResult < T > result;
    result.refFrame =
            refFrame ?
                    refFrame.value() :
                    this->findReferenceImage(N, shiftX, shiftY);
    result.shifts.reserve(settings.n);

    // compute total shift in respect to reference frame
    for (size_t i = 0; i < settings.n; ++i) {
        T x, y;
        this->computeTotalShift(result.refFrame, i, shiftX, shiftY, x, y);
        result.shifts.emplace_back(x, y);
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::applyShiftsComputeAverage(
        const MetaData& movie, const Image<T>& dark, const Image<T>& gain,
        Image<T>& initialMic, size_t& Ninitial, Image<T>& averageMicrograph,
        size_t& N) {
    // Apply shifts and compute average
    Image<T> frame, croppedFrame, reducedFrame, shiftedFrame;
    Matrix1D<T> shift(2);
    FileName fnFrame;
    int j = 0;
    int n = 0;
    Ninitial = N = 0;
    GeoShiftTransformer<T> transformer;
    FOR_ALL_OBJECTS_IN_METADATA(movie)
    {
        if (n >= this->nfirstSum && n <= this->nlastSum) {
            movie.getValue(MDL_IMAGE, fnFrame, __iter.objId);
            movie.getValue(MDL_SHIFT_X, XX(shift), __iter.objId);
            movie.getValue(MDL_SHIFT_Y, YY(shift), __iter.objId);

            std::cout << fnFrame << " shiftX=" << XX(shift) << " shiftY="
                    << YY(shift) << std::endl;
            frame.read(fnFrame);
            if (XSIZE(dark()) > 0)
                frame() -= dark();
            if (XSIZE(gain()) > 0)
                frame() *= gain();
            if (this->yDRcorner != -1)
                frame().window(croppedFrame(), this->yLTcorner, this->xLTcorner,
                        this->yDRcorner, this->xDRcorner);
            else
                croppedFrame() = frame();
            if (this->bin > 0) {
                // FIXME add templates to respective functions/classes to avoid type casting
                Image<double> croppedFrameDouble;
                Image<double> reducedFrameDouble;
                typeCast(croppedFrame(), croppedFrameDouble());

                scaleToSizeFourier(1, floor(YSIZE(croppedFrame()) / this->bin),
                        floor(XSIZE(croppedFrame()) / this->bin),
                        croppedFrameDouble(), reducedFrameDouble());

                typeCast(reducedFrameDouble(), reducedFrame());

                shift /= this->bin;
                croppedFrame() = reducedFrame();
            }

            if (this->fnInitialAvg != "") {
                if (j == 0)
                    initialMic() = croppedFrame();
                else
                    initialMic() += croppedFrame();
                Ninitial++;
            }

            if (this->fnAligned != "" || this->fnAvg != "") {
                if (this->outsideMode == OUTSIDE_WRAP) {
//                    Matrix2D<T> tmp;
//                    translation2DMatrix(shift, tmp, true);
                    transformer.initLazy(croppedFrame().xdim,
                            croppedFrame().ydim, 1, device);
                    transformer.applyShift(shiftedFrame(), croppedFrame(), XX(shift), YY(shift));
//                    transformer.applyGeometry(this->BsplineOrder,
//                            shiftedFrame(), croppedFrame(), tmp, IS_INV, WRAP);
                } else if (this->outsideMode == OUTSIDE_VALUE)
                    translate(this->BsplineOrder, shiftedFrame(),
                            croppedFrame(), shift, DONT_WRAP,
                            this->outsideValue);
                else
                    translate(this->BsplineOrder, shiftedFrame(),
                            croppedFrame(), shift, DONT_WRAP,
                            (T) croppedFrame().computeAvg());
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
            j++;
        }
        n++;
    }
}

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

template<typename T>
T* ProgMovieAlignmentCorrelationGPU<T>::loadToRAM(const MetaData& movie,
        int noOfImgs, const Image<T>& dark, const Image<T>& gain,
        bool cropInput) {
    // allocate enough memory for the images. Since it will be reused, it has to be big
    // enough to store either all FFTs or all input images
    T* imgs = new T[noOfImgs * inputOptSizeY
            * std::max(inputOptSizeX, inputOptSizeFFTX * 2)]();
    Image<T> frame;

    int movieImgIndex = -1;
    FOR_ALL_OBJECTS_IN_METADATA(movie)
    {
        // update variables
        movieImgIndex++;
        if (movieImgIndex < this->nfirst)
            continue;
        if (movieImgIndex > this->nlast)
            break;

        // load image
        loadFrame(movie, __iter.objId, frame);
        if (XSIZE(dark()) > 0)
            frame() -= dark();
        if (XSIZE(gain()) > 0)
            frame() *= gain();

        // copy line by line, adding offset at the end of each line
        // result is the same image, padded in the X and Y dimensions
        T* dest = imgs
                + ((movieImgIndex - this->nfirst) * inputOptSizeX
                        * inputOptSizeY); // points to first float in the image
        for (size_t i = 0; i < inputOptSizeY; ++i) {
            memcpy(dest + (inputOptSizeX * i),
                    frame.data.data + i * frame.data.xdim,
                    inputOptSizeX * sizeof(T));
        }
    }
    return imgs;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::setSizes(Image<T> &frame,
        int noOfImgs) {

    std::string UUID = getUUID(device);

    int maxFilterSize = getMaxFilterSize(frame);
    size_t availableMemMB = getFreeMem(device);
    correlationBufferSizeMB = availableMemMB / 3; // divide available memory to 3 parts (2 buffers + 1 FFT)

    if (! getStoredSizes(frame, noOfImgs, UUID)) {
        runBenchmark(frame, noOfImgs, UUID);
        storeSizes(frame, noOfImgs, UUID);
    }

    T corrSizeMB = ((size_t) croppedOptSizeFFTX * croppedOptSizeY
            * sizeof(std::complex<T>)) / (1024 * 1024.);
    correlationBufferImgs = std::ceil(correlationBufferSizeMB / corrSizeMB);
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::runBenchmark(Image<T> &frame,
        int noOfImgs, std::string &uuid) {
    // get best sizes
    int maxFilterSize = getMaxFilterSize(frame);
    if (this->verbose)
        std::cerr << "Benchmarking cuFFT ..." << std::endl;

    size_t noOfCorrelations = (noOfImgs * (noOfImgs - 1)) / 2;

    // we also need enough memory for filter
    getBestFFTSize(noOfImgs, frame.data.xdim, frame.data.ydim, inputOptBatchSize,
            true,
            inputOptSizeX, inputOptSizeY, maxFilterSize, this->verbose, device,
            frame().xdim == frame().ydim, 10); // allow max 10% change

    inputOptSizeFFTX = inputOptSizeX / 2 + 1;

    getBestFFTSize(noOfCorrelations, this->newXdim, this->newYdim,
            croppedOptBatchSize, false, croppedOptSizeX, croppedOptSizeY,
            correlationBufferSizeMB * 2, this->verbose, device,
            this->newXdim == this->newYdim, 10);

    croppedOptSizeFFTX = croppedOptSizeX / 2 + 1;
}


template<typename T>
bool ProgMovieAlignmentCorrelationGPU<T>::getStoredSizes(Image<T> &frame,
        int noOfImgs, std::string &uuid) {
    bool res = true;
    size_t neededMem;
    res = res && UserSettings::get(storage).find(*this,
        getKey(uuid, inputOptSizeXStr, frame, noOfImgs, true), inputOptSizeX);
    res = res && UserSettings::get(storage).find(*this,
        getKey(uuid, inputOptSizeYStr, frame, noOfImgs, true), inputOptSizeY);
    res = res && UserSettings::get(storage).find(*this,
        getKey(uuid, inputOptBatchSizeStr, frame, noOfImgs, true), inputOptBatchSize);
    inputOptSizeFFTX =  inputOptSizeX / 2 + 1;
    res = res && UserSettings::get(storage).find(*this,
        getKey(uuid, availableMemoryStr, frame, noOfImgs, true), neededMem);
    res = res && neededMem <= getFreeMem(device);

    res = res && UserSettings::get(storage).find(*this,
        getKey(uuid, croppedOptSizeXStr, this->newXdim, this->newYdim, noOfImgs, false), croppedOptSizeX);
    res = res && UserSettings::get(storage).find(*this,
        getKey(uuid, croppedOptSizeYStr, this->newXdim, this->newYdim, noOfImgs, false), croppedOptSizeY);
    res = res && UserSettings::get(storage).find(*this,
        getKey(uuid, croppedOptBatchSizeStr, this->newXdim, this->newYdim, noOfImgs, false),
        croppedOptBatchSize);
    croppedOptSizeFFTX =  croppedOptSizeX / 2 + 1;
    res = res && UserSettings::get(storage).find(*this,
        getKey(uuid, availableMemoryStr, this->newXdim, this->newYdim, noOfImgs, false), neededMem);
    res = res && neededMem <= getFreeMem(device);

    return res;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::storeSizes(Image<T> &frame,
        int noOfImgs, std::string &uuid) {
    UserSettings::get(storage).insert(*this,
        getKey(uuid, inputOptSizeXStr, frame, noOfImgs, true), inputOptSizeX);
    UserSettings::get(storage).insert(*this,
        getKey(uuid, inputOptSizeYStr, frame, noOfImgs, true), inputOptSizeY);
    UserSettings::get(storage).insert(*this,
        getKey(uuid, inputOptBatchSizeStr, frame, noOfImgs, true),
        inputOptBatchSize);
    UserSettings::get(storage).insert(*this,
        getKey(uuid, availableMemoryStr, frame, noOfImgs, true), getFreeMem(device));

    UserSettings::get(storage).insert(*this,
        getKey(uuid, croppedOptSizeXStr, this->newXdim, this->newYdim, noOfImgs, false),
        croppedOptSizeX);
    UserSettings::get(storage).insert(*this,
        getKey(uuid, croppedOptSizeYStr, this->newXdim, this->newYdim, noOfImgs, false),
        croppedOptSizeY);
    UserSettings::get(storage).insert(*this,
        getKey(uuid, croppedOptBatchSizeStr,
        this->newXdim, this->newYdim, noOfImgs, false),
        croppedOptBatchSize);
    UserSettings::get(storage).insert(*this,
        getKey(uuid, availableMemoryStr, this->newXdim, this->newYdim, noOfImgs, false),
        getFreeMem(device));
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testFFT() {

    double delta = 0.00001;
    size_t x, y;
    x = y = 2304;
    size_t order = 10000;

    srand(42);

    Image<double> inputDouble(x, y); // keep sync with values
    Image<float> inputFloat(x, y); // keep sync with values
    size_t pixels = inputDouble.data.xdim * inputDouble.data.ydim;
    for (size_t y = 0; y < inputDouble.data.ydim; ++y) {
        for (size_t x = 0; x < inputDouble.data.xdim; ++x) {
            size_t index = y * inputDouble.data.xdim + x;
            double value = rand() / (RAND_MAX / 2000.);
            inputDouble.data.data[index] = value;
            inputFloat.data.data[index] = (float) value;
        }
    }

    // CPU part

    MultidimArray<std::complex<double> > tmpFFTCpu;
    FourierTransformer transformer;

    transformer.FourierTransform(inputDouble(), tmpFFTCpu, true);

    // store results to drive
    Image<double> fftCPU(tmpFFTCpu.xdim, tmpFFTCpu.ydim);
    size_t fftPixels = fftCPU.data.yxdim;
    for (size_t i = 0; i < fftPixels; i++) {
        fftCPU.data.data[i] = tmpFFTCpu.data[i].real();
    }
    fftCPU.write("testFFTCpu.vol");

    // GPU part

    GpuMultidimArrayAtGpu<float> gpuIn(inputFloat.data.xdim,
            inputFloat.data.ydim);
    gpuIn.copyToGpu(inputFloat.data.data);
    GpuMultidimArrayAtGpu<std::complex<float> > gpuFFT;
    mycufftHandle handle;
    gpuIn.fft(gpuFFT, handle);

    fftPixels = gpuFFT.yxdim;
    std::complex<float>* tmpFFTGpu = new std::complex<float>[fftPixels];
    gpuFFT.copyToCpu(tmpFFTGpu);

    // store results to drive
    Image<float> fftGPU(gpuFFT.Xdim, gpuFFT.Ydim);
    float norm = inputFloat.data.yxdim;
    for (size_t i = 0; i < fftPixels; i++) {
        fftGPU.data.data[i] = tmpFFTGpu[i].real() / norm;
    }
    fftGPU.write("testFFTGpu.vol");

    ////////////////////////////////////////

    if (fftCPU.data.xdim != fftGPU.data.xdim) {
        printf("wrong size: X cpu %lu X gpu %lu\n", fftCPU.data.xdim,
                fftGPU.data.xdim);
    }
    if (fftCPU.data.ydim != fftGPU.data.ydim) {
        printf("wrong size: Y cpu %lu Y gpu %lu\n", fftCPU.data.xdim,
                fftGPU.data.xdim);
    }

    for (size_t i = 0; i < fftCPU.data.yxdim; ++i) {
        float cpuReal = tmpFFTCpu.data[i].real();
        float cpuImag = tmpFFTCpu.data[i].imag();
        float gpuReal = tmpFFTGpu[i].real() / norm;
        float gpuImag = tmpFFTGpu[i].imag() / norm;
        if ((std::abs(cpuReal - gpuReal) > delta)
                || (std::abs(cpuImag - gpuImag) > delta)) {
            printf("ERROR FFT: %lu cpu (%f, %f) gpu (%f, %f)\n", i, cpuReal,
                    cpuImag, gpuReal, gpuImag);
        }
    }

    delete[] tmpFFTGpu;

}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testFilterAndScale() {
    double delta = 0.00001;
    size_t xIn, yIn, xOut, yOut, xOutFFT;
    xIn = yIn = 4096;
    xOut = yOut = 2275;
    xOutFFT = xOut / 2 + 1;

    size_t fftPixels = xOutFFT * yOut;
    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[fftPixels];
    float* filter = new float[fftPixels];
    for (size_t i = 0; i < fftPixels; ++i) {
        filter[i] = (rand() * 100) / (float) RAND_MAX;
    }

    srand(42);

    Image<double> inputDouble(xIn, yIn); // keep sync with values
    Image<float> inputFloat(xIn, yIn); // keep sync with values
    size_t pixels = inputDouble.data.xdim * inputDouble.data.ydim;
    for (size_t y = 0; y < inputDouble.data.ydim; ++y) {
        for (size_t x = 0; x < inputDouble.data.xdim; ++x) {
            size_t index = y * inputDouble.data.xdim + x;
            double value = rand() > (RAND_MAX / 2) ? -1 : 1; // ((int)(1000 * (double)rand() / (RAND_MAX))) / 1000.f;
            inputDouble.data.data[index] = value;
            inputFloat.data.data[index] = (float) value;
        }
    }
//	inputDouble(0,0) = 1;
//	inputFloat(0,0) = 1;
    Image<double> outputDouble(xOut, yOut);
    Image<double> reducedFrame;

    // CPU part

    scaleToSizeFourier(1, yOut, xOut, inputDouble(), reducedFrame());
//	inputDouble().printStats();
//	printf("\n");
//	reducedFrame().printStats();
//	printf("\n");
    // Now do the Fourier transform and filter
    MultidimArray<std::complex<double> > *tmpFFTCpuOut = new MultidimArray<
            std::complex<double> >;
    MultidimArray<std::complex<double> > *tmpFFTCpuOutFull = new MultidimArray<
            std::complex<double> >;
    FourierTransformer transformer;

    transformer.FourierTransform(inputDouble(), *tmpFFTCpuOutFull);
//	std::cout << *tmpFFTCpuOutFull<< std::endl;

    transformer.FourierTransform(reducedFrame(), *tmpFFTCpuOut, true);
    for (size_t nn = 0; nn < fftPixels; ++nn) {
        double wlpf = filter[nn];
        DIRECT_MULTIDIM_ELEM(*tmpFFTCpuOut,nn) *= wlpf;
    }

    // store results to drive
    Image<double> fftCPU(tmpFFTCpuOut->xdim, tmpFFTCpuOut->ydim);
    fftPixels = tmpFFTCpuOut->yxdim;
    for (size_t i = 0; i < fftPixels; i++) {
        fftCPU.data.data[i] = tmpFFTCpuOut->data[i].real();
        if (fftCPU.data.data[i] > 10)
            fftCPU.data.data[i] = 0;
    }
    fftCPU.write("testFFTCpuScaledFiltered.vol");

    // GPU part

    float* d_filter = loadToGPU(filter, fftPixels);

    GpuMultidimArrayAtGpu<float> gpuIn(inputFloat.data.xdim,
            inputFloat.data.ydim);
    gpuIn.copyToGpu(inputFloat.data.data);
    GpuMultidimArrayAtGpu<std::complex<float> > gpuFFT;
    mycufftHandle handle;

//    processInput(gpuIn, gpuFFT, handle, xIn, yIn, 1, xOutFFT, yOut, d_filter,
//            tmpFFTGpuOut); // FIXME test

    // store results to drive
    Image<float> fftGPU(xOutFFT, yOut);
    float norm = inputFloat.data.yxdim;
    for (size_t i = 0; i < fftPixels; i++) {
        fftGPU.data.data[i] = tmpFFTGpuOut[i].real() / norm;
        if (fftGPU.data.data[i] > 10)
            fftGPU.data.data[i] = 0;
    }
    fftGPU.write("testFFTGpuScaledFiltered.vol");

    ////////////////////////////////////////

    if (fftCPU.data.xdim != fftGPU.data.xdim) {
        printf("wrong size: X cpu %lu X gpu %lu\n", fftCPU.data.xdim,
                fftGPU.data.xdim);
    }
    if (fftCPU.data.ydim != fftGPU.data.ydim) {
        printf("wrong size: Y cpu %lu Y gpu %lu\n", fftCPU.data.xdim,
                fftGPU.data.xdim);
    }
    if (tmpFFTCpuOut->ydim != yOut) {
        printf("wrong size tmpFFTCpuOut: Y cpu %lu Y gpu %lu\n",
                tmpFFTCpuOut->ydim, yOut);
    }
    if (tmpFFTCpuOut->xdim != xOutFFT) {
        printf("wrong size tmpFFTCpuOut: X cpu %lu X gpu %lu\n",
                tmpFFTCpuOut->xdim, xOutFFT);
    }

    for (size_t i = 0; i < fftCPU.data.yxdim; ++i) {
        float cpuReal = tmpFFTCpuOut->data[i].real();
        float cpuImag = tmpFFTCpuOut->data[i].imag();
        float gpuReal = tmpFFTGpuOut[i].real() / norm;
        float gpuImag = tmpFFTGpuOut[i].imag() / norm;
        if ((std::abs(cpuReal - gpuReal) > delta)
                || (std::abs(cpuImag - gpuImag) > delta)) {
            printf("ERROR FILTER: %lu cpu (%f, %f) gpu (%f, %f)\n", i, cpuReal,
                    cpuImag, gpuReal, gpuImag);
        }
    }
    delete[] tmpFFTGpuOut;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testScalingGpuOO() {
    double delta = 0.000001;
    size_t xIn, yIn, xOut, yOut, xOutFFT, xInFFT;
    xIn = yIn = 9;
    xOut = yOut = 5;
    xOutFFT = xOut / 2 + 1; // == 3
    xInFFT = xIn / 2 + 1; // == 5

    std::complex<float>* tmpFFTGpuIn = new std::complex<float>[yIn * xInFFT];
    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[yOut * xOutFFT];
    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
    for (size_t y = 0; y < yIn; ++y) {
        for (size_t x = 0; x < xInFFT; ++x) {
            size_t index = y * xInFFT + x;
            tmpFFTGpuIn[index] = std::complex<float>(y, x);
        }
    }

    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);

    tmpFFTCpuOutExpected[3] = std::complex<double>(1, 0);
    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 1);
    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 2);

    tmpFFTCpuOutExpected[6] = std::complex<double>(2, 0);
    tmpFFTCpuOutExpected[7] = std::complex<double>(2, 1);
    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 2);

    tmpFFTCpuOutExpected[9] = std::complex<double>(7, 0);
    tmpFFTCpuOutExpected[10] = std::complex<double>(7, 1);
    tmpFFTCpuOutExpected[11] = std::complex<double>(7, 2);

    tmpFFTCpuOutExpected[12] = std::complex<double>(8, 0);
    tmpFFTCpuOutExpected[13] = std::complex<double>(8, 1);
    tmpFFTCpuOutExpected[14] = std::complex<double>(8, 2);

//    applyFilterAndCrop<float>(tmpFFTGpuIn, tmpFFTGpuOut, 1, xInFFT, yIn,
//            xOutFFT, yOut, NULL); // FIXME test

    ////////////////////////////////////////

    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
        float cpuReal = tmpFFTGpuOut[i].real();
        float cpuImag = tmpFFTGpuOut[i].imag();
        float expReal = tmpFFTCpuOutExpected[i].real();
        float expImag = tmpFFTCpuOutExpected[i].imag();
        if ((std::abs(cpuReal - expReal) > delta)
                || (std::abs(cpuImag - expImag) > delta)) {
            printf("ERROR SCALE GPU OO: %lu gpu (%f, %f) exp (%f, %f)\n", i,
                    cpuReal, cpuImag, expReal, expImag);
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testScalingGpuEO() {
    double delta = 0.000001;
    size_t xIn, yIn, xOut, yOut, xOutFFT, xInFFT;
    xIn = yIn = 10;
    xOut = yOut = 5;
    xOutFFT = xOut / 2 + 1; // == 3
    xInFFT = xIn / 2 + 1; // == 6

    std::complex<float>* tmpFFTGpuIn = new std::complex<float>[yIn * xInFFT];
    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[yOut * xOutFFT];
    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
    for (size_t y = 0; y < yIn; ++y) {
        for (size_t x = 0; x < xInFFT; ++x) {
            size_t index = y * xInFFT + x;
            tmpFFTGpuIn[index] = std::complex<float>(y, x);
        }
    }

    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);

    tmpFFTCpuOutExpected[3] = std::complex<double>(1, 0);
    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 1);
    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 2);

    tmpFFTCpuOutExpected[6] = std::complex<double>(2, 0);
    tmpFFTCpuOutExpected[7] = std::complex<double>(2, 1);
    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 2);

    tmpFFTCpuOutExpected[9] = std::complex<double>(8, 0);
    tmpFFTCpuOutExpected[10] = std::complex<double>(8, 1);
    tmpFFTCpuOutExpected[11] = std::complex<double>(8, 2);

    tmpFFTCpuOutExpected[12] = std::complex<double>(9, 0);
    tmpFFTCpuOutExpected[13] = std::complex<double>(9, 1);
    tmpFFTCpuOutExpected[14] = std::complex<double>(9, 2);

//    applyFilterAndCrop<float>(tmpFFTGpuIn, tmpFFTGpuOut, 1, xInFFT, yIn,
//            xOutFFT, yOut, NULL); // FIXME test

    ////////////////////////////////////////

    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
        float cpuReal = tmpFFTGpuOut[i].real();
        float cpuImag = tmpFFTGpuOut[i].imag();
        float expReal = tmpFFTCpuOutExpected[i].real();
        float expImag = tmpFFTCpuOutExpected[i].imag();
        if ((std::abs(cpuReal - expReal) > delta)
                || (std::abs(cpuImag - expImag) > delta)) {
            printf("ERROR SCALE GPU EO: %lu gpu (%f, %f) exp (%f, %f)\n", i,
                    cpuReal, cpuImag, expReal, expImag);
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testScalingGpuOE() {
    double delta = 0.000001;
    size_t xIn, yIn, xOut, yOut, xOutFFT, xInFFT;
    xIn = yIn = 9;
    xOut = yOut = 6;
    xOutFFT = xOut / 2 + 1; // == 4
    xInFFT = xIn / 2 + 1; // == 5

    std::complex<float>* tmpFFTGpuIn = new std::complex<float>[yIn * xInFFT];
    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[yOut * xOutFFT];
    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
    for (size_t y = 0; y < yIn; ++y) {
        for (size_t x = 0; x < xInFFT; ++x) {
            size_t index = y * xInFFT + x;
            tmpFFTGpuIn[index] = std::complex<float>(y, x);
        }
    }
    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);
    tmpFFTCpuOutExpected[3] = std::complex<double>(0, 3);

    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 0);
    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 1);
    tmpFFTCpuOutExpected[6] = std::complex<double>(1, 2);
    tmpFFTCpuOutExpected[7] = std::complex<double>(1, 3);

    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 0);
    tmpFFTCpuOutExpected[9] = std::complex<double>(2, 1);
    tmpFFTCpuOutExpected[10] = std::complex<double>(2, 2);
    tmpFFTCpuOutExpected[11] = std::complex<double>(2, 3);

    tmpFFTCpuOutExpected[12] = std::complex<double>(3, 0);
    tmpFFTCpuOutExpected[13] = std::complex<double>(3, 1);
    tmpFFTCpuOutExpected[14] = std::complex<double>(3, 2);
    tmpFFTCpuOutExpected[15] = std::complex<double>(3, 3);

    tmpFFTCpuOutExpected[16] = std::complex<double>(7, 0);
    tmpFFTCpuOutExpected[17] = std::complex<double>(7, 1);
    tmpFFTCpuOutExpected[18] = std::complex<double>(7, 2);
    tmpFFTCpuOutExpected[19] = std::complex<double>(7, 3);

    tmpFFTCpuOutExpected[20] = std::complex<double>(8, 0);
    tmpFFTCpuOutExpected[21] = std::complex<double>(8, 1);
    tmpFFTCpuOutExpected[22] = std::complex<double>(8, 2);
    tmpFFTCpuOutExpected[23] = std::complex<double>(8, 3);

//    applyFilterAndCrop<float>(tmpFFTGpuIn, tmpFFTGpuOut, 1, xInFFT, yIn,
//            xOutFFT, yOut, NULL); // FIXME test

    ////////////////////////////////////////

    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
        float cpuReal = tmpFFTGpuOut[i].real();
        float cpuImag = tmpFFTGpuOut[i].imag();
        float expReal = tmpFFTCpuOutExpected[i].real();
        float expImag = tmpFFTCpuOutExpected[i].imag();
        if ((std::abs(cpuReal - expReal) > delta)
                || (std::abs(cpuImag - expImag) > delta)) {
            printf("ERROR SCALE GPU OE: %lu gpu (%f, %f) exp (%f, %f)\n", i,
                    cpuReal, cpuImag, expReal, expImag);
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testScalingGpuEE() {
    double delta = 0.000001;
    size_t xIn, yIn, xOut, yOut, xOutFFT, xInFFT;
    xIn = yIn = 10;
    xOut = yOut = 6;
    xOutFFT = xOut / 2 + 1; // == 4
    xInFFT = xIn / 2 + 1; // == 6

    std::complex<float>* tmpFFTGpuIn = new std::complex<float>[yIn * xInFFT];
    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[yOut * xOutFFT];
    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
    for (size_t y = 0; y < yIn; ++y) {
        for (size_t x = 0; x < xInFFT; ++x) {
            size_t index = y * xInFFT + x;
            tmpFFTGpuIn[index] = std::complex<float>(y, x);
        }
    }

    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);
    tmpFFTCpuOutExpected[3] = std::complex<double>(0, 3);

    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 0);
    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 1);
    tmpFFTCpuOutExpected[6] = std::complex<double>(1, 2);
    tmpFFTCpuOutExpected[7] = std::complex<double>(1, 3);

    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 0);
    tmpFFTCpuOutExpected[9] = std::complex<double>(2, 1);
    tmpFFTCpuOutExpected[10] = std::complex<double>(2, 2);
    tmpFFTCpuOutExpected[11] = std::complex<double>(2, 3);

    tmpFFTCpuOutExpected[12] = std::complex<double>(3, 0);
    tmpFFTCpuOutExpected[13] = std::complex<double>(3, 1);
    tmpFFTCpuOutExpected[14] = std::complex<double>(3, 2);
    tmpFFTCpuOutExpected[15] = std::complex<double>(3, 3);

    tmpFFTCpuOutExpected[16] = std::complex<double>(8, 0);
    tmpFFTCpuOutExpected[17] = std::complex<double>(8, 1);
    tmpFFTCpuOutExpected[18] = std::complex<double>(8, 2);
    tmpFFTCpuOutExpected[19] = std::complex<double>(8, 3);

    tmpFFTCpuOutExpected[20] = std::complex<double>(9, 0);
    tmpFFTCpuOutExpected[21] = std::complex<double>(9, 1);
    tmpFFTCpuOutExpected[22] = std::complex<double>(9, 2);
    tmpFFTCpuOutExpected[23] = std::complex<double>(9, 3);

//    applyFilterAndCrop<float>(tmpFFTGpuIn, tmpFFTGpuOut, 1, xInFFT, yIn,
//            xOutFFT, yOut, NULL); // FIXME test

    ////////////////////////////////////////

    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
        float cpuReal = tmpFFTGpuOut[i].real();
        float cpuImag = tmpFFTGpuOut[i].imag();
        float expReal = tmpFFTCpuOutExpected[i].real();
        float expImag = tmpFFTCpuOutExpected[i].imag();
        if ((std::abs(cpuReal - expReal) > delta)
                || (std::abs(cpuImag - expImag) > delta)) {
            printf("ERROR SCALE GPU EE: %lu gpu (%f, %f) exp (%f, %f)\n", i,
                    cpuReal, cpuImag, expReal, expImag);
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testScalingCpuOO() {
    double delta = 0.000001;
    size_t xIn, yIn, xOut, yOut, xOutFFT, xInFFT;
    xIn = yIn = 9;
    xOut = yOut = 5;
    xOutFFT = xOut / 2 + 1; // == 3

    Image<double> inputDouble(xIn, yIn);
    Image<double> outputDouble(xOut, yOut);
    MultidimArray<std::complex<double> > tmpFFTCpuIn(yIn, xIn / 2 + 1);
    MultidimArray<std::complex<double> > tmpFFTCpuOut(yOut, xOutFFT);
    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
    for (size_t y = 0; y < tmpFFTCpuIn.ydim; ++y) {
        for (size_t x = 0; x < tmpFFTCpuIn.xdim; ++x) {
            size_t index = y * tmpFFTCpuIn.xdim + x;
            tmpFFTCpuIn.data[index] = std::complex<double>(y, x);
        }
    }

    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);

    tmpFFTCpuOutExpected[3] = std::complex<double>(1, 0);
    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 1);
    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 2);

    tmpFFTCpuOutExpected[6] = std::complex<double>(2, 0);
    tmpFFTCpuOutExpected[7] = std::complex<double>(2, 1);
    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 2);

    tmpFFTCpuOutExpected[9] = std::complex<double>(7, 0);
    tmpFFTCpuOutExpected[10] = std::complex<double>(7, 1);
    tmpFFTCpuOutExpected[11] = std::complex<double>(7, 2);

    tmpFFTCpuOutExpected[12] = std::complex<double>(8, 0);
    tmpFFTCpuOutExpected[13] = std::complex<double>(8, 1);
    tmpFFTCpuOutExpected[14] = std::complex<double>(8, 2);

    scaleToSizeFourier(inputDouble(), outputDouble(), tmpFFTCpuIn,
            tmpFFTCpuOut);

    ////////////////////////////////////////

    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
        float cpuReal = tmpFFTCpuOut.data[i].real();
        float cpuImag = tmpFFTCpuOut.data[i].imag();
        float expReal = tmpFFTCpuOutExpected[i].real();
        float expImag = tmpFFTCpuOutExpected[i].imag();
        if ((std::abs(cpuReal - expReal) > delta)
                || (std::abs(cpuImag - expImag) > delta)) {
            printf("ERROR SCALE CPU OO: %lu cpu (%f, %f) exp (%f, %f)\n", i,
                    cpuReal, cpuImag, expReal, expImag);
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testScalingCpuEO() {
    double delta = 0.000001;
    size_t xIn, yIn, xOut, yOut, xOutFFT;
    xIn = yIn = 10;
    xOut = yOut = 5;
    xOutFFT = xOut / 2 + 1; // == 3

    Image<double> inputDouble(xIn, yIn);
    Image<double> outputDouble(xOut, yOut);
    MultidimArray<std::complex<double> > tmpFFTCpuIn(yIn, xIn / 2 + 1);
    MultidimArray<std::complex<double> > tmpFFTCpuOut(yOut, xOutFFT);
    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
    for (size_t y = 0; y < tmpFFTCpuIn.ydim; ++y) {
        for (size_t x = 0; x < tmpFFTCpuIn.xdim; ++x) {
            size_t index = y * tmpFFTCpuIn.xdim + x;
            tmpFFTCpuIn.data[index] = std::complex<double>(y, x);
        }
    }

    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);

    tmpFFTCpuOutExpected[3] = std::complex<double>(1, 0);
    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 1);
    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 2);

    tmpFFTCpuOutExpected[6] = std::complex<double>(2, 0);
    tmpFFTCpuOutExpected[7] = std::complex<double>(2, 1);
    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 2);

    tmpFFTCpuOutExpected[9] = std::complex<double>(8, 0);
    tmpFFTCpuOutExpected[10] = std::complex<double>(8, 1);
    tmpFFTCpuOutExpected[11] = std::complex<double>(8, 2);

    tmpFFTCpuOutExpected[12] = std::complex<double>(9, 0);
    tmpFFTCpuOutExpected[13] = std::complex<double>(9, 1);
    tmpFFTCpuOutExpected[14] = std::complex<double>(9, 2);

    scaleToSizeFourier(inputDouble(), outputDouble(), tmpFFTCpuIn,
            tmpFFTCpuOut);

    ////////////////////////////////////////

    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
        float cpuReal = tmpFFTCpuOut.data[i].real();
        float cpuImag = tmpFFTCpuOut.data[i].imag();
        float expReal = tmpFFTCpuOutExpected[i].real();
        float expImag = tmpFFTCpuOutExpected[i].imag();
        if ((std::abs(cpuReal - expReal) > delta)
                || (std::abs(cpuImag - expImag) > delta)) {
            printf("ERROR SCALE CPU EO: %lu cpu (%f, %f) exp (%f, %f)\n", i,
                    cpuReal, cpuImag, expReal, expImag);
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testScalingCpuOE() {
    double delta = 0.000001;
    size_t xIn, yIn, xOut, yOut, xOutFFT;
    xIn = yIn = 9;
    xOut = yOut = 6;
    xOutFFT = xOut / 2 + 1; // == 4

    Image<double> inputDouble(xIn, yIn);
    Image<double> outputDouble(xOut, yOut);
    MultidimArray<std::complex<double> > tmpFFTCpuIn(yIn, xIn / 2 + 1);
    MultidimArray<std::complex<double> > tmpFFTCpuOut(yOut, xOutFFT);
    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
    for (size_t y = 0; y < tmpFFTCpuIn.ydim; ++y) {
        for (size_t x = 0; x < tmpFFTCpuIn.xdim; ++x) {
            size_t index = y * tmpFFTCpuIn.xdim + x;
            tmpFFTCpuIn.data[index] = std::complex<double>(y, x);
        }
    }

    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);
    tmpFFTCpuOutExpected[3] = std::complex<double>(0, 3);

    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 0);
    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 1);
    tmpFFTCpuOutExpected[6] = std::complex<double>(1, 2);
    tmpFFTCpuOutExpected[7] = std::complex<double>(1, 3);

    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 0);
    tmpFFTCpuOutExpected[9] = std::complex<double>(2, 1);
    tmpFFTCpuOutExpected[10] = std::complex<double>(2, 2);
    tmpFFTCpuOutExpected[11] = std::complex<double>(2, 3);

    tmpFFTCpuOutExpected[12] = std::complex<double>(3, 0);
    tmpFFTCpuOutExpected[13] = std::complex<double>(3, 1);
    tmpFFTCpuOutExpected[14] = std::complex<double>(3, 2);
    tmpFFTCpuOutExpected[15] = std::complex<double>(3, 3);

    tmpFFTCpuOutExpected[16] = std::complex<double>(7, 0);
    tmpFFTCpuOutExpected[17] = std::complex<double>(7, 1);
    tmpFFTCpuOutExpected[18] = std::complex<double>(7, 2);
    tmpFFTCpuOutExpected[19] = std::complex<double>(7, 3);

    tmpFFTCpuOutExpected[20] = std::complex<double>(8, 0);
    tmpFFTCpuOutExpected[21] = std::complex<double>(8, 1);
    tmpFFTCpuOutExpected[22] = std::complex<double>(8, 2);
    tmpFFTCpuOutExpected[23] = std::complex<double>(8, 3);

    scaleToSizeFourier(inputDouble(), outputDouble(), tmpFFTCpuIn,
            tmpFFTCpuOut);

    ////////////////////////////////////////

    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
        float cpuReal = tmpFFTCpuOut.data[i].real();
        float cpuImag = tmpFFTCpuOut.data[i].imag();
        float expReal = tmpFFTCpuOutExpected[i].real();
        float expImag = tmpFFTCpuOutExpected[i].imag();
        if ((std::abs(cpuReal - expReal) > delta)
                || (std::abs(cpuImag - expImag) > delta)) {
            printf("ERROR SCALE CPU OE: %lu cpu (%f, %f) exp (%f, %f)\n", i,
                    cpuReal, cpuImag, expReal, expImag);
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testScalingCpuEE() {
    double delta = 0.000001;
    size_t xIn, yIn, xOut, yOut, xOutFFT;
    xIn = yIn = 10;
    xOut = yOut = 6;
    xOutFFT = xOut / 2 + 1; // == 4

    Image<double> inputDouble(xIn, yIn);
    Image<double> outputDouble(xOut, yOut);
    MultidimArray<std::complex<double> > tmpFFTCpuIn(yIn, xIn / 2 + 1);
    MultidimArray<std::complex<double> > tmpFFTCpuOut(yOut, xOutFFT);
    MultidimArray<std::complex<double> > tmpFFTCpuOutExpected(yOut, xOutFFT);
    for (size_t y = 0; y < tmpFFTCpuIn.ydim; ++y) {
        for (size_t x = 0; x < tmpFFTCpuIn.xdim; ++x) {
            size_t index = y * tmpFFTCpuIn.xdim + x;
            tmpFFTCpuIn.data[index] = std::complex<double>(y, x);
        }
    }

    tmpFFTCpuOutExpected[0] = std::complex<double>(0, 0);
    tmpFFTCpuOutExpected[1] = std::complex<double>(0, 1);
    tmpFFTCpuOutExpected[2] = std::complex<double>(0, 2);
    tmpFFTCpuOutExpected[3] = std::complex<double>(0, 3);

    tmpFFTCpuOutExpected[4] = std::complex<double>(1, 0);
    tmpFFTCpuOutExpected[5] = std::complex<double>(1, 1);
    tmpFFTCpuOutExpected[6] = std::complex<double>(1, 2);
    tmpFFTCpuOutExpected[7] = std::complex<double>(1, 3);

    tmpFFTCpuOutExpected[8] = std::complex<double>(2, 0);
    tmpFFTCpuOutExpected[9] = std::complex<double>(2, 1);
    tmpFFTCpuOutExpected[10] = std::complex<double>(2, 2);
    tmpFFTCpuOutExpected[11] = std::complex<double>(2, 3);

    tmpFFTCpuOutExpected[12] = std::complex<double>(3, 0);
    tmpFFTCpuOutExpected[13] = std::complex<double>(3, 1);
    tmpFFTCpuOutExpected[14] = std::complex<double>(3, 2);
    tmpFFTCpuOutExpected[15] = std::complex<double>(3, 3);

    tmpFFTCpuOutExpected[16] = std::complex<double>(8, 0);
    tmpFFTCpuOutExpected[17] = std::complex<double>(8, 1);
    tmpFFTCpuOutExpected[18] = std::complex<double>(8, 2);
    tmpFFTCpuOutExpected[19] = std::complex<double>(8, 3);

    tmpFFTCpuOutExpected[20] = std::complex<double>(9, 0);
    tmpFFTCpuOutExpected[21] = std::complex<double>(9, 1);
    tmpFFTCpuOutExpected[22] = std::complex<double>(9, 2);
    tmpFFTCpuOutExpected[23] = std::complex<double>(9, 3);

    scaleToSizeFourier(inputDouble(), outputDouble(), tmpFFTCpuIn,
            tmpFFTCpuOut);

    ////////////////////////////////////////

    for (size_t i = 0; i < tmpFFTCpuOutExpected.yxdim; ++i) {
        float cpuReal = tmpFFTCpuOut.data[i].real();
        float cpuImag = tmpFFTCpuOut.data[i].imag();
        float expReal = tmpFFTCpuOutExpected[i].real();
        float expImag = tmpFFTCpuOutExpected[i].imag();
        if ((std::abs(cpuReal - expReal) > delta)
                || (std::abs(cpuImag - expImag) > delta)) {
            printf("ERROR SCALE CPU EE: %lu cpu (%f, %f) exp (%f, %f)\n", i,
                    cpuReal, cpuImag, expReal, expImag);
        }
    }
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::testFFTAndScale() {
    double delta = 0.00001;
    size_t xIn, yIn, xOut, yOut, xOutFFT;
    xIn = yIn = 4096;
    xOut = yOut = 2276;
    xOutFFT = xOut / 2 + 1;
    size_t order = 10000;
    size_t fftPixels = xOutFFT * yOut;

    srand(42);

    Image<double> inputDouble(xIn, yIn); // keep sync with values
    Image<float> inputFloat(xIn, yIn); // keep sync with values
    size_t pixels = inputDouble.data.xdim * inputDouble.data.ydim;
    for (size_t y = 0; y < inputDouble.data.ydim; ++y) {
        for (size_t x = 0; x < inputDouble.data.xdim; ++x) {
            size_t index = y * inputDouble.data.xdim + x;
            double value = rand() / (RAND_MAX / 2000.);
            inputDouble.data.data[index] = value;
            inputFloat.data.data[index] = (float) value;
        }
    }

    float* filter = new float[fftPixels];
    for (size_t i = 0; i < fftPixels; ++i) {
        filter[i] = rand() / (float) RAND_MAX;
    }

    // CPU part
    Image<double> outputDouble(xOut, yOut);
    MultidimArray<std::complex<double> > tmpFFTCpuIn;
    MultidimArray<std::complex<double> > tmpFFTCpuOut(yOut, xOutFFT);
    FourierTransformer transformer;

    transformer.FourierTransform(inputDouble(), tmpFFTCpuIn, true);
    scaleToSizeFourier(inputDouble(), outputDouble(), tmpFFTCpuIn,
            tmpFFTCpuOut);

    for (size_t nn = 0; nn < fftPixels; ++nn) {
        double wlpf = filter[nn];
        DIRECT_MULTIDIM_ELEM(tmpFFTCpuOut,nn) *= wlpf;
    }

    // store results to drive
    Image<double> fftCPU(tmpFFTCpuOut.xdim, tmpFFTCpuOut.ydim);
    for (size_t i = 0; i < fftPixels; i++) {
        fftCPU.data.data[i] = tmpFFTCpuOut.data[i].real();
    }
    fftCPU.write("testFFTCpuScaled.vol");

    // GPU part

    std::complex<float>* tmpFFTGpuOut = new std::complex<float>[fftPixels];
    float* d_filter = loadToGPU(filter, fftPixels);

    GpuMultidimArrayAtGpu<float> gpuIn(inputFloat.data.xdim,
            inputFloat.data.ydim);
    gpuIn.copyToGpu(inputFloat.data.data);
    GpuMultidimArrayAtGpu<std::complex<float> > gpuFFT;
    mycufftHandle handle;

//    processInput(gpuIn, gpuFFT, handle, xIn, yIn, 1, xOutFFT, yOut, d_filter,
//            tmpFFTGpuOut); FIXME test

    // store results to drive
    Image<float> fftGPU(xOutFFT, yOut);
    float norm = inputFloat.data.yxdim;
    for (size_t i = 0; i < fftPixels; i++) {
        fftGPU.data.data[i] = tmpFFTGpuOut[i].real() / norm;
    }
    fftGPU.write("testFFTGpuScaled.vol");

    ////////////////////////////////////////

    if (fftCPU.data.xdim != fftGPU.data.xdim) {
        printf("wrong size: X cpu %lu X gpu %lu\n", fftCPU.data.xdim,
                fftGPU.data.xdim);
    }
    if (fftCPU.data.ydim != fftGPU.data.ydim) {
        printf("wrong size: Y cpu %lu Y gpu %lu\n", fftCPU.data.xdim,
                fftGPU.data.xdim);
    }

    for (size_t i = 0; i < fftCPU.data.yxdim; ++i) {
        float cpuReal = tmpFFTCpuOut.data[i].real();
        float cpuImag = tmpFFTCpuOut.data[i].imag();
        float gpuReal = tmpFFTGpuOut[i].real() / norm;
        float gpuImag = tmpFFTGpuOut[i].imag() / norm;
        if ((std::abs(cpuReal - gpuReal) > delta)
                || (std::abs(cpuImag - gpuImag) > delta)) {
            printf("ERROR SCALE: %lu cpu (%f, %f) gpu (%f, %f)\n", i, cpuReal,
                    cpuImag, gpuReal, gpuImag);
        }
    }
    delete[] tmpFFTGpuOut;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::loadData(const MetaData& movie,
        const Image<T>& dark, const Image<T>& gain, T targetOccupancy,
        const MultidimArray<T>& lpf) {

    setDevice(device);

    bool cropInput = (this->yDRcorner != -1);
    int noOfImgs = this->nlast - this->nfirst + 1;

    // get frame info
    Image<T> frame;
    loadFrame(movie, movie.firstObject(), frame);
    setSizes(frame, noOfImgs);
    // prepare filter
    MultidimArray<T> filter;
    filter.initZeros(croppedOptSizeY, croppedOptSizeFFTX);
    this->scaleLPF(lpf, croppedOptSizeX, croppedOptSizeY, targetOccupancy,
            filter);

    // load all frames to RAM
    // reuse memory
    frameFourier = (std::complex<T>*)loadToRAM(movie, noOfImgs, dark, gain, cropInput);
    // scale and transform to FFT on GPU
    performFFTAndScale((T*)frameFourier, noOfImgs, inputOptSizeX,
            inputOptSizeY, inputOptBatchSize, croppedOptSizeFFTX,
            croppedOptSizeY, filter);
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::computeShifts(size_t N,
        const Matrix1D<T>& bX, const Matrix1D<T>& bY, const Matrix2D<T>& A) {
    setDevice(device);

    T* correlations;
    size_t centerSize = std::ceil(this->maxShift * 2 + 1);
    computeCorrelations(centerSize, N, frameFourier, croppedOptSizeFFTX,
            croppedOptSizeX, croppedOptSizeY, correlationBufferImgs,
            croppedOptBatchSize, correlations);

    // since we are using different size of FFT, we need to scale results to
    // 'expected' size
    T localSizeFactorX = this->sizeFactor
            / (croppedOptSizeX / (T) inputOptSizeX);
    T localSizeFactorY = this->sizeFactor
            / (croppedOptSizeY / (T) inputOptSizeY);

    int idx = 0;
    MultidimArray<T> Mcorr(centerSize, centerSize);
    for (size_t i = 0; i < N - 1; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            size_t offset = idx * centerSize * centerSize;
            Mcorr.data = correlations + offset;
            Mcorr.setXmippOrigin();
            bestShift(Mcorr, bX(idx), bY(idx), NULL, this->maxShift);
            bX(idx) *= localSizeFactorX; // scale to expected size
            bY(idx) *= localSizeFactorY;
            if (this->verbose)
                std::cerr << "Frame " << i + this->nfirst << " to Frame "
                        << j + this->nfirst << " -> ("
                        << bX(idx) / this->sizeFactor << ","
                        << bY(idx) / this->sizeFactor << ")" << std::endl;
            for (int ij = i; ij < j; ij++)
                A(idx, ij) = 1;

            idx++;
        }
    }
    Mcorr.data = NULL;
    delete[] frameFourier;
}

// explicit specialization
template class ProgMovieAlignmentCorrelationGPU<float> ;
