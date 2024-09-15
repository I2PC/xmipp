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

#include <algorithm>
#include <limits>
#include "reconstruction/movie_alignment_correlation_base.h"
#include "core/xmipp_image_generic.h"

template<typename T>
void AProgMovieAlignmentCorrelation<T>::readParams() {
    fnMovie = getParam("-i");
    fnOut = getParam("-o");
    fnInitialAvg = getParam("--oavgInitial");
    fnDark = getParam("--dark");
    fnGain = getParam("--gain");
    binning = getFloatParam("--bin");
    if (binning < 1.0)
        REPORT_ERROR(ERR_ARG_INCORRECT, "Binning must be >= 1");
    Ts = getFloatParam("--sampling") * binning;
    maxShift = getFloatParam("--maxShift") / Ts;
    maxResForCorrelation = getFloatParam("--maxResForCorrelation");
    fnAligned = getParam("--oaligned");
    fnAvg = getParam("--oavg");
    nfirst = getIntParam("--frameRange", 0);
    nlast = getIntParam("--frameRange", 1);
    nfirstSum = getIntParam("--frameRangeSum", 0);
    nlastSum = getIntParam("--frameRangeSum", 1);
    skipLocalAlignment = checkParam("--skipLocalAlignment");
    minLocalRes = getIntParam("--minLocalRes");

    // read control points
    Dimensions cPoints(
            this->getIntParam("--controlPoints", 0),
            this->getIntParam("--controlPoints", 1),
            1,
            this->getIntParam("--controlPoints", 2));
    if ((cPoints.x() < 3) || (cPoints.y() < 3) || (cPoints.n() < 3))
        REPORT_ERROR(ERR_ARG_INCORRECT,
            "All control points has to be bigger than 2");
    localAlignmentControlPoints = cPoints;

}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::checkSettings() {
    if ((nfirstSum < nfirst) || (nlastSum > nlast)) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Summing frames that were not aligned is not allowed. "
                "Check the intervals of the alignment and summation "
                "(--frameRange and --frameRangeSum).");
    }
    if (getScaleFactor() >= 1) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "The correlation scale factor is bigger than one. "
                "Check that the sampling rate (--sampling) and maximal resolution to align "
                "(--maxResForCorrelation) are correctly set. For current sampling, you can "
                "use maximal resolution of " + std::to_string(this->Ts * 8 * getC()) + " or higher.");
    }
    if (!skipLocalAlignment) {
        if (this->localAlignPatches.first <= this->localAlignmentControlPoints.x()
            || this->localAlignPatches.second <= this->localAlignmentControlPoints.y()) {
                REPORT_ERROR(ERR_LOGIC_ERROR, "More control points than patches. Decrease the number of control points.");
        }
    }
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::show() {
    if (!verbose)
        return;
    std::cout
            << "Input movie:           " << fnMovie << std::endl
            << "Output metadata:       " << fnOut << std::endl
            << "Dark image:            " << fnDark << std::endl
            << "Gain image:            " << fnGain << std::endl
            << "Max. Shift (A / px):   " << (maxShift * Ts) << " / " << maxShift << std::endl
            << "Max resolution (A):    " << maxResForCorrelation << std::endl
            << "Sampling:              " << Ts << std::endl
            << "Aligned movie:         " << fnAligned << std::endl
            << "Aligned micrograph:    " << fnAvg << std::endl
            << "Unaligned micrograph:  " << fnInitialAvg << std::endl
            << "Frame range alignment: " << nfirst << " " << nlast << std::endl
            << "Frame range sum:       " << nfirstSum << " " << nlastSum << std::endl
            << "Binning factor:        " << binning << std::endl
            << "Skip local alignment:  " << (skipLocalAlignment ? "yes" : "no") << std::endl
            << "Control points:        " << this->localAlignmentControlPoints << std::endl
            << "No of patches:         " << this->localAlignPatches.first << " * " << localAlignPatches.second << std::endl;
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::defineParams() {
    addUsageLine("Align a set of frames by cross-correlation of the frames");
    addParamsLine(
            "   -i <metadata>               : Metadata with the list of frames to align");
    addParamsLine(
            "  [-o <fn=\"out.xmd\">]        : Metadata with the shifts of each frame.");
    addParamsLine(
            "                               : If no filename is given, the input is rewritten");
    addParamsLine(
            "  [--bin <s=1>]                : Binning factor, it may be any floating number > 1.");
    addParamsLine(
            "                               : Binning is applied during the data loading, i.e. the program will processed and store binned data.");
    addParamsLine(
            "  [--maxShift <s=50>]          : Maximum shift allowed in A");
    addParamsLine(
            "  [--maxResForCorrelation <R=30>]: Maximum resolution to align (in Angstroms)");
    addParamsLine(
            "  [--sampling <Ts=1>]          : Sampling rate (A/pixel)");
    addParamsLine(
            "  [--oaligned <fn=\"\">]       : Aligned movie consists of aligned frames used for micrograph generation");
    addParamsLine(
            "  [--oavgInitial <fn=\"\">]    : Give the name of a micrograph to generate an unaligned (initial) micrograph");
    addParamsLine(
            "  [--oavg <fn=\"\">]           : Give the name of a micrograph to generate an aligned micrograph");
    addParamsLine(
            "  [--frameRange <n0=-1> <nF=-1>]  : First and last frame to align, frame numbers start at 0");
    addParamsLine(
            "  [--frameRangeSum <n0=-1> <nF=-1>]  : First and last frame to sum, frame numbers start at 0");
    addParamsLine("  [--dark <fn=\"\">]           : Dark correction image");
    addParamsLine("  [--gain <fn=\"\">]           : Gain correction image (we will multiply by it)");
    addParamsLine(
            "  [--skipLocalAlignment]       : If used, only global alignment will be performed. It's faster, but gives worse results.");
    addParamsLine(
            "  [--controlPoints <x=6> <y=6> <t=5>]: Number of control points (including end points) used for defining the BSpline");
    addParamsLine(
            "  [--patches <x=7> <y=7>]: Number of patches used for local alignment");
    addParamsLine(
            "  [--minLocalRes <R=500>]      : Minimal resolution (in A) of patches during local alignment");
    addExampleLine("A typical example", false);
    addSeeAlsoLine("xmipp_movie_optical_alignment_cpu");
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::loadFrame(const MetaData &movie,
        const Image<T> &dark, const Image<T> &igain, size_t objId,
            Image<T> &out) const {
    FileName fnFrame;
    movie.getValue(MDL_IMAGE, fnFrame, objId);
    out.read(fnFrame);
    if (XSIZE(dark()) > 0) {
        if ((XSIZE(dark()) != XSIZE(out()))
                || (YSIZE(dark()) != YSIZE(out()))) {
            REPORT_ERROR(ERR_ARG_INCORRECT,
                            "The dark image size does not match the movie frame size.");
        }
        out() -= dark();
    }
    if (XSIZE(igain()) > 0) {
        if ((XSIZE(igain()) != XSIZE(out()))
                || (YSIZE(igain()) != YSIZE(out()))) {
            REPORT_ERROR(ERR_ARG_INCORRECT,
                            "The gain image size does not match the movie frame size.");
        }
        out() *= igain();
    }
}

template<typename T>
float AProgMovieAlignmentCorrelation<T>::getPixelResolution(float scaleFactor) const {
    return this->Ts / scaleFactor;
}

template<typename T>
MultidimArray<T> AProgMovieAlignmentCorrelation<T>::createLPF(T Ts, const Dimensions &dims) {
    assert(Ts >= this->Ts);

    // Construct 1D profile of the lowpass filter
    MultidimArray<T> lpf(dims.x());
    createLPF(Ts, lpf);

    // scale 1D filter to 2D
    MultidimArray<T> result;
    result.initZeros(dims.y(), (dims.x() / 2) + 1);
    scaleLPF(lpf, dims, result);
    return result;
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::createLPF(T Ts, MultidimArray<T> &filter) {
    // from formula
    // e^(-1/2 * (omega^2 / sigma^2)) = 1/2; omega = Ts / max_resolution
    // sigma = Ts / max_resolution * sqrt(1/-2log(1/2))
    // c = sqrt(1/-2log(1/2))
    // sigma = Ts / max_resolution * c
    const size_t length = filter.xdim;
    T iX = 1 / (T)length;
    T sigma = (Ts * getC()) / maxResForCorrelation;
    for (size_t x = 0; x < length; ++x) {
        T w = x * iX;
        filter.data[x] = (exp(-0.5*(w*w)/(sigma*sigma)));
    }
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::scaleLPF(const MultidimArray<T>& lpf,
        const Dimensions &dims, MultidimArray<T>& result) {
    Matrix1D<T> w(2);
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(result)
    {
        FFT_IDX2DIGFREQ(i, dims.y(), YY(w));
        FFT_IDX2DIGFREQ(j, dims.x(), XX(w));
        T wabs = w.module();
        A2D_ELEM(result, i, j) = lpf.interpolatedElement1D(wabs * dims.x());
    }
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::computeTotalShift(int iref, int j,
        const Matrix1D<T> &shiftX, const Matrix1D<T> &shiftY, T &totalShiftX,
        T &totalShiftY) {
    totalShiftX = totalShiftY = 0;
    if (iref < j) {
        for (int jj = j - 1; jj >= iref; --jj) {
            totalShiftX -= shiftX(jj);
            totalShiftY -= shiftY(jj);
        }
    } else if (iref > j) {
        for (int jj = j; jj <= iref - 1; ++jj) {
            totalShiftX += shiftX(jj);
            totalShiftY += shiftY(jj);
        }
    }
}

template<typename T>
int AProgMovieAlignmentCorrelation<T>::findReferenceImage(size_t N,
        const Matrix1D<T>& shiftX, const Matrix1D<T>& shiftY) {
    int bestIref = -1;
    // Choose reference image as the minimax of shifts
    T worstShiftEver = std::numeric_limits<T>::max();
    for (int iref = 0; iref < N; ++iref) {
        T worstShift = -1;
        for (int j = 0; j < N; ++j) {
            T totalShiftX, totalShiftY;
            computeTotalShift(iref, j, shiftX, shiftY, totalShiftX,
                    totalShiftY);
            if (fabs(totalShiftX) > worstShift)
                worstShift = fabs(totalShiftX);
        }
        if (worstShift < worstShiftEver) {
            worstShiftEver = worstShift;
            bestIref = iref;
        }
    }
    return bestIref;
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::loadDarkCorrection(Image<T>& dark) {
    if (fnDark.isEmpty())
        return;
    dark.read(fnDark);
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::loadGainCorrection(Image<T>& igain) {
    if (fnGain.isEmpty())
        return;
    igain.read(fnGain);
    T avg = igain().computeAvg();
    if (std::isinf(avg) || std::isnan(avg))
        REPORT_ERROR(ERR_ARG_INCORRECT,
                "The input gain image is incorrect, it contains infinite or nan");
}

template<typename T>
float AProgMovieAlignmentCorrelation<T>::getC() const {
    // from formula
    // e^(-1/2 * (omega^2 / sigma^2)) = 1/2; omega = Ts / max_resolution
    // sigma = Ts / max_resolution * sqrt(1/-2log(1/2))
    static const float c = std::sqrt(-1.f / (2.f * std::log(0.5f)));
    return c;
}

template<typename T>
float AProgMovieAlignmentCorrelation<T>::getTsPrime() const {
    // from formula
    // e^(-1/2 * (omega^2 / sigma^2)) = 1/2; omega = Ts / max_resolution
    // sigma = Ts / max_resolution * sqrt(1/-2log(1/2))
    // c = sqrt(1/-2log(1/2))
    // sigma = Ts / max_resolution * c
    // then we want to find a resolution at 4 sigma (because values there will be almost zero anyway)
    // omega4 = 4 * sigma -> Ts/R4 = 4 Ts / max_resolution * c
    // R4 = max_resolution / (4 * c)
    // new pixel size Ts' = R4 / 2 (to preserve Nyquist frequency)
    return maxResForCorrelation / (8.f * getC());
}

template<typename T>
float AProgMovieAlignmentCorrelation<T>::getScaleFactor() const {
    // scale is ration between original pixel size and new pixel size
    return Ts / getTsPrime();
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::readMovie(MetaData& movie) {
    //if input is an stack create a metadata.
    if (fnMovie.isMetaData())
        movie.read(fnMovie);
    else {
        ImageGeneric movieStack;
        movieStack.read(fnMovie, HEADER);
        size_t Xdim, Ydim, Zdim, Ndim;
        movieStack.getDimensions(Xdim, Ydim, Zdim, Ndim);
        if (fnMovie.getExtension() == "mrc" and Ndim == 1)
            Ndim = Zdim;
        size_t id;
        FileName fn;
        for (size_t i = 0; i < Ndim; i++) {
            id = movie.addObject();
            fn.compose(i + FIRST_IMAGE, fnMovie);
            movie.setValue(MDL_IMAGE, fn, id);
        }
    }
}

template<typename T>
Dimensions AProgMovieAlignmentCorrelation<T>::getMovieSizeRaw() {
    if (this->movieSizeRaw) return movieSizeRaw.value();
    int noOfImgs = this->nlast - this->nfirst + 1;
    auto fn = fnMovie;
    if (fnMovie.isMetaData()) {
        MetaDataVec md;
        md.read(fnMovie);
        md.getValue(MDL_IMAGE, fn, md.firstRowId()); // assuming all frames have the same resolution
    }
    ImageGeneric movieStack;
    movieStack.read(fn, HEADER);
    size_t xdim, ydim, zdim, ndim;
    movieStack.getDimensions(xdim, ydim, zdim, ndim);
    this->movieSizeRaw = Dimensions(xdim, ydim, 1, noOfImgs);
    return movieSizeRaw.value();
}

template<typename T>
Dimensions AProgMovieAlignmentCorrelation<T>::getMovieSize() {
    if (movieSize) return movieSize.value();
    auto full = getMovieSizeRaw();
    if (applyBinning()) {
        // to make FFT fast, we want the size to be a multiple of 2
        auto x = ((static_cast<float>(full.x()) / binning) / 2.f) * 2.f;
        auto y = ((static_cast<float>(full.y()) / binning) / 2.f) * 2.f;
        movieSize = Dimensions(static_cast<size_t>(x), static_cast<size_t>(y), 1L, full.n());
    } else {
        movieSize = full;
    }
    return movieSize.value();
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::storeGlobalShifts(
        const AlignmentResult<T> &alignment, MetaData &movie) {
    int j = 0;
    int n = 0;
    auto negateToDouble = [binning=binning] (T v) {return (double) (v * -1) * binning;};
        for (size_t objId : movie.ids())
        {
            if (n >= nfirst && n <= nlast) {
                auto shift = alignment.shifts.at(j);
                // we should store shift that should be applied
                movie.setValue(MDL_SHIFT_X, negateToDouble(shift.x), objId);
                movie.setValue(MDL_SHIFT_Y, negateToDouble(shift.y), objId);
                j++;
            movie.setValue(MDL_ENABLED, 1, objId);
        } else {
            movie.setValue(MDL_ENABLED, -1, objId);
            movie.setValue(MDL_SHIFT_X, 0.0, objId);
            movie.setValue(MDL_SHIFT_Y, 0.0, objId);
        }
        movie.setValue(MDL_WEIGHT, 1.0, objId);
        n++;
    }
    MetaDataVec mdIref;
    mdIref.setValue(MDL_REF, (int)(nfirst + alignment.refFrame), mdIref.addObject());
    mdIref.write((FileName) "referenceFrame@" + fnOut, MD_APPEND);
}

template<typename T>
AlignmentResult<T> AProgMovieAlignmentCorrelation<T>::computeAlignment(
        Matrix1D<T> &bX, Matrix1D<T> &bY, Matrix2D<T> &A,
        const core::optional<size_t> &refFrame, size_t N, int verbose) {
    // now get the estimated shift (from the equation system)
    // from each frame to successive frame
    Matrix1D<T> shiftX, shiftY;
    EquationSystemSolver::solve(bX, bY, A, shiftX, shiftY, verbose, solverIterations);
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
void AProgMovieAlignmentCorrelation<T>::storeResults(Image<T>& initialMic,
        size_t Ninitial, Image<T>& averageMicrograph, size_t N,
        const MetaData& movie, int bestIref) {
    if (fnInitialAvg != "") {
        initialMic() /= Ninitial;
        initialMic.write(fnInitialAvg);
    }
    if (fnAvg != "") {
        averageMicrograph() /= N;
        averageMicrograph.write(fnAvg);
    }
    movie.write((FileName) "frameShifts@" + fnOut, MD_APPEND);
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::correctLoopIndices(
        const MetaData& movie) {
    nfirst = std::max(nfirst, 0);
    nfirstSum = std::max(nfirstSum, 0);
    if (nlast < 0)
        nlast = movie.size() - 1;

    if (nlastSum < 0)
        nlastSum = movie.size() - 1;
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::printGlobalShift(
        const AlignmentResult<T> &globAlignment) {
    std::cout << "Reference frame: " << globAlignment.refFrame << "\n";
    std::cout << "Estimated global shifts (in px, from the reference frame";
    std::cout << (applyBinning() ? ", ignoring binning" : "");
    std::cout << "):\n";
    for (auto &s : globAlignment.shifts) {
        printf("X: %07.4f Y: %07.4f\n", s.x * binning, s.y * binning);
    }
    std::cout << std::endl;
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::storeResults(
        const LocalAlignmentResult<T> &alignment) {
    if (fnOut.isEmpty()) {
        return;
    }
    if ( ! alignment.bsplineRep) {
        REPORT_ERROR(ERR_VALUE_INCORRECT,
            "Missing BSpline representation. This should not happen. Please contact developers.");
    }
    // Store average
    std::vector<double> shifts;
    for (auto &&p : alignment.shifts) {
        int tileCenterX = p.first.rec.getCenter().x;
        int tileCenterY = p.first.rec.getCenter().y;
        int tileIdxT = p.first.id_t;
        auto shift = BSplineHelper::getShift(
                alignment.bsplineRep.value(), alignment.movieDim,
                tileCenterX, tileCenterY, tileIdxT);
        auto globalShift = alignment.globalHint.shifts.at(tileIdxT);
        shifts.emplace_back(hypot(shift.first - globalShift.x, shift.second - globalShift.y));
    }
    MetaDataVec mdIref;
    size_t id = mdIref.addObject();
    // Store confidence interval
    std::sort(shifts.begin(), shifts.end(), std::less<double>());
    size_t indexL = shifts.size() * 0.025;
    size_t indexU = shifts.size() * 0.975;
    mdIref.setValue(MDL_LOCAL_ALIGNMENT_CONF_2_5_PERC, shifts.at(indexL), id);
    mdIref.setValue(MDL_LOCAL_ALIGNMENT_CONF_97_5_PERC, shifts.at(indexU), id);
    // Store patches
    mdIref.setValue(MDL_LOCAL_ALIGNMENT_PATCHES,
        std::vector<size_t>{localAlignPatches.first, localAlignPatches.second}, id);
    // Store coefficients
    std::vector<double> tmpX;
    std::vector<double> tmpY;
    size_t size = alignment.bsplineRep.value().getCoeffsX().size();
    for (int i = 0; i < size; ++i) {
        tmpX.push_back(alignment.bsplineRep.value().getCoeffsX()(i));
        tmpY.push_back(alignment.bsplineRep.value().getCoeffsY()(i));
    }
    mdIref.setValue(MDL_LOCAL_ALIGNMENT_COEFFS_X, tmpX, id);
    mdIref.setValue(MDL_LOCAL_ALIGNMENT_COEFFS_Y, tmpY, id);
    // Store control points
    mdIref.setValue(MDL_LOCAL_ALIGNMENT_CONTROL_POINTS,
            std::vector<size_t>{
                localAlignmentControlPoints.x(),
                localAlignmentControlPoints.y(),
                localAlignmentControlPoints.n()},
            id);
    // Safe to file
    mdIref.write("localAlignment@" + fnOut, MD_APPEND);
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::setNoOfPatches() {
        // set number of patches
    if (checkParam("--patches")) {
        localAlignPatches = {getIntParam("--patches", 0), getIntParam("--patches", 1)};
    } else {
        const auto &movieDim = getMovieSize();
        auto patchDim = getRequestedPatchSize();
        localAlignPatches = {
                std::ceil(static_cast<float>(movieDim.x()) / static_cast<float>(patchDim.first)),
                std::ceil(static_cast<float>(movieDim.y()) / static_cast<float>(patchDim.second))};
    }
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::run() {
    // preprocess input data
    MetaDataVec movie;
    readMovie(movie);
    correctLoopIndices(movie);

    setNoOfPatches();
    show();
    checkSettings();

    Image<T> dark, igain;
    loadDarkCorrection(dark);
    loadGainCorrection(igain);

    auto globalAlignment = AlignmentResult<T>();
    std::cout << "Computing global alignment ...\n";
    globalAlignment = computeGlobalAlignment(movie, dark, igain);

    if ( ! fnOut.isEmpty()) {
        storeGlobalShifts(globalAlignment, movie);
    }

    if (verbose) printGlobalShift(globalAlignment);

    size_t N, Ninitial;
    Image<T> initialMic, averageMicrograph;
    // Apply shifts and compute average
    if (skipLocalAlignment) {
        applyShiftsComputeAverage(movie, dark, igain, initialMic, Ninitial,
                    averageMicrograph, N, globalAlignment);
    } else {
        std::cout << "Computing local alignment ...\n";
        auto localAlignment = computeLocalAlignment(movie, dark, igain, globalAlignment);
        applyShiftsComputeAverage(movie, dark, igain, initialMic, Ninitial,
                    averageMicrograph, N, localAlignment);
        storeResults(localAlignment);
    }

    storeResults(initialMic, Ninitial, averageMicrograph, N, movie, globalAlignment.refFrame);

    releaseAll();
}

template class AProgMovieAlignmentCorrelation<float> ;
template class AProgMovieAlignmentCorrelation<double> ;
