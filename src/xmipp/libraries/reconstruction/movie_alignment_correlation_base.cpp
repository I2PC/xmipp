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

#include "reconstruction/movie_alignment_correlation_base.h"

template<typename T>
void AProgMovieAlignmentCorrelation<T>::readParams() {
    fnMovie = getParam("-i");
    fnOut = getParam("-o");
    fnInitialAvg = getParam("--oavgInitial");
    fnDark = getParam("--dark");
    fnGain = getParam("--gain");
    maxShift = getDoubleParam("--max_shift");
    Ts = getDoubleParam("--sampling");
    maxResForCorrelation = getDoubleParam("--maxResForCorrelation");
    solverIterations = getIntParam("--solverIterations");
    fnAligned = getParam("--oaligned");
    fnAvg = getParam("--oavg");
    nfirst = getIntParam("--frameRange", 0);
    nlast = getIntParam("--frameRange", 1);
    nfirstSum = getIntParam("--frameRangeSum", 0);
    nlastSum = getIntParam("--frameRangeSum", 1);
    xLTcorner = getIntParam("--cropULCorner", 0);
    yLTcorner = getIntParam("--cropULCorner", 1);
    xDRcorner = getIntParam("--cropDRCorner", 0);
    yDRcorner = getIntParam("--cropDRCorner", 1);
    useInputShifts = checkParam("--useInputShifts");
    outputBinning = getDoubleParam("--bin");
    BsplineOrder = getIntParam("--Bspline");
    processLocalShifts = checkParam("--processLocalShifts");
    minLocalRes = getIntParam("--minLocalRes");

    String outside = getParam("--outside");
    if (outside == "wrap")
        outsideMode = OUTSIDE_WRAP;
    else if (outside == "avg")
        outsideMode = OUTSIDE_AVG;
    else if (outside == "value") {
        outsideMode = OUTSIDE_VALUE;
        outsideValue = getDoubleParam("--outside", 1);
    }
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

    addParamsLine("  [--minLocalRes <R=500>]            : Minimal resolution (in A) of patches during local alignment");
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::checkSettings() {
    if ((nfirstSum < nfirst) || (nlastSum > nlast)) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Summing frames that were not aligned is not allowed. "
                "Check the intervals of the alignment and summation "
                "(--frameRange and --frameRangeSum).");
    }
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::show() {
    if (!verbose)
        return;
    std::cout
            << "Input movie:         " << fnMovie << std::endl
            << "Output metadata:     " << fnOut << std::endl
            << "Dark image:          " << fnDark << std::endl
            << "Gain image:          " << fnGain << std::endl
            << "Max. Shift:          " << maxShift << std::endl
            << "Max resolution (A):  " << maxResForCorrelation << std::endl
            << "Sampling:            " << Ts << std::endl
            << "Solver iterations:   " << solverIterations << std::endl
            << "Aligned movie:       " << fnAligned << std::endl
            << "Aligned micrograph:  " << fnAvg << std::endl
            << "Unaligned micrograph: " << fnInitialAvg << std::endl
            << "Frame range alignment: " << nfirst << " " << nlast << std::endl
            << "Frame range sum:       " << nfirstSum << " " << nlastSum << std::endl
            << "Crop corners  " << "(" << xLTcorner << ", "
            << yLTcorner << ") " << "(" << xDRcorner << ", " << yDRcorner
            << ") " << std::endl
            << "Use input shifts:    " << useInputShifts << std::endl
            << "Output Binning factor: " << outputBinning << std::endl
            << "Bspline:             " << BsplineOrder << std::endl
            << "Local shift correction: " << (processLocalShifts ? "yes" : "no") << std::endl
            << "Control points:      " << this->localAlignmentControlPoints << std::endl
            << "Patches:             " << this->localAlignPatches.first << " x " << this->localAlignPatches.second << std::endl;
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
            "  [--bin <s=-1>]               : Binning factor, it may be any floating number");
    addParamsLine(
            "                               :+Binning in Fourier is the first operation, so that");
    addParamsLine(
            "                               :+crop parameters are referred to the binned images");
    addParamsLine(
            "                               :+By default, -1, the binning is automatically calculated ");
    addParamsLine(
            "                               :+as a function of max_freq.");
    addParamsLine(
            "  [--max_shift <s=30>]         : Maximum shift allowed in pixels");
    addParamsLine(
            "  [--maxResForCorrelation <R=30>]: Maximum resolution to align (in Angstroms)");
    addParamsLine(
            "  [--sampling <Ts=1>]          : Sampling rate (A/pixel)");
    addParamsLine(
            "  [--solverIterations <N=2>]   : Number of robust least squares iterations");
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
    addParamsLine(
            "  [--cropULCorner <x=0> <y=0>]    : crop up left corner (unit=px, index starts at 0)");
    addParamsLine(
            "  [--cropDRCorner <x=-1> <y=-1>]    : crop down right corner (unit=px, index starts at 0), -1 -> no crop");
    addParamsLine("  [--dark <fn=\"\">]           : Dark correction image");
    addParamsLine("  [--gain <fn=\"\">]           : Gain correction image (we will multiply by it)");
    addParamsLine(
            "  [--useInputShifts]           : Do not calculate shifts and use the ones in the input file");
    addParamsLine(
            "  [--Bspline <order=3>]        : B-spline order for the final interpolation (1 or 3)");
    addParamsLine(
            "  [--outside <mode=wrap> <v=0>]: How to deal with borders (wrap, substitute by avg, or substitute by value)");
    addParamsLine("      where <mode>");
    addParamsLine(
            "             wrap              : Wrap the image to deal with borders");
    addParamsLine(
            "             avg               : Fill borders with the average of the frame");
    addParamsLine(
            "             value             : Fill borders with a specific value v");
    addParamsLine(
            "  [--processLocalShifts]       : Calculate and correct local shifts");
    addParamsLine(
            "  [--controlPoints <x=6> <y=6> <t=5>]: Number of control points (including end points) used for defining the BSpline");
    addParamsLine(
            "  [--minLocalRes <R=500>]      : Minimal resolution (in A) of patches during local alignment");
    addExampleLine("A typical example", false);
    addSeeAlsoLine("xmipp_movie_optical_alignment_cpu");
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::loadFrame(const MetaData& movie,
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
void AProgMovieAlignmentCorrelation<T>::loadFrame(const MetaData &movie,
        const Image<T> &dark, const Image<T> &igain, size_t objId,
            Image<T> &out) {
    loadFrame(movie, objId, out);
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
T AProgMovieAlignmentCorrelation<T>::getPixelResolution(T scaleFactor) {
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
    if (yDRcorner != -1)
        dark().selfWindow(yLTcorner, xLTcorner, yDRcorner, xDRcorner);
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::loadGainCorrection(Image<T>& igain) {
    if (fnGain.isEmpty())
        return;
    igain.read(fnGain);
    if (yDRcorner != -1)
        igain().selfWindow(yLTcorner, xLTcorner, yDRcorner, xDRcorner);
    T avg = igain().computeAvg();
    if (std::isinf(avg) || std::isnan(avg))
        REPORT_ERROR(ERR_ARG_INCORRECT,
                "The input gain image is incorrect, it contains infinite or nan");
}

template<typename T>
T AProgMovieAlignmentCorrelation<T>::getC() {
    // from formula
    // e^(-1/2 * (omega^2 / sigma^2)) = 1/2; omega = Ts / max_resolution
    // sigma = Ts / max_resolution * sqrt(1/-2log(1/2))
    constexpr T c = sqrt(-(T)1 / (2 * log((T)0.5)));
    return c;
}

template<typename T>
T AProgMovieAlignmentCorrelation<T>::getTsPrime() {
    // from formula
    // e^(-1/2 * (omega^2 / sigma^2)) = 1/2; omega = Ts / max_resolution
    // sigma = Ts / max_resolution * sqrt(1/-2log(1/2))
    // c = sqrt(1/-2log(1/2))
    // sigma = Ts / max_resolution * c
    // then we want to find a resolution at 4 sigma (because values there will be almost zero anyway)
    // omega4 = 4 * sigma -> Ts/R4 = 4 Ts / max_resolution * c
    // R4 = max_resolution / (4 * c)
    // new pixel size Ts' = R4 / 2 (to preserve Nyquist frequency)
    T TsPrime = maxResForCorrelation / (8 * getC());
    return TsPrime;
}

template<typename T>
T AProgMovieAlignmentCorrelation<T>::getScaleFactor() {
    // scale is ration between original pixel size and new pixel size
    T scale = Ts / getTsPrime();
    return scale;
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
void AProgMovieAlignmentCorrelation<T>::storeGlobalShifts(
        const AlignmentResult<T> &alignment, MetaData &movie) {
    int j = 0;
    int n = 0;
    auto negateToDouble = [] (T v) {return (double) (v * -1);};
        FOR_ALL_OBJECTS_IN_METADATA(movie)
        {
            if (n >= nfirst && n <= nlast) {
                auto shift = alignment.shifts.at(j);
                // we should store shift that should be applied
                movie.setValue(MDL_SHIFT_X, negateToDouble(shift.x),
                        __iter.objId);
                movie.setValue(MDL_SHIFT_Y, negateToDouble(shift.y),
                        __iter.objId);
                j++;
            movie.setValue(MDL_ENABLED, 1, __iter.objId);
        } else {
            movie.setValue(MDL_ENABLED, -1, __iter.objId);
            movie.setValue(MDL_SHIFT_X, 0.0, __iter.objId);
            movie.setValue(MDL_SHIFT_Y, 0.0, __iter.objId);
        }
        movie.setValue(MDL_WEIGHT, 1.0, __iter.objId);
        n++;
    }
    MetaData mdIref;
    mdIref.setValue(MDL_REF, (int)(nfirst + alignment.refFrame), mdIref.addObject());
    mdIref.write((FileName) ("referenceFrame@") + fnOut, MD_APPEND);
}

template<typename T>
AlignmentResult<T> AProgMovieAlignmentCorrelation<T>::loadGlobalShifts(MetaData &movie) {
    auto alignment = AlignmentResult<T>();
    int n = 0;
    T shiftX;
    T shiftY;
    FOR_ALL_OBJECTS_IN_METADATA(movie)
    {
        if (n >= nfirst && n <= nlast) {
            movie.getValue(MDL_SHIFT_X, shiftX, __iter.objId);
            movie.getValue(MDL_SHIFT_Y, shiftY, __iter.objId);

            // loaded values are shift that should be applied
            // so we have to negate it
            alignment.shifts.emplace_back(-shiftX, -shiftY);
        }
        n++;
    }
//    FIXME load reference frame
    return alignment;
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::setZeroShift(MetaData& movie) {
    // assuming movie does not contain MDL_SHIFT_X label
    movie.addLabel(MDL_SHIFT_X);
    movie.addLabel(MDL_SHIFT_Y);
    movie.fillConstant(MDL_SHIFT_X, "0.0");
    movie.fillConstant(MDL_SHIFT_Y, "0.0");
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
    std::cout << (useInputShifts ? "Loaded" : "Estimated")
            << " global shifts (from the reference frame):\n";
    for (auto &&s : globAlignment.shifts) {
        printf("X: %07.4f Y: %07.4f\n", s.x, s.y);
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
    MetaData mdIref;
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
    mdIref.write((FileName) ("localAlignment@") + fnOut, MD_APPEND);
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::setNoOfPaches(const Dimensions &movieDim,
        const Dimensions &patchDim) {
    // read patches
    localAlignPatches = {
            std::ceil(movieDim.x() / (float)patchDim.x()),
            std::ceil(movieDim.y() / (float)patchDim.y())};
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::run() {
    show();
    checkSettings();
    // preprocess input data
    MetaData movie;
    readMovie(movie);
    correctLoopIndices(movie);

    Image<T> dark, igain;
    loadDarkCorrection(dark);
    loadGainCorrection(igain);

    auto globalAlignment = AlignmentResult<T>();
    if (useInputShifts) {
        if (!movie.containsLabel(MDL_SHIFT_X)) {
            setZeroShift(movie);
        }
        globalAlignment = loadGlobalShifts(movie);
    } else {
        globalAlignment = computeGlobalAlignment(movie, dark, igain);
    }

    if ( ! fnOut.isEmpty()) {
        storeGlobalShifts(globalAlignment, movie);
    }

    if (verbose) printGlobalShift(globalAlignment);

    size_t N, Ninitial;
    Image<T> initialMic, averageMicrograph;
    // Apply shifts and compute average
    if (processLocalShifts) {
        auto localAlignment = computeLocalAlignment(movie, dark, igain, globalAlignment);
        applyShiftsComputeAverage(movie, dark, igain, initialMic, Ninitial,
                    averageMicrograph, N, localAlignment);
        storeResults(localAlignment);
    } else {
        applyShiftsComputeAverage(movie, dark, igain, initialMic, Ninitial,
                    averageMicrograph, N, globalAlignment);
    }

    storeResults(initialMic, Ninitial, averageMicrograph, N, movie, globalAlignment.refFrame);

    releaseAll();
}

template class AProgMovieAlignmentCorrelation<float> ;
template class AProgMovieAlignmentCorrelation<double> ;
