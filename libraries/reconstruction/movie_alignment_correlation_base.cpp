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
void AProgMovieAlignmentCorrelation<T>::scaleLPF(const MultidimArray<T>& lpf,
        int xSize, int ySize, T targetOccupancy, MultidimArray<T>& result) {
    Matrix1D<T> w(2);
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(result)
    {
        FFT_IDX2DIGFREQ(i, ySize, YY(w));
        FFT_IDX2DIGFREQ(j, xSize, XX(w));
        T wabs = w.module();
        if (wabs <= targetOccupancy)
            A2D_ELEM(result, i, j) = lpf.interpolatedElement1D(wabs * xSize);
    }
}

template<typename T>
MultidimArray<T> AProgMovieAlignmentCorrelation<T>::createLPF(T targetOccupancy,
        size_t xSize,
        size_t xFFTSize,
        size_t ySize) {
    // Construct 1D profile of the lowpass filter
    MultidimArray<T> lpf(xSize);
    constructLPF(targetOccupancy, lpf);

    MultidimArray<T> result;
    result.initZeros(ySize, xFFTSize);

    scaleLPF(lpf, xSize, ySize, targetOccupancy, result);
    return result;
}

// Read arguments ==========================================================
template<typename T>
void AProgMovieAlignmentCorrelation<T>::readParams() {
    fnMovie = getParam("-i");
    fnOut = getParam("-o");
    fnInitialAvg = getParam("--oavgInitial");
    fnDark = getParam("--dark");
    fnGain = getParam("--gain");
    maxShift = getDoubleParam("--max_shift");
    Ts = getDoubleParam("--sampling");
    maxFreq = getDoubleParam("--max_freq");
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
    bin = getDoubleParam("--bin");
    BsplineOrder = getIntParam("--Bspline");
    processLocalShifts = checkParam("--processLocalShifts");

    String outside = getParam("--outside");
    if (outside == "wrap")
        outsideMode = OUTSIDE_WRAP;
    else if (outside == "avg")
        outsideMode = OUTSIDE_AVG;
    else if (outside == "value") {
        outsideMode = OUTSIDE_VALUE;
        outsideValue = getDoubleParam("--outside", 1);
    }
}

// Show ====================================================================
template<typename T>
void AProgMovieAlignmentCorrelation<T>::show() {
    if (!verbose)
        return;
    std::cout << "Input movie:         " << fnMovie << std::endl
            << "Output metadata:     " << fnOut << std::endl
            << "Dark image:          " << fnDark << std::endl
            << "Gain image:          " << fnGain << std::endl
            << "Max. Shift:          " << maxShift << std::endl
            << "Max. Scale:          " << maxFreq << std::endl
            << "Sampling:            " << Ts << std::endl
            << "Solver iterations:   " << solverIterations << std::endl
            << "Aligned movie:       " << fnAligned << std::endl
            << "Aligned micrograph:  " << fnAvg << std::endl
            << "Unaligned micrograph: " << fnInitialAvg << std::endl
            << "Frame range alignment: " << nfirst << " " << nlast << std::endl
            << "Frame range sum:       " << nfirstSum << " " << nlastSum
            << std::endl << "Crop corners  " << "(" << xLTcorner << ", "
            << yLTcorner << ") " << "(" << xDRcorner << ", " << yDRcorner
            << ") " << std::endl << "Use input shifts:    " << useInputShifts
            << std::endl << "Binning factor:      " << bin << std::endl
            << "Bspline:             " << BsplineOrder << std::endl
            << "Local shift correction: " << (processLocalShifts ? "yes" : "no")
            << std::endl;
}

// usage ===================================================================
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
            "  [--max_shift <s=-1>]         : Maximum shift allowed in pixels");
    addParamsLine(
            "  [--max_freq <s=4>]           : Maximum resolution to align (in Angstroms)");
    addParamsLine("  [--sampling <Ts=1>]          : Sampling rate (A/pixel)");
    addParamsLine(
            "  [--solverIterations <N=2>]   : Number of robust least squares iterations");
    addParamsLine(
            "  [--oaligned <fn=\"\">]       : Give the name of a stack if you want to generate an aligned movie");
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
    addParamsLine("  [--gain <fn=\"\">]           : Gain correction image");
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
            "  [--processLocalShifts]           : Calculate and correct local shifts");
    addExampleLine("A typical example", false);
    addSeeAlsoLine("xmipp_movie_optical_alignment_cpu");
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
    if (verbose)
        std::cout << "Reference frame: " << bestIref + nfirst
            << std::endl;

    return bestIref;
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::solveEquationSystem(Matrix1D<T>& bXt,
        Matrix1D<T>& bYt, Matrix2D<T>& At, Matrix1D<T>& shiftXt,
        Matrix1D<T>& shiftYt) {
    Matrix1D<double> ex, ey;
    WeightedLeastSquaresHelper helper;
    Matrix2D<double> A;
    Matrix1D<double> bX, bY, shiftX, shiftY;
    typeCast(At, helper.A);
    typeCast(bXt, bX);
    typeCast(bYt, bY);
    typeCast(shiftXt, shiftX);
    typeCast(shiftYt, shiftY);

    helper.w.initZeros(VEC_XSIZE(bX));
    helper.w.initConstant(1);

    int it = 0;
    double mean, varbX, varbY;
    bX.computeMeanAndStddev(mean, varbX);
    varbX *= varbX;
    bY.computeMeanAndStddev(mean, varbY);
    varbY *= varbY;
    if (verbose)
        std::cout << "\nSolving for the shifts ...\n";
    do {
        // Solve the equation system
        helper.b = bX;
        weightedLeastSquares(helper, shiftX);
        helper.b = bY;
        weightedLeastSquares(helper, shiftY);

        // Compute residuals
        ex = bX - helper.A * shiftX;
        ey = bY - helper.A * shiftY;

        // Compute R2
        double mean, vareX, vareY;
        ex.computeMeanAndStddev(mean, vareX);
        vareX *= vareX;
        ey.computeMeanAndStddev(mean, vareY);
        vareY *= vareY;
        double R2x = 1 - vareX / varbX;
        double R2y = 1 - vareY / varbY;
        if (verbose)
            std::cout << "Iteration " << it << " R2x=" << R2x << " R2y=" << R2y
                    << std::endl;

        // Identify outliers
        double oldWeightSum = helper.w.sum();
        double stddeveX = sqrt(vareX);
        double stddeveY = sqrt(vareY);
        FOR_ALL_ELEMENTS_IN_MATRIX1D (ex)
            if (fabs(VEC_ELEM(ex, i)) > 3 * stddeveX
                    || fabs(VEC_ELEM(ey, i)) > 3 * stddeveY)
                VEC_ELEM(helper.w, i) = 0.0;
        double newWeightSum = helper.w.sum();
        if (newWeightSum == oldWeightSum) {
            std::cout << "No outlier found\n";
            break;
        } else
            std::cout << "Found " << (int) (oldWeightSum - newWeightSum)
                    << " outliers\n";

        it++;
    } while (it < solverIterations);

    typeCast(shiftX, shiftXt);
    typeCast(shiftY, shiftYt);
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
void AProgMovieAlignmentCorrelation<T>::loadGainCorrection(Image<T>& gain) {
    if (fnGain.isEmpty())
        return;
    gain.read(fnGain);
    if (yDRcorner != -1)
        gain().selfWindow(yLTcorner, xLTcorner, yDRcorner, xDRcorner);
    gain() = (T) 1 / gain();
    T avg = gain().computeAvg();
    if (std::isinf(avg) || std::isnan(avg))
        REPORT_ERROR(ERR_ARG_INCORRECT,
                "The input gain image is incorrect, its inverse produces infinite or nan");
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::constructLPFold(T targetOccupancy,
        const MultidimArray<T>& lpf) {
    T iNewXdim = 1.0 / newXdim;
    T sigma = targetOccupancy / 6; // So that from -targetOccupancy to targetOccupancy there is 6 sigma
    T K = -0.5 / (sigma * sigma);
    for (int i = STARTINGX(lpf); i <= FINISHINGX(lpf); ++i) {
        T w = i * iNewXdim;
        A1D_ELEM(lpf, i) = exp(K * (w * w));
    }
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::constructLPF(T targetOccupancy,
        const MultidimArray<T>& lpf) {
    T iNewXdim = 1.0 / lpf.xdim;
    T sigma = targetOccupancy / 6; // So that from -targetOccupancy to targetOccupancy there is 6 sigma
    T K = -0.5 / (sigma * sigma);
    for (int i = STARTINGX(lpf); i <= FINISHINGX(lpf); ++i) {
        T w = i * iNewXdim;
        A1D_ELEM(lpf, i) = exp(K * (w * w));
    }
}


template<typename T>
void AProgMovieAlignmentCorrelation<T>::computeSizeFactor(T& targetOccupancy) {
    if (bin < 0) {
        targetOccupancy = 0.9; // Set to 1 if you want fmax maps onto 1/(2*newTs) // FIXME make function
        // Determine target size of the images
        newTs = targetOccupancy * maxFreq / 2;
        newTs = std::max(newTs, Ts);

        sizeFactor = Ts / newTs;
        std::cout << "Estimated binning factor = " << 1 / sizeFactor
                << std::endl;
    } else {
        newTs = bin * Ts;
        sizeFactor = 1.0 / bin;
        targetOccupancy = 2 * newTs / maxFreq;
    }
}

template<typename T>
T AProgMovieAlignmentCorrelation<T>::computeSizeFactor() {
    if (bin < 0) {
        T targetOccupancy = 0.9; // Set to 1 if you want fmax maps onto 1/(2*newTs)
        // Determine target size of the images
        newTs = targetOccupancy * maxFreq / 2;
        newTs = std::max(newTs, Ts);

        sizeFactor = Ts / newTs;
        std::cout << "Estimated binning factor = " << 1 / sizeFactor
                << std::endl;
    } else {
        newTs = bin * Ts;
        sizeFactor = 1.0 / bin;
    }
    return sizeFactor;
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::setNewDimensions(
        const MetaData& movie) {

    size_t Xdim, Ydim, Zdim, Ndim;
    getImageSize(movie, Xdim, Ydim, Zdim, Ndim);
    if (yDRcorner != -1) {
        Xdim = xDRcorner - xLTcorner + 1;
        Ydim = yDRcorner - yLTcorner + 1;
    }
    if (Zdim != 1)
        REPORT_ERROR(ERR_ARG_INCORRECT,
                "This program is meant to align 2D frames, not 3D");
    newXdim = (int(Xdim * sizeFactor) / 2) * 2; // we need odd size of the input, to be able to
    newYdim = (int(Ydim * sizeFactor) / 2) * 2; // compute FFT more efficiently (and e.g. perform shift by multiplication)
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
int AProgMovieAlignmentCorrelation<T>::findShiftsAndStore(MetaData& movie,
        Image<T>& dark, Image<T>& gain) {
    T targetOccupancy = 1.0; // Set to 1 if you want fmax maps onto 1/(2*newTs)

    computeSizeFactor(targetOccupancy);
    setNewDimensions(movie);
    // Construct 1D profile of the lowpass filter
    MultidimArray<T> lpf(newXdim / 2);
    constructLPFold(targetOccupancy, lpf);

    size_t N = nlast - nfirst + 1; // no of images to process
    Matrix2D<T> A(N * (N - 1) / 2, N - 1);
    Matrix1D<T> bX(N * (N - 1) / 2), bY(N * (N - 1) / 2);
    if (verbose)
        std::cout << "Loading frames ..." << std::endl;
    // Compute the Fourier transform of all input images
    loadData(movie, dark, gain, targetOccupancy, lpf);
    if (verbose)
        std::cout << "Computing shifts between frames ..." << std::endl;
    // Now compute all shifts
    computeShifts(N, bX, bY, A); // notice that the shifts bX and bY are scaled by the sizeFactor

    Matrix1D<T> shiftX, shiftY;
    solveEquationSystem(bX, bY, A, shiftX, shiftY);

    // Choose reference image as the minimax of shifts
    int bestIref = findReferenceImage(N, shiftX, shiftY);
//    storeRelativeShifts(bestIref, shiftX, shiftY, movie);
    return bestIref;
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
    movie.write((FileName) ("frameShifts@") + fnOut);
    if (bestIref >= 0) {
        MetaData mdIref;
        mdIref.setValue(MDL_REF, nfirst + bestIref, mdIref.addObject());
        mdIref.write((FileName) ("referenceFrame@") + fnOut, MD_APPEND);
    }
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::correctLoopIndices(
        const MetaData& movie) {
    nfirst = std::max(nfirst, 0);
    nfirstSum = std::max(nfirstSum, 0);
    if (nlast < 0)
        nlast = movie.size();

    if (nlastSum < 0)
        nlastSum = movie.size();
}

template<typename T>
void AProgMovieAlignmentCorrelation<T>::run() {
    show();
    // preprocess input data
    MetaData movie;
    readMovie(movie);
    correctLoopIndices(movie);

    Image<T> dark, gain;
    loadDarkCorrection(dark);
    loadGainCorrection(gain);

    AlignmentResult<T> globalAlignment;
    if (useInputShifts) {
        if (!movie.containsLabel(MDL_SHIFT_X)) {
            setZeroShift(movie);
        }
        // FIXME load from file
    } else {
        globalAlignment = computeGlobalAlignment(movie, dark, gain);
    }

    // FIXME if -o, store global alignment
//    core::optional<LocalAlignmentResult<T>> localAlignment;
    if (processLocalShifts) {
        computeLocalAlignment(movie, dark, gain);
    }

    size_t N, Ninitial;
    Image<T> initialMic, averageMicrograph;
    // Apply shifts and compute average
    applyShiftsComputeAverage(movie, dark, gain, initialMic, Ninitial, // FIXME implement for local alignment
            averageMicrograph, N);

    int bestIref = 0;
    storeResults(initialMic, Ninitial, averageMicrograph, N, movie, bestIref);
}

template class AProgMovieAlignmentCorrelation<float> ;
template class AProgMovieAlignmentCorrelation<double> ;
