/***************************************************************************
 *
 * Authors:    Martin Horacek (horacek1martin@gmail.com)
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
#include "volume_halves_restoration_gpu.h"

template< typename T >
void ProgVolumeHalvesRestorationGpu<T>::readParams() {
    readFilenames();
    readDenoisingParams();
    readDeconvolutionParams();
    readFilterBankParams();
    readDifferenceParams();
    readMaskParams();
}

template< typename T >
void ProgVolumeHalvesRestorationGpu<T>::readFilenames() {
    fnV1 = getParam("--i1");
    fnV2 = getParam("--i2");
    fnRoot = getParam("--oroot");
}

template< typename T >
void ProgVolumeHalvesRestorationGpu<T>::readDenoisingParams() {
    const int iters = getIntParam("--denoising");
    if (iters < 0) {
        REPORT_ERROR(ERR_ARG_BADCMDLINE, "`denoising N` has to be non-negative integer");
    }

    builder.setDenoising(iters);
}

template< typename T >
void ProgVolumeHalvesRestorationGpu<T>::readDeconvolutionParams() {
    const int iters = getIntParam("--deconvolution");
    const T sigma = getDoubleParam("--deconvolution", 1);
    const T lambda = getDoubleParam("--deconvolution", 2);
    if (iters < 0) {
        REPORT_ERROR(ERR_ARG_BADCMDLINE, "`deconvolution N` has to be non-negative integer");
    }

    builder.setDeconvolution(iters, sigma, lambda);
}

template< typename T >
void ProgVolumeHalvesRestorationGpu<T>::readFilterBankParams() {
    const T bankStep = getDoubleParam("--filterBank", 0);
    const T bankOverlap = getDoubleParam("--filterBank", 1);
    const int weightFun = getIntParam("--filterBank", 2);
    const T weightPower = getDoubleParam("--filterBank", 3);
    if (bankStep < 0 || bankStep > 0.5001) {
        REPORT_ERROR(ERR_ARG_BADCMDLINE, "`filterBank step` parameter has to be in interval [0, 0.5].");
    }
    if (bankOverlap < 0 || bankOverlap > 1.001) {
        REPORT_ERROR(ERR_ARG_BADCMDLINE, "`filterBank overlap` parameter has to be in interval [0, 1]");
    }

    if (weightFun < 0 || weightFun > 3) {
        REPORT_ERROR(ERR_ARG_BADCMDLINE, "`filterBank weightFun` parameter has to be 0, 1 or 2");
    }

    builder.setFilterBank(bankStep, bankOverlap, weightFun, weightPower);
}

template< typename T >
void ProgVolumeHalvesRestorationGpu<T>::readDifferenceParams() {
    const int iters = getIntParam("--difference");
    const T Kdiff = getDoubleParam("--difference", 1);
    if (iters < 0) {
        REPORT_ERROR(ERR_ARG_BADCMDLINE, "`difference N` has to be non-negative integer");
    }

    builder.setDifference(iters, Kdiff);
}

template< typename T >
void ProgVolumeHalvesRestorationGpu<T>::readMaskParams() {
    if (checkParam("--mask")) {
        mask.fn_mask = getParam("--mask");
        mask.mask_type = "binary_file";
        mask.type = READ_BINARY_MASK;

        if (checkParam("--center")) {
            mask.x0 = getDoubleParam("--center", 0);
            mask.y0 = getDoubleParam("--center", 1);
            mask.z0 = getDoubleParam("--center", 2);
        }
    }
}

template< typename T >
void ProgVolumeHalvesRestorationGpu<T>::show(const VolumeHalvesRestorator<T>& restorator) {
    if (!verbose)
        return;
    std::cout
    << "Input/Ouput filenames:" << std::endl
    << "    Volume1:  " << fnV1 << std::endl
    << "    Volume2:  " << fnV2 << std::endl
    << "    Rootname: " << fnRoot << std::endl
    ;
    std::cout << restorator;

    mask.show();
}

template< typename T >
void ProgVolumeHalvesRestorationGpu<T>::defineParams() {
    addUsageLine("Given two halves of a volume (and an optional mask), produce a better estimate of the volume underneath");
    addParamsLine("   --i1 <volume1>              : First half");
    addParamsLine("   --i2 <volume2>              : Second half");
    addParamsLine("  [--oroot <root=\"volumeRestored\">] : Output rootname");
    addParamsLine("  [--denoising <N=0>]          : Number of iterations of denoising in real space");
    addParamsLine("  [--deconvolution <N=0> <sigma0=0.2> <lambda=0.001>]   : Number of iterations of deconvolution in Fourier space, initial sigma and lambda");
    addParamsLine("  [--filterBank <step=0> <overlap=0.5> <weightFun=1> <weightPower=3>] : Frequency step for the filter bank (typically, 0.01; between 0 and 0.5)");
    addParamsLine("                                        : filter overlap is between 0 (no overlap) and 1 (full overlap)");
    addParamsLine("                                : Weight function (0=mean, 1=min, 2=mean*diff");
    addParamsLine("  [--difference <N=0> <K=1.5>]  : Number of iterations of difference evaluation in real space");
    addParamsLine("  [--mask <binary_file>]        : Read from file and cast to binary");
    addParamsLine("  [--center <x0=0> <y0=0> <z0=0>]           : Mask center");
}

template< typename T >
void ProgVolumeHalvesRestorationGpu<T>::run() {
    builder.setVerbosity(verbose);
    auto restorator = builder.build();

    show(restorator);

    readData();
    restorator.apply(V1(), V2(), maskData);

    saveResults(restorator);
}

template< typename T >
void ProgVolumeHalvesRestorationGpu<T>::readData() {
    V1.read(fnV1);
    V2.read(fnV2);
    V1().setXmippOrigin();
    V2().setXmippOrigin();

    checkInputDimensions();

    if (!mask.fn_mask.isEmpty()) {
        mask.generate_mask();
        maskData = mask.get_binary_mask().data;
    }
}

template< typename T >
void ProgVolumeHalvesRestorationGpu<T>::checkInputDimensions() {
    if (XSIZE(V1()) != XSIZE(V2()) || YSIZE(V1()) != YSIZE(V2())
        || ZSIZE(V1()) != ZSIZE(V2())) {
        REPORT_ERROR(ERR_MATRIX_DIM, "Input volumes have different dimensions");
    }
    if (maskData) {
        auto& maskArray = mask.get_binary_mask();
        if (XSIZE(V1()) != XSIZE(maskArray) || YSIZE(V1()) != YSIZE(maskArray)
            || ZSIZE(V1()) != ZSIZE(maskArray)) {
            REPORT_ERROR(ERR_MATRIX_DIM, "Mask and input volumes have different dimensions");
        }
    }
}

template< typename T >
void ProgVolumeHalvesRestorationGpu<T>::saveResults(const VolumeHalvesRestorator<T>& restorator) {
    V1() = restorator.getReconstructedVolume1();
    saveImage(V1, "_restored1.vol");
    V1() = restorator.getReconstructedVolume2();
    saveImage(V1, "_restored2.vol");
    V1() = restorator.getFilterBankVolume();
    saveImage(V1, "_filterBank.vol");
    V1() = restorator.getDeconvolvedS();
    saveImage(V1, "_deconvolved.vol");
    V1() = restorator.getConvolvedS();
    saveImage(V1, "_convolved.vol");
    V1() = restorator.getAverageDifference();
    saveImage(V1, "_avgDiff.vol");
}

template< typename T >
void ProgVolumeHalvesRestorationGpu<T>::saveImage(Image<T>& image, std::string&& filename) {
    if (XSIZE(image()) != 0) {
        image.write(fnRoot + filename);
    }
}

template class ProgVolumeHalvesRestorationGpu<double>;
template class ProgVolumeHalvesRestorationGpu<float>;