/***************************************************************************
 *
 * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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

#include "reconstruct_noise.h"

#include <core/transformations.h>
#include <core/metadata_extension.h>
#include <core/xmipp_fft.h>

#include <cassert>

void ProgReconstructNoise::defineParams() {
    addUsageLine("Compute the spectral noise of an image set based on its reconstruction");

    addParamsLine("   -i <md_file>                    : Metadata file with the experimental images");
    addParamsLine("   -r <md_file>                    : Reference volume");
    addParamsLine("   --oroot <directory>             : Output directory");
    
    addParamsLine("   --padding <padding>             : Padding factor");
    addParamsLine("   --max_resolution <resolution>   : Resolution limit");

    addParamsLine("   [--useCTF]                      : Consider the CTF when comparing images");
    m_ctfDesc.defineParams(this);

    addParamsLine("   --thr <threads>                 : Number of threads");

}

void ProgReconstructNoise::readParams() {
    fnExperimentalMetadata = getParam("-i");
    fnReferenceVolume = getParam("-r");
    fnOutputRoot = getParam("--oroot");

    paddingFactor = getDoubleParam("--padding");
    maxResolution = getDoubleParam("--max_resolution");

    useCtf = checkParam("--useCTF");
    if(useCtf) m_ctfDesc.readParams(this);

    nThreads = getIntParam("--thr");
}

void ProgReconstructNoise::show() const {
    if (verbose < 1) return;

    std::cout << "Experimanetal metadata      : " << fnExperimentalMetadata << "\n";
    std::cout << "Reference volume            : " << fnReferenceVolume << "\n";
    std::cout << "Output root                 : " << fnOutputRoot << "\n";

    std::cout << "Use CTF                     : " << useCtf << "\n";
    if(useCtf) {
        std::cout << m_ctfDesc; 
    }

    std::cout << "Number of threads           : " << nThreads << "\n";
    std::cout.flush();
}

void ProgReconstructNoise::run() {
    readReference();
    createReferenceProjector();
    computeNoise();
}
    
    
    
    
    
    
    
    
void ProgReconstructNoise::readReference() {
    m_reference.read(fnReferenceVolume);
    m_reference().setXmippOrigin();
}

void ProgReconstructNoise::createReferenceProjector() {
    m_referenceProjector = std::make_unique<FourierProjector>(
        m_reference(), paddingFactor, maxResolution, 
        xmipp_transformation::BSPLINE3
    );
}

void ProgReconstructNoise::computeNoise() {
    MetaDataVec mdExperimental;
    mdExperimental.read(fnExperimentalMetadata);

    FourierTransformer fourierTransformer;
    Image<Real> experimentalImage;
    MultidimArray<Real> ctf, averageImagePsd, averageCtfPsd, averageNoisePsd;
    MultidimArray<Complex> experimentalImageFourier;
    averageImagePsd.initZeros(m_referenceProjector->projectionFourier);
    averageNoisePsd.initZeros(m_referenceProjector->projectionFourier);
    averageCtfPsd.initZeros(m_referenceProjector->projectionFourier);
    for(const auto& row : mdExperimental) {
        // Read the metadata
        const auto rot = row.getValue<double>(MDL_ANGLE_ROT);
        const auto tilt = row.getValue<double>(MDL_ANGLE_TILT);
        const auto psi = row.getValue<double>(MDL_ANGLE_PSI);
        const auto shiftX = row.getValue<double>(MDL_SHIFT_X);
        const auto shiftY = row.getValue<double>(MDL_SHIFT_Y);
        experimentalImage.read(row.getValue<String>(MDL_IMAGE));

        // Generate the CTF image if necessary
        if(useCtf) {
            m_ctfDesc.readFromMdRow(row); 
            m_ctfDesc.produceSideInfo();
            m_ctfDesc.generateCTF(experimentalImage(), ctf);
            updatePsd(averageCtfPsd, ctf);
        }

        // Project the volume
        const auto& proj = projectReference(rot, tilt, psi, useCtf ? &ctf : nullptr);

        // Compute the Fourier transform of the image
        fourierTransformer.FourierTransform(
            experimentalImage(),
            experimentalImageFourier,
            false
        );
        shiftSpectra(experimentalImageFourier, shiftX, shiftY);
        updatePsd(averageImagePsd, experimentalImageFourier);

        //Compute the error in place
        assert(proj.sameShape(experimentalImageFourier));
        experimentalImageFourier -= proj;
        updatePsd(averageNoisePsd, experimentalImageFourier);
    }

    averageImagePsd /= mdExperimental.size();
    averageNoisePsd /= mdExperimental.size();
    averageCtfPsd /= mdExperimental.size();
    Image<Real>(averageImagePsd).write(fnOutputRoot + "averageImagePsd.stk");
    Image<Real>(averageNoisePsd).write(fnOutputRoot + "averageNoisePsd.stk");
    Image<Real>(averageCtfPsd).write(fnOutputRoot + "averageCtfPsd.stk");
}

const MultidimArray<ProgReconstructNoise::Complex>& 
ProgReconstructNoise::projectReference( double rot, double tilt, double psi, 
                                        const MultidimArray<double>* ctf )
{
    m_referenceProjector->projectToFourier(
        rot, tilt, psi,
        ctf
    );
    return m_referenceProjector->projectionFourier;
}

void ProgReconstructNoise::shiftSpectra(MultidimArray<Complex>& spectra, double shiftX, double shiftY) {
    const int ny = YSIZE(spectra);
    const int ny_2 = ny / 2;
    const auto ny_inv = 1.0 / ny;
    const int nx_2 = (XSIZE(spectra) - 1);
    const int nx = nx_2 * 2;
    const auto nx_inv = 1.0 / nx;

    // Normalize the displacement
    const auto dy = (-2 * M_PI) * shiftY;
    const auto dx = (-2 * M_PI) * shiftX;

    // Compute the Fourier Transform of delta[i-y, j-x]
    double fy, fx;
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(spectra) {
        // Convert the indices to fourier coefficients
        FFT_IDX2DIGFREQ_FAST(static_cast<int>(i), ny, ny_2, ny_inv, fy);
        FFT_IDX2DIGFREQ_FAST(static_cast<int>(j), nx, nx_2, nx_inv, fx);

        const auto theta = fy*dy + fx*dx; // Dot product of (dx, dy) and (j, i)
        DIRECT_A2D_ELEM(spectra, i, j) *= std::polar(1.0, theta); //e^(i*theta)
    }
}

void ProgReconstructNoise::updatePsd(   MultidimArray<Real>& psd, 
                                        const MultidimArray<Real>& diff )
{
    assert(XSIZE(psd) <= XSIZE(diff));
    assert(YSIZE(psd) <= YSIZE(diff));

    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(psd) {
        const auto& x = DIRECT_A2D_ELEM(diff, i, j);
        DIRECT_A2D_ELEM(psd, i, j) += x*x;
    }
}

void ProgReconstructNoise::updatePsd(   MultidimArray<Real>& psd, 
                                        const MultidimArray<Complex>& diff )
{
    assert(XSIZE(psd) <= XSIZE(diff));
    assert(YSIZE(psd) <= YSIZE(diff));

    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(psd) {
        const auto& x = DIRECT_A2D_ELEM(diff, i, j);
        DIRECT_A2D_ELEM(psd, i, j) += x.real()*x.real() + x.imag()*x.imag();
    }
}