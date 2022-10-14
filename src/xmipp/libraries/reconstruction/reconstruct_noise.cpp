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

    each_image_produces_an_output = true;
    XmippMetadataProgram::defineParams();

    addParamsLine("   -r <md_file>                    : Reference volume");
    
    addParamsLine("   --padding <padding>             : Padding factor");
    addParamsLine("   --max_resolution <resolution>   : Resolution limit");

    addParamsLine("   [--useCTF]                      : Consider the CTF when comparing images");
    m_ctfDesc.defineParams(this);

    addParamsLine("   --thr <threads>                 : Number of threads");

}

void ProgReconstructNoise::readParams() {
    XmippMetadataProgram::readParams();
    fnReferenceVolume = getParam("-r");

    paddingFactor = getDoubleParam("--padding");
    maxResolution = getDoubleParam("--max_resolution");

    useCtf = checkParam("--useCTF");
    if(useCtf) m_ctfDesc.readParams(this);
}

void ProgReconstructNoise::show() const {
    if (verbose < 1) return;

    std::cout << "Reference volume            : " << fnReferenceVolume << "\n";

    std::cout << "Use CTF                     : " << useCtf << "\n";
    if(useCtf) {
        std::cout << m_ctfDesc; 
    }

    std::cout.flush();
}

void ProgReconstructNoise::preProcess() {
    readReference();
    createReferenceProjector();

    // Initialize all the averages to zeros with the appropiate size
    m_avgExpImagePsd.initZeros(m_referenceProjector->projectionFourier);
    m_avgRefImagePsd.initZeros(m_referenceProjector->projectionFourier);
    m_avgNoisePsd.initZeros(m_referenceProjector->projectionFourier);
    m_avgCtfPsd.initZeros(m_referenceProjector->projectionFourier);
}

void ProgReconstructNoise::postProcess() {
    m_avgExpImagePsd /= mdInSize;
    m_avgRefImagePsd /= mdInSize;
    m_avgNoisePsd /= mdInSize;
    if(useCtf) m_avgCtfPsd /= mdInSize;
}

void ProgReconstructNoise::writeOutput() {
    XmippMetadataProgram::writeOutput();
    Image<Real>(m_avgExpImagePsd).write(oroot + "avgExperimentalPsd.stk");
    Image<Real>(m_avgRefImagePsd).write(oroot + "avgReferencePsd.stk");
    Image<Real>(m_avgNoisePsd).write(oroot + "avgNoisePsd.stk");
    if(useCtf) Image<Real>(m_avgCtfPsd).write(oroot + "avgCtfPsd.stk");
}

void ProgReconstructNoise::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut) {
    // Read the image
    m_experimental.read(fnImg);
    
    // Read the metadata
    const auto rot = rowIn.getValue<double>(MDL_ANGLE_ROT);
    const auto tilt = rowIn.getValue<double>(MDL_ANGLE_TILT);
    const auto psi = rowIn.getValue<double>(MDL_ANGLE_PSI);
    const auto shiftX = rowIn.getValue<double>(MDL_SHIFT_X);
    const auto shiftY = rowIn.getValue<double>(MDL_SHIFT_Y);

    // Generate the CTF image if necessary
    if(useCtf) {
        m_ctfDesc.readFromMdRow(rowIn); 
        m_ctfDesc.produceSideInfo();
        m_ctfDesc.generateCTF(m_experimental(), m_ctf);
        updatePsd(m_avgCtfPsd, m_ctf);
    }

    // Project the volume
    const auto& proj = projectReference(rot, tilt, psi, useCtf ? &m_ctf : nullptr);
    updatePsd(m_avgCtfPsd, proj);

    // Compute the Fourier transform of the image
    m_fourier.FourierTransform(
        m_experimental(),
        m_experimentalFourier,
        false
    );
    shiftSpectra(m_experimentalFourier, shiftX, shiftY);
    updatePsd(m_avgExpImagePsd, m_experimentalFourier);

    //Compute the noise in place
    assert(proj.sameShape(m_experimentalFourier));
    m_experimentalFourier -= proj;
    updatePsd(m_avgNoisePsd, m_experimentalFourier);
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