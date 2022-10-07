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

#include <data/ctf.h>

#include <core/transformations.h>
#include <core/metadata_extension.h>

#include <cassert>

void ProgReconstructNoise::defineParams() {
    addUsageLine("Compute the spectral noise of an image set based on its reconstruction");

    addParamsLine("   -i <md_file>                    : Metadata file with the experimental images");
    addParamsLine("   -r <md_file>                    : Reference volume");
    addParamsLine("   --oroot <directory>             : Output directory");
    
    addParamsLine("   --padding <padding>             : Padding factor");
    addParamsLine("   --max_resolution <resolution>   : Resolution limit");

    addParamsLine("   --thr <threads>                 : Number of threads");
}

void ProgReconstructNoise::readParams() {
    fnExperimentalMetadata = getParam("-i");
    fnReferenceVolume = getParam("-r");
    fnOutputRoot = getParam("--oroot");

    paddingFactor = getDoubleParam("--padding");
    maxResolution = getDoubleParam("--max_resolution");

    nThreads = getIntParam("--thr");
}

void ProgReconstructNoise::show() const {
    if (verbose < 1) return;

    std::cout << "Experimanetal metadata      : " << fnExperimentalMetadata << "\n";
    std::cout << "Reference volume            : " << fnReferenceVolume << "\n";
    std::cout << "Output root                 : " << fnOutputRoot << "\n";

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

    Image<Real> experimentalImage;
    CTFDescription ctf;
    MultidimArray<Real> ctfImage, averageCtfVariance;
    FourierTransformer fourierTransformer;
    MultidimArray<Complex> experimentalImageFourier;
    MultidimArray<Real> noiseVariance, averageNoiseVariance;
    experimentalImageFourier.initZeros(m_referenceProjector->projectionFourier);
    averageCtfVariance.initZeros(experimentalImageFourier);
    averageNoiseVariance.initZeros(experimentalImageFourier);
    for(const auto& row : mdExperimental) {
        // Read the metadata
        const auto rot = row.getValue<double>(MDL_ANGLE_ROT);
        const auto tilt = row.getValue<double>(MDL_ANGLE_TILT);
        const auto psi = row.getValue<double>(MDL_ANGLE_PSI);
        const auto shiftX = row.getValue<double>(MDL_SHIFT_X);
        const auto shiftY = row.getValue<double>(MDL_SHIFT_Y);
        experimentalImage.read(row.getValue<String>(MDL_IMAGE));
        ctf.readFromMdRow(row); ctf.produceSideInfo();

        // Project the volume
        ctf.generateCTF(experimentalImage(), ctfImage);
        const auto& proj = projectReference(rot, tilt, psi, &ctfImage);

        // Compute the Fourier transform of the image
        fourierTransformer.FourierTransform(
            experimentalImage(),
            experimentalImageFourier,
            false
        );
        shiftSpectra(experimentalImageFourier, shiftX, shiftY);

        //Compute the error in place
        assert(proj.sameShape(experimentalImageFourier));
        experimentalImageFourier -= proj;
        experimentalImageFourier.getAbs2(noiseVariance);

        // Accumulate
        //ctfImage *= ctfImage; //Square to obtain the variance (mean is zero)
        //averageCtfVariance += ctfImage;
        averageNoiseVariance += noiseVariance;
    }

    Image<Real>(ctfImage).write(fnOutputRoot + "averageCtf.stk");
    Image<Real>(averageNoiseVariance).write(fnOutputRoot + "averageNoise.stk");
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
    size_t ny = YSIZE(spectra);
    size_t nx = (XSIZE(spectra) - 1) * 2;

    // Normalize the displacement
    const auto dy = (-2 * M_PI) * shiftX / ny;
    const auto dx = (-2 * M_PI) * shiftY / nx;

    // Compute the Fourier Transform of delta[i-y, j-x]
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(spectra) {
        const auto theta = i*dy + j*dx; // Dot product of (dx, dy) and (j, i)
        DIRECT_A2D_ELEM(spectra, i, j) *= std::polar(1.0, theta); //e^(i*theta)
    }
}