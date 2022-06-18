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

#include "align_spectral.h"

#include <core/transformations.h>
#include <core/metadata_extension.h>

#include <iostream>
#include <cmath>
#include <cassert>


namespace Alignment {

void ProgAlignSpectral::defineParams() {
    addUsageLine("Find alignment of the experimental images in respect to a set of references");

    addParamsLine("   -i <md_file>                    : Metadata file with the experimental images");
    addParamsLine("   -r <md_file>                    : Metadata file with the reference images");
    addParamsLine("   -o <md_file>                    : Resulting metadata file with the aligned images");
    
    addParamsLine("   --rotations <rotations>         : Number of rotations to consider");
    addParamsLine("   --translations <transtions>     : Number of translations to consider");
    addParamsLine("   --maxShift <maxShift>           : Maximum translation in percentage relative to the image size");
}

void ProgAlignSpectral::readParams() {
    auto& param = m_parameters;

    param.fnExperimental = getParam("-i");
    param.fnReference = getParam("-r");
    param.fnOutput = getParam("-o");

    param.nRotations = getIntParam("--rotations");
    param.nTranslations = getIntParam("--translations");
    param.maxShift = getDoubleParam("--maxShift") / 100;
}

void ProgAlignSpectral::show() const {
    if (verbose < 1) return;
    auto& param = m_parameters;

    std::cout << "Experimanetal metadata      : " << param.fnExperimental << "\n";
    std::cout << "Reference metadata          : " << param.fnReference << "\n";
    std::cout << "Output metadata             : " << param.fnOutput << "\n";

    std::cout << "Rotations                   : " << param.nRotations << "\n";
    std::cout << "Translations                : " << param.nTranslations << "\n";
    std::cout << "Maximum shift               : " << param.maxShift*100 << "%\n";

    std::cout.flush();
}

void ProgAlignSpectral::run() {
    readInput();
    learnReferences();
    learnExperimental();
    projectReferences();
    projectExperimental();
}



void ProgAlignSpectral::TranslationFilter::computeCoefficients()
{
    // Shorthands
    auto& coeff = m_coefficients;
    size_t ny = YSIZE(coeff);
    size_t nx = (XSIZE(coeff) - 1) * 2;

    // Normalize the displacement and magnitude
    const auto dy = m_dy / ny;
    const auto dx = m_dx / nx;
    const auto mag = 1.0 / (nx*ny);

    // Compute the Fourier Transform of delta[i-y, j-x]
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(coeff) {
        const auto r2 = i*dy + j*dx; // Dot product of (dx, dy) and (j, i)
        const auto theta = (-2 * M_PI) * r2;
        DIRECT_A2D_ELEM(coeff, i, j) = std::polar(mag, theta); //e^(i*theta)
    }
}

void ProgAlignSpectral::TranslationFilter::operator()(  const MultidimArray<std::complex<double>>& in, 
                                                        MultidimArray<std::complex<double>>& out) const
{
    // Shorthands
    auto& coeff = m_coefficients;

    // Input and filter should be equal
    assert(SAME_SHAPE2D(in, coeff));

    // Reshape the output to be the same as the input and the filter
    out.resizeNoCopy(in);
    
    // Multiply coefficient by coefficient
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(coeff) {
        const auto& x = DIRECT_A2D_ELEM(in, i, j);
        const auto& h = DIRECT_A2D_ELEM(coeff, i, j);
        auto& y = DIRECT_A2D_ELEM(out, i, j);
        y = x*h;
    }
}



template<typename F>
void ProgAlignSpectral::ImageTransformer::forEachInPlaneTransform(  const MultidimArray<double>& img,
                                                                    size_t nRotations,
                                                                    const std::vector<TranslationFilter>& translations,
                                                                    F&& func )
{
    const auto step = 360.0 / nRotations;

    for (size_t i = 0; i < nRotations; ++i) {
        // Rotate the input image into the cached image
        const auto rotation = i*step;
        rotate(
            xmipp_transformation::LINEAR, 
            m_rotated, img,
            rotation, 'Z',
            xmipp_transformation::WRAP
        );

        // Apply all translations to the rotated image
        forEachInPlaneTranslation(
            m_rotated,
            translations,
            std::forward<F>(func)
        );
    }

}

template<typename F>
void ProgAlignSpectral::ImageTransformer::forEachInPlaneTranslation(const MultidimArray<double>& img,
                                                                    const std::vector<TranslationFilter>& translations,
                                                                    F&& func )
{
    // Compute the fourier transform of the input image
    m_fourier.FourierTransform(
        const_cast<MultidimArray<double>&>(img), //HACK although it won't be written
        m_dft, 
        false
    );

    // Compute all translations of it
    for (const auto& translation : translations) {
        // Perform the translation
        translation(m_dft, m_translatedDft);

        // Call the provided function
        std::forward<F>(func)(m_translatedDft);
    }
}



void ProgAlignSpectral::readInput() {
    readMetadata(m_parameters.fnExperimental, m_mdExperimental);
    readMetadata(m_parameters.fnReference, m_mdReference);
}

void ProgAlignSpectral::calculateTranslations() {
    // Shorthands
    const auto& nTranslations = m_parameters.nTranslations;
    const auto& maxShift = m_parameters.maxShift;
    const auto& md = m_mdReference;
    
    // Determine the image size
    size_t nx, ny, nz, nn;
    getImageSize(md, nx, ny, nz, nn);

    // Pre-compute the translation filters in the freq. domain
    m_translations = computeTranslationFilters(nx, ny, nTranslations, maxShift);
}

void ProgAlignSpectral::learnReferences() {
    // Shorthands
    const auto& md = m_mdReference;
    const auto& nRotations = m_parameters.nRotations;
    const auto& translations = m_translations;

    Image<double> image;
    ImageTransformer transformer;
    for(const auto& row : md) {
        const FileName& fnImage = row.getValue<String>(MDL_IMAGE);
        readImage(fnImage, image);
        transformer.forEachInPlaneTransform(
            image(),
            nRotations,
            translations,
            [] (const auto& x) {
                // TODO learn
            }
        );
    }
}

void ProgAlignSpectral::learnExperimental() {

}

void ProgAlignSpectral::projectReferences() {

}

void ProgAlignSpectral::projectExperimental() {

}



void ProgAlignSpectral::readMetadata(const FileName& fn, MetaDataVec& result) {
    result.read(fn);
    result.removeDisabled();
}

void ProgAlignSpectral::readImage(const FileName& fn, Image<double>& result) {
    result.read(fn);
}



std::vector<ProgAlignSpectral::TranslationFilter> 
ProgAlignSpectral::computeTranslationFilters(   size_t nx, 
                                                size_t ny,
                                                size_t nTranslations,
                                                double maxShift )
{
    std::vector<ProgAlignSpectral::TranslationFilter> result;
    result.reserve(nTranslations);

    // Shorthands
    const auto n = std::max(nx, ny);
    const auto maxRadius = n*maxShift;
    constexpr auto PHI = (std::sqrt(5)+1)/2;
    constexpr auto PHI2 = PHI*PHI;

    // Compute some evenly spaced points using a spiral
    for (size_t i = 0; i < nTranslations; ++i) {
        const auto r = maxRadius * std::sqrt(i+0.5) / std::sqrt(nTranslations-0.5);
        const auto theta = M_2_PI * i / PHI2;
        const auto point = std::polar(r, theta);
        result.emplace_back(std::round(point.real()), std::round(point.imag()), nx, ny);
    }

    return result;
}

}