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
#include <thread>
#include <atomic>
#include <limits>
#include <algorithm>
#include <numeric>
#include <set>


namespace Alignment {

void ProgAlignSpectral::defineParams() {
    addUsageLine("Find alignment of the experimental images in respect to a set of references");

    addParamsLine("   -i <md_file>                    : Metadata file with the experimental images. All images should have similar CTFs");
    addParamsLine("   -r <md_file>                    : Metadata file with the reference images");
    addParamsLine("   -o <md_file>                    : Resulting metadata file with the aligned images");
    addParamsLine("   --oroot <directory>             : Root directory for auxiliary output files");
    
    addParamsLine("   --bands <image_file>            : Image containing a band map");
    addParamsLine("   --weights <text_file>           : Text file containing space separated values used as weights for each band");
    addParamsLine("   --ctf <image_file>              : Image containing the CTF applied to the experimental images");
    addParamsLine("   --pca <stack_file>              : Stack containing the PCA basis");
    
    addParamsLine("   --rotations <rotations>         : Number of rotations to consider");
    addParamsLine("   --translations <transtions>     : Number of translations to consider");
    addParamsLine("   --maxShift <maxShift>           : Maximum translation in percentage relative to the image size");
    
    addParamsLine("   --thr <threads>                 : Number of threads");
    addParamsLine("   --memory <memory>               : Amount of memory used for storing references");
}

void ProgAlignSpectral::readParams() {
    fnExperimentalMetadata = getParam("-i");
    fnReferenceMetadata = getParam("-r");
    fnOutputMetadata = getParam("-o");
    fnOroot = getParam("--oroot");
    
    fnBands = getParam("--bands");
    fnWeights = getParam("--weights");
    fnCtf = getParam("--ctf");
    fnPca = getParam("--pca");
    
    nRotations = getIntParam("--rotations");
    nTranslations = getIntParam("--translations");
    maxShift = getDoubleParam("--maxShift");

    nThreads = getIntParam("--thr");
    maxMemory = getDoubleParam("--memory");
}

void ProgAlignSpectral::show() const {
    if (verbose < 1) return;

    std::cout << "Experimanetal metadata      : " << fnExperimentalMetadata << "\n";
    std::cout << "Reference metadata          : " << fnReferenceMetadata << "\n";
    std::cout << "Output metadata             : " << fnOutputMetadata << "\n";
    std::cout << "Output root                 : " << fnOroot << "\n";

    std::cout << "Band map                    : " << fnBands << "\n";
    std::cout << "Band weights                : " << fnWeights << "\n";
    std::cout << "CTF                         : " << fnCtf << "\n";
    std::cout << "PCA                         : " << fnPca << "\n";

    std::cout << "Rotations                   : " << nRotations << "\n";
    std::cout << "Translations                : " << nTranslations << "\n";
    std::cout << "Maximum shift               : " << maxShift << "%\n";

    std::cout << "Number of threads           : " << nThreads << "\n";
    std::cout << "Max memory                  : " << maxMemory << "\n";
    std::cout.flush();
}

void ProgAlignSpectral::run() {
    readInputMetadata();
    readBandMap();
    readBases();
    applyWeightsToBases();
    applyCtfToBases();
    generateTranslations();
    projectReferences();
    classifyExperimental();
    generateBandSsnr();
    generateOutput();
}





void ProgAlignSpectral::BandMap::flatten(   const MultidimArray<std::complex<Real>>& spectrum,
                                            std::vector<Matrix1D<Real>>& data,
                                            size_t image ) const
{
    data.resize(m_sizes.size());
    for(size_t i = 0; i < m_sizes.size(); ++i) {
        flatten(spectrum, i, data[i], image);
    }
}

void ProgAlignSpectral::BandMap::flattenOddEven(const MultidimArray<Real>& spectrum,
                                                std::vector<Matrix1D<Real>>& data,
                                                size_t oddEven,
                                                size_t image ) const
{
    data.resize(m_sizes.size());
    for(size_t i = 0; i < m_sizes.size(); ++i) {
        flattenOddEven(spectrum, i, data[i], oddEven, image);
    }
}

void ProgAlignSpectral::BandMap::flatten(   const MultidimArray<std::complex<Real>>& spectrum,
                                            size_t band,
                                            Matrix1D<Real>& data,
                                            size_t image ) const
{
    if (XSIZE(m_bands) != XSIZE(spectrum) || 
        YSIZE(m_bands) != YSIZE(spectrum) || 
        ZSIZE(m_bands) != ZSIZE(spectrum) ) 
    {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Spectrum and band map must coincide in shape");
    }

    data.resizeNoCopy(m_sizes[band]);
    auto* wrPtr = MATRIX1D_ARRAY(data);
    const auto* rdPtr = MULTIDIM_ARRAY(spectrum) + image*spectrum.zyxdim;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(m_bands) {
        if(DIRECT_MULTIDIM_ELEM(m_bands, n) == band) {
            *(wrPtr++) = rdPtr->real();
            *(wrPtr++) = rdPtr->imag();
        }
        ++rdPtr;
    }
    assert(wrPtr == MATRIX1D_ARRAY(data) + VEC_XSIZE(data));
}

void ProgAlignSpectral::BandMap::flattenOddEven(const MultidimArray<Real>& spectrum,
                                                size_t band,
                                                Matrix1D<Real>& data,
                                                size_t oddEven,
                                                size_t image ) const
{
    if (XSIZE(m_bands) != XSIZE(spectrum) || 
        YSIZE(m_bands) != YSIZE(spectrum) || 
        ZSIZE(m_bands) != ZSIZE(spectrum) ) 
    {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Spectrum and band map must coincide in shape");
    }

    data.resizeNoCopy(m_sizes[band]);
    auto* wrPtr = MATRIX1D_ARRAY(data);
    const auto* rdPtr = MULTIDIM_ARRAY(spectrum) + image*spectrum.zyxdim;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(m_bands) {
        if(DIRECT_MULTIDIM_ELEM(m_bands, n) == band) {
            *(wrPtr + oddEven) = *rdPtr;
            wrPtr += 2;
        }
        ++rdPtr;
    }
    assert(wrPtr == MATRIX1D_ARRAY(data) + VEC_XSIZE(data));
}



void ProgAlignSpectral::TranslationFilter::getTranslation(double& dx, double& dy) const {
    dx = m_dx;
    dy = m_dy;
}

void ProgAlignSpectral::TranslationFilter::operator()(  const MultidimArray<std::complex<Real>>& in, 
                                                        MultidimArray<std::complex<Real>>& out) const
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

void ProgAlignSpectral::TranslationFilter::computeCoefficients()
{
    // Shorthands
    auto& coeff = m_coefficients;
    size_t ny = YSIZE(coeff);
    size_t nx = (XSIZE(coeff) - 1) * 2;

    // Normalize the displacement
    const auto dy = m_dy / ny;
    const auto dx = m_dx / nx;

    // Compute the Fourier Transform of delta[i-y, j-x]
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(coeff) {
        const auto r2 = i*dy + j*dx; // Dot product of (dx, dy) and (j, i)
        const auto theta = (-2 * M_PI) * r2;
        DIRECT_A2D_ELEM(coeff, i, j) = std::polar(1.0, theta); //e^(i*theta)
    }
}



template<typename F>
void ProgAlignSpectral::ImageTransformer::forEachInPlaneTransform(  const MultidimArray<Real>& img,
                                                                    size_t nRotations,
                                                                    const std::vector<TranslationFilter>& translations,
                                                                    F&& func )
{
    // Perform all but the trivial rotations (0...360)
    const auto step = 360.0 / nRotations;
    for (size_t i = 1; i < nRotations; ++i) {
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
            std::bind(std::ref(func), std::placeholders::_1, rotation, std::placeholders::_2, std::placeholders::_3)
        );
    }

    // The first one (0 deg) does not need any rotate operation
    forEachInPlaneTranslation(
        img,
        translations,
        std::bind(std::forward<F>(func), std::placeholders::_1, 0.0, std::placeholders::_2, std::placeholders::_3)
    );
}

template<typename F>
void ProgAlignSpectral::ImageTransformer::forEachInPlaneTranslation(const MultidimArray<Real>& img,
                                                                    const std::vector<TranslationFilter>& translations,
                                                                    F&& func )
{
    // Compute the fourier transform of the input image
    m_fourier.FourierTransform(
        const_cast<MultidimArray<Real>&>(img), //HACK although it won't be written
        m_dft, 
        false
    );

    // Compute all translations of it
    for (const auto& translation : translations) {
        double sx, sy;
        translation.getTranslation(sx, sy);

        if(sx || sy) {
            // Perform the translation
            translation(m_dft, m_translatedDft);

            // Call the provided function
            func(m_translatedDft, sx, sy);
        } else {
            // Call the provided function with the DFT
            func(m_dft, 0.0, 0.0);
        }
    }
}

template<typename F>
void ProgAlignSpectral::ImageTransformer::forFourierTransform(  const MultidimArray<double>& img,
                                                                F&& func )
{
    m_fourier.FourierTransform(
        const_cast<MultidimArray<Real>&>(img), //HACK although it won't be written
        m_dft, 
        false
    );
    func(m_dft);
}





ProgAlignSpectral::BandMap::BandMap(const MultidimArray<int>& bands) 
    : m_bands(bands)
    , m_sizes(computeBandSizes(m_bands))
{
}

void ProgAlignSpectral::BandMap::reset(const MultidimArray<int>& bands) {
    m_bands = bands;
    m_sizes = computeBandSizes(m_bands);
}

const MultidimArray<int>& ProgAlignSpectral::BandMap::getBands() const {
    return m_bands;
}

const std::vector<size_t>& ProgAlignSpectral::BandMap::getBandSizes() const {
    return m_sizes;
}

std::vector<size_t> ProgAlignSpectral::BandMap::computeBandSizes(const MultidimArray<int>& bands) {
    std::vector<size_t> sizes;

    // Compute the band count
    const auto nBands = bands.computeMax() + 1;
    sizes.reserve(nBands);
    
    // Obtain the element count for each band
    for(size_t band = 0; band < nBands; ++band) {
        size_t count = 0;
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bands) {
            if (DIRECT_MULTIDIM_ELEM(bands, n) == band) {
                ++count;
            }
        }
        sizes.emplace_back(count*2); // *2 because complex numbers
    }
    assert(sizes.size() == nBands);

    return sizes;
}





ProgAlignSpectral::ReferencePcaProjections::ReferencePcaProjections(size_t nImages, 
                                                                    const std::vector<size_t>& bandSizes )
{
    reset(nImages, bandSizes);
}

void ProgAlignSpectral::ReferencePcaProjections::reset( size_t nImages, 
                                                        const std::vector<size_t>& bandSizes )
{
    m_projections.resize(bandSizes.size());
    for(size_t i = 0; i < bandSizes.size(); ++i) {
        m_projections[i].resizeNoCopy(nImages, bandSizes[i]);
    }

}
        
size_t ProgAlignSpectral::ReferencePcaProjections::getImageCount() const {
    return m_projections.front().Ydim();
}

size_t ProgAlignSpectral::ReferencePcaProjections::getBandCount() const {
    return m_projections.size();
}

size_t ProgAlignSpectral::ReferencePcaProjections::getComponentCount(size_t i) const {
    return MAT_XSIZE(m_projections.at(i));
}

void ProgAlignSpectral::ReferencePcaProjections::getPcaProjection(  size_t i, 
                                                                    std::vector<Matrix1D<Real>>& r) 
{
    r.resize(m_projections.size());
    for(size_t j = 0; j < m_projections.size(); ++j) {
        m_projections[j].getRowAlias(i, r[j]);
    }
}

size_t ProgAlignSpectral::ReferencePcaProjections::matchPcaProjection(  const std::vector<Matrix1D<Real>>& experimentalBands,
                                                                        const Matrix1D<Real>& weights) const 
{
    Matrix1D<Real> referenceBand;

    // Compare it with all the reference images
    auto best = getImageCount();
    auto bestScore = std::numeric_limits<Real>::infinity();
    for(size_t i = 0; i < getImageCount(); ++i) {
        // Add band by band
        double score = 0.0;
        for (size_t j = 0; j < getBandCount() && score < bestScore; ++j) {
            const auto& experimentalBand = experimentalBands[j];
            const_cast<Matrix2D<double>&>(m_projections[j]).getRowAlias(i, referenceBand);

            // Compute the difference between the bands and increment the score
            score += weights[j] * euclideanDistance2(experimentalBand, referenceBand);
        }

        // Update the score if necessary
        if (score < bestScore) {
            best = i;
            bestScore = score;
        }
    }

    if (best >= getImageCount()) {
        REPORT_ERROR(ERR_DEBUG_TEST,
            "Could not find a best match. This is probably due "
            "to the input being empty or the basis having NaNs"
        );
    }

    assert(best < getImageCount());
    return best;
}

size_t ProgAlignSpectral::ReferencePcaProjections::matchPcaProjectionBaB(   const std::vector<Matrix1D<Real>>& experimentalBands,
                                                                            const Matrix1D<Real>& weights ) const 
{
    Matrix1D<Real> referenceBand;

    // Compare band-by-band using the branch and bound approach
    auto best = getImageCount();
    auto bestScore = std::numeric_limits<double>::infinity();
    auto bestCandidate = 0; // Arbitrary
    std::vector<double> scores(getImageCount(), 0.0);

    for(size_t i = 0; i < getBandCount() && bestCandidate != best; ++i) {
        // Evaluate the "complete" score for the best candidate image
        for(size_t j = i; j < getBandCount() && scores[bestCandidate] < bestScore; ++j) {
            const auto& experimentalBand = experimentalBands[j];
            const_cast<Matrix2D<Real>&>(m_projections[j]).getRowAlias(bestCandidate, referenceBand);
            scores[bestCandidate] += VEC_ELEM(weights, j) * euclideanDistance2(experimentalBand, referenceBand);
        }

        // Update the best score
        if (scores[bestCandidate] < bestScore) {
            best = bestCandidate;
            bestScore = scores[bestCandidate];
        }

        // Determine the best candidate for the next iteration
        bestCandidate = best;
        auto bestCandidateScore = bestScore;
        const auto& experimentalBand = experimentalBands[i];
        for(size_t j = 0; j < getImageCount(); ++j) {
            // Only consider this image if it is below the threshold
            if(scores[j] < bestScore) {
                const_cast<Matrix2D<Real>&>(m_projections[i]).getRowAlias(j, referenceBand);
                scores[j] += VEC_ELEM(weights, i) * euclideanDistance2(experimentalBand, referenceBand);
                if(scores[j] < bestCandidateScore) {
                    bestCandidate = j;
                    bestCandidateScore = scores[j];
                }
            }
        }
    }

    // Update the score with the last best candidate
    best = bestCandidate;
    bestScore = scores[bestCandidate];

    if (best >= getImageCount()) {
        REPORT_ERROR(ERR_DEBUG_TEST,
            "Could not find a best match. This is probably due "
            "to the input being empty or the basis having NaNs"
        );
    }

    assert(best < getImageCount());
    return best;
}





ProgAlignSpectral::ReferenceMetadata::ReferenceMetadata(size_t rowId, 
                                                        double rotation, 
                                                        double shiftx, 
                                                        double shifty )
    : m_rowId(rowId)
    , m_rotation(rotation)
    , m_shiftX(shiftx)
    , m_shiftY(shifty)
{
}

void ProgAlignSpectral::ReferenceMetadata::setRowId(size_t rowId) {
    m_rowId = rowId;
}

size_t ProgAlignSpectral::ReferenceMetadata::getRowId() const {
    return m_rowId;
}

void ProgAlignSpectral::ReferenceMetadata::setRotation(double rotation) {
    m_rotation = rotation;
}

double ProgAlignSpectral::ReferenceMetadata::getRotation() const {
    return m_rotation;
}

void ProgAlignSpectral::ReferenceMetadata::setShiftX(double sx) {
    m_shiftX = sx;
}

double ProgAlignSpectral::ReferenceMetadata::getShiftX() const {
    return m_shiftX;
}

void ProgAlignSpectral::ReferenceMetadata::setShiftY(double sy) {
    m_shiftY = sy;
}

double ProgAlignSpectral::ReferenceMetadata::getShiftY() const {
    return m_shiftY;
}





void ProgAlignSpectral::readInputMetadata() {
    m_mdExperimental.read(fnExperimentalMetadata);
    m_mdReference.read(fnReferenceMetadata);
}

void ProgAlignSpectral::readBandMap() {
    Image<int> map;
    map.read(fnBands);
    m_bandMap.reset(map());
}

void ProgAlignSpectral::readBases() {
    Image<Real> bases;
    bases.read(fnPca);

    // Generate the Basis matrices with the maximum possible size
    const auto nMaxProjections = NSIZE(bases())/2;
    const auto& bandSizes = m_bandMap.getBandSizes();
    m_bases.resize(bandSizes.size());
    for(size_t i = 0; i < bandSizes.size(); ++i) {
        m_bases[i].resizeNoCopy(bandSizes[i], nMaxProjections);
    }

    // Convert all elements into matrix form
    std::vector<Matrix1D<Real>> columns;
    for(size_t i = 0; i < nMaxProjections; ++i) {
        m_bandMap.flattenOddEven(bases(), columns, 0, i*2 + 0);
        m_bandMap.flattenOddEven(bases(), columns, 1, i*2 + 1);

        for(size_t j = 0; j < columns.size(); ++j) {
            auto& basis = m_bases[j];
            const auto& column = columns[j];
            if(i < MAT_XSIZE(basis)) {
                if(column.sum2() > 0) {
                    // Non-zero column. Write it
                    basis.setCol(i, column);
                } else {
                    // Zero column. End. Shrink the matrix. 
                    // This will prevent further evaluation
                    basis.resize(MAT_YSIZE(basis), i);
                }
            }
        }
    }
}

void ProgAlignSpectral::applyWeightsToBases() {
    // Read
    Image<Real> weights;
    weights.read(fnWeights);

    // Remove the symmetric part
    if(XSIZE(weights()) > XSIZE(m_bandMap.getBands())) {
        weights().resize(
            NSIZE(weights()), ZSIZE(weights()), YSIZE(weights()), 
            XSIZE(m_bandMap.getBands())
        );
    }

    // Write the same in for real and imaginary values
    std::vector<Matrix1D<Real>> bandWeights;
    m_bandMap.flattenOddEven(weights(), bandWeights, 0, 0);
    m_bandMap.flattenOddEven(weights(), bandWeights, 1, 0);

    // Scale the basis accordingly
    for(size_t i = 0; i < m_bases.size(); ++i) {
        multiplyToAllColumns(m_bases[i], bandWeights[i]);
    } 
}

void ProgAlignSpectral::applyCtfToBases() {
    Image<Real> ctf;
    ctf.read(fnCtf);

    // Remove the symmetric part
    if(XSIZE(ctf()) > XSIZE(m_bandMap.getBands())) {
        ctf().resize(
            NSIZE(ctf()), ZSIZE(ctf()), YSIZE(ctf()), 
            XSIZE(m_bandMap.getBands())
        );
    }

    std::vector<Matrix1D<Real>> bands;
    m_bandMap.flattenOddEven(ctf(), bands, 0, 0);
    m_bandMap.flattenOddEven(ctf(), bands, 1, 0);

    // Start from the clean bases
    m_ctfBases = m_bases;
    for(size_t i = 0; i < m_ctfBases.size(); ++i) {
        multiplyToAllColumns(m_ctfBases[i], bands[i]);
    }
}

void ProgAlignSpectral::generateTranslations() {
    // TODO determine optimally based on a heat map
    size_t nx, ny, nz, nn;
    getImageSize(m_mdExperimental, nx, ny, nz, nn);

    m_translations = computeTranslationFiltersRectangle(
        nx, ny,
        nTranslations, maxShift
    );
}

void ProgAlignSpectral::projectReferences() {
    // Allocate space
    std::vector<size_t> projectionSizes(m_bases.size());
    for(size_t i = 0; i < m_bases.size(); ++i) projectionSizes[i] = MAT_XSIZE(m_bases[i]);
    m_references.reset(
        m_mdReference.size() * nRotations * m_translations.size(),
        projectionSizes
    );
    m_referenceData.resize(m_references.getImageCount());
    
    struct ThreadData {
        Image<Real> image;
        ImageTransformer transformer;
        std::vector<Matrix1D<Real>> bandCoefficients;
        std::vector<Matrix1D<Real>> bandProjections;
    };

    // Create a lambda to run in parallel
    std::vector<ThreadData> threadData(nThreads);
    const auto func = [this, &threadData] (size_t threadId, size_t i, const MDRowVec& row) {
        auto& data = threadData[threadId];

        // Read an image from disk
        const size_t rowId = row.getValue<size_t>(MDL_OBJID);
        const FileName& fnImage = row.getValue<String>(MDL_IMAGE);
        data.image.read(fnImage);

        // For each in-plane transformation train the PCA
        const auto offset = i * nRotations * m_translations.size();
        data.transformer.forEachInPlaneTransform(
            data.image(),
            nRotations,
            m_translations,
            [this, &data, rowId, counter=offset] (const auto& x, auto rot, auto sx, auto sy) mutable {
                // Obtain the resulting memory area
                const auto index = counter++;
                m_references.getPcaProjection(index, data.bandProjections);

                // Project the image
                m_bandMap.flatten(x, data.bandCoefficients);
                //m_pca.project(data.bandCoefficients, data.bandProjections); //TODO replace

                // Write the metadata
                m_referenceData[index] = ReferenceMetadata(
                    rowId, 
                    (rot > 180) ? rot - 360 : rot, 
                    -sx, -sy //Opposite transform
                );
            }
        );
    };

    processRowsInParallel(m_mdReference, func, threadData.size());
}

void ProgAlignSpectral::classifyExperimental() {
    // Initialize the classification vector with invalid 
    // data and the appropiate size
    // TODO uncomment
    /*m_classification.clear();
    m_classification.resize(
        m_mdExperimental.size(), // size
        m_references.getImageCount() // invalid value
    );
    m_ssnr.resizeNoCopy(m_classification.size(), m_pca.getBandCount());*/

    struct ThreadData {
        Image<Real> image;
        FourierTransformer fourier;
        MultidimArray<std::complex<Real>> spectrum;
        std::vector<Matrix1D<Real>> bandCoefficients;
        std::vector<Matrix1D<Real>> bandProjections;
        std::vector<Matrix1D<Real>> referenceBandProjections;
        Matrix1D<Real> ssnr;
    };

    // Create a lambda to run in parallel
    std::vector<ThreadData> threadData(nThreads);
    const auto func = [this, &threadData] (size_t threadId, size_t i, const MDRowVec& row) {
        auto& data = threadData[threadId];

        // Read an image
        const FileName& fnImage = row.getValue<String>(MDL_IMAGE);
        data.image.read(fnImage);

        // Project the image
        data.fourier.FourierTransform(data.image(), data.spectrum, false);
        m_bandMap.flatten(data.spectrum, data.bandCoefficients);

        //m_pca.project(data.bandCoefficients, data.bandProjections); //TODO replace

        // Compare the projection to find a match //TODO uncomment
        //const auto classification = m_references.matchPcaProjection(data.bandProjections, m_weights);
        //m_classification[i] = classification; //TODO uncomment

        // Compute the SSNR TODO uncomment
        //m_references.getPcaProjection(classification, data.referenceBandProjections);
        //m_ssnr.getRowAlias(i, data.ssnr);
        //calculateBandSsnr(data.referenceBandProjections, data.bandProjections, data.ssnr);
    };

    processRowsInParallel(m_mdExperimental, func, threadData.size());
}

void ProgAlignSpectral::generateBandSsnr() {
    //TODO uncomment
    /*// Calculate the average SSNR on each band
    Matrix1D<Real> totalSsnr;
    m_ssnr.colSum(totalSsnr);
    totalSsnr /= MAT_YSIZE(m_ssnr);

    // Write it to disk
    totalSsnr.write(m_parameters.fnOroot + "ssnr");
    m_ssnr.write(m_parameters.fnOroot + "image_ssnr");*/
}

void ProgAlignSpectral::generateOutput() {
    auto mdOut = m_mdExperimental; //TODO overwrite

    // Modify the experimental data
    //assert(mdOut.size() == m_classification.size()); //TODO uncomment
    for(size_t i = 0; i < mdOut.size(); ++i) {
        auto row = mdOut.getRowVec(mdOut.getRowId(i));
        //updateRow(row, m_classification[i]); //TODO uncomment
    }

    // Write the data
    mdOut.write(fnOutputMetadata);
}





void ProgAlignSpectral::updateRow(MDRowVec& row, size_t matchIndex) const {
    // Obtain the metadata
    const auto& data = m_referenceData[matchIndex];
    const auto refRow = m_mdReference.getRowVec(data.getRowId());
    const auto expRow = m_mdExperimental.getRowVec(row.getValue<size_t>(MDL_OBJID));

    // Calculate the values to be written
    const auto rot = refRow.getValue<double>(MDL_ANGLE_ROT);
    const auto tilt = refRow.getValue<double>(MDL_ANGLE_TILT);
    const auto psi = data.getRotation();
    const auto shiftX = data.getShiftX();
    const auto shiftY = data.getShiftY();

    // Write the old shift and pose values to the second MD labels
    if (expRow.containsLabel(MDL_ANGLE_ROT)) {
        const auto oldRot = expRow.getValue<double>(MDL_ANGLE_ROT);
        row.setValue(MDL_ANGLE_ROT2, oldRot);
    } 
    if (expRow.containsLabel(MDL_ANGLE_TILT)) {
        const auto oldTilt = expRow.getValue<double>(MDL_ANGLE_TILT);
        row.setValue(MDL_ANGLE_TILT2, oldTilt);
    } 
    if (expRow.containsLabel(MDL_ANGLE_PSI)) {
        const auto oldPsi = expRow.getValue<double>(MDL_ANGLE_PSI);
        row.setValue(MDL_ANGLE_PSI2, oldPsi);
    } 
    if (expRow.containsLabel(MDL_SHIFT_X)) {
        const auto oldShiftX = expRow.getValue<double>(MDL_SHIFT_X);
        const auto deltaShiftX = shiftX - oldShiftX;
        row.setValue(MDL_SHIFT_X2, oldShiftX);
        row.setValue(MDL_SHIFT_X_DIFF, deltaShiftX);
    } 
    if (expRow.containsLabel(MDL_SHIFT_Y)) {
        const auto oldShiftY = expRow.getValue<double>(MDL_SHIFT_Y);
        const auto deltaShiftY = shiftY - oldShiftY;
        row.setValue(MDL_SHIFT_Y2, oldShiftY);
        row.setValue(MDL_SHIFT_Y_DIFF, deltaShiftY);
    } 

    // Write the new pose
    row.setValue(MDL_ANGLE_ROT, rot);
    row.setValue(MDL_ANGLE_TILT, tilt);
    row.setValue(MDL_ANGLE_PSI, psi);
    row.setValue(MDL_SHIFT_X, shiftX);
    row.setValue(MDL_SHIFT_Y, shiftY);

    // Write the reference image
    row.setValue(MDL_IMAGE_REF, refRow.getValue<std::string>(MDL_IMAGE));
}



template<typename F>
void ProgAlignSpectral::processRowsInParallel(  const MetaDataVec& md, 
                                                F&& func, 
                                                size_t nThreads ) 
{
    if(nThreads < 1) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "There needs to be at least one thread");
    }

    std::atomic<size_t> currRowNum(0);
    const auto mdSize = md.size();

    // Create a worker function which atomically aquires a row and
    // dispatches the provided function
    const auto workerFunc = [&md, &func, &currRowNum, mdSize] (size_t threadId) {
        auto rowNum = currRowNum++;

        while(rowNum < mdSize) {
            // Process a row
            const auto row = md.getRowVec(md.getRowId(rowNum));
            func(threadId, rowNum, row);

            // Update the progress bar only from the first thread 
            // due to concurrency issues
            if (threadId == 0) {
                progress_bar(rowNum+1);
            }

            // Aquire the next row
            rowNum = currRowNum++;
        }
    };

    // Initialzie the progress bar
    init_progress_bar(mdSize);

    // Create some workers
    std::vector<std::thread> threads;
    threads.reserve(nThreads - 1);
    for(size_t i = 1; i < nThreads; ++i) {
        threads.emplace_back(workerFunc, i);
    }

    // Use the local thread
    workerFunc(0);

    //Wait for the others to finish
    for (auto& thread : threads) {
        thread.join();
    }

    // Set the progress bar as finished
    progress_bar(mdSize);
}





size_t ProgAlignSpectral::getImageProjectionSize(const std::vector<Matrix2D<Real>>& bases) {
    size_t coeffs = 0;
    for(const auto& m : bases) {
        coeffs += MAT_XSIZE(m);
    }
    return coeffs * sizeof(Real);
}

std::vector<ProgAlignSpectral::TranslationFilter> 
ProgAlignSpectral::computeTranslationFiltersRectangle(  const size_t nx, 
                                                        const size_t ny,
                                                        const size_t nTranslations,
                                                        const double maxShift )
{
    std::vector<ProgAlignSpectral::TranslationFilter> result;
    result.reserve(nTranslations*nTranslations);
    
    if(nTranslations > 1) {
        const auto n = std::max(nx, ny);
        const auto maxRadius = n*maxShift;
        const auto step = 2*maxRadius / (nTranslations - 1);
        
        // Create a grid with all the considered shifts. Use
        // a set in order to avoid duplicates in case the step
        // is smaller than 1px
        std::set<std::array<double, 2>> shifts;
        for (size_t i = 0; i < nTranslations; ++i) {
            const auto dx = std::round(i*step - maxRadius);
            for (size_t j = 0; j < nTranslations; ++j) {
                const auto dy = std::round(j*step - maxRadius);
                shifts.insert({dx, dy});
            }
        }

        // Transform all the points into an array of translation filters
        std::transform(
            shifts.cbegin(), shifts.cend(),
            std::back_inserter(result),
            [nx, ny] (const std::array<double, 2>& shift) -> TranslationFilter {
                return TranslationFilter(shift[0], shift[1], nx, ny);
            }
        );
    } else {
        // Only one translation, use the identity filter
        result.emplace_back(0, 0, nx, ny);
    }

    return result;
}

std::vector<ProgAlignSpectral::TranslationFilter> 
ProgAlignSpectral::computeTranslationFiltersSunflower(  const size_t nx, 
                                                        const size_t ny,
                                                        const size_t nTranslations,
                                                        const double maxShift )
{
    std::vector<ProgAlignSpectral::TranslationFilter> result;
    result.reserve(nTranslations);

    if(nTranslations > 1) {
        // Shorthands
        const auto n = std::max(nx, ny);
        const auto maxRadius = n*maxShift;
        constexpr auto PHI = (std::sqrt(5)+1)/2;
        constexpr auto PHI2 = PHI*PHI;
        
        // Create a sunflower with all the considered shifts. Use
        // a set in order to avoid duplicates in case the step
        // is smaller than 1px
        std::set<std::array<double, 2>> shifts;
        for (size_t i = 0; i < nTranslations; ++i) {
            const auto r = maxRadius * std::sqrt(i+0.5) / std::sqrt(nTranslations-0.5);
            const auto theta = M_2_PI * i / PHI2;
            const auto point = std::polar(r, theta);
            shifts.insert({std::round(point.real()), std::round(point.imag())});
        }

        // Transform all the points into an array of translation filters
        std::transform(
            shifts.cbegin(), shifts.cend(),
            std::back_inserter(result),
            [nx, ny] (const std::array<double, 2>& shift) -> TranslationFilter {
                return TranslationFilter(shift[0], shift[1], nx, ny);
            }
        );
    } else {
        // Only one translation, use the identity filter
        result.emplace_back(0, 0, nx, ny);
    }

    return result;
}

void ProgAlignSpectral::calculateBandSsnr(  const std::vector<Matrix1D<Real>>& reference, 
                                            const std::vector<Matrix1D<Real>>& experimental, 
                                            Matrix1D<Real>& ssnr )
{

    assert(experimental.size() == reference.size());
    assert(VEC_XSIZE(ssnr) == reference.size());

    for(size_t i = 0; i < reference.size(); ++i) {
        const auto& referenceBand = reference[i];
        const auto& experimentalBand = experimental[i];
        const auto noiseEnergy = euclideanDistance2(referenceBand, experimentalBand);
        //const auto signalEnergy = referenceBand.sum2();
        //VEC_ELEM(ssnr, i) = signalEnergy / noiseEnergy;
        VEC_ELEM(ssnr, i) = VEC_XSIZE(experimentalBand) / noiseEnergy;
    }
}

}