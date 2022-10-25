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
#include <core/xmipp_fft.h>

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
    generateRotations();
    alignImages();
    generateOutput();
}





void ProgAlignSpectral::BandMap::flatten(   const MultidimArray<Complex>& spectrum,
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

void ProgAlignSpectral::BandMap::flatten(   const MultidimArray<Complex>& spectrum,
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

void ProgAlignSpectral::TranslationFilter::operator()(  const MultidimArray<Complex>& in, 
                                                        MultidimArray<Complex>& out) const
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
    auto& coeff = m_coefficients;
    const int ny = YSIZE(coeff);
    const int ny_2 = ny / 2;
    const auto ny_inv = 1.0 / ny;
    const int nx_2 = (XSIZE(coeff) - 1);
    const int nx = nx_2 * 2;
    const auto nx_inv = 1.0 / nx;

    // Normalize the displacement
    const auto dy = (-2 * M_PI) * m_dy;
    const auto dx = (-2 * M_PI) * m_dx;

    // Compute the Fourier Transform of delta[i-y, j-x]
    double fy, fx;
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(coeff) {
        // Convert the indices to fourier coefficients
        FFT_IDX2DIGFREQ_FAST(static_cast<int>(i), ny, ny_2, ny_inv, fy);
        FFT_IDX2DIGFREQ_FAST(static_cast<int>(j), nx, nx_2, nx_inv, fx);

        const auto theta = fy*dy + fx*dx; // Dot product of (dx, dy) and (j, i)
        DIRECT_A2D_ELEM(coeff, i, j) = std::polar(1.0, theta); //e^(i*theta)
    }
}





ProgAlignSpectral::Rotation::Rotation(double angle)
    : m_angle(angle)
{
    computeMatrix();
}

void ProgAlignSpectral::Rotation::operator()(   const MultidimArray<Real>& in, 
                                                MultidimArray<Real>& out ) const
{
    applyGeometry(
        xmipp_transformation::LINEAR, 
        out, in,
        m_matrix, xmipp_transformation::IS_NOT_INV, 
        xmipp_transformation::WRAP, 0.0
    );
}

double ProgAlignSpectral::Rotation::getAngle() const {
    return m_angle;
}

void ProgAlignSpectral::Rotation::computeMatrix() {
    rotation2DMatrix(m_angle, m_matrix);
}





template<typename F>
void ProgAlignSpectral::ImageTransformer::forEachInPlaneTransform(  const MultidimArray<Real>& img,
                                                                    const std::vector<Rotation>& rotations,
                                                                    const std::vector<TranslationFilter>& translations,
                                                                    F&& func )
{
    forEachInPlaneRotation(
        img, rotations,
        [this, &func, &translations] (const MultidimArray<Complex>& dft, double angle) {
            forEachInPlaneTranslation(
                dft, translations,
                [&func, angle] (const MultidimArray<Complex>& dft, double sx, double sy) {
                    func(dft, angle, sx, sy);
                }
            );
        }
    );
}

template<typename F>
void ProgAlignSpectral::ImageTransformer::forEachInPlaneRotation(   const MultidimArray<Real>& img,
                                                                    const std::vector<Rotation>& rotations,
                                                                    F&& func )
{
    for(const auto& rotation : rotations) {
        const auto angle = rotation.getAngle();
        if(angle) {
            // Apply the rotation
            rotation(img, m_rotated);

            // Compute the fourier transform
            m_fourierRotated.setReal(m_rotated);
            m_fourierRotated.FourierTransform();

            func(m_fourierRotated.fFourier, angle);
        } else {
            // Compute the fourier transform of the clean image
            m_fourierClean.setReal(const_cast<MultidimArray<Real>&>(img)); // HACK although it wont be written
            m_fourierClean.FourierTransform();

            func(m_fourierClean.fFourier, angle);
        }
    }
}
template<typename F>
void ProgAlignSpectral::ImageTransformer::forEachInPlaneTranslation(const MultidimArray<Real>& img,
                                                                    const std::vector<TranslationFilter>& translations,
                                                                    F&& func )
{
    m_fourierClean.setReal(const_cast<MultidimArray<Real>&>(img)); // HACK although it wont be written
    m_fourierClean.FourierTransform();
    forEachInPlaneTranslation(m_fourierClean.fFourier, translations, std::forward<F>(func));
}

template<typename F>
void ProgAlignSpectral::ImageTransformer::forEachInPlaneTranslation(const MultidimArray<Complex>& img,
                                                                    const std::vector<TranslationFilter>& translations,
                                                                    F&& func )
{
    // Compute all translations of it
    for (const auto& translation : translations) {
        double sx, sy;
        translation.getTranslation(sx, sy);

        if(sx || sy) {
            // Perform the translation
            translation(img, m_translated);

            // Call the provided function
            func(m_translated, sx, sy);
        } else {
            // Call the provided function with the DFT
            func(img, 0.0, 0.0);
        }
    }
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

size_t ProgAlignSpectral::ReferencePcaProjections::matchPcaProjection(const std::vector<Matrix1D<Real>>& experimentalBands, Real& bestDistance) const 
{
    Matrix1D<Real> referenceBand;

    // Compare it with all the reference images
    auto best = getImageCount();
    for(size_t i = 0; i < getImageCount(); ++i) {
        // Add band by band
        double distance = 0.0;
        for (size_t j = 0; j < getBandCount() && distance < bestDistance; ++j) {
            const auto& experimentalBand = experimentalBands[j];
            const_cast<Matrix2D<double>&>(m_projections[j]).getRowAlias(i, referenceBand);

            // Compute the difference between the bands and increment the distance
            distance += euclideanDistance2(experimentalBand, referenceBand);
        }

        // Update the distance if necessary
        if (distance < bestDistance) {
            best = i;
            bestDistance = distance;
        }
    }

    return best;
}

size_t ProgAlignSpectral::ReferencePcaProjections::matchPcaProjectionBaB(const std::vector<Matrix1D<Real>>& experimentalBands, Real& bestDistance) const 
{
    Matrix1D<Real> referenceBand;

    // Compare band-by-band using the branch and bound approach
    auto best = getImageCount();
    auto bestCandidate = 0; // Arbitrary
    std::vector<double> distances(getImageCount(), 0.0);

    for(size_t i = 0; i < getBandCount() && bestCandidate != best; ++i) {
        // Evaluate the "complete" distance for the best candidate image
        for(size_t j = i; j < getBandCount() && distances[bestCandidate] < bestDistance; ++j) {
            const auto& experimentalBand = experimentalBands[j];
            const_cast<Matrix2D<Real>&>(m_projections[j]).getRowAlias(bestCandidate, referenceBand);
            distances[bestCandidate] += euclideanDistance2(experimentalBand, referenceBand);
        }

        // Update the best distance
        if (distances[bestCandidate] < bestDistance) {
            best = bestCandidate;
            bestDistance = distances[bestCandidate];
        }

        // Determine the best candidate for the next iteration
        bestCandidate = best;
        auto bestCandidateDistance = bestDistance;
        const auto& experimentalBand = experimentalBands[i];
        for(size_t j = 0; j < getImageCount(); ++j) {
            // Only consider this image if it is below the threshold
            if(distances[j] < bestDistance) {
                const_cast<Matrix2D<Real>&>(m_projections[i]).getRowAlias(j, referenceBand);
                distances[j] += euclideanDistance2(experimentalBand, referenceBand);
                if(distances[j] < bestCandidateDistance) {
                    bestCandidate = j;
                    bestCandidateDistance = distances[j];
                }
            }
        }
    }

    // Update the best distance
    if (distances[bestCandidate] < bestDistance) {
        best = bestCandidate;
        bestDistance = distances[bestCandidate];
    }

    assert(best < getImageCount());
    return best;
}





ProgAlignSpectral::ReferenceMetadata::ReferenceMetadata(size_t rowId, 
                                                        double rotation, 
                                                        double shiftx, 
                                                        double shifty,
                                                        double distance )
    : m_rowId(rowId)
    , m_rotation(rotation)
    , m_shiftX(shiftx)
    , m_shiftY(shifty)
    , m_distance(distance)
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

void ProgAlignSpectral::ReferenceMetadata::setDistance(double distance) {
    m_distance = distance;
}

double ProgAlignSpectral::ReferenceMetadata::getDistance() const {
    return m_distance;
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

    // Write the projection sizes
    getProjectionSizes(m_bases, m_projectionSizes);
}

void ProgAlignSpectral::applyWeightsToBases() {
    Image<Real> weights;
    weights.read(fnWeights);
    removeFourierSymmetry(weights());
    multiplyBases(m_bases, weights());
}

void ProgAlignSpectral::applyCtfToBases() {
    Image<Real> ctf;
    ctf.read(fnCtf);
    m_ctfBases = m_bases;
    removeFourierSymmetry(ctf());
    multiplyBases(m_ctfBases, ctf());
}

void ProgAlignSpectral::generateTranslations() {
    // TODO determine optimally based on a heat map
    size_t nx, ny, nz, nn;
    getImageSize(m_mdExperimental, nx, ny, nz, nn);

    m_translations = computeTranslationFiltersRectangle(
        nx, ny,
        nTranslations, maxShift/100.0
    );
}

void ProgAlignSpectral::generateRotations() {
    m_rotations.clear();
    m_rotations.reserve(nRotations);

    const auto step = 360.0 / nRotations;
    for(size_t i = 0; i < nRotations; ++i) {
        const auto angle = step * i;
        m_rotations.emplace_back(angle);
    }
}

void ProgAlignSpectral::alignImages() {
    // Convert sizes
    constexpr size_t megabytes2bytes = 1024*1024; 
    const auto memorySizeBytes = static_cast<size_t>(megabytes2bytes*maxMemory);
    const auto imageProjCoeffCount = std::accumulate(m_projectionSizes.cbegin(), m_projectionSizes.cend(), 0UL);
    const auto imageProjSizeBytes = imageProjCoeffCount * sizeof(Real);

    // Decide if it is convenient to split transformations
    const auto splitTransform = m_mdExperimental.size() < m_mdReference.size()*m_rotations.size();
    const auto nRefTransform = splitTransform ? m_rotations.size() : m_rotations.size()*m_translations.size();

    // Decide the batch sizes
    const auto batchSize = getBatchSize(memorySizeBytes, imageProjSizeBytes, nRefTransform);
    const auto batchCount = (m_mdReference.size() + batchSize - 1) / batchSize; // Round up

    // Initialize the classification vector with invalid 
    // data and the appropiate size
    m_classification.clear();
    m_classification.resize(m_mdExperimental.size());

    std::cout << "Applying rotations to reference images\n";
    std::cout << "Applying shifts to " << (splitTransform ? "experimental" : "references") << " images\n";

    // Process in batches
    for(size_t i = 0; i < batchCount; ++i) {
        const auto start = i*batchSize;
        const auto count = std::min(batchSize, m_mdReference.size() - start);
        std::cout << "Reference batch " << i+1 << "/" << batchCount << " (" << count << " images)\n";
        
        std::cout << "Projecting reference batch\n";
        if(splitTransform) projectReferencesRot(start, count); else projectReferencesRotShift(start, count);
        std::cout << std::endl;

        std::cout << "Classifying reference batch\n";
        if(splitTransform) classifyExperimentalShift(); else classifyExperimental();
        std::cout << std::endl;
    }
}

void ProgAlignSpectral::generateOutput() {
    auto mdOut = m_mdExperimental;

    // Modify the experimental data
    assert(mdOut.size() == m_classification.size());
    for(size_t i = 0; i < mdOut.size(); ++i) {
        auto row = mdOut.getRowVec(mdOut.getRowId(i));
        updateRow(row, m_classification[i]);
    }

    // Write the data
    mdOut.write(fnOutputMetadata);
}



void ProgAlignSpectral::projectReferences(  size_t start, 
                                            size_t count ) 
{
    // Allocate space
    const auto nTransform = 1UL;
    m_references.reset(count * nTransform, m_projectionSizes);
    m_referenceData.resize(m_references.getImageCount());
    
    struct ThreadData {
        Image<Real> image;
        FourierTransformer fourier;
        std::vector<Matrix1D<Real>> bandCoefficients;
        std::vector<Matrix1D<Real>> bandProjections;
    };

    // Create a lambda to run in parallel
    std::vector<ThreadData> threadData(nThreads);
    const auto func = [this, &threadData, start] (size_t threadId, size_t i, const MDRowVec& row) {
        auto& data = threadData[threadId];
        const auto index = i - start;

        // Read an image from disk
        const size_t rowId = row.getValue<size_t>(MDL_OBJID);
        const FileName& fnImage = row.getValue<String>(MDL_IMAGE);
        data.image.read(fnImage);

        // Compute the spectrum of the image
        data.fourier.setReal(data.image());
        data.fourier.FourierTransform();

        // Obtain the corresponding memory area
        m_references.getPcaProjection(index, data.bandProjections);

        // Project the image
        m_bandMap.flatten(data.fourier.fFourier, data.bandCoefficients);
        project(m_ctfBases, data.bandCoefficients, data.bandProjections);

        // Write the metadata
        m_referenceData[index] = ReferenceMetadata(rowId, 0.0, 0.0, 0.0); 
    };

    processRowsInParallel(m_mdReference, func, threadData.size(), start, count);
}

void ProgAlignSpectral::projectReferencesRot(   size_t start, 
                                                size_t count ) 
{
    // Allocate space
    const auto nTransform = m_rotations.size();
    m_references.reset(count * nTransform, m_projectionSizes);
    m_referenceData.resize(m_references.getImageCount());
    
    struct ThreadData {
        Image<Real> image;
        ImageTransformer transformer;
        std::vector<Matrix1D<Real>> bandCoefficients;
        std::vector<Matrix1D<Real>> bandProjections;
    };

    // Create a lambda to run in parallel
    std::vector<ThreadData> threadData(nThreads);
    const auto func = [this, &threadData, start, nTransform] (size_t threadId, size_t i, const MDRowVec& row) {
        auto& data = threadData[threadId];

        // Read an image from disk
        const size_t rowId = row.getValue<size_t>(MDL_OBJID);
        const FileName& fnImage = row.getValue<String>(MDL_IMAGE);
        data.image.read(fnImage);

        // Project all the rotations
        auto counter = nTransform * (i - start);
        data.transformer.forEachInPlaneRotation(
            data.image(), m_rotations,
            [this, rowId, &data, &counter] (const MultidimArray<Complex>& spectrum, double angle) {
                const auto index = counter++;

                // Obtain the corresponding memory area
                m_references.getPcaProjection(index, data.bandProjections);

                // Project the image
                m_bandMap.flatten(spectrum, data.bandCoefficients);
                project(m_ctfBases, data.bandCoefficients, data.bandProjections);

                // Write the metadata
                m_referenceData[index] = ReferenceMetadata(rowId, standardizeAngle(angle)); 
            }
        );
        assert(counter == nTransform * (i - start + 1)); // Ensure all has been written
    };

    processRowsInParallel(m_mdReference, func, threadData.size(), start, count);
}

void ProgAlignSpectral::projectReferencesRotShift(  size_t start, 
                                                    size_t count ) 
{
    // Allocate space
    const auto nTransform = m_rotations.size()*m_translations.size();
    m_references.reset(count * nTransform, m_projectionSizes);
    m_referenceData.resize(m_references.getImageCount());
    
    struct ThreadData {
        Image<Real> image;
        ImageTransformer transformer;
        std::vector<Matrix1D<Real>> bandCoefficients;
        std::vector<Matrix1D<Real>> bandProjections;
    };

    // Create a lambda to run in parallel
    std::vector<ThreadData> threadData(nThreads);
    const auto func = [this, &threadData, start, nTransform] (size_t threadId, size_t i, const MDRowVec& row) {
        auto& data = threadData[threadId];

        // Read an image from disk
        const size_t rowId = row.getValue<size_t>(MDL_OBJID);
        const FileName& fnImage = row.getValue<String>(MDL_IMAGE);
        data.image.read(fnImage);

        // Project all the rotations
        auto counter = nTransform * (i - start);
        data.transformer.forEachInPlaneTransform(
            data.image(), m_rotations, m_translations,
            [this, rowId, &data, &counter] (const MultidimArray<Complex>& spectrum, double angle, double sx, double sy) {
                const auto index = counter++;

                // Obtain the corresponding memory area
                m_references.getPcaProjection(index, data.bandProjections);

                // Project the image
                m_bandMap.flatten(spectrum, data.bandCoefficients);
                project(m_ctfBases, data.bandCoefficients, data.bandProjections);

                // Write the metadata
                m_referenceData[index] = ReferenceMetadata(rowId, standardizeAngle(angle), -sx, -sy); 
            }
        );
        assert(counter == nTransform * (i - start + 1)); // Ensure all has been written
    };

    processRowsInParallel(m_mdReference, func, threadData.size(), start, count);
}

void ProgAlignSpectral::classifyExperimental() {
    struct ThreadData {
        Image<Real> image;
        FourierTransformer fourier;
        std::vector<Matrix1D<Real>> bandCoefficients;
        std::vector<Matrix1D<Real>> bandProjections;
    };

    // Create a lambda to run in parallel
    std::vector<ThreadData> threadData(nThreads);
    const auto func = [this, &threadData] (size_t threadId, size_t i, const MDRowVec& row) {
        auto& data = threadData[threadId];

        // Read an image
        const FileName& fnImage = row.getValue<String>(MDL_IMAGE);
        data.image.read(fnImage);

        // Compute the fourier transform
        data.fourier.setReal(data.image());
        data.fourier.FourierTransform();

        // Project the image
        m_bandMap.flatten(data.fourier.fFourier, data.bandCoefficients);
        project(m_bases, data.bandCoefficients, data.bandProjections);

        // Compare the projection to find a match
        auto& classification = m_classification[i];
        auto distance = classification.getDistance();
        const auto match = m_references.matchPcaProjection(data.bandProjections, distance);
        if(match < m_references.getImageCount()) {
            classification = m_referenceData[match];
            classification.setDistance(distance);
        }
    };

    processRowsInParallel(m_mdExperimental, func, threadData.size());
}

void ProgAlignSpectral::classifyExperimentalShift() {
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

        // Read an image
        const FileName& fnImage = row.getValue<String>(MDL_IMAGE);
        data.image.read(fnImage);

        // Compare against all the references for each translation
        data.transformer.forEachInPlaneTranslation(
            data.image(), m_translations,
            [this, &data, i] (const MultidimArray<Complex>& spectrum, double sx, double sy) {
                // Project the spectrum
                m_bandMap.flatten(spectrum, data.bandCoefficients);
                project(m_bases, data.bandCoefficients, data.bandProjections);

                // Compare the projection to find a match
                auto& classification = m_classification[i];
                auto distance = classification.getDistance();
                const auto match = m_references.matchPcaProjection(data.bandProjections, distance);
                if(match < m_references.getImageCount()) {
                    const auto rowId = m_referenceData[match].getRowId();
                    const auto angle = m_referenceData[match].getRotation();
                    classification = ReferenceMetadata(rowId, angle, sx, sy, distance);
                }
                assert(classification.getDistance() == distance);
            }
        );
    };

    processRowsInParallel(m_mdExperimental, func, threadData.size());
}





void ProgAlignSpectral::removeFourierSymmetry(MultidimArray<Real>& spectrum) const {
    if(XSIZE(spectrum) > XSIZE(m_bandMap.getBands())) {
        spectrum.resize(
            NSIZE(spectrum), ZSIZE(spectrum), YSIZE(spectrum), 
            XSIZE(m_bandMap.getBands())
        );
    }
}

void ProgAlignSpectral::multiplyBases(  std::vector<Matrix2D<Real>>& bases,
                                        const MultidimArray<Real>& spectrum ) const
{
    // Write the same in for real and imaginary values
    // as multiplying a complex number by a real one is
    // equivalent to scaling both components by the same
    // factor
    std::vector<Matrix1D<Real>> bands;
    m_bandMap.flattenOddEven(spectrum, bands, 0, 0);
    m_bandMap.flattenOddEven(spectrum, bands, 1, 0);

    // Scale the basis accordingly
    for(size_t i = 0; i < bases.size(); ++i) {
        multiplyToAllColumns(bases[i], bands[i]);
    } 
}

void ProgAlignSpectral::updateRow(MDRowVec& row, const ReferenceMetadata& data) const {
    // Obtain the metadata
    const auto refRow = m_mdReference.getRowVec(data.getRowId());
    const auto expRow = m_mdExperimental.getRowVec(row.getValue<size_t>(MDL_OBJID));

    // Calculate the values to be written
    const auto rot = refRow.getValue<double>(MDL_ANGLE_ROT);
    const auto tilt = refRow.getValue<double>(MDL_ANGLE_TILT);
    const auto psi = data.getRotation();
    const auto shiftX = data.getShiftX();
    const auto shiftY = data.getShiftY();
    const auto score = data.getDistance();

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
    row.setValue(MDL_SCORE_BY_ALIGNABILITY_NOISE, score);

    // Write the reference image
    row.setValue(MDL_IMAGE_REF, refRow.getValue<std::string>(MDL_IMAGE));
}



template<typename F>
void ProgAlignSpectral::processRowsInParallel(  const MetaDataVec& md, 
                                                F&& func, 
                                                size_t nThreads,
                                                size_t start, size_t count ) 
{
    if(nThreads < 1) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "There needs to be at least one thread");
    }

    // Set the starting and ending points
    const auto firstRowNum = start;
    const auto lastRowNum = firstRowNum + std::min(count, md.size() - start);
    const auto rowCount = lastRowNum - firstRowNum;
    std::atomic<size_t> currRowNum(firstRowNum);

    // Create a worker function which atomically aquires a row and
    // dispatches the provided function
    const auto workerFunc = [&md, &func, &currRowNum, lastRowNum, firstRowNum] (size_t threadId) {
        auto rowNum = currRowNum++;

        while(rowNum < lastRowNum) {
            // Process a row
            const auto row = md.getRowVec(md.getRowId(rowNum));
            func(threadId, rowNum, row);

            // Update the progress bar only from the first thread 
            // due to concurrency issues
            if (threadId == 0) {
                progress_bar(rowNum - firstRowNum + 1);
            }

            // Aquire the next row
            rowNum = currRowNum++;
        }
    };

    // Initialzie the progress bar
    init_progress_bar(rowCount);

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
    progress_bar(rowCount);
}





void ProgAlignSpectral::getProjectionSizes( const std::vector<Matrix2D<Real>>& bases, 
                                            std::vector<size_t>& result ) 
{
    result.clear();
    result.reserve(bases.size());
    std::transform(
        bases.cbegin(), bases.cend(),
        std::back_inserter(result),
        [] (const Matrix2D<Real>& m) -> size_t {
            return MAT_XSIZE(m);
        }
    );
}

size_t ProgAlignSpectral::getBatchSize(size_t memorySize, size_t imageProjSize, size_t nTransform) {
    return std::max(memorySize / (imageProjSize * nTransform), 1UL);
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
            const auto dx = i*step - maxRadius;
            for (size_t j = 0; j < nTranslations; ++j) {
                const auto dy = j*step - maxRadius;
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
            shifts.insert({point.real(), point.imag()});
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

void ProgAlignSpectral::project(const std::vector<Matrix2D<Real>>& bases,
                                const std::vector<Matrix1D<Real>>& bands,
                                std::vector<Matrix1D<Real>>& projections )
{
    assert(bases.size() == bands.size());
    projections.resize(bases.size());
    for(size_t i = 0; i < bands.size(); ++i) {
        assert(MAT_YSIZE(bases[i]) == VEC_XSIZE(bands[i]));
        matrixOperation_Atx(bases[i], bands[i], projections[i]);
    }
}

double ProgAlignSpectral::standardizeAngle(double angle) {
    return (angle > 180.0) ? (angle - 360.0) : angle;
}

}