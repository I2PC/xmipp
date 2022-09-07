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

    addParamsLine("   -r <md_file>                    : Metadata file with the reference images");
    addParamsLine("   -i <md_file>                    : Metadata file with the experimental images");
    addParamsLine("   -o <md_file>                    : Resulting metadata file with the aligned images");
    addParamsLine("   --oroot <directory>             : Root directory for auxiliary output files");
    
    addParamsLine("   --rotations <rotations>         : Number of rotations to consider");
    addParamsLine("   --translations <transtions>     : Number of translations to consider");
    addParamsLine("   --maxShift <maxShift>           : Maximum translation in percentage relative to the image size");
    
    addParamsLine("   --pc <pc>                       : Number of principal components to consider in each band");
    addParamsLine("   --bands <bands>                 : Number of principal components to consider in each band");
    addParamsLine("   --lowRes <low_resolution>       : Lowest resolution to consider [0, 1] in terms of the Nyquist freq.");
    addParamsLine("   --highRes <high_resolution>     : Highest resolution to consider [0, 1] in terms of the Nyquist freq.");
    
    addParamsLine("   --training <percentage>         : Percentage of images used when training the PCAs");

    addParamsLine("   --thr <threads>                 : Number of threads");
}

void ProgAlignSpectral::readParams() {
    auto& param = m_parameters;

    param.fnReference = getParam("-r");
    param.fnExperimental = getParam("-i");
    param.fnOutput = getParam("-o");
    param.fnOroot = getParam("--oroot");

    param.nRotations = getIntParam("--rotations");
    param.nTranslations = getIntParam("--translations");
    param.maxShift = getDoubleParam("--maxShift") / 100;

    param.nBandPc = getIntParam("--pc");
    param.nBands = getIntParam("--bands");
    param.lowResLimit = getDoubleParam("--lowRes") * (2*M_PI);
    param.highResLimit = getDoubleParam("--highRes") * (2*M_PI);
    
    param.training = getDoubleParam("--training") / 100;

    param.nThreads = getIntParam("--thr");
}

void ProgAlignSpectral::show() const {
    if (verbose < 1) return;
    auto& param = m_parameters;

    std::cout << "Experimanetal metadata      : " << param.fnExperimental << "\n";
    std::cout << "Reference metadata          : " << param.fnReference << "\n";
    std::cout << "Output metadata             : " << param.fnOutput << "\n";
    std::cout << "Output root                 : " << param.fnOroot << "\n";

    std::cout << "Rotations                   : " << param.nRotations << "\n";
    std::cout << "Translations                : " << param.nTranslations << "\n";
    std::cout << "Maximum shift               : " << param.maxShift*100 << "%\n";

    std::cout << "Number of PC per band       : " << param.nBandPc << "\n";
    std::cout << "Number of bands             : " << param.nBands << "\n";
    std::cout << "Low resolution limit        : " << param.lowResLimit << "rad\n";
    std::cout << "High resolution limit       : " << param.highResLimit << "rad\n";

    std::cout << "Training percentage         : " << param.training*100 << "%\n";

    std::cout << "Number of threads           : " << param.nThreads << "\n";
    std::cout.flush();
}

void ProgAlignSpectral::run() {
    readInput();
    calculateTranslationFilters();
    calculateBands();
    trainPcas();
    projectReferences();
    classifyExperimental();
    generateBandSsnr();
    generateOutput();
}





void ProgAlignSpectral::TranslationFilter::getTranslation(double& dx, double& dy) const {
    dx = m_dx;
    dy = m_dy;
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

void ProgAlignSpectral::TranslationFilter::computeCoefficients()
{
    // Shorthands
    auto& coeff = m_coefficients;
    size_t ny = YSIZE(coeff);
    size_t nx = fromFourierXSize(XSIZE(coeff));

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
void ProgAlignSpectral::ImageTransformer::forEachInPlaneTransform(  const MultidimArray<double>& img,
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
        const_cast<MultidimArray<double>&>(img), //HACK although it won't be written
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

void ProgAlignSpectral::BandMap::flattenForPca( const MultidimArray<std::complex<double>>& spectrum,
                                                std::vector<Matrix1D<double>>& data ) const
{
    data.resize(m_sizes.size());
    for(size_t i = 0; i < m_sizes.size(); ++i) {
        flattenForPca(spectrum, i, data[i]);
    }
}

void ProgAlignSpectral::BandMap::flattenForPca( const MultidimArray<std::complex<double>>& spectrum,
                                                size_t band,
                                                Matrix1D<double>& data ) const
{
    if (!spectrum.sameShape(m_bands)) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Spectrum and band map must coincide in shape");
    }

    data.resizeNoCopy(m_sizes[band]);
    auto* wrPtr = MATRIX1D_ARRAY(data);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(m_bands) {
        if(DIRECT_MULTIDIM_ELEM(m_bands, n) == band) {
            const auto& value = DIRECT_MULTIDIM_ELEM(spectrum, n);
            *(wrPtr++) = value.real();
            *(wrPtr++) = value.imag();
        }
    }
    assert(wrPtr == MATRIX1D_ARRAY(data) + VEC_XSIZE(data));
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





ProgAlignSpectral::SpectralPca::SpectralPca(const std::vector<size_t>& sizes,
                                            double initialCompression,
                                            double initialBatch )
{
    reset(sizes, initialCompression, initialBatch);
}



size_t ProgAlignSpectral::SpectralPca::getBandCount() const {
    return m_bandPcas.size();
}

size_t ProgAlignSpectral::SpectralPca::getBandSize(size_t i) const {
    return m_bandPcas.at(i).getComponentCount();
}

size_t ProgAlignSpectral::SpectralPca::getProjectionSize(size_t i) const {
    return m_bandPcas.at(i).getPrincipalComponentCount();
}

void ProgAlignSpectral::SpectralPca::getMean(size_t i, Matrix1D<double>& v) const {
    return m_bandPcas.at(i).getMean(v);
}

void ProgAlignSpectral::SpectralPca::getVariance(size_t i, Matrix1D<double>& v) const {
    return m_bandPcas.at(i).getVariance(v);
}

void ProgAlignSpectral::SpectralPca::getAxisVariance(size_t i, Matrix1D<double>& v) const {
    return m_bandPcas.at(i).getAxisVariance(v);
}

void ProgAlignSpectral::SpectralPca::getBasis(size_t i, Matrix2D<double>& b) const {
    return m_bandPcas.at(i).getBasis(b);
}

double ProgAlignSpectral::SpectralPca::getError(size_t i) const {
    return m_bandPcas.at(i).getError();
}


void ProgAlignSpectral::SpectralPca::getErrorFunction(size_t i, Matrix1D<double>& errFn) {
    Matrix1D<double> variances;
    getVariance(i, variances);
    getAxisVariance(i, errFn);
    calculateErrorFunction(errFn, variances);
}


void ProgAlignSpectral::SpectralPca::reset() {
    for (auto& pca : m_bandPcas) {
        pca.reset();
    }
}


void ProgAlignSpectral::SpectralPca::reset( const std::vector<size_t>& sizes,
                                            double initialCompression, 
                                            double initialBatch ) 
{
    // Setup PCAs
    m_bandPcas.clear();
    m_bandPcas.reserve(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) {
        const auto nPc = static_cast<size_t>(initialCompression*sizes[i]);
        const auto nInitialBatch = static_cast<size_t>(initialBatch*nPc);
        m_bandPcas.emplace_back(sizes[i], nPc, nInitialBatch);
    }

    // Setup mutexes for PCAs
    m_bandMutex = std::vector<std::mutex>(m_bandPcas.size());
}

void ProgAlignSpectral::SpectralPca::learn(const std::vector<Matrix1D<double>>& bands) {
    if (bands.size() != getBandCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Received band count does not match");
    }

    for (size_t i = 0; i < getBandCount(); ++i) {
        m_bandPcas[i].learn(bands[i]);
    } 
}

void ProgAlignSpectral::SpectralPca::learnConcurrent(const std::vector<Matrix1D<double>>& bands) {
    if (bands.size() != getBandCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Received band count does not match");
    }

    // Learn backwawds so that hopefully only once is blocked
    // HACK the odd condition works because of overflow (condition i >= 0 does
    // not work because of the same reason)
    for (size_t i = getBandCount() - 1; i < getBandCount(); --i) {
        std::lock_guard<std::mutex> lock(m_bandMutex[i]);
        m_bandPcas[i].learn(bands[i]);
    } 
}

void ProgAlignSpectral::SpectralPca::finalize() {
    for (auto& pca : m_bandPcas) {
        pca.finalize();
    }
}


void ProgAlignSpectral::SpectralPca::equalizeError(double precision) {
    Matrix1D<double> errorFunction;
    for(size_t i = 0; i < getBandCount(); ++i) {
        getErrorFunction(i, errorFunction);
        m_bandPcas[i].shrink(calculateRequiredComponents(errorFunction, precision));
    }   
}

double ProgAlignSpectral::SpectralPca::optimizeError(size_t totalSize) {
    std::vector<size_t> sizeDistribution(getBandCount());
    std::vector<Matrix1D<double>> errorFunctions(getBandCount());
    std::vector<Matrix1D<double>> errorFunctionDerivatives(getBandCount());

    // Get the error functions
    for(size_t i = 0; i < getBandCount(); ++i) {
        getErrorFunction(i, errorFunctions[i]);
        errorFunctions[i].numericalDerivative(errorFunctionDerivatives[i]);
    }

    // Calculate the initial precision for ascending
    std::fill(sizeDistribution.begin(), sizeDistribution.end(), 0UL);
    size_t sum = getBandCount(); // (0+1)*nBands
    double precision = 0.0;
    
    // Perform a gradient descent
    while (sum != totalSize) {
        const auto error = static_cast<double>(totalSize) - static_cast<double>(sum);

        // Compute the gradient for the current iteration
        double gradient = 0;
        for(size_t i = 0; i < getBandCount(); ++i) {
            gradient += VEC_ELEM(errorFunctionDerivatives[i], sizeDistribution[i]);
        }
        std::cout << "Precision: " << precision << " Gradient: " << gradient << " Sum: " << sum << std::endl;
        if(!gradient) {
            REPORT_ERROR(ERR_NUMERICAL, "Could not optimize PCAs due to a zero gradient");
        }

        // Update the precision
        precision += gradient * error;

        // Determine the required component count for the given precision
        for(size_t i = 0; i < getBandCount(); ++i) {
            sizeDistribution[i] = calculateRequiredComponents(errorFunctions[i], precision);
        }
        sum = std::accumulate(sizeDistribution.cbegin(), sizeDistribution.cend(), 0UL);
    }

    // Shrink
    for(size_t i = 0; i < sizeDistribution.size(); ++i) {
        m_bandPcas[i].shrink(sizeDistribution[i]);
    }

    return precision;
}



void ProgAlignSpectral::SpectralPca::centerAndProject(  std::vector<Matrix1D<double>>& bands, 
                                                        std::vector<Matrix1D<double>>& projections) const
{
    if (bands.size() != getBandCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Received band count does not match");
    }

    // Project row by row
    projections.resize(getBandCount());
    for (size_t i = 0; i < getBandCount(); ++i) {
        m_bandPcas[i].centerAndProject(bands[i], projections[i]);
    }
}

void ProgAlignSpectral::SpectralPca::unprojectAndUncenter(  const std::vector<Matrix1D<double>>& projections,
                                                            std::vector<Matrix1D<double>>& bands ) const
{
    if (projections.size() != getBandCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Received band count does not match");
    }

    // Unproject row by row
    bands.resize(getBandCount());
    for (size_t i = 0; i < getBandCount(); ++i) {
        m_bandPcas[i].unprojectAndUncenter(projections[i], bands[i]);
    }
}

void ProgAlignSpectral::SpectralPca::calculateErrorFunction(Matrix1D<double>& lambdas, 
                                                            const Matrix1D<double>& variances )
{
    // Integrate in-place the represented variances
    std::partial_sum(
        MATRIX1D_ARRAY(lambdas),
        MATRIX1D_ARRAY(lambdas) + VEC_XSIZE(lambdas),
        MATRIX1D_ARRAY(lambdas)
    );
    
    // Normalize
    lambdas /= variances.sum();
}

size_t ProgAlignSpectral::SpectralPca::calculateRequiredComponents( const Matrix1D<double>& errFn,
                                                                    double precision )
{
    const auto ite = std::lower_bound(
        MATRIX1D_ARRAY(errFn),
        MATRIX1D_ARRAY(errFn) + VEC_XSIZE(errFn),
        precision
    );

    const auto index = std::distance(
        MATRIX1D_ARRAY(errFn),
        ite
    );
    assert(index >= 0);

    auto s = static_cast<size_t>(index) + 1;

    // Use the next multiple of 16 for faster SIMD
    constexpr size_t SIMD_SIZE = 16;
    s = (s + SIMD_SIZE - 1) / SIMD_SIZE * SIMD_SIZE;

    return std::min(s, VEC_XSIZE(errFn));
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
                                                                    std::vector<Matrix1D<double>>& r) 
{
    r.resize(m_projections.size());
    for(size_t j = 0; j < m_projections.size(); ++j) {
        m_projections[j].getRowAlias(i, r[j]);
    }
}

size_t ProgAlignSpectral::ReferencePcaProjections::matchPcaProjection(  const std::vector<Matrix1D<double>>& experimentalBands,
                                                                        const Matrix1D<double>& weights) const 
{
    Matrix1D<double> referenceBand;

    // Compare it with all the reference images
    auto best = getImageCount();
    auto bestScore = std::numeric_limits<double>::infinity();
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

size_t ProgAlignSpectral::ReferencePcaProjections::matchPcaProjectionBaB(   const std::vector<Matrix1D<double>>& experimentalBands,
                                                                            const Matrix1D<double>& weights ) const 
{
    Matrix1D<double> referenceBand;

    // Compare band-by-band using the branch and bound approach
    auto best = getImageCount();
    auto bestScore = std::numeric_limits<double>::infinity();
    auto bestCandidate = 0; // Arbitrary
    std::vector<double> scores(getImageCount(), 0.0);

    for(size_t i = 0; i < getBandCount() && bestCandidate != best; ++i) {
        // Evaluate the "complete" score for the best candidate image
        for(size_t j = i; j < getBandCount() && scores[bestCandidate] < bestScore; ++j) {
            const auto& experimentalBand = experimentalBands[j];
            const_cast<Matrix2D<double>&>(m_projections[j]).getRowAlias(bestCandidate, referenceBand);
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
                const_cast<Matrix2D<double>&>(m_projections[i]).getRowAlias(j, referenceBand);
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





ProgAlignSpectral::ReferenceMetadata::ReferenceMetadata(size_t index, 
                                                        double rotation, 
                                                        double shiftx, 
                                                        double shifty )
    : m_index(index)
    , m_rotation(rotation)
    , m_shiftX(shiftx)
    , m_shiftY(shifty)
{
}

void ProgAlignSpectral::ReferenceMetadata::setIndex(size_t index) {
    m_index = index;
}

size_t ProgAlignSpectral::ReferenceMetadata::getIndex() const {
    return m_index;
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





void ProgAlignSpectral::readInput() {
    readMetadata(m_parameters.fnReference, m_mdReference);
    readMetadata(m_parameters.fnExperimental, m_mdExperimental);

    // TODO determine correctly
    m_weights.resizeNoCopy(m_parameters.nBands);
    const auto& bandSizes = m_bandMap.getBandSizes();
    FOR_ALL_ELEMENTS_IN_MATRIX1D(m_weights) {
        VEC_ELEM(m_weights, i) = 1.0;
        //VEC_ELEM(m_weights, i) = std::exp(-static_cast<double>(i));
        //VEC_ELEM(m_weights, i) bandSizes[i] * std::exp(-static_cast<double>(i));
    }
}

void ProgAlignSpectral::calculateTranslationFilters() {
    // Shorthands
    const auto& nTranslations = m_parameters.nTranslations;
    const auto& maxShift = m_parameters.maxShift;
    const auto& md = m_mdReference;
    
    // Determine the image size
    size_t nx, ny, nz, nn;
    getImageSize(md, nx, ny, nz, nn);

    // Pre-compute the translation filters in the freq. domain
    m_translations = computeTranslationFiltersRectangle(nx, ny, nTranslations, maxShift);
}

void ProgAlignSpectral::calculateBands() {
    const auto& md = m_mdReference;

    // Determine the image size
    size_t nx, ny, nz, nn;
    getImageSize(md, nx, ny, nz, nn);

    const auto frequencies = computeArithmeticBandFrecuencies(
        m_parameters.lowResLimit, 
        m_parameters.highResLimit,
        m_parameters.nBands
    );
    const auto bands = computeBands(
        nx, ny,
        frequencies 
    );
    m_bandMap.reset(bands);

    // Show info
    const auto& sizes = m_bandMap.getBandSizes();
    assert(frequencies.size() == sizes.size()+1);
    std::cout << "Bands frequencies:\n";
    for(size_t i = 0; i < sizes.size(); ++i) {
        std::cout << "\t- Band " << i << ": ";
        std::cout << frequencies[i] << " - " << frequencies[i+1] << "rad ";
        std::cout << "(" << sizes[i] << " coeffs)\n"; 
    }
}

void ProgAlignSpectral::trainPcas() {
    struct ThreadData {
        Image<double> image;
        ImageTransformer transformer;
        std::vector<Matrix1D<double>> bandCoefficients;
    };

    // Create a MD with a subset of all the images (experimental and reference)
    MetaDataVec mdAll;
    for(auto& inRow : m_mdReference) {
        auto outRow = mdAll.getRowVec(mdAll.addObject());
        outRow.setValue(MDL_IMAGE, inRow.getValue<String>(MDL_IMAGE));
        outRow.setValue(MDL_REF, 1);
    }
    for(auto& inRow : m_mdExperimental) {
        auto outRow = mdAll.getRowVec(mdAll.addObject());
        outRow.setValue(MDL_IMAGE, inRow.getValue<String>(MDL_IMAGE));
        outRow.setValue(MDL_REF, 0);
    }
    mdAll.randomize(mdAll);
    const auto nTraining = static_cast<size_t>(m_parameters.training * mdAll.size());
    while(mdAll.size() >= nTraining) mdAll.removeObject(mdAll.lastRowId());

    // Setup PCAs
    m_pca.reset(
        m_bandMap.getBandSizes(), 
        0.75, 8.0
    );

    // Create a lambda to run in parallel for each image to be learnt
    std::vector<ThreadData> threadData(m_parameters.nThreads);
    const auto func = [this, &threadData] (size_t threadId, size_t i, const MDRowVec& row) {
        auto& data = threadData[threadId];

        // Read an image from disk
        const auto& isReference = row.getValue<int>(MDL_REF);
        const auto& fnImage = row.getValue<String>(MDL_IMAGE);
        readImage(fnImage, data.image);

        const auto learnFunc = [this, &data] (const auto& dft) {
            m_bandMap.flattenForPca(dft, data.bandCoefficients);
            m_pca.learnConcurrent(data.bandCoefficients);
        };

        // Depending on if it is a reference, learn all its in-plane
        // transformations
        if(isReference) {
            data.transformer.forEachInPlaneTransform(
                data.image(),
                m_parameters.nRotations,
                m_translations,
                std::bind(std::ref(learnFunc), std::placeholders::_1)
            );
        } else {
            data.transformer.forFourierTransform(
                data.image(),
                learnFunc
            );
        }
    };

    // Dispatch training
    processRowsInParallel(mdAll, func, threadData.size());

    // Finalize training
    m_pca.finalize();
    //m_pca.equalizeError(0.75);
    //m_pca.optimizeError(m_parameters.nBandPc);

    // Show info
    std::cout << "PCA error:\n";
    for(size_t i = 0; i < m_pca.getBandCount(); ++i) {
        std::cout   << "\t- Band " << i << ": " << m_pca.getError(i)
                    << " (" << m_pca.getProjectionSize(i) << ")\n";
    }

    // Write the PCA to disk
    Matrix1D<double> variances;
    Matrix1D<double> axisVariances;
    Matrix2D<double> basis;
    for(size_t i = 0; i < m_pca.getBandCount(); ++i) {
        m_pca.getVariance(i, variances);
        m_pca.getAxisVariance(i, axisVariances);
        m_pca.getBasis(i, basis);
        variances.write(m_parameters.fnOroot + "variances_" + std::to_string(i));
        axisVariances.write(m_parameters.fnOroot + "axis_variances_" + std::to_string(i));
        basis.write(m_parameters.fnOroot + "basis_" + std::to_string(i));
    }
    
}

void ProgAlignSpectral::projectReferences() {
    // Allocate space
    std::vector<size_t> projectionSizes(m_pca.getBandCount());
    for(size_t i = 0; i < m_pca.getBandCount(); ++i) projectionSizes[i] = m_pca.getProjectionSize(i);
    m_references.reset(
        m_mdReference.size() * m_parameters.nRotations * m_translations.size(),
        projectionSizes
    );
    m_referenceData.resize(m_references.getImageCount());
    
    struct ThreadData {
        Image<double> image;
        ImageTransformer transformer;
        std::vector<Matrix1D<double>> bandCoefficients;
        std::vector<Matrix1D<double>> bandProjections;
    };

    // Create a lambda to run in parallel
    std::vector<ThreadData> threadData(m_parameters.nThreads);
    const auto func = [this, &threadData] (size_t threadId, size_t i, const MDRowVec& row) {
        auto& data = threadData[threadId];

        // Read an image from disk
        const FileName& fnImage = row.getValue<String>(MDL_IMAGE);
        readImage(fnImage, data.image);

        // For each in-plane transformation train the PCA
        const auto offset = i * m_parameters.nRotations * m_translations.size();
        data.transformer.forEachInPlaneTransform(
            data.image(),
            m_parameters.nRotations,
            m_translations,
            [this, &data, i, counter=offset] (const auto& x, auto rot, auto sx, auto sy) mutable {
                // Obtain the resulting memory area
                const auto index = counter++;
                m_references.getPcaProjection(index, data.bandProjections);

                // Project the image
                m_bandMap.flattenForPca(x, data.bandCoefficients);
                m_pca.centerAndProject(data.bandCoefficients, data.bandProjections);

                // Write the metadata
                m_referenceData[index] = ReferenceMetadata(i, -rot, -sx, -sy); //Opposite transform
            }
        );
    };

    processRowsInParallel(m_mdReference, func, threadData.size());
}

void ProgAlignSpectral::classifyExperimental() {
    // Initialize the classification vector with invalid 
    // data and the appropiate size
    m_classification.clear();
    m_classification.resize(
        m_mdExperimental.size(), // size
        m_references.getImageCount() // invalid value
    );
    m_ssnr.resizeNoCopy(m_classification.size(), m_pca.getBandCount());

    struct ThreadData {
        Image<double> image;
        FourierTransformer fourier;
        MultidimArray<std::complex<double>> spectrum;
        std::vector<Matrix1D<double>> bandCoefficients;
        std::vector<Matrix1D<double>> bandProjections;
        std::vector<Matrix1D<double>> referenceBandProjections;
        Matrix1D<double> ssnr;
    };

    // Create a lambda to run in parallel
    std::vector<ThreadData> threadData(m_parameters.nThreads);
    const auto func = [this, &threadData] (size_t threadId, size_t i, const MDRowVec& row) {
        auto& data = threadData[threadId];

        // Read an image
        const FileName& fnImage = row.getValue<String>(MDL_IMAGE);
        readImage(fnImage, data.image);

        // Project the image
        data.fourier.FourierTransform(data.image(), data.spectrum, false);
        m_bandMap.flattenForPca(data.spectrum, data.bandCoefficients);
        m_pca.centerAndProject(data.bandCoefficients, data.bandProjections);

        // Compare the projection to find a match
        const auto classification = m_references.matchPcaProjection(data.bandProjections, m_weights);
        //assert(classification == m_references.matchPcaProjection(data.projection, m_weights));
        m_classification[i] = classification;

        // Compute the SSNR
        m_references.getPcaProjection(classification, data.referenceBandProjections);
        m_ssnr.getRowAlias(i, data.ssnr);
        calculateBandSsnr(data.referenceBandProjections, data.bandProjections, data.ssnr);

    };

    processRowsInParallel(m_mdExperimental, func, threadData.size());
}
    
void ProgAlignSpectral::generateBandSsnr() {
    // Calculate the average SSNR on each band
    Matrix1D<double> totalSsnr;
    m_ssnr.colSum(totalSsnr);
    totalSsnr /= MAT_YSIZE(m_ssnr);

    // Write it to disk
    totalSsnr.write(m_parameters.fnOroot + "ssnr");
    m_ssnr.write(m_parameters.fnOroot + "image_ssnr");
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
    mdOut.write(m_parameters.fnOutput);
}





void ProgAlignSpectral::updateRow(MDRowVec& row, size_t matchIndex) const {
    // Obtain the metadata
    const auto& data = m_referenceData[matchIndex];
    const auto refRow = m_mdReference.getRowVec(m_mdReference.getRowId(data.getIndex()));
    const auto expRow = m_mdExperimental.getRowVec(row.getValue<size_t>(MDL_OBJID));

    // Shift the old pose values to the second MD labels
    if (expRow.containsLabel(MDL_ANGLE_ROT)) row.setValue(MDL_ANGLE_ROT2, expRow.getValue<double>(MDL_ANGLE_ROT));
    if (expRow.containsLabel(MDL_ANGLE_TILT)) row.setValue(MDL_ANGLE_TILT2, expRow.getValue<double>(MDL_ANGLE_TILT));
    if (expRow.containsLabel(MDL_ANGLE_PSI)) row.setValue(MDL_ANGLE_PSI2, expRow.getValue<double>(MDL_ANGLE_PSI));
    if (expRow.containsLabel(MDL_SHIFT_X)) row.setValue(MDL_SHIFT_X2, expRow.getValue<double>(MDL_SHIFT_X));
    if (expRow.containsLabel(MDL_SHIFT_Y)) row.setValue(MDL_SHIFT_Y2, expRow.getValue<double>(MDL_SHIFT_Y));

    // Write the new pose
    row.setValue(MDL_ANGLE_ROT, refRow.getValue<double>(MDL_ANGLE_ROT));
    row.setValue(MDL_ANGLE_TILT, refRow.getValue<double>(MDL_ANGLE_TILT));
    row.setValue(MDL_ANGLE_PSI, data.getRotation());
    row.setValue(MDL_SHIFT_X, data.getShiftX());
    row.setValue(MDL_SHIFT_Y, data.getShiftY());

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





void ProgAlignSpectral::readMetadata(const FileName& fn, MetaDataVec& result) {
    result.read(fn);
    result.removeDisabled();
}

void ProgAlignSpectral::readImage(const FileName& fn, Image<double>& result) {
    result.read(fn);
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



std::vector<double> ProgAlignSpectral::computeArithmeticBandFrecuencies(double lowResLimit,
                                                                        double highResLimit,
                                                                        size_t nBands )
{
    std::vector<double> result;
    result.reserve(nBands+1);

    const auto delta = highResLimit - lowResLimit;
    const auto step = delta / nBands;
    for(size_t i = 0; i <= nBands; ++i) {
        result.push_back(lowResLimit + i*step);
    }

    return result;
}

std::vector<double> ProgAlignSpectral::computeGeometricBandFrecuencies( double lowResLimit,
                                                                        double highResLimit,
                                                                        size_t nBands )
{
    std::vector<double> result;
    result.reserve(nBands+1);

    const auto ratio = highResLimit / lowResLimit;
    const auto step = std::pow(ratio, 1.0/nBands);
    result.push_back(lowResLimit);
    for(size_t i = 1; i <= nBands; ++i) {
        result.push_back(result.back()*step);
    }

    return result;
}

MultidimArray<int> ProgAlignSpectral::computeBands( const size_t nx, 
                                                    const size_t ny, 
                                                    const std::vector<double>& frecuencies )
{
    // Determine the band in which falls each frequency
    MultidimArray<int> bands(ny, toFourierXSize(nx));
    const auto yStep = (2*M_PI) / ny;
    const auto xStep = (2*M_PI) / nx;
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(bands) {
        // Determine the angular frequency
        auto wy = yStep*i;
        auto wx = xStep*j;
        wy = (wy > M_PI) ? wy - (2*M_PI) : wy;
        wx = (wx > M_PI) ? wx - (2*M_PI) : wx;
        auto w = std::sqrt(wx*wx + wy*wy);

        // Determine the band in which belongs this frequency
        int band = 0;
        while(band < frecuencies.size() && w > frecuencies[band]) {
            ++band;
        }
        if(band >= frecuencies.size()) {
            band = -1; // Larger than the last threshold
        } else {
            --band; // Undo the last increment
        }

        DIRECT_A2D_ELEM(bands, i, j) = band;
    }

    return bands;
}

void ProgAlignSpectral::calculateBandSsnr(  const std::vector<Matrix1D<double>>& reference, 
                                            const std::vector<Matrix1D<double>>& experimental, 
                                            Matrix1D<double>& ssnr )
{

    assert(experimental.size() == reference.size());
    assert(VEC_XSIZE(ssnr) == reference.size());

    for(size_t i = 0; i < reference.size(); ++i) {
        const auto& referenceBand = reference[i];
        const auto& experimentalBand = experimental[i];
        const auto noiseEnergy = euclideanDistance2(referenceBand, experimentalBand);
        const auto signalEnergy = referenceBand.sum2();
        VEC_ELEM(ssnr, i) = signalEnergy / noiseEnergy;
    }
}

}