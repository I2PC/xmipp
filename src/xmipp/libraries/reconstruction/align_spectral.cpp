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


namespace Alignment {

void ProgAlignSpectral::defineParams() {
    addUsageLine("Find alignment of the experimental images in respect to a set of references");

    addParamsLine("   -r <md_file>                    : Metadata file with the reference images");
    addParamsLine("   -i <md_file>                    : Metadata file with the experimental images");
    addParamsLine("   -o <md_file>                    : Resulting metadata file with the aligned images");
    
    addParamsLine("   --rotations <rotations>         : Number of rotations to consider");
    addParamsLine("   --translations <transtions>     : Number of translations to consider");
    addParamsLine("   --maxShift <maxShift>           : Maximum translation in percentage relative to the image size");
    
    addParamsLine("   --pc <pc>                       : Number of principal components to consider in each band");
    addParamsLine("   --lowRes <low_resolution>       : Lowest resolution to consider [0, 1] in terms of the Nyquist freq.");
    addParamsLine("   --highRes <high_resolution>     : Highest resolution to consider [0, 1] in terms of the Nyquist freq.");

    addParamsLine("   --thr <threads>                 : Number of threads");
}

void ProgAlignSpectral::readParams() {
    auto& param = m_parameters;

    param.fnReference = getParam("-r");
    param.fnExperimental = getParam("-i");
    param.fnOutput = getParam("-o");

    param.nRotations = getIntParam("--rotations");
    param.nTranslations = getIntParam("--translations");
    param.maxShift = getDoubleParam("--maxShift") / 100;

    param.nBandPc = getIntParam("--pc");
    param.lowResLimit = getDoubleParam("--lowRes") * M_PI;
    param.highResLimit = getDoubleParam("--highRes") * M_PI;

    param.nThreads = getIntParam("--thr");
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

    std::cout << "Number of PC per band       : " << param.nBandPc << "\n";
    std::cout << "Low resolution limit        : " << param.lowResLimit << "rad\n";
    std::cout << "High resolution limit       : " << param.highResLimit << "rad\n";

    std::cout << "Number of threads           : " << param.nThreads << "\n";
    std::cout.flush();
}

void ProgAlignSpectral::run() {
    readInput();
    calculateTranslationFilters();
    calculateBands();
    initPcas();
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
    size_t nx = fromFourierXSize(XSIZE(coeff));

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

    data.resizeNoCopy(m_sizes[band]*2);
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
        sizes.emplace_back(count);
    }
    assert(sizes.size() == nBands);

    return sizes;
}





ProgAlignSpectral::SpectralPca::SpectralPca(const std::vector<size_t>& sizes, 
                                            size_t nPc )
{
    reset(sizes, nPc);
}



size_t ProgAlignSpectral::SpectralPca::getBandCount() const {
    return m_bandPcas.size();
}

size_t ProgAlignSpectral::SpectralPca::getBandPrincipalComponentCount() const {
    return m_bandPcas.front().getPrincipalComponentCount();
}

size_t ProgAlignSpectral::SpectralPca::getTotalPrincipalComponentCount() const {
    return getBandCount() * getBandPrincipalComponentCount();
}



void ProgAlignSpectral::SpectralPca::reset() {
    for (auto& pca : m_bandPcas) {
        pca.reset();
    }
}


void ProgAlignSpectral::SpectralPca::reset(const std::vector<size_t>& sizes, size_t nPc) {
    m_bandPcas.clear();
    m_bandPcas.reserve(sizes.size());

    for (const auto& size : sizes) {
        m_bandPcas.emplace_back(size*2, nPc); //*2 as we are are using complex numbers
    }
    assert(m_bandPcas.size() == sizes.size());

    m_bandMutex = std::vector<std::mutex>(m_bandPcas.size());
}

void ProgAlignSpectral::SpectralPca::learn(const std::vector<Matrix1D<double>>& bands) {
    if (bands.size() != getBandCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Received band count does not match");
    }

    for (size_t i = 0; i < getBandCount(); ++i) {
        m_bandPcas[i].learnNoEigenValues(bands[i]);
    } 
}

void ProgAlignSpectral::SpectralPca::learnConcurrent(const std::vector<Matrix1D<double>>& bands) {
    if (bands.size() != getBandCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Received band count does not match");
    }

    // Learn backwawds so that hopefully only once is blocked
    for (size_t i = getBandCount() - 1; i < getBandCount(); --i) {
        std::lock_guard<std::mutex> lock(m_bandMutex[i]);
        m_bandPcas[i].learnNoEigenValues(bands[i]);
    } 
}

void ProgAlignSpectral::SpectralPca::project(   const std::vector<Matrix1D<double>>& bands, 
                                                MultidimArray<double>& projections) const
{
    if (bands.size() != getBandCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Received band count does not match");
    }

    projections.resizeNoCopy(getBandCount(), getBandPrincipalComponentCount());

    // Create an alias for the rows
    Matrix1D<double> rowAlias;
    aliasFirstRow(projections, rowAlias);

    // Project row by row
    for (size_t i = 0; i < getBandCount(); ++i) {
        m_bandPcas[i].project(bands[i], rowAlias);
        aliasNextRow(rowAlias);
    }
    assert(rowAlias.vdata == MULTIDIM_ARRAY(projections) + MULTIDIM_SIZE(projections));
}

void ProgAlignSpectral::SpectralPca::unproject( const MultidimArray<double>& projections,
                                                std::vector<Matrix1D<double>>& bands ) const
{
    if (YSIZE(projections) != getBandCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Received band count does not match");
    }
    if (XSIZE(projections) != getBandPrincipalComponentCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Received principal component count does not match");
    }
    if (YXSIZE(projections) != MULTIDIM_SIZE(projections)) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Input projection array must have 2 dimensions");
    }

    bands.resize(getBandCount());

    // Create an alias for the rows
    Matrix1D<double> rowAlias;
    aliasFirstRow(projections, rowAlias);

    // Unproject row by row
    for (size_t i = 0; i < getBandCount(); ++i) {
        m_bandPcas[i].unproject(rowAlias, bands[i]);
        aliasNextRow(rowAlias);
    }
    assert(rowAlias.vdata == MULTIDIM_ARRAY(projections) + MULTIDIM_SIZE(projections));
}





void ProgAlignSpectral::readInput() {
    readMetadata(m_parameters.fnReference, m_mdReference);
    readMetadata(m_parameters.fnExperimental, m_mdExperimental);
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
    m_translations = computeTranslationFilters(nx, ny, nTranslations, maxShift);
}

void ProgAlignSpectral::calculateBands() {
    const auto& md = m_mdReference;

    // Determine the image size
    size_t nx, ny, nz, nn;
    getImageSize(md, nx, ny, nz, nn);

    const auto bands = computeBands(
        nx, ny, 
        m_parameters.lowResLimit, 
        m_parameters.highResLimit
    );
    m_bandMap.reset(bands);
}

void ProgAlignSpectral::initPcas() {
    m_pca.reset(
        m_bandMap.getBandSizes(), 
        m_parameters.nBandPc
    );
}

void ProgAlignSpectral::learnReferences() {
    // Shorthands
    const auto& md = m_mdReference;
    const auto nImages = md.size()*m_parameters.nRotations*m_translations.size();

    struct ThreadData {
        Image<double> image;
        ImageTransformer transformer;
        std::vector<Matrix1D<double>> bandCoefficients;
    };

    // Create a lambda to run in parallel
    const auto func = [this] (size_t i, const MDRowVec& row, ThreadData& data) {
        // Read an image from disk
        const FileName& fnImage = row.getValue<String>(MDL_IMAGE);
        readImage(fnImage, data.image);

        // For each in-plane transformation train the PCA
        data.transformer.forEachInPlaneTransform(
            data.image(),
            m_parameters.nRotations,
            m_translations,
            [this, &data] (const auto& x) {
                this->m_bandMap.flattenForPca(x, data.bandCoefficients);
                this->m_pca.learnConcurrent(data.bandCoefficients);
            }
        );
    };

    std::vector<ThreadData> threadData(m_parameters.nThreads);
    processRowsInParallel(md, func, threadData);
}

void ProgAlignSpectral::learnExperimental() {
    // Shorthands
    const auto& md = m_mdExperimental;
    const auto nImages = md.size();

    struct ThreadData {
        Image<double> image;
        FourierTransformer fourier;
        MultidimArray<std::complex<double>> spectrum;
        std::vector<Matrix1D<double>> bandCoefficients;
    };

    // Create a lambda to run in parallel
    const auto func = [this] (size_t i, const MDRowVec& row, ThreadData& data) {
        // Read an image
        const FileName& fnImage = row.getValue<String>(MDL_IMAGE);
        readImage(fnImage, data.image);

        // Use the image to train the pca
        data.fourier.FourierTransform(data.image(), data.spectrum, false);
        m_bandMap.flattenForPca(data.spectrum, data.bandCoefficients);
        m_pca.learnConcurrent(data.bandCoefficients);
    };

    std::vector<ThreadData> threadData(m_parameters.nThreads);
    processRowsInParallel(md, func, threadData);
}

void ProgAlignSpectral::projectReferences() {
    // Shorthands
    const auto& md = m_mdReference;
    auto& proj = m_referenceProjections;
    const auto nImages = md.size()*m_parameters.nRotations*m_translations.size();

    // Allocate space
    proj.resizeNoCopy(
        nImages,
        m_pca.getBandCount(),
        m_pca.getBandPrincipalComponentCount()
    );
    
    struct ThreadData {
        Image<double> image;
        ImageTransformer transformer;
        std::vector<Matrix1D<double>> bandCoefficients;
    };

    // Create a lambda to run in parallel
    const auto func = [this, &proj] (size_t i, const MDRowVec& row, ThreadData& data) {
        // Read an image from disk
        const FileName& fnImage = row.getValue<String>(MDL_IMAGE);
        readImage(fnImage, data.image);

        // For each in-plane transformation train the PCA
        data.transformer.forEachInPlaneTransform(
            data.image(),
            m_parameters.nRotations,
            m_translations,
            [this, i, &data, &proj] (const auto& x) {
                MultidimArray<double> slice;

                // Alias an slice of the input array
                slice.aliasSlice(proj, i);

                // Project the image
                this->m_bandMap.flattenForPca(x, data.bandCoefficients);
                this->m_pca.project(data.bandCoefficients, slice);
            }
        );
    };

    std::vector<ThreadData> threadData(m_parameters.nThreads);
    processRowsInParallel(md, func, threadData);
}

void ProgAlignSpectral::projectExperimental() {
    // Shorthands
    const auto& md = m_mdExperimental;
    const auto nImages = md.size();

    struct ThreadData {
        Image<double> image;
        FourierTransformer fourier;
        MultidimArray<std::complex<double>> spectrum;
        std::vector<Matrix1D<double>> bandCoefficients;
        MultidimArray<double> projection;
    };

    // Create a lambda to run in parallel
    const auto func = [this] (size_t i, const MDRowVec& row, ThreadData& data) {
        // Read an image
        const FileName& fnImage = row.getValue<String>(MDL_IMAGE);
        readImage(fnImage, data.image);

        // Project the image
        data.fourier.FourierTransform(data.image(), data.spectrum, false);
        m_bandMap.flattenForPca(data.spectrum, data.bandCoefficients);
        m_pca.project(data.bandCoefficients, data.projection);

        // Compare the projection to find a match
        compareProjection(data.projection);
    };

    std::vector<ThreadData> threadData(m_parameters.nThreads);
    processRowsInParallel(md, func, threadData);
}



template<typename F, typename T>
void ProgAlignSpectral::processRowsInParallel(  const MetaDataVec& md, 
                                                F&& func, 
                                                std::vector<T>& threadData ) 
{
    std::atomic<size_t> currRowNum(0);
    const auto mdSize = md.size();

    // Create a worker function which atomically aquires a row and
    // dispatches the provided function
    const auto workerFunc = [&md, &func, &currRowNum, mdSize] (T& data, bool first) {
        auto rowNum = currRowNum++;
        while(rowNum < mdSize) {
            // Process a row
            const auto row = md.getRowVec(md.getRowId(rowNum));
            func(rowNum, row, data);

            // Update the progress bar only from the first thread 
            // due to concurrency issues
            if (first) {
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
    threads.reserve(threadData.size());
    for(size_t i = 0; i < threadData.size(); ++i) {
        threads.emplace_back(workerFunc, std::ref(threadData[i]), i==0);
    }

    //Wait for them to finish
    for (auto& thread : threads) {
        thread.join();
    }

    // Set the progress bar as finished
    progress_bar(mdSize);
}

void ProgAlignSpectral::compareProjection(const MultidimArray<double>& experimentalBands) {
    MultidimArray<double> referenceBands;
    Matrix1D<double> experimentalBand, referenceBand, diffrenceBand;

    // Compare it with all the reference images using the B&B approach
    size_t best;
    double bestScore = std::numeric_limits<double>::infinity();
    for(size_t i = 0; i < ZSIZE(m_referenceProjections); ++i) {
        // Obtain the current reference particle's PCA projection
        referenceBands.aliasSlice(m_referenceProjections, i);
        assert(referenceBands.sameShape(experimentalBands));
        
        // Setup aliases
        aliasFirstRow(experimentalBands, experimentalBand);
        aliasFirstRow(referenceBands, referenceBand);

        // Add band by band using Branch and Bound
        double score = 0.0;
        for (size_t j = 0; j < YSIZE(experimentalBands) && score < bestScore; ++j) {
            // Compute the difference between the bands
            diffrenceBand = experimentalBand;
            diffrenceBand -= referenceBand;
            
            //Increment the score
            const auto weight = 1.0; //TODO
            const auto bandDistance = diffrenceBand.sum2();
            score += weight * bandDistance;
            
            // Continue
            aliasNextRow(experimentalBand);
            aliasNextRow(referenceBand);
        }

        // Update the score if necessary
        if (score < bestScore) {
            best = i;
            bestScore = score;
        }
    }
}



void ProgAlignSpectral::readMetadata(const FileName& fn, MetaDataVec& result) {
    result.read(fn);
    result.removeDisabled();
}

void ProgAlignSpectral::readImage(const FileName& fn, Image<double>& result) {
    result.read(fn);
}

void ProgAlignSpectral::aliasFirstRow(const MultidimArray<double>& md, Matrix1D<double>& v) {
    v.vdata = MULTIDIM_ARRAY(md);
    v.destroyData = false;
    v.vdim = XSIZE(md);
}

void ProgAlignSpectral::aliasNextRow(Matrix1D<double>& v) {
    v.vdata += v.vdim;
}



std::vector<ProgAlignSpectral::TranslationFilter> 
ProgAlignSpectral::computeTranslationFilters(   const size_t nx, 
                                                const size_t ny,
                                                const size_t nTranslations,
                                                const double maxShift )
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

MultidimArray<int> ProgAlignSpectral::computeBands( const size_t nx, 
                                                    const size_t ny, 
                                                    const double lowCutoffLimit,
                                                    const double highCutoffLimit )
{
    // Compute the frequency thresholds for the bands
    std::vector<double> thresholds;
    thresholds.emplace_back(lowCutoffLimit);
    while (thresholds.back() < highCutoffLimit) {
        thresholds.push_back(std::min(thresholds.back()*2, highCutoffLimit));
    }

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
        while(band < thresholds.size() && w > thresholds[band]) {
            ++band;
        }
        if(band >= thresholds.size()) {
            band = -1; // Larger than the last threshold
        } else {
            --band; // Undo the last increment
        }

        DIRECT_A2D_ELEM(bands, i, j) = band;
    }

    return bands;
}

}