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

#include "image_spectral_pca.h"

#include <core/metadata_extension.h>

#include <numeric>
#include <cassert>

void ProgImageSpectralPca::defineParams() {
    addUsageLine("Compute the basis for each band of a set of images");

    addParamsLine("   -i <md_file>                    : Stack with the images");
    addParamsLine("   -b <image>                      : Image with the band map");
    addParamsLine("   --oroot <directory>             : Root directory for the output files");
    
    addParamsLine("   --training <percentage>         : Percentage of images used when training the PCAs");
    addParamsLine("   --efficiency <percentage>       : Target PCA efficiency");
    addParamsLine("   --initial <percentage>          : Ratio of the images used for the initial batch");

    addParamsLine("   --thr <threads>                 : Number of threads");
}

void ProgImageSpectralPca::readParams() {
    fnImages = getParam("-i");
    fnBandMap = getParam("-b");
    fnOroot = getParam("--oroot");
    pcaTraining = getDoubleParam("--training");
    pcaEfficiency = getDoubleParam("--efficiency");
    pcaInitialBatch = getDoubleParam("--initial");
    nThreads = getIntParam("--thr");
}

void ProgImageSpectralPca::show() const {
    if (verbose < 1) return;

    std::cout << "Input stack                 : " << fnImages << "\n";
    std::cout << "Band map                    : " << fnBandMap << "\n";
    std::cout << "Output root                 : " << fnOroot << "\n";
    std::cout << "PCA training percentage     : " << pcaTraining << "\n";
    std::cout << "Target PCA efficiency       : " << pcaEfficiency << "\n";
    std::cout << "PCA initial batch ratio     : " << pcaInitialBatch << "\n";
    std::cout << "Number of threads           : " << nThreads << "\n";
    std::cout.flush();
}

void ProgImageSpectralPca::run() {
    //TODO
}







ProgImageSpectralPca::BandMap::BandMap(const MultidimArray<int>& bands) 
    : m_bands(bands)
    , m_sizes(computeBandSizes(m_bands))
{
}

void ProgImageSpectralPca::BandMap::reset(const MultidimArray<int>& bands) {
    m_bands = bands;
    m_sizes = computeBandSizes(m_bands);
}

const MultidimArray<int>& ProgImageSpectralPca::BandMap::getBands() const {
    return m_bands;
}

const std::vector<size_t>& ProgImageSpectralPca::BandMap::getBandSizes() const {
    return m_sizes;
}

void ProgImageSpectralPca::BandMap::flattenForPca( const MultidimArray<std::complex<double>>& spectrum,
                                                std::vector<Matrix1D<double>>& data ) const
{
    data.resize(m_sizes.size());
    for(size_t i = 0; i < m_sizes.size(); ++i) {
        flattenForPca(spectrum, i, data[i]);
    }
}

void ProgImageSpectralPca::BandMap::flattenForPca( const MultidimArray<std::complex<double>>& spectrum,
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

std::vector<size_t> ProgImageSpectralPca::BandMap::computeBandSizes(const MultidimArray<int>& bands) {
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





ProgImageSpectralPca::SpectralPca::SpectralPca(const std::vector<size_t>& sizes,
                                            double initialCompression,
                                            double initialBatch )
{
    reset(sizes, initialCompression, initialBatch);
}



size_t ProgImageSpectralPca::SpectralPca::getBandCount() const {
    return m_bandPcas.size();
}

size_t ProgImageSpectralPca::SpectralPca::getBandSize(size_t i) const {
    return m_bandPcas.at(i).getComponentCount();
}

size_t ProgImageSpectralPca::SpectralPca::getProjectionSize(size_t i) const {
    return m_bandPcas.at(i).getPrincipalComponentCount();
}

void ProgImageSpectralPca::SpectralPca::getMean(size_t i, Matrix1D<double>& v) const {
    return m_bandPcas.at(i).getMean(v);
}

void ProgImageSpectralPca::SpectralPca::getVariance(size_t i, Matrix1D<double>& v) const {
    return m_bandPcas.at(i).getVariance(v);
}

void ProgImageSpectralPca::SpectralPca::getProjectionVariance(size_t i, Matrix1D<double>& v) const {
    return m_bandPcas.at(i).getAxisVariance(v); //TODO rename in the class
}

void ProgImageSpectralPca::SpectralPca::getBasis(size_t i, Matrix2D<double>& b) const {
    return m_bandPcas.at(i).getBasis(b);
}

double ProgImageSpectralPca::SpectralPca::getError(size_t i) const {
    return m_bandPcas.at(i).getError();
}


void ProgImageSpectralPca::SpectralPca::getErrorFunction(size_t i, Matrix1D<double>& errFn) {
    Matrix1D<double> variances;
    getVariance(i, variances);
    getProjectionVariance(i, errFn);
    calculateErrorFunction(errFn, variances.sum());
}


void ProgImageSpectralPca::SpectralPca::reset(  const std::vector<size_t>& sizes,
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


void ProgImageSpectralPca::SpectralPca::learn(const std::vector<Matrix1D<double>>& bands) {
    if (bands.size() != getBandCount()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Received band count does not match");
    }

    for (size_t i = 0; i < getBandCount(); ++i) {
        m_bandPcas[i].learn(bands[i]);
    } 
}

void ProgImageSpectralPca::SpectralPca::learnConcurrent(const std::vector<Matrix1D<double>>& bands) {
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

void ProgImageSpectralPca::SpectralPca::finalize() {
    for (auto& pca : m_bandPcas) {
        pca.finalize();
    }
}


void ProgImageSpectralPca::SpectralPca::equalizeError(double precision) {
    Matrix1D<double> errorFunction;
    for(size_t i = 0; i < getBandCount(); ++i) {
        getErrorFunction(i, errorFunction);
        m_bandPcas[i].shrink(calculateRequiredComponents(errorFunction, precision));
    }   
}


void ProgImageSpectralPca::SpectralPca::calculateErrorFunction(Matrix1D<double>& lambdas, 
                                                            double totalVariance)
{
    // Integrate in-place the represented variances
    std::partial_sum(
        MATRIX1D_ARRAY(lambdas),
        MATRIX1D_ARRAY(lambdas) + VEC_XSIZE(lambdas),
        MATRIX1D_ARRAY(lambdas)
    );
    
    // Normalize
    const auto gain = 1.0 / totalVariance;
    lambdas *= gain;
}

size_t ProgImageSpectralPca::SpectralPca::calculateRequiredComponents( const Matrix1D<double>& errFn,
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


void ProgImageSpectralPca::readBandMap() {
    Image<int> image;
    image.read(fnBandMap);
    m_bandMap.reset(image());
}

void ProgImageSpectralPca::trainPca() {
    struct ThreadData {
        Image<double> image;
        FourierTransformer fourier;
        MultidimArray<std::complex<double>> spectrum;
        std::vector<Matrix1D<double>> bandCoefficients;
    };

    // Read a MD vec with a subset of the input particles
    MetaDataVec md;
    md.read(fnImages);
    md.removeDisabled();
    subset(md, static_cast<size_t>(md.size()*pcaTraining/100));

    // Setup PCAs
    m_pca.reset(
        m_bandMap.getBandSizes(), 
        0.9, 4.0 //TODO parameters
    );

    // Create a lambda to run in parallel for each image to be learnt
    std::vector<ThreadData> threadData(nThreads);
    const auto func = [this, &threadData] (size_t threadId, size_t i, const MDRowVec& row) {
        auto& data = threadData[threadId];

        // Read an image from disk
        const auto& fnImage = row.getValue<String>(MDL_IMAGE);
        data.image.read(fnImage);

        // Learn the image        
        data.fourier.FourierTransform(data.image(), data.spectrum, false);
        m_bandMap.flattenForPca(data.spectrum, data.bandCoefficients);
        m_pca.learnConcurrent(data.bandCoefficients);
    };

    // Dispatch training
    //processRowsInParallel(md, func, threadData.size()); //TODO

    // Finalize training
    m_pca.finalize();
    m_pca.equalizeError(pcaEfficiency / 100.0);

    // Show info
    std::cout << "PCA coefficients:\n";
    size_t totalBandSize = 0, totalProjSize = 0;
    for(size_t i = 0; i < m_pca.getBandCount(); ++i) {
        const auto bandSize = m_pca.getBandSize(i); totalBandSize += bandSize;
        const auto projSize = m_pca.getProjectionSize(i); totalProjSize += projSize;
        std::cout   << "\t- Band " << i << ": " << projSize
                    << " (" << 100.0*projSize/bandSize << "%)\n";
    }
    std::cout   << "\t- Total: " << totalProjSize
                << " (" << 100.0*totalProjSize/totalBandSize << "%)\n";

    std::cout << "PCA error:\n";
    for(size_t i = 0; i < m_pca.getBandCount(); ++i) {
        std::cout   << "\t- Band " << i << ": " << m_pca.getError(i) << "\n";
    }
}

void ProgImageSpectralPca::generateOutput() {
    Matrix1D<double> v;
    Matrix2D<double> m;

    for(size_t i = 0; i < m_pca.getBandCount(); ++i) {
        const auto num = std::to_string(i);
        m_pca.getBasis(i, m); m.write(fnOroot + "basis_" + num + ".txt");
        m_pca.getMean(i, v); v.write(fnOroot + "mean_" + num + ".txt");
        m_pca.getVariance(i, v); v.write(fnOroot + "variance_" + num + ".txt");
        m_pca.getProjectionVariance(i, v); v.write(fnOroot + "projection_variance_" + num + ".txt");
    }
}

void ProgImageSpectralPca::subset(MetaDataVec& md, size_t n) {
    MetaDataVec aux;
    aux.randomize(md);
    md.selectPart(aux, 0, n);
}