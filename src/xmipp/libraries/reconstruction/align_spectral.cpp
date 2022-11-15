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
#include <map>


namespace Alignment {

template<typename T>
void ProgAlignSpectral<T>::defineParams() {
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

template<typename T>
void ProgAlignSpectral<T>::readParams() {
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

template<typename T>
void ProgAlignSpectral<T>::show() const {
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

template<typename T>
void ProgAlignSpectral<T>::run() {
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





template<typename T>
ProgAlignSpectral<T>::BandMap::BandMap(const MultidimArray<int>& bands) 
    : m_bands(bands)
    , m_sizes(computeBandSizes(m_bands))
{
}

template<typename T>
void ProgAlignSpectral<T>::BandMap::reset(const MultidimArray<int>& bands) {
    m_bands = bands;
    m_sizes = computeBandSizes(m_bands);
}

template<typename T>
const MultidimArray<int>& ProgAlignSpectral<T>::BandMap::getBands() const {
    return m_bands;
}

template<typename T>
const std::vector<size_t>& ProgAlignSpectral<T>::BandMap::getBandSizes() const {
    return m_sizes;
}

template<typename T>
std::vector<size_t> ProgAlignSpectral<T>::BandMap::computeBandSizes(const MultidimArray<int>& bands) {
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

template<typename T>
template<typename Q, typename P>
void ProgAlignSpectral<T>::BandMap::flatten(const MultidimArray<Q>& spectrum,
                                            std::vector<MultidimArray<P>>& data,
                                            size_t image ) const
{
    data.resize(m_sizes.size());
    for(size_t i = 0; i < m_sizes.size(); ++i) {
        flatten(spectrum, i, data[i], image);
    }
}

template<typename T>
template<typename Q, typename P>
void ProgAlignSpectral<T>::BandMap::flatten(const MultidimArray<Q>& spectrum,
                                            size_t band,
                                            MultidimArray<P>& data,
                                            size_t image ) const
{
    if (XSIZE(m_bands) != XSIZE(spectrum) || 
        YSIZE(m_bands) != YSIZE(spectrum) || 
        ZSIZE(m_bands) != ZSIZE(spectrum) ) 
    {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Spectrum and band map must coincide in shape");
    }

    data.resizeNoCopy(m_sizes[band]);
    auto* wrPtr = MULTIDIM_ARRAY(data);
    const auto* rdPtr = MULTIDIM_ARRAY(spectrum) + image*spectrum.zyxdim;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(m_bands) {
        if(DIRECT_MULTIDIM_ELEM(m_bands, n) == band) {
            *(wrPtr++) = *rdPtr;
        }
        ++rdPtr;
    }
    assert(wrPtr == MULTIDIM_ARRAY(data) + NZYXSIZE(data));
    assert(rdPtr == MULTIDIM_ARRAY(spectrum) + (image+1)*spectrum.zyxdim);
}





template<typename T>
ProgAlignSpectral<T>::Rotation::Rotation(double angle)
    : m_angle(angle)
{
    computeMatrix();
}

template<typename T>
void ProgAlignSpectral<T>::Rotation::operator()(const MultidimArray<Real>& in, 
                                                MultidimArray<Real>& out ) const
{
    applyGeometry(
        xmipp_transformation::LINEAR, 
        out, in,
        m_matrix, xmipp_transformation::IS_NOT_INV, 
        xmipp_transformation::WRAP, 0.0
    );
}

template<typename T>
double ProgAlignSpectral<T>::Rotation::getAngle() const {
    return m_angle;
}

template<typename T>
void ProgAlignSpectral<T>::Rotation::computeMatrix() {
    rotation2DMatrix(m_angle, m_matrix);
}





template<typename T>
ProgAlignSpectral<T>::BandShiftFilters::BandShiftFilters(   std::vector<Shift>&& shifts, 
                                                            const BandMap& bands )
    : m_shifts(std::move(shifts))
{
    computeFlattenedCoefficients(m_shifts, bands, m_coefficients);
}



template<typename T>
void ProgAlignSpectral<T>::BandShiftFilters::operator()(size_t index,
                                                        const std::vector<MultidimArray<Complex>>& in, 
                                                        std::vector<MultidimArray<Complex>>& out ) const
{
    if(in.size() != m_coefficients.size()) REPORT_ERROR(ERR_ARG_INCORRECT, "Input band count does not coincide");
    out.resize(in.size());

    MultidimArray<Complex> bandShift;
    for(size_t i = 0; i < in.size(); ++i) {
        bandShift.aliasRow(const_cast<MultidimArray<Complex>&>(m_coefficients[i]), index);
        if(!in[i].sameShape(bandShift)) REPORT_ERROR(ERR_MULTIDIM_SIZE, "Band " + std::to_string(i) + " has incorrect size");

        out[i].resizeNoCopy(in[i]);
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(out[i]) {
            DIRECT_MULTIDIM_ELEM(out[i], n) = DIRECT_MULTIDIM_ELEM(in[i], n) * DIRECT_MULTIDIM_ELEM(bandShift, n);
        }
    }
}

template<typename T>
const typename ProgAlignSpectral<T>::BandShiftFilters::Shift&
ProgAlignSpectral<T>::BandShiftFilters::getShift(size_t index) const {
    return m_shifts.at(index);
}

template<typename T>
size_t ProgAlignSpectral<T>::BandShiftFilters::getShiftCount() const {
    return m_shifts.size();
}

template<typename T>
void ProgAlignSpectral<T>::BandShiftFilters::computeCoefficients(   const Shift& shift, 
                                                                    MultidimArray<Complex>& result )
{
    const int ny = YSIZE(result);
    const int ny_2 = ny / 2;
    const auto ny_inv = 1.0 / ny;
    const int nx_2 = (XSIZE(result) - 1);
    const int nx = nx_2 * 2;
    const auto nx_inv = 1.0 / nx;

    // Normalize the displacement
    const auto dy = (-2 * M_PI) * shift[1];
    const auto dx = (-2 * M_PI) * shift[0];

    // Compute the Fourier Transform of delta[i-y, j-x]
    double fy, fx;
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(result) {
        // Convert the indices to fourier coefficients
        FFT_IDX2DIGFREQ_FAST(static_cast<int>(i), ny, ny_2, ny_inv, fy);
        FFT_IDX2DIGFREQ_FAST(static_cast<int>(j), nx, nx_2, nx_inv, fx);

        const auto theta = fy*dy + fx*dx; // Dot product of (dx, dy) and (j, i)
        DIRECT_A2D_ELEM(result, i, j) = std::polar(1.0, theta); //e^(i*theta)
    }
}

template<typename T>
void ProgAlignSpectral<T>::BandShiftFilters::computeFlattenedCoefficients(  const std::vector<Shift>& shifts,
                                                                            const BandMap& bands,
                                                                            std::vector<MultidimArray<Complex>>& result )
{
    const auto& bandSizes = bands.getBandSizes();

    // Initialize the output
    result.clear();
    result.reserve(bandSizes.size());
    for(size_t i = 0; i < bandSizes.size(); ++i) {
        result.emplace_back(shifts.size(), bandSizes[i]);
    }
    
    std::vector<MultidimArray<Complex>> bandCoefficients(result.size());
    MultidimArray<Complex> coefficients;
    coefficients.resizeNoCopy(bands.getBands());
    for(size_t i = 0; i < shifts.size(); ++i) {
        // Compute the translation filter
        computeCoefficients(shifts[i], coefficients);

        // Flatten the coefficients to their corresponding bands
        for(size_t j = 0; j < result.size(); ++j) {
            bandCoefficients[j].aliasRow(result[j], i);
        }
        bands.flatten(coefficients, bandCoefficients);
    }
}





template<typename T>
template<typename F>
void ProgAlignSpectral<T>::ImageRotationTransformer::forEachInPlaneRotation(const MultidimArray<Real>& img,
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

            func(m_fourierClean.fFourier, 0.0);
        }
    }
}





template<typename T>
template<typename F>
void ProgAlignSpectral<T>::BandShiftTransformer::forEachInPlaneTranslation( const std::vector<MultidimArray<Complex>>& in,
                                                                            const BandShiftFilters& shifts,
                                                                            F&& func )
{
    for(size_t i = 0; i < shifts.getShiftCount(); ++i) {
        const auto& shift = shifts.getShift(i);
        const auto sx = shift[0], sy = shift[1];

        if(sx || sy) {
            // Translate and call
            shifts(i, in, m_shifted);
            func(m_shifted, sx, sy);
        } else {
            // No shift applied. simply call
            func(in, 0.0, 0.0);
        }
    }
}





template<typename T>
ProgAlignSpectral<T>::ReferencePcaProjections::ReferencePcaProjections( size_t nImages, 
                                                                        const std::vector<size_t>& bandSizes )
{
    reset(nImages, bandSizes);
}

template<typename T>
void ProgAlignSpectral<T>::ReferencePcaProjections::reset(  size_t nImages, 
                                                            const std::vector<size_t>& bandSizes )
{
    m_projections.resize(bandSizes.size());
    for(size_t i = 0; i < bandSizes.size(); ++i) {
        m_projections[i].resizeNoCopy(nImages, bandSizes[i]);
    }

}

template<typename T>
void ProgAlignSpectral<T>::ReferencePcaProjections::buildTrees() {
    m_trees.clear();
    m_trees.reserve(m_projections.size());

    for(const auto& samples : m_projections) {
        m_trees.emplace_back(samples);
    }
}

template<typename T>
size_t ProgAlignSpectral<T>::ReferencePcaProjections::getImageCount() const {
    return m_projections.front().Ydim();
}

template<typename T>
size_t ProgAlignSpectral<T>::ReferencePcaProjections::getBandCount() const {
    return m_projections.size();
}

template<typename T>
size_t ProgAlignSpectral<T>::ReferencePcaProjections::getComponentCount(size_t i) const {
    return MAT_XSIZE(m_projections.at(i));
}

template<typename T>
void ProgAlignSpectral<T>::ReferencePcaProjections::getPcaProjection(   size_t i, 
                                                                        std::vector<Matrix1D<Real>>& r) 
{
    r.resize(m_projections.size());
    for(size_t j = 0; j < m_projections.size(); ++j) {
        m_projections[j].getRowAlias(i, r[j]);
    }
}

template<typename T>
size_t ProgAlignSpectral<T>::ReferencePcaProjections::matchPcaProjection(   const std::vector<Matrix1D<Real>>& experimentalBands, 
                                                                            Real& bestDistance) const 
{
    Matrix1D<Real> referenceBand;

    // Compare it with all the reference images
    auto best = getImageCount();
    for(size_t i = 0; i < getImageCount(); ++i) {
        // Add band by band
        Real distance = 0.0;
        for (size_t j = 0; j < getBandCount() && distance < bestDistance; ++j) {
            const auto& experimentalBand = experimentalBands[j];
            const_cast<Matrix2D<Real>&>(m_projections[j]).getRowAlias(i, referenceBand);

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

template<typename T>
size_t ProgAlignSpectral<T>::ReferencePcaProjections::matchPcaProjectionKDTree( const std::vector<Matrix1D<Real>>& experimentalBands, 
                                                                                Real& bestDistance) const 
{
    if(experimentalBands.size() != 1) {
        REPORT_ERROR(ERR_NOT_IMPLEMENTED, "kd-Tree search is not implemented for multiple bands");
    }
    
    // Perform a lookup in the tree
    Real distance;
    size_t index = m_trees.front().nearest(experimentalBands.front(), distance);

    // Evaluate if it is a better mach
    if(distance < bestDistance) {
        bestDistance = distance;
    } else {
        index = getImageCount();
    }

    return index;
}

template<typename T>
size_t ProgAlignSpectral<T>::ReferencePcaProjections::matchPcaProjectionBnB(const std::vector<Matrix1D<Real>>& experimentalBands, 
                                                                            Real& bestDistance,
                                                                            std::list<std::pair<size_t, Real>>& ws ) const 
{
    Matrix1D<Real> referenceBand, experimentalBand;

    // Initialize the working set. Use the provided distance as the best one
    ws.resize(getImageCount() + 1);
    auto best = ws.begin();
    for(size_t i = 0; i < getImageCount(); ++i) {
        *(best++) = std::make_pair(i, Real(0));
    }
    *best = std::make_pair(getImageCount(), bestDistance);
    assert(std::next(best) == ws.cend());

    // Compare band-by-band using the branch and bound approach
    constexpr auto BAND_ATOM_SIZE = 128UL;
    for(size_t i = 0; i < getBandCount(); ++i) {
        const auto bandSize = MAT_XSIZE(m_projections[i]);

        size_t offset = 0;
        size_t remaining = bandSize;
        while(remaining > 0) {
            // Determine how many items will be processed
            const auto count = std::min(BAND_ATOM_SIZE, remaining);

            // Determine the best candidate for the this iteration
            auto bestCandidate = best;
            experimentalBand.alias(MATRIX1D_ARRAY(experimentalBands[i]) + offset, count);
            for(auto ite = ws.begin(); ite != best; /*EMPTY ON PURPOSE*/) {
                // Add the distance of this band atom to all elements
                const_cast<Matrix2D<Real>&>(m_projections[i]).getRowAlias(ite->first, referenceBand);
                referenceBand.vdata += offset;
                referenceBand.vdim = count;
                ite->second += euclideanDistance2(experimentalBand, referenceBand);
                
                // Check if it is still eligible
                if(ite->second < best->second) {
                    if(ite->second < bestCandidate->second) {
                        bestCandidate = ite;
                    }
                    ++ite;
                } else {
                    // Not a candidate anymore. Move it to the end to
                    // stop considering it
                    ws.splice(ws.cend(), ws, ite++);
                }
            }


            if (bestCandidate != best) {
                // Advance the counters 
                offset += count;
                remaining -= count;

                // Evaluate the "complete" distance for the best candidate image
                experimentalBand.alias(MATRIX1D_ARRAY(experimentalBands[i]) + offset, remaining);
                const_cast<Matrix2D<Real>&>(m_projections[i]).getRowAlias(bestCandidate->first, referenceBand);
                referenceBand.vdata += offset;
                referenceBand.vdim = remaining;
                bestCandidate->second += euclideanDistance2(experimentalBand, referenceBand);
                for(size_t j = i+1; j < getBandCount() && bestCandidate->second < best->second; ++j) {
                    const auto& experimentalBand = experimentalBands[j];
                    const_cast<Matrix2D<Real>&>(m_projections[j]).getRowAlias(bestCandidate->first, referenceBand);
                    bestCandidate->second += euclideanDistance2(experimentalBand, referenceBand);
                }

                // Update the best distance
                if (bestCandidate->second < best->second) {
                    // Best candidate is better than the previous best.
                    // Insert it before best so that it is the new best
                    ws.splice(best, ws, bestCandidate);
                    best = bestCandidate;
                } else {
                    // Best candidate is not a candidate anymore. 
                    // Move it to the end to stop considering it
                    ws.splice(ws.cend(), ws, bestCandidate);
                }

            } else {
                // No more candidates left. Stop
                assert(ws.cbegin() == best);
                break;
            }
        }

        if(ws.cbegin() == best) {
            break;
        }
    }

    assert(ws.size() == (getImageCount() + 1)); //Nothing should have been deleted
    bestDistance = best->second;
    return best->first;
}





template<typename T>
ProgAlignSpectral<T>::ReferenceMetadata::ReferenceMetadata( size_t rowId, 
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

template<typename T>
void ProgAlignSpectral<T>::ReferenceMetadata::setRowId(size_t rowId) {
    m_rowId = rowId;
}

template<typename T>
size_t ProgAlignSpectral<T>::ReferenceMetadata::getRowId() const {
    return m_rowId;
}

template<typename T>
void ProgAlignSpectral<T>::ReferenceMetadata::setRotation(double rotation) {
    m_rotation = rotation;
}

template<typename T>
double ProgAlignSpectral<T>::ReferenceMetadata::getRotation() const {
    return m_rotation;
}

template<typename T>
void ProgAlignSpectral<T>::ReferenceMetadata::setShiftX(double sx) {
    m_shiftX = sx;
}

template<typename T>
double ProgAlignSpectral<T>::ReferenceMetadata::getShiftX() const {
    return m_shiftX;
}

template<typename T>
void ProgAlignSpectral<T>::ReferenceMetadata::setShiftY(double sy) {
    m_shiftY = sy;
}

template<typename T>
double ProgAlignSpectral<T>::ReferenceMetadata::getShiftY() const {
    return m_shiftY;
}

template<typename T>
void ProgAlignSpectral<T>::ReferenceMetadata::setDistance(double distance) {
    m_distance = distance;
}

template<typename T>
double ProgAlignSpectral<T>::ReferenceMetadata::getDistance() const {
    return m_distance;
}





template<typename T>
void ProgAlignSpectral<T>::readInputMetadata() {
    m_mdExperimental.read(fnExperimentalMetadata);
    m_mdReference.read(fnReferenceMetadata);
}

template<typename T>
void ProgAlignSpectral<T>::readBandMap() {
    Image<int> map;
    map.read(fnBands);
    m_bandMap.reset(map());
}

template<typename T>
void ProgAlignSpectral<T>::readBases() {
    Image<Real> basesImg;
    basesImg.read(fnPca);


    // Generate the Basis matrices with the maximum possible size
    constexpr auto realToComplex = 2UL;
    const auto nMaxProjections = NSIZE(basesImg())/2;
    const auto& bandSizes = m_bandMap.getBandSizes();
    std::vector<Matrix2D<Real>> bases(bandSizes.size());
    for(size_t i = 0; i < bandSizes.size(); ++i) {
        bases[i].resizeNoCopy(bandSizes[i]*2UL, nMaxProjections); // * 2 because of complex numbers
    }

    // Convert all elements into matrix form
    std::vector<MultidimArray<Complex>> bandCoefficients(bandSizes.size());
    std::vector<MultidimArray<Real>> bandCoefficientsRe(bandSizes.size()), bandCoefficientsIm(bandSizes.size());
    for(size_t i = 0; i < nMaxProjections; ++i) {
        // Read the real and imaginary parts from even and odd images
        m_bandMap.flatten(basesImg(), bandCoefficientsRe, i*2 + 0);
        m_bandMap.flatten(basesImg(), bandCoefficientsIm, i*2 + 1);

        Matrix1D<Real> column;
        for(size_t j = 0; j < bases.size(); ++j) {
            // Interleave real and complex parts
            composeComplex(
                bandCoefficientsRe[j], 
                bandCoefficientsIm[j], 
                bandCoefficients[j]
            );

            // Reinterpret the complex data as a vector
            aliasComplexElements(bandCoefficients[j], column);
            
            // Write the column
            auto& basis = bases[j];
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

    // Write
    getProjectionSizes(bases, m_projectionSizes);
    m_bases = std::move(bases);
}

template<typename T>
void ProgAlignSpectral<T>::applyWeightsToBases() {
    Image<Real> weights;
    weights.read(fnWeights);
    removeFourierSymmetry(weights());
    multiplyBases(m_bases, weights());
}

template<typename T>
void ProgAlignSpectral<T>::applyCtfToBases() {
    Image<Real> ctf;
    ctf.read(fnCtf);
    m_ctfBases = m_bases;
    removeFourierSymmetry(ctf());
    multiplyBases(m_ctfBases, ctf());
}

template<typename T>
void ProgAlignSpectral<T>::generateRotations() {
    m_rotations.clear();
    m_rotations.reserve(nRotations);

    const auto step = 360.0 / nRotations;
    for(size_t i = 0; i < nRotations; ++i) {
        const auto angle = step * i;
        m_rotations.emplace_back(angle);
    }
}

template<typename T>
void ProgAlignSpectral<T>::generateTranslations() {
    size_t nx, ny, nz, nn;
    getImageSize(m_mdExperimental, nx, ny, nz, nn);

    auto shifts = computeTranslationFiltersRectangle(
        nx, ny,
        nTranslations, maxShift/100.0
    );

    m_translations = BandShiftFilters(std::move(shifts), m_bandMap);
}

template<typename T>
void ProgAlignSpectral<T>::alignImages() {
    // Convert sizes
    constexpr size_t megabytes2bytes = 1024*1024; 
    const auto memorySizeBytes = static_cast<size_t>(megabytes2bytes*maxMemory);
    const auto imageProjCoeffCount = std::accumulate(m_projectionSizes.cbegin(), m_projectionSizes.cend(), 0UL);
    const auto imageProjSizeBytes = imageProjCoeffCount * sizeof(Real);

    // Decide if it is convenient to split transformations
    constexpr auto splitTransform = false;
    //const auto splitTransform = m_mdExperimental.size() < m_mdReference.size()*m_rotations.size();
    const auto nRefTransform = splitTransform ? m_rotations.size() : m_rotations.size()*m_translations.getShiftCount();

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

template<typename T>
void ProgAlignSpectral<T>::generateOutput() {
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





template<typename T>
void ProgAlignSpectral<T>::projectReferencesRot(size_t start, 
                                                size_t count ) 
{
    // Allocate space
    const auto nTransform = m_rotations.size();
    m_references.reset(count * nTransform, m_projectionSizes);
    m_referenceData.resize(m_references.getImageCount());
    
    struct ThreadData {
        Image<Real> image;
        ImageRotationTransformer rotator;
        std::vector<MultidimArray<Complex>> bandCoefficients;
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
        data.rotator.forEachInPlaneRotation(
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
    m_references.buildTrees();
}

template<typename T>
void ProgAlignSpectral<T>::projectReferencesRotShift(   size_t start, 
                                                        size_t count ) 
{
    // Allocate space
    const auto nTransform = m_rotations.size()*m_translations.getShiftCount();
    m_references.reset(count * nTransform, m_projectionSizes);
    m_referenceData.resize(m_references.getImageCount());
    
    struct ThreadData {
        Image<Real> image;
        ImageRotationTransformer rotator;
        BandShiftTransformer shifter;
        std::vector<MultidimArray<Complex>> bandCoefficients;
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

        // Project all the rotations and shifts
        auto counter = nTransform * (i - start);
        data.rotator.forEachInPlaneRotation(
            data.image(), m_rotations,
            [this, rowId, &data, &counter] (const MultidimArray<Complex>& spectrum, double angle) {
                // Flatten the spectrum in bands
                m_bandMap.flatten(spectrum, data.bandCoefficients);

                // Consider shifts
                data.shifter.forEachInPlaneTranslation(
                    data.bandCoefficients, m_translations,
                    [this, rowId, angle, &data, &counter] 
                    (const std::vector<MultidimArray<Complex>>& bandCoefficients, double sx, double sy) {
                        const auto index = counter++;

                        // Obtain the corresponding memory area
                        m_references.getPcaProjection(index, data.bandProjections);

                        // Project the image
                        project(m_ctfBases, bandCoefficients, data.bandProjections);

                        // Write the metadata
                        // Invert the translation as we're referring to the experimental image
                        m_referenceData[index] = ReferenceMetadata(rowId, standardizeAngle(angle), -sx, -sy); 
                    }
                );
            }
        );
        assert(counter == nTransform * (i - start + 1)); // Ensure all has been written
    };

    processRowsInParallel(m_mdReference, func, threadData.size(), start, count);
    m_references.buildTrees();
}

template<typename T>
void ProgAlignSpectral<T>::classifyExperimental() {
    struct ThreadData {
        Image<Real> image;
        FourierTransformer fourier;
        std::vector<MultidimArray<Complex>> bandCoefficients;
        std::vector<Matrix1D<Real>> bandProjections;
        std::list<std::pair<size_t, Real>> ws;
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
        auto distance2 = distance;
        //const auto match = m_references.matchPcaProjectionBnB(data.bandProjections, distance, data.ws);
        //const auto match = m_references.matchPcaProjection(data.bandProjections, distance);
        const auto match = m_references.matchPcaProjectionKDTree(data.bandProjections, distance);
        assert(match == m_references.matchPcaProjection(data.bandProjections, distance2));
        assert(distance == distance2);
        if(match < m_references.getImageCount()) {
            classification = m_referenceData[match];
            classification.setDistance(distance);
        }
    };

    processRowsInParallel(m_mdExperimental, func, threadData.size());
}

template<typename T>
void ProgAlignSpectral<T>::classifyExperimentalShift() {
    struct ThreadData {
        Image<Real> image;
        FourierTransformer fourier;
        BandShiftTransformer shifter;
        std::vector<MultidimArray<Complex>> bandCoefficients;
        std::vector<Matrix1D<Real>> bandProjections;
    };

    // Create a lambda to run in parallel
    std::vector<ThreadData> threadData(nThreads);
    const auto func = [this, &threadData] (size_t threadId, size_t i, const MDRowVec& row) {
        auto& data = threadData[threadId];

        // Read an image
        const FileName& fnImage = row.getValue<String>(MDL_IMAGE);
        data.image.read(fnImage);
        
        // Compute the fourier transform of the image
        data.fourier.setReal(data.image());
        data.fourier.FourierTransform();

        // Flatten the bands
        m_bandMap.flatten(data.fourier.fFourier, data.bandCoefficients);

        // Compare against all the references for each translation
        data.shifter.forEachInPlaneTranslation(
            data.bandCoefficients, m_translations,
            [this, &data, i] (const std::vector<MultidimArray<Complex>>& bandCoefficients, double sx, double sy) {
                // Project the coefficients to the PCA
                project(m_bases, bandCoefficients, data.bandProjections);

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





template<typename T>
void ProgAlignSpectral<T>::removeFourierSymmetry(MultidimArray<Real>& spectrum) const {
    if(XSIZE(spectrum) > XSIZE(m_bandMap.getBands())) {
        spectrum.resize(
            NSIZE(spectrum), ZSIZE(spectrum), YSIZE(spectrum), 
            XSIZE(m_bandMap.getBands())
        );
    }
}

template<typename T>
void ProgAlignSpectral<T>::multiplyBases(   std::vector<Matrix2D<Real>>& bases,
                                            const MultidimArray<Real>& spectrum ) const
{
    // Read the spectrum to the real part of a complex array
    std::vector<MultidimArray<Complex>> scales;
    m_bandMap.flatten(spectrum, scales);

    Matrix1D<Real> alias;
    for(size_t i = 0; i < bases.size(); ++i) {
        auto& scale = scales[i];
        auto& basis = bases[i];

        // Modify the scale to have the same value in real and complex
        // components (scaling factor when multiplying elementwise)
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(scale) {
            auto& elem = DIRECT_MULTIDIM_ELEM(scale, n);
            elem.imag(elem.real());
        }

        // Reinterpret the scale as a real array
        aliasComplexElements(scale, alias);

        // Scale accordingly
        multiplyToAllColumns(basis, alias);
    } 
}

template<typename T>
void ProgAlignSpectral<T>::updateRow(   MDRowVec& row, 
                                        const ReferenceMetadata& data ) const 
{
    // Obtain the metadata
    const auto refRow = m_mdReference.getRowVec(data.getRowId());
    const auto expRow = m_mdExperimental.getRowVec(row.getValue<size_t>(MDL_OBJID));

    // Calculate the values to be written
    const auto rot = refRow.template getValue<double>(MDL_ANGLE_ROT);
    const auto tilt = refRow.template getValue<double>(MDL_ANGLE_TILT);
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
    row.setValue(MDL_IMAGE_REF, refRow.template getValue<std::string>(MDL_IMAGE));
}



template<typename T>
template<typename F>
void ProgAlignSpectral<T>::processRowsInParallel(   const MetaDataVec& md, 
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





template<typename T>
void ProgAlignSpectral<T>::getProjectionSizes(  const std::vector<Matrix2D<Real>>& bases, 
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

template<typename T>
size_t ProgAlignSpectral<T>::getBatchSize(  size_t memorySize, 
                                            size_t imageProjSize, 
                                            size_t nTransform ) 
{
    return std::max(memorySize / (imageProjSize * nTransform), 1UL);
}

template<typename T>
std::vector<typename ProgAlignSpectral<T>::BandShiftFilters::Shift> 
ProgAlignSpectral<T>::computeTranslationFiltersRectangle(   const size_t nx, 
                                                            const size_t ny,
                                                            const size_t nTranslations,
                                                            const double maxShift )
{
    std::vector<typename BandShiftFilters::Shift> result; 
    result.reserve(nTranslations*nTranslations);
    
    if(nTranslations > 1) {
        const auto n = std::max(nx, ny);
        const auto maxRadius = n*maxShift;
        const auto step = 2*maxRadius / (nTranslations - 1);
        
        // Create a grid
        for (size_t i = 0; i < nTranslations; ++i) {
            const auto dx = i*step - maxRadius;
            for (size_t j = 0; j < nTranslations; ++j) {
                const auto dy = j*step - maxRadius;
                result.push_back(typename BandShiftFilters::Shift{dx, dy});
            }
        }
    } else {
        // Only one translation, use the identity filter
        result.emplace_back();
    }

    return result;
}

template<typename T>
std::vector<typename ProgAlignSpectral<T>::BandShiftFilters::Shift> 
ProgAlignSpectral<T>::computeTranslationFiltersSunflower(   const size_t nx, 
                                                            const size_t ny,
                                                            const size_t nTranslations,
                                                            const double maxShift )
{
    std::vector<typename BandShiftFilters::Shift> result; 
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
        for (size_t i = 0; i < nTranslations; ++i) {
            const auto r = maxRadius * std::sqrt(i+0.5) / std::sqrt(nTranslations-0.5);
            const auto theta = M_2_PI * i / PHI2;
            const auto point = std::polar(r, theta);
            result.push_back(typename BandShiftFilters::Shift{point.real(), point.imag()});
        }
    } else {
        // Only one translation, use the identity filter
        result.emplace_back();
    }

    return result;
}

template<typename T>
void ProgAlignSpectral<T>::project( const std::vector<Matrix2D<Real>>& bases,
                                    const std::vector<MultidimArray<Complex>>& bands,
                                    std::vector<Matrix1D<Real>>& projections )
{
    assert(bases.size() == bands.size());
    projections.resize(bases.size());

    Matrix1D<Real> realBand; // The other one was fake
    for(size_t i = 0; i < bands.size(); ++i) {
        // Alias the band as interleaved real values
        aliasComplexElements(const_cast<MultidimArray<Complex>&>(bands[i]), realBand);

        // Project
        assert(MAT_YSIZE(bases[i]) == VEC_XSIZE(realBand));
        matrixOperation_Atx(bases[i], realBand, projections[i]);
    }
}

template<typename T>
void ProgAlignSpectral<T>::composeComplex(  const MultidimArray<Real>& re, 
                                            const MultidimArray<Real>& im, 
                                            MultidimArray<Complex>& result )
{
    if(!re.sameShape(im)) REPORT_ERROR(ERR_ARG_INCORRECT, "Input components must have the same size");
    result.resizeNoCopy(re);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(result) {
        DIRECT_MULTIDIM_ELEM(result, n) = {
            DIRECT_MULTIDIM_ELEM(re, n),
            DIRECT_MULTIDIM_ELEM(im, n)
        };
    }
}

template<typename T>
void ProgAlignSpectral<T>::aliasComplexElements(MultidimArray<Complex>& x, 
                                                Matrix1D<Real>& result ) 
{
    result.alias(
        reinterpret_cast<Real*>(MULTIDIM_ARRAY(x)),
        NZYXSIZE(x) * 2UL
    );
}

template<typename T>
double ProgAlignSpectral<T>::standardizeAngle(double angle) {
    return (angle > 180.0) ? (angle - 360.0) : angle;
}

// explicit instantiation
template class ProgAlignSpectral<double>;

}