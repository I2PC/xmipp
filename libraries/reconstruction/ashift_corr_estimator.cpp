/***************************************************************************
 *
 * Authors:    David Strelak (davidstrelak@gmail.com)
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
#include "ashift_corr_estimator.h"

namespace Alignment {

template<typename T>
void AShiftCorrEstimator<T>::setDefault() {
    AShiftEstimator<T>::setDefault();

    m_settingsInv = nullptr;
    m_centerSize = 0;

    // host memory
    m_h_centers = nullptr;

    // flags
    m_includingBatchFT = false;
    m_includingSingleFT = false;
    m_is_ref_FD_loaded = false;
}

template<typename T>
void AShiftCorrEstimator<T>::release() {
    delete m_settingsInv;
    // host memory
    delete[] m_h_centers;

    AShiftEstimator<T>::release();

    AShiftCorrEstimator<T>::setDefault();
}

template<typename T>
void AShiftCorrEstimator<T>::init2D(AlignType type,
        const FFTSettingsNew<T> &dims, size_t maxShift,
        bool includingBatchFT, bool includingSingleFT) {
    AShiftEstimator<T>::init2D(type, dims.sDim(), dims.batch(),
            Point2D<size_t>(maxShift, maxShift));

    m_settingsInv = new FFTSettingsNew<T>(dims);
    m_includingBatchFT = includingBatchFT;
    m_includingSingleFT = includingSingleFT;
    m_centerSize = 2 * maxShift + 1;

    check();

    switch (type) {
        case AlignType::OneToN:
            init2DOneToN();
            break;
        default:
            REPORT_ERROR(ERR_NOT_IMPLEMENTED, "This alignment type is not supported yet");
    }
}

template<typename T>
void AShiftCorrEstimator<T>::init2DOneToN() {
    // allocate helper objects
    m_h_centers = new T[m_centerSize * m_centerSize * m_settingsInv->batch()]();
}

template<typename T>
void AShiftCorrEstimator<T>::check() {
    using memoryUtils::operator ""_GB;

    if (this->m_settingsInv->isForward()) {
        REPORT_ERROR(ERR_VALUE_INCORRECT, "Inverse transform expected");
    }
    if (this->m_settingsInv->isInPlace()) {
        REPORT_ERROR(ERR_VALUE_INCORRECT, "In-place transform only supported");
    }
    if (this->m_settingsInv->fBytesBatch() >= (4_GB)) {
       REPORT_ERROR(ERR_VALUE_INCORRECT, "Batch is bigger than max size (4GB)");
    }
    if ((0 == this->m_settingsInv->fDim().size())
        || (0 == this->m_settingsInv->sDim().size())) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "Fourier or Spatial domain dimension is zero (0)");
    }
    if ((m_centerSize > this->m_settingsInv->sDim().x())
        || m_centerSize > this->m_settingsInv->sDim().y()) {
            REPORT_ERROR(ERR_VALUE_INCORRECT, "The maximum shift (and hence the shift area: 2 * shift + 1) "
                "must be sharply smaller than the smallest dimension");
    }
    if ((0 != (this->m_settingsInv->sDim().x() % 2))
        || (0 != (this->m_settingsInv->sDim().y() % 2))) {
        // while performing IFT of the correlation, we center the signal using multiplication
        // in the FD. This, however, works only for even signal.
            REPORT_ERROR(ERR_VALUE_INCORRECT,
                    "The X and Y dimensions have to be multiple of two. Crop your signal");
    }

    switch (this->m_type) {
        case AlignType::OneToN:
            break;
        default:
            REPORT_ERROR(ERR_VALUE_INCORRECT,
               "This type is not supported.");
    }
}


template<typename T>
std::vector<T> AShiftCorrEstimator<T>::findMaxAroundCenter(
        const T *correlations,
        const Dimensions &dims,
        const Point3D<size_t> &maxShift,
        std::vector<Point2D<float>> &shifts) {
    size_t xHalf = dims.x() / 2;
    size_t yHalf = dims.y() / 2;

    assert(0 == shifts.size());
    assert(2 <= dims.x());
    assert(2 <= dims.y());
    assert(1 == dims.z());
    assert(nullptr != correlations);
    assert(maxShift.x <= xHalf);
    assert(maxShift.y <= yHalf);
    assert(0 < maxShift.x);
    assert(0 < maxShift.y);
    assert( ! dims.isPadded());

    auto result = std::vector<T>();
    shifts.reserve(dims.n());
    result.reserve(dims.n());

    // FIXME DS implement support for Z dimension
    size_t maxDist = maxShift.x * maxShift.y;
    for (size_t n = 0; n < dims.n(); ++n) {
        size_t offsetN = n * dims.xyz();
        // reset values
        float maxX;
        float maxY;
        T val = std::numeric_limits<T>::lowest();
        // iterate through the center
        for (size_t y = yHalf - maxShift.y; y <= yHalf + maxShift.y; ++y) {
            size_t offsetY = y * dims.x();
            int logicY = (int)y - yHalf;
            T ySq = logicY * logicY;
            for (size_t x = xHalf - maxShift.x; x <= xHalf + maxShift.x; ++x) {
                int logicX = (int)x - yHalf;
                // continue if the Euclidean distance is too far
                if ((ySq + (logicX * logicX)) > maxDist) continue;
                // get current value and update, if necessary
                T tmp = correlations[offsetN + offsetY + x];
                if (tmp > val) {
                    val = tmp;
                    maxX = logicX;
                    maxY = logicY;
                }
            }
        }
        // store results
        result.push_back(val);
        shifts.emplace_back(maxX, maxY);
    }
    return result;
}

template<typename T>
std::vector<T> AShiftCorrEstimator<T>::findMaxAroundCenter(
        const T *correlations,
        const Dimensions &dims,
        size_t maxShift,
        std::vector<Point2D<float>> &shifts) {
    return findMaxAroundCenter(correlations, dims, Point3D<size_t>(maxShift, maxShift, maxShift), shifts);
}

template<typename T>
std::vector<T> AShiftCorrEstimator<T>::findMaxAroundCenter(
        const T *correlations,
        const Dimensions &dims,
        const Point2D<size_t> &maxShift,
        std::vector<Point2D<float>> &shifts) {
    return findMaxAroundCenter(correlations, dims, Point3D<size_t>(maxShift.x, maxShift.y, 1), shifts);
}

// explicit instantiation
template class AShiftCorrEstimator<float>;
template class AShiftCorrEstimator<double>;

} /* namespace Alignment */
