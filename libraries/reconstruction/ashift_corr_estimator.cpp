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

    // flags
    m_includingBatchFT = false;
    m_includingSingleFT = false;
    m_is_ref_FD_loaded = false;
}

template<typename T>
void AShiftCorrEstimator<T>::release() {
    delete m_settingsInv;
    AShiftEstimator<T>::release();

    AShiftCorrEstimator<T>::setDefault();
}

template<typename T>
void AShiftCorrEstimator<T>::init2D(AlignType type,
        const FFTSettingsNew<T> &dims, size_t maxShift,
        bool includingBatchFT, bool includingSingleFT) {
    AShiftEstimator<T>::init2D(type, dims.sDim(), dims.batch(), maxShift);

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
void AShiftCorrEstimator<T>::check() {
    using memoryUtils::operator "" _GB;

    if (this->m_settingsInv->isForward()) {
        REPORT_ERROR(ERR_VALUE_INCORRECT, "Inverse transform expected");
    }
    if (this->m_settingsInv->fBytesBatch() >= 4_GB) {
       REPORT_ERROR(ERR_VALUE_INCORRECT, "Batch is bigger than max size (4GB)");
    }
    if ((0 != (this->m_settingsInv->sDim().x() % 2))
        || (0 != (this->m_settingsInv->sDim().y() % 2))) {
        // while performing IFT of the correlation, we center the signal using multiplication
        // in the FD. This, however, works only for even signal.
            REPORT_ERROR(ERR_VALUE_INCORRECT,
                    "The X and Y dimensions have to be multiple of two. Crop your signal");
    }
}

template<typename T>
std::vector<T> AShiftCorrEstimator<T>::findMaxAroundCenter(
        const T *correlations,
        const Dimensions &dims,
        size_t maxShift,
        std::vector<Point2D<float>> &shifts) {
    assert(0 == shifts.size());
    assert(nullptr != correlations);

    assert(0 < dims.x());
    assert(0 < dims.y());
    assert(1 == dims.zPadded());
    assert(0 < dims.n());
    assert(0 < maxShift);

    auto result = std::vector<T>();
    shifts.reserve(dims.n());
    result.reserve(dims.n());
    int xHalf = dims.x() / 2;
    int yHalf = dims.y() / 2;

    auto min = std::pair<size_t, size_t>(
        std::max(0, xHalf - (int)maxShift),
        std::max(0, yHalf - (int)maxShift)
    );

    auto max = std::pair<size_t, size_t>(
        std::min((int)dims.x() - 1, xHalf + (int)maxShift),
        std::min((int)dims.y() - 1, yHalf + (int)maxShift)
    );

    size_t maxDistSq = maxShift * maxShift;
    for (size_t n = 0; n < dims.n(); ++n) {
        size_t offsetN = n * dims.xyzPadded();
        // reset values
        float maxX;
        float maxY;
        T val = std::numeric_limits<T>::lowest();
        // iterate through the center
        for (size_t y = min.second; y <= max.second; ++y) {
            size_t offsetY = y * dims.xPadded();
            int logicY = (int)y - yHalf;
            size_t ySq = logicY * logicY;
            for (size_t x = min.first; x <= max.first; ++x) {
                int logicX = (int)x - xHalf;
                // continue if the Euclidean distance is too far
                if ((ySq + (logicX * logicX)) > maxDistSq) continue;
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

// explicit instantiation
template class AShiftCorrEstimator<float>;
template class AShiftCorrEstimator<double>;

} /* namespace Alignment */
